//! Multi-dimensional scoring, speed estimation and fit analysis.
//!
//! LLM inference is memory-bandwidth bound: every generated token requires
//! reading the active weights once. Throughput is therefore estimated as
//! `bandwidth / bytes_read_per_token * efficiency`, falling back to a
//! per-backend constant when the GPU's bandwidth is unknown.

use serde::Serialize;
use std::fmt;

use crate::hardware::Hardware;
use crate::models::{Model, Quant, UseCase};

/// Assumed system-memory bandwidth (GB/s) for CPU-resident weights.
const CPU_MEM_BANDWIDTH_GB_S: f64 = 60.0;

/// System memory bandwidth as a fraction of a typical discrete GPU's, used by
/// the fallback throughput model when the GPU is unrecognised.
const CPU_BANDWIDTH_RATIO: f64 = 0.2;

/// Fraction of VRAM a model may occupy and still count as a comfortable fit.
const VRAM_COMFORTABLE: f64 = 0.95;

/// Extra headroom on top of the estimate required for a `Perfect` verdict.
const RECOMMENDED_HEADROOM: f64 = 1.15;

/// Throughput (tok/s) mapped to a speed score of 100. Set at roughly twice
/// reading speed: past this point extra tokens per second buy nothing, and
/// letting them keep earning points would trade away quantization quality for
/// speed nobody can use.
const SPEED_SCORE_CEILING: f64 = 40.0;

/// Curve exponent for the speed score. Below 1 the curve is concave, so the
/// first tokens per second matter far more than the last.
const SPEED_SCORE_EXPONENT: f64 = 0.5;

/// Throughput below which a model is not really usable interactively. Scores
/// are scaled down towards zero underneath it, so a placement that technically
/// fits but crawls cannot out-rank one that actually runs.
const MIN_USABLE_TPS: f64 = 3.0;

/// Parameter count (billions) mapped to a size score of 100.
const QUALITY_SIZE_CEILING: f64 = 70.0;

/// Shortest context worth proposing.
const MIN_CONTEXT: u32 = 4_096;

// ---------------------------------------------------------------------------
// Tunables (TUI "Advanced Config")
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SpeedConfig {
    /// Global efficiency factor for the bandwidth-based estimate.
    pub efficiency: f64,
    pub gpu_factor: f64,
    pub cpu_offload_factor: f64,
    pub moe_offload_factor: f64,
    pub tensor_parallel_factor: f64,
    pub cpu_only_factor: f64,
    /// Upper bound on the context used for memory estimation.
    pub context_cap: Option<u32>,
}

impl Default for SpeedConfig {
    fn default() -> Self {
        SpeedConfig {
            efficiency: 0.55,
            gpu_factor: 1.0,
            cpu_offload_factor: 0.5,
            moe_offload_factor: 0.8,
            tensor_parallel_factor: 0.9,
            cpu_only_factor: 0.3,
            context_cap: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Run modes and fit levels
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RunMode {
    /// Everything resident in VRAM.
    Gpu,
    /// MoE with inactive experts offloaded to system RAM.
    Moe,
    /// Dense weights partially offloaded to system RAM.
    CpuGpu,
    /// No usable GPU; weights live in system RAM.
    Cpu,
}

impl RunMode {
    pub fn label(self) -> &'static str {
        match self {
            RunMode::Gpu => "GPU",
            RunMode::Moe => "MoE",
            RunMode::CpuGpu => "CPU+GPU",
            RunMode::Cpu => "CPU",
        }
    }

    pub fn parse(s: &str) -> Option<RunMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gpu" => Some(RunMode::Gpu),
            "moe" => Some(RunMode::Moe),
            "cpu+gpu" | "hybrid" => Some(RunMode::CpuGpu),
            "cpu" => Some(RunMode::Cpu),
            _ => None,
        }
    }
}

impl fmt::Display for RunMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum FitLevel {
    TooTight,
    Marginal,
    Good,
    Perfect,
}

impl FitLevel {
    pub fn label(self) -> &'static str {
        match self {
            FitLevel::Perfect => "Perfect",
            FitLevel::Good => "Good",
            FitLevel::Marginal => "Marginal",
            FitLevel::TooTight => "Too Tight",
        }
    }

    pub fn parse(s: &str) -> Option<FitLevel> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace(['_', '-', ' '], "")
            .as_str()
        {
            "perfect" => Some(FitLevel::Perfect),
            "good" => Some(FitLevel::Good),
            "marginal" => Some(FitLevel::Marginal),
            "tootight" => Some(FitLevel::TooTight),
            _ => None,
        }
    }

    pub fn is_runnable(self) -> bool {
        self != FitLevel::TooTight
    }
}

impl fmt::Display for FitLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Scores {
    pub quality: f64,
    pub speed: f64,
    pub fit: f64,
    pub context: f64,
    pub composite: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FitResult {
    pub model_id: String,
    pub name: String,
    pub provider: String,
    pub params_b: f64,
    pub active_params_b: Option<f64>,
    pub use_case: UseCase,
    pub license: String,
    pub released: String,
    pub gguf: bool,
    pub ollama: Option<String>,
    pub max_context: u32,

    pub quant: Quant,
    pub mode: RunMode,
    pub fit: FitLevel,
    /// Context length the estimate was made at.
    pub context: u32,
    /// Total memory the model needs in this configuration.
    pub required_gb: f64,
    /// Memory that must be resident on the accelerator.
    pub resident_gb: f64,
    /// Percentage of the constraining memory pool that is used.
    pub mem_percent: f64,
    /// Size of the weights alone — what the download costs and what the file
    /// occupies on disk. Excludes the KV cache and runtime overhead, which
    /// exist only while the model is loaded.
    pub download_gb: f64,
    pub tokens_per_second: f64,
    pub scores: Scores,
}

impl FitResult {
    pub fn is_runnable(&self) -> bool {
        self.fit.is_runnable()
    }

    /// True when the placement had to run the model below its native context.
    ///
    /// Worth surfacing: a model advertised at 128k that only fits at 8k here
    /// is a different tool than the one on the model card.
    pub fn context_is_reduced(&self) -> bool {
        self.context < self.max_context
    }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Context lengths to try, longest first.
///
/// On long-context models the KV cache dwarfs the weights — Qwen3-4B at its
/// full 256k context needs ~36 GB of cache against 2.4 GB of Q4 weights — so
/// full and half context alone would push almost every modern model into CPU
/// offload. Below half context the ladder steps through the context lengths
/// people actually serve at. Running short is scored honestly: `context_score`
/// penalises whatever the model ends up placed at.
fn context_ladder(full: u32) -> Vec<u32> {
    const RUNGS: [u32; 5] = [65_536, 32_768, 16_384, 8_192, 4_096];
    let mut contexts = vec![full];
    let half = full / 2;
    if half >= MIN_CONTEXT {
        contexts.push(half);
    }
    for rung in RUNGS {
        if rung < half && rung >= MIN_CONTEXT {
            contexts.push(rung);
        }
    }
    contexts.dedup();
    contexts
}

/// Candidate placement of a model, before scoring.
struct Placement {
    quant: Quant,
    mode: RunMode,
    fit: FitLevel,
    context: u32,
    required_gb: f64,
    resident_gb: f64,
    mem_percent: f64,
}

impl Placement {
    /// Share of the footprint that stays on the accelerator.
    fn gpu_fraction(&self) -> f64 {
        if self.required_gb <= 0.0 {
            return 1.0;
        }
        (self.resident_gb / self.required_gb).clamp(0.0, 1.0)
    }
}

/// Analyse one model against the given hardware.
pub fn analyze(model: &Model, hw: &Hardware, target: UseCase, cfg: &SpeedConfig) -> FitResult {
    let (placement, tps, scores) = place(model, hw, target, cfg);

    FitResult {
        model_id: model.id.clone(),
        name: model.name.clone(),
        provider: model.provider.clone(),
        params_b: model.params_b,
        active_params_b: model.active_params_b,
        use_case: model.use_case,
        license: model.license.clone(),
        released: model.released.clone(),
        gguf: model.gguf,
        ollama: model.ollama.clone(),
        max_context: model.context_length,
        quant: placement.quant,
        mode: placement.mode,
        fit: placement.fit,
        context: placement.context,
        required_gb: placement.required_gb,
        resident_gb: placement.resident_gb,
        mem_percent: placement.mem_percent,
        download_gb: model.weights_gb(placement.quant),
        tokens_per_second: tps,
        scores,
    }
}

/// Analyse every model and sort best-first, with unrunnable models last.
pub fn analyze_all(
    models: &[Model],
    hw: &Hardware,
    target: UseCase,
    cfg: &SpeedConfig,
) -> Vec<FitResult> {
    let mut results: Vec<FitResult> = models.iter().map(|m| analyze(m, hw, target, cfg)).collect();
    sort_by_score(&mut results);
    results
}

pub fn sort_by_score(results: &mut [FitResult]) {
    results.sort_by(|a, b| {
        b.is_runnable()
            .cmp(&a.is_runnable())
            .then(b.scores.composite.total_cmp(&a.scores.composite))
            .then(a.name.cmp(&b.name))
    });
}

/// Pick the best (run mode, context, quantization) triple for this model.
///
/// Every viable combination is scored and the highest composite wins, rather
/// than walking a fixed priority order. That keeps one rule instead of three:
/// the trade-offs between a longer context, a higher quantization and a faster
/// run mode are already expressed by the four score dimensions and the
/// use-case weights, so the placement optimises the same objective the ranking
/// does. A fixed order would, for instance, take Q2_K at full context over
/// Q4_K_M at half — quality the score model says is not worth the context.
fn place(
    model: &Model,
    hw: &Hardware,
    target: UseCase,
    cfg: &SpeedConfig,
) -> (Placement, f64, Scores) {
    let full_context = cfg
        .context_cap
        .map_or(model.context_length, |cap| cap.min(model.context_length));
    let contexts = context_ladder(full_context);

    let vram = hw.total_vram_gb();
    let ram = hw.usable_ram_gb();

    let mut modes = Vec::with_capacity(4);
    if hw.has_gpu() {
        modes.push(RunMode::Gpu);
        if model.is_moe() {
            modes.push(RunMode::Moe);
        }
        modes.push(RunMode::CpuGpu);
    }
    modes.push(RunMode::Cpu);

    let mut best: Option<(Placement, f64, Scores)> = None;
    for mode in modes {
        for &context in &contexts {
            for quant in Quant::HIERARCHY {
                let Some(placement) = try_place(model, mode, quant, context, vram, ram) else {
                    continue;
                };
                let tps = estimate_tps(model, hw, quant, mode, placement.gpu_fraction(), cfg);
                let scores = score(model, &placement, tps, target);
                let better = best
                    .as_ref()
                    .is_none_or(|(_, _, best_scores)| scores.composite > best_scores.composite);
                if better {
                    best = Some((placement, tps, scores));
                }
            }
        }
    }

    if let Some(found) = best {
        return found;
    }

    // Nothing fits anywhere: report the cheapest configuration as Too Tight.
    let quant = Quant::Q2K;
    let context = *contexts.last().unwrap();
    let required = model.total_memory_gb(quant, context);
    let pool = if hw.has_gpu() { vram + ram } else { ram };
    let placement = Placement {
        quant,
        mode: if hw.has_gpu() {
            RunMode::CpuGpu
        } else {
            RunMode::Cpu
        },
        fit: FitLevel::TooTight,
        context,
        required_gb: required,
        resident_gb: required,
        mem_percent: if pool > 0.0 {
            required / pool * 100.0
        } else {
            100.0
        },
    };
    let tps = estimate_tps(
        model,
        hw,
        quant,
        placement.mode,
        placement.gpu_fraction(),
        cfg,
    );
    let scores = score(model, &placement, tps, target);
    (placement, tps, scores)
}

fn try_place(
    model: &Model,
    mode: RunMode,
    quant: Quant,
    context: u32,
    vram: f64,
    ram: f64,
) -> Option<Placement> {
    let total = model.total_memory_gb(quant, context);

    match mode {
        RunMode::Gpu => {
            if total > vram * VRAM_COMFORTABLE {
                return None;
            }
            let percent = total / vram * 100.0;
            let fit = if total * RECOMMENDED_HEADROOM <= vram {
                FitLevel::Perfect
            } else if total <= vram * 0.90 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            };
            Some(Placement {
                quant,
                mode,
                fit,
                context,
                required_gb: total,
                resident_gb: total,
                mem_percent: percent,
            })
        }
        RunMode::Moe => {
            // Active experts stay in VRAM, the rest streams from system RAM.
            let resident = model.moe_resident_gb(quant, context);
            if resident > vram * VRAM_COMFORTABLE || total > vram + ram {
                return None;
            }
            // MoE offloading tops out at Good: it is never as clean as pure GPU.
            let fit = if resident * RECOMMENDED_HEADROOM <= vram {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            };
            Some(Placement {
                quant,
                mode,
                fit,
                context,
                required_gb: total,
                resident_gb: resident,
                mem_percent: resident / vram * 100.0,
            })
        }
        RunMode::CpuGpu => {
            if total > vram + ram {
                return None;
            }
            // Hybrid execution also tops out at Good.
            let gpu_fraction = (vram / total).min(1.0);
            let fit = if gpu_fraction >= 0.5 {
                FitLevel::Good
            } else {
                FitLevel::Marginal
            };
            Some(Placement {
                quant,
                mode,
                fit,
                context,
                required_gb: total,
                resident_gb: vram.min(total),
                mem_percent: total / (vram + ram) * 100.0,
            })
        }
        RunMode::Cpu => {
            if total > ram {
                return None;
            }
            // CPU-only always caps at Marginal, however much RAM is free.
            Some(Placement {
                quant,
                mode,
                fit: FitLevel::Marginal,
                context,
                required_gb: total,
                resident_gb: total,
                mem_percent: total / ram * 100.0,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Speed
// ---------------------------------------------------------------------------

/// Estimated generation throughput in tokens per second.
///
/// `gpu_fraction` is the share of the footprint that stays on the accelerator;
/// the remainder is read across the much slower system memory bus.
pub fn estimate_tps(
    model: &Model,
    hw: &Hardware,
    quant: Quant,
    mode: RunMode,
    gpu_fraction: f64,
    cfg: &SpeedConfig,
) -> f64 {
    // Bytes read per token: only the active weights for MoE models.
    let read_gb = if model.is_moe() {
        model.active_weights_gb(quant)
    } else {
        model.weights_gb(quant)
    }
    .max(0.01);

    let resident = gpu_fraction.clamp(0.0, 1.0);
    let base = match (mode, hw.primary_bandwidth()) {
        // Known GPU: use its real memory bandwidth.
        (RunMode::Gpu | RunMode::Moe, Some(bw)) => bw / read_gb * cfg.efficiency,
        (RunMode::CpuGpu, Some(bw)) => {
            let blended = resident * bw + (1.0 - resident) * CPU_MEM_BANDWIDTH_GB_S;
            blended / read_gb * cfg.efficiency
        }
        (RunMode::Cpu, _) => CPU_MEM_BANDWIDTH_GB_S / read_gb * cfg.efficiency,
        // Unknown GPU: per-backend constant, scaled by how much of the model
        // actually stays resident. System memory is roughly a fifth as fast as
        // a discrete GPU's, so spilling weights costs most of the throughput.
        (mode, None) => {
            let raw = hw.backend.speed_constant() / model.active_params().max(0.1)
                * quant.speed_multiplier();
            match mode {
                RunMode::CpuGpu => raw * (resident + (1.0 - resident) * CPU_BANDWIDTH_RATIO),
                _ => raw,
            }
        }
    };

    let mode_factor = match mode {
        RunMode::Gpu => {
            if hw.gpus.len() > 1 {
                cfg.gpu_factor * cfg.tensor_parallel_factor
            } else {
                cfg.gpu_factor
            }
        }
        RunMode::Moe => cfg.moe_offload_factor,
        RunMode::CpuGpu => cfg.cpu_offload_factor,
        RunMode::Cpu => cfg.cpu_only_factor,
    };

    (base * mode_factor).max(0.0)
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Per-use-case weights for `(quality, speed, fit, context)`.
fn weights(target: UseCase) -> (f64, f64, f64, f64) {
    match target {
        UseCase::General => (0.40, 0.25, 0.20, 0.15),
        UseCase::Coding => (0.35, 0.20, 0.15, 0.30),
        UseCase::Reasoning => (0.55, 0.15, 0.15, 0.15),
        UseCase::Chat => (0.30, 0.35, 0.20, 0.15),
        UseCase::Multimodal => (0.40, 0.25, 0.20, 0.15),
        UseCase::Embedding => (0.35, 0.35, 0.20, 0.10),
    }
}

/// How well a model's own specialty serves the requested use case.
fn use_case_affinity(model: UseCase, target: UseCase) -> f64 {
    if model == target {
        return 1.0;
    }
    // Embedding models are not interchangeable with generative ones.
    if model == UseCase::Embedding || target == UseCase::Embedding {
        return 0.30;
    }
    match (model, target) {
        // A generalist stands in for a specialist at a real discount: the gap
        // between a general model and a code-tuned one of the same size is
        // large on coding benchmarks. The reverse costs less — a code model is
        // still a competent general model.
        (UseCase::General, _) => 0.70,
        (_, UseCase::General) => 0.85,
        (UseCase::Reasoning, UseCase::Coding) | (UseCase::Coding, UseCase::Reasoning) => 0.80,
        (UseCase::Chat, _) | (_, UseCase::Chat) => 0.75,
        (UseCase::Multimodal, _) => 0.70,
        (_, UseCase::Multimodal) => 0.45,
        _ => 0.70,
    }
}

fn quality_score(model: &Model, quant: Quant, target: UseCase) -> f64 {
    // For MoE models the geometric mean of total and active parameters tracks
    // observed quality better than either number alone.
    let effective_params = match model.active_params_b {
        Some(active) => (model.params_b * active).sqrt(),
        None => model.params_b,
    };
    // `ln_1p` rather than `(1.0 + x).ln()`: the latter loses precision for the
    // sub-billion models at the bottom of the catalog, where `1.0 + x` rounds
    // away most of the value being measured.
    let size = 100.0 * effective_params.ln_1p() / QUALITY_SIZE_CEILING.ln_1p();
    let tier = 0.80 + f64::from(model.quality_tier.clamp(1, 5)) * 0.04;
    let score =
        size.min(100.0) * tier * quant.quality_factor() * use_case_affinity(model.use_case, target);
    score.clamp(0.0, 100.0)
}

/// Throughput score.
///
/// [`COMFORTABLE_TPS`] — roughly twice reading speed — scores
/// [`COMFORTABLE_SCORE`] rather than a full 100. Past it the curve keeps
/// rising, but logarithmically: the difference between 20 and 40 tok/s is felt,
/// the difference between 80 and 100 is not.
///
/// The earlier version clamped at 40 tok/s, which put 43% of the catalog —
/// and 15 of the top 20 — on exactly 100. A dimension that returns the same
/// number for most of the field cannot contribute to the ranking, so the
/// clamp is now a soft knee.
fn speed_score(tps: f64) -> f64 {
    if tps <= 0.0 {
        return 0.0;
    }
    if tps <= COMFORTABLE_TPS {
        COMFORTABLE_SCORE * (tps / COMFORTABLE_TPS).powf(SPEED_SCORE_EXPONENT)
    } else {
        let headroom = (tps / COMFORTABLE_TPS).ln() / SPEED_SATURATION.ln();
        (COMFORTABLE_SCORE + (100.0 - COMFORTABLE_SCORE) * headroom).min(100.0)
    }
}

/// Memory-efficiency score. Well under half the pool means the hardware is
/// being left on the table — a bigger model or a higher quantization would
/// have fit — so it is scored down. The plateau runs all the way to 95%
/// because filling VRAM is the point; only genuinely cutting it fine is
/// penalised. A tighter plateau would quietly favour a smaller quantization
/// over a better one purely for using less memory.
/// Headroom score: how much room the placement leaves.
///
/// This dimension answers "how tight is this?", not "did you use the card
/// well". The latter is already carried by quality and context — a model that
/// fills VRAM with better weights and a longer window scores higher there —
/// and rewarding memory use twice was what produced the old plateau.
///
/// Headroom is real information: a placement at 95% will run out when the
/// context actually fills or the desktop compositor asks for VRAM back, while
/// one at 70% will not. [`COMFORTABLE_HEADROOM`] percent free is treated as
/// all the room anyone needs; below that the score falls away steeply.
fn fit_score(mem_percent: f64) -> f64 {
    let headroom = (100.0 - mem_percent).max(0.0);
    let ratio = (headroom / COMFORTABLE_HEADROOM).min(1.0);
    100.0 * ratio.powf(HEADROOM_EXPONENT)
}

/// Context capacity score. Sub-target context is penalised on a square-root
/// curve: the marginal value of context falls off, so 16k against a 64k target
/// is worth half the points rather than a quarter.
fn context_score(context: u32, target: UseCase) -> f64 {
    let ratio = f64::from(context) / target.target_context();
    if ratio >= 1.0 {
        100.0
    } else {
        (100.0 * ratio.sqrt()).clamp(0.0, 100.0)
    }
}

fn score(model: &Model, placement: &Placement, tps: f64, target: UseCase) -> Scores {
    let quality = quality_score(model, placement.quant, target);
    let speed = speed_score(tps);
    let fit = fit_score(placement.mem_percent);
    let context = context_score(placement.context, target);

    let (wq, ws, wf, wc) = weights(target);
    let composite = if placement.fit == FitLevel::TooTight {
        0.0
    } else {
        let weighted = quality * wq + speed * ws + fit * wf + context * wc;
        // A placement that fits but generates a token every few seconds is not
        // a usable answer, however good its quality and context scores are.
        let usability = (tps / MIN_USABLE_TPS).clamp(0.0, 1.0);
        weighted * usability
    };

    Scores {
        quality,
        speed,
        fit,
        context,
        composite,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{Backend, Hardware};
    use crate::models::ModelDb;

    fn hw(vram: f64, ram: f64) -> Hardware {
        let mut h = Hardware {
            cpu_brand: "Test CPU".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            arch: "x86_64".into(),
            total_ram_gb: ram,
            available_ram_gb: ram,
            gpus: Vec::new(),
            backend: Backend::CpuX86,
            simulated: true,
        };
        if vram > 0.0 {
            h.set_vram(vram);
        }
        h
    }

    #[test]
    fn small_model_fits_perfectly_on_a_big_gpu() {
        let db = ModelDb::embedded();
        let m = db.find("meta-llama/Llama-3.2-3B-Instruct").unwrap();
        let r = analyze(
            m,
            &hw(24.0, 64.0),
            UseCase::General,
            &SpeedConfig::default(),
        );
        assert_eq!(r.mode, RunMode::Gpu);
        assert_eq!(r.fit, FitLevel::Perfect);
        assert!(r.tokens_per_second > 0.0);
    }

    #[test]
    fn huge_model_does_not_fit_a_small_machine() {
        let db = ModelDb::embedded();
        let m = db.find("meta-llama/Llama-3.3-70B-Instruct").unwrap();
        let r = analyze(m, &hw(8.0, 16.0), UseCase::General, &SpeedConfig::default());
        assert_eq!(r.fit, FitLevel::TooTight);
        assert_eq!(r.scores.composite, 0.0);
    }

    #[test]
    fn quantization_steps_down_until_it_fits() {
        let db = ModelDb::embedded();
        let m = db.find("Qwen/Qwen2.5-14B-Instruct").unwrap();
        let big = analyze(
            m,
            &hw(48.0, 64.0),
            UseCase::General,
            &SpeedConfig::default(),
        );
        let small = analyze(
            m,
            &hw(10.0, 32.0),
            UseCase::General,
            &SpeedConfig::default(),
        );
        assert_eq!(big.quant, Quant::Q8_0);
        assert!(
            small.quant.bits_per_weight() < big.quant.bits_per_weight(),
            "tight VRAM should force a smaller quantization, got {}",
            small.quant
        );
    }

    #[test]
    fn cpu_only_machine_caps_at_marginal() {
        let db = ModelDb::embedded();
        let m = db.find("meta-llama/Llama-3.2-1B-Instruct").unwrap();
        let r = analyze(m, &hw(0.0, 32.0), UseCase::General, &SpeedConfig::default());
        assert_eq!(r.mode, RunMode::Cpu);
        assert_eq!(r.fit, FitLevel::Marginal);
    }

    #[test]
    fn moe_offloading_beats_giving_up() {
        let db = ModelDb::embedded();
        let m = db.find("Qwen/Qwen3-30B-A3B").unwrap();
        // 30B total does not fit 12 GB of VRAM, but 3.3B active does.
        let r = analyze(
            m,
            &hw(12.0, 64.0),
            UseCase::Reasoning,
            &SpeedConfig::default(),
        );
        assert_eq!(r.mode, RunMode::Moe);
        assert!(r.resident_gb < r.required_gb);
        assert!(r.is_runnable());
    }

    #[test]
    fn gpu_placement_is_preferred_over_full_context_on_cpu() {
        let db = ModelDb::embedded();
        let m = db.find("Qwen/Qwen3-4B-Instruct-2507").unwrap();
        // 262k context never fits 8 GB, but half context on the GPU does.
        let r = analyze(m, &hw(8.0, 32.0), UseCase::General, &SpeedConfig::default());
        assert_eq!(r.mode, RunMode::Gpu);
        assert!(r.context < m.context_length);
    }

    #[test]
    fn download_size_is_the_weights_alone() {
        let db = ModelDb::embedded();
        let m = db.find("meta-llama/Llama-3.1-8B-Instruct").unwrap();
        let r = analyze(
            m,
            &hw(24.0, 64.0),
            UseCase::General,
            &SpeedConfig::default(),
        );
        // The download is the weight file; the KV cache and runtime overhead
        // only exist once the model is loaded, so they must not be counted.
        assert!((r.download_gb - m.weights_gb(r.quant)).abs() < 1e-9);
        assert!(
            r.download_gb < r.required_gb,
            "download {:.2} should be under the {:.2} GB needed to run it",
            r.download_gb,
            r.required_gb
        );
    }

    #[test]
    fn reduced_context_is_flagged() {
        let db = ModelDb::embedded();
        let m = db.find("Qwen/Qwen3-4B-Instruct-2507").unwrap();
        // 262k never fits 8 GB, so the placement runs short and says so.
        let tight = analyze(m, &hw(8.0, 32.0), UseCase::General, &SpeedConfig::default());
        assert!(tight.context_is_reduced());
        assert!(tight.context < tight.max_context);

        let roomy = analyze(
            m,
            &hw(160.0, 256.0),
            UseCase::General,
            &SpeedConfig::default(),
        );
        assert!(!roomy.context_is_reduced(), "full context should fit here");
    }

    #[test]
    fn ranking_puts_unrunnable_models_last() {
        let db = ModelDb::embedded();
        let ranked = analyze_all(
            &db.models,
            &hw(8.0, 16.0),
            UseCase::General,
            &SpeedConfig::default(),
        );
        let first_unrunnable = ranked.iter().position(|r| !r.is_runnable());
        if let Some(idx) = first_unrunnable {
            assert!(ranked[idx..].iter().all(|r| !r.is_runnable()));
        }
    }

    fn rank_of(results: &[FitResult], id: &str) -> usize {
        results
            .iter()
            .position(|r| r.model_id == id)
            .expect("model present in results")
    }

    #[test]
    fn use_case_weighting_promotes_specialists() {
        let db = ModelDb::embedded();
        let machine = hw(24.0, 64.0);
        let general = analyze_all(
            &db.models,
            &machine,
            UseCase::General,
            &SpeedConfig::default(),
        );
        let coding = analyze_all(
            &db.models,
            &machine,
            UseCase::Coding,
            &SpeedConfig::default(),
        );

        let coder = "Qwen/Qwen2.5-Coder-32B-Instruct";
        assert!(
            rank_of(&coding, coder) < rank_of(&general, coder),
            "a coder model should rank higher under a coding query"
        );
        assert!(
            coding[..5].iter().any(|r| r.use_case == UseCase::Coding),
            "a coding query should surface a coding model in the top 5, got {:?}",
            coding[..5]
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn reasoning_weighting_promotes_reasoning_models() {
        let db = ModelDb::embedded();
        let machine = hw(24.0, 64.0);
        let general = analyze_all(
            &db.models,
            &machine,
            UseCase::General,
            &SpeedConfig::default(),
        );
        let reasoning = analyze_all(
            &db.models,
            &machine,
            UseCase::Reasoning,
            &SpeedConfig::default(),
        );
        let r1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B";
        assert!(rank_of(&reasoning, r1) < rank_of(&general, r1));
    }
}

// ---------------------------------------------------------------------------
// Hardware Planning (inverse fit analysis)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct HardwarePlan {
    pub model_name: String,
    pub params_b: f64,
    pub context_length: u32,
    pub quantization: Quant,
    /// The weight file itself — what the download costs.
    pub weights_gb: f64,
    /// KV cache at this context, which exists only while the model is loaded.
    pub kv_cache_gb: f64,
    pub min_vram_gb: f64,
    pub recommended_vram_gb: f64,
    pub min_ram_gb: f64,
    pub tps_gpu: f64,
    pub tps_cpu: f64,
    pub viable_modes: Vec<&'static str>,
}

/// Plan hardware requirements for a model with a given configuration.
pub fn plan(model: &Model, quant: Quant, context: u32, cfg: &SpeedConfig) -> HardwarePlan {
    let weights_size = model.weights_gb(quant);
    let kv_cache_size = model.kv_cache_gb(context);
    let total_gpu = weights_size + kv_cache_size;
    let min_vram = total_gpu.ceil();
    let recommended_vram = (total_gpu * RECOMMENDED_HEADROOM).ceil();

    let mut viable_modes = Vec::new();
    if min_vram <= 256.0 {
        viable_modes.push("GPU");
    }

    if model.is_moe() {
        viable_modes.push("MoE");
    }

    viable_modes.push("CPU+GPU");
    viable_modes.push("CPU");

    let mut hw_gpu = Hardware::detect();
    hw_gpu.set_vram(256.0);
    let tps_gpu = estimate_tps(model, &hw_gpu, quant, RunMode::Gpu, 1.0, cfg);

    let mut hw_cpu = Hardware::detect();
    hw_cpu.gpus.clear();
    let tps_cpu = estimate_tps(model, &hw_cpu, quant, RunMode::Cpu, 0.0, cfg);

    HardwarePlan {
        model_name: model.name.clone(),
        params_b: model.params_b,
        context_length: context,
        quantization: quant,
        weights_gb: weights_size,
        kv_cache_gb: kv_cache_size,
        min_vram_gb: min_vram,
        recommended_vram_gb: recommended_vram,
        min_ram_gb: total_gpu.ceil(),
        tps_gpu,
        tps_cpu,
        viable_modes,
    }
}
