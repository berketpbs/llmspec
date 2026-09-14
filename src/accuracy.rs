//! How close the estimates land to measured throughput.
//!
//! Every tok/s figure llmspec prints outside `bench` is a prediction, and a
//! prediction is only worth printing if it is checked against reality. The
//! tests here hold the speed model to published and locally measured decode
//! throughput, so a change that makes the ranking look better but the numbers
//! less true fails the build instead of shipping.
//!
//! Accuracy is judged in log space. A 2x overestimate and a 2x underestimate
//! are equally wrong, and a linear error would let a handful of fast GPUs
//! dominate the mean while the slow machines — where a wrong answer costs a
//! wasted download — barely register.

#![cfg(test)]

use serde::{Deserialize, Serialize};

use crate::fit::{RunMode, SpeedConfig, estimate_at_quant};
use crate::hardware::{Backend, Hardware};
use crate::models::{ModelDb, Quant};

/// One measured decode throughput on a named machine.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Measurement {
    /// GPU as reported on the measuring machine, or empty for CPU-only.
    pub gpu: String,
    pub vram_gb: f64,
    pub ram_gb: f64,
    #[serde(default)]
    pub cores: Option<usize>,
    /// `cuda`, `rocm`, `metal` or `cpu`.
    pub backend: String,
    /// The model as the runtime named it.
    pub model: String,
    pub quant: String,
    #[serde(default = "default_context")]
    pub context: u32,
    pub measured_tps: f64,
}

fn default_context() -> u32 {
    4_096
}

/// One measurement set against the estimate for it.
#[derive(Debug, Clone, Serialize)]
pub struct Scored {
    pub measurement: Measurement,
    pub catalog_id: String,
    pub mode: String,
    pub estimated_tps: f64,
    /// `ln(estimated / measured)`: zero is exact, positive is optimistic.
    pub log_error: f64,
    /// The inputs the estimate was built from, so a miss can be traced to
    /// the term responsible for it.
    pub gpu_bandwidth: Option<f64>,
    pub ram_bandwidth: f64,
    pub read_gb: f64,
    pub total_gb: f64,
    pub vram_gb: f64,
}

/// Summary statistics over a set of scored measurements.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub count: usize,
    /// Median of `|ln(est / measured)|`, reported as a ratio (1.25 = 25% off).
    pub median_ratio: f64,
    /// Geometric mean of `est / measured`: above 1 is optimistic overall.
    pub bias: f64,
    /// Share of estimates within 25% and within 50% of the measurement.
    pub within_25: f64,
    pub within_50: f64,
}

impl Summary {
    pub fn of(scored: &[Scored]) -> Summary {
        let count = scored.len();
        if count == 0 {
            return Summary {
                count,
                median_ratio: f64::NAN,
                bias: f64::NAN,
                within_25: 0.0,
                within_50: 0.0,
            };
        }
        let mut abs: Vec<f64> = scored.iter().map(|s| s.log_error.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let median = if count % 2 == 1 {
            abs[count / 2]
        } else {
            (abs[count / 2 - 1] + abs[count / 2]) / 2.0
        };
        let mean = scored.iter().map(|s| s.log_error).sum::<f64>() / count as f64;
        let share = |limit: f64| {
            scored
                .iter()
                .filter(|s| s.log_error.abs() <= limit.ln())
                .count() as f64
                / count as f64
        };
        Summary {
            count,
            median_ratio: median.exp(),
            bias: mean.exp(),
            within_25: share(1.25),
            within_50: share(1.5),
        }
    }
}

/// The machine a measurement was taken on, as llmspec would have detected it.
pub fn hardware_for(m: &Measurement) -> Hardware {
    let mut hw = Hardware::reference_cpu();
    hw.total_ram_gb = m.ram_gb;
    hw.available_ram_gb = m.ram_gb;
    hw.cpu_cores = m.cores.unwrap_or(8);
    hw.cpu_threads = hw.cpu_cores;
    hw.arch = if m.backend == "metal" {
        "aarch64".to_string()
    } else {
        "x86_64".to_string()
    };
    hw.backend = if m.backend == "metal" {
        Backend::CpuArm
    } else {
        Backend::CpuX86
    };
    if m.backend != "cpu" {
        if hw.simulate_gpu(&m.gpu, 1).is_err() {
            // An unrecognised card still has its reported VRAM, and llmspec
            // reads it at its backend's assumed bandwidth, as it would on the
            // real machine.
            let vram = if m.backend == "metal" {
                m.ram_gb * 0.75
            } else {
                m.vram_gb
            };
            hw.set_vram(vram);
            hw.backend = match m.backend.as_str() {
                "rocm" => Backend::Rocm,
                "metal" => Backend::Metal,
                _ => Backend::Cuda,
            };
        } else if m.backend != "metal" && m.vram_gb > 0.0 {
            // The table's VRAM is the common configuration; the measuring
            // machine reported its own.
            hw.set_vram(m.vram_gb);
        }
    }
    hw
}

/// Score every measurement that resolves to a catalog entry.
pub fn score_all(measurements: &[Measurement], cfg: &SpeedConfig) -> Vec<Scored> {
    let db = ModelDb::embedded();
    measurements
        .iter()
        .filter_map(|m| {
            let model = db.find_for_runtime(&m.model)?;
            let quant = Quant::parse(&m.quant)?;
            let hw = hardware_for(m);
            let (mode, estimated_tps) = estimate_at_quant(model, &hw, quant, m.context, cfg)?;
            (estimated_tps > 0.0 && m.measured_tps > 0.0).then(|| Scored {
                measurement: m.clone(),
                catalog_id: model.id.clone(),
                mode: mode_code(mode).to_string(),
                estimated_tps,
                log_error: (estimated_tps / m.measured_tps).ln(),
                gpu_bandwidth: hw.primary_bandwidth(),
                ram_bandwidth: hw.ram_bandwidth(),
                read_gb: model.active_weights_gb(quant),
                total_gb: model.total_memory_gb_with(
                    quant,
                    m.context.min(model.context_length),
                    cfg.kv_quant,
                ),
                vram_gb: hw.total_vram_gb(),
            })
        })
        .collect()
}

fn mode_code(mode: RunMode) -> &'static str {
    match mode {
        RunMode::Gpu => "gpu",
        RunMode::Moe => "moe",
        RunMode::CpuGpu => "cpu+gpu",
        RunMode::Cpu => "cpu",
    }
}

/// Evaluate an external measurement set and write the per-row results.
///
/// Not part of the normal run: it needs a dataset that does not ship with the
/// repository. Point `LLMSPEC_EVAL_IN` at a JSON array of [`Measurement`]s and
/// `LLMSPEC_EVAL_OUT` at where the scored rows should go, then run
/// `cargo test evaluate_external -- --ignored --nocapture`.
#[test]
#[ignore]
fn evaluate_external_measurements() {
    let input = std::env::var("LLMSPEC_EVAL_IN").expect("set LLMSPEC_EVAL_IN");
    let text = std::fs::read_to_string(&input).expect("readable measurement file");
    let measurements: Vec<Measurement> = serde_json::from_str(&text).expect("valid measurements");
    let scored = score_all(&measurements, &SpeedConfig::default());
    println!(
        "resolved {} of {} measurements",
        scored.len(),
        measurements.len()
    );
    println!("all: {:?}", Summary::of(&scored));
    for mode in ["gpu", "moe", "cpu+gpu", "cpu"] {
        let subset: Vec<Scored> = scored.iter().filter(|s| s.mode == mode).cloned().collect();
        println!("{mode}: {:?}", Summary::of(&subset));
    }
    if let Ok(out) = std::env::var("LLMSPEC_EVAL_OUT") {
        std::fs::write(out, serde_json::to_string_pretty(&scored).unwrap()).unwrap();
    }
}
