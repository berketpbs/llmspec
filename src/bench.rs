//! Real throughput measurement against a locally running runtime.
//!
//! Everything else in llmspec is an estimate from a bandwidth model. `bench`
//! is the ground truth: it asks the runtime to actually generate tokens and
//! reports what came back, so an estimate can be checked rather than trusted.

use serde::Serialize;

use crate::fit::FitResult;
use crate::hardware::Hardware;
use crate::providers::{DiscoveredRuntime, ProviderRegistry, Runtime, RuntimeKind, Sample};

/// Prompt used for every benchmark run.
///
/// Short enough that prompt processing does not dominate, and open-ended
/// enough that the model will not stop before the token budget is spent.
const BENCH_PROMPT: &str = "Write a detailed technical explanation of how virtual memory paging works \
     in a modern operating system. Cover page tables, the TLB, and page faults.";

/// Tokens generated per measured run.
const DEFAULT_TOKENS: u32 = 128;

/// An untimed run that warms caches and forces the model to load.
const WARMUP_TOKENS: u32 = 8;

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub model_ref: String,
    pub runtime: &'static str,
    pub runs: usize,
    /// Median throughput across runs, in tokens per second.
    pub tokens_per_second: f64,
    /// Slowest and fastest run, to show the spread.
    pub tps_min: f64,
    pub tps_max: f64,
    /// Median time to first token, when the runtime reports prompt timing.
    pub ttft_seconds: Option<f64>,
    pub output_tokens: u32,
    pub prompt_tokens: u32,
    /// Throughput llmspec predicted for this model, when it is in the catalog.
    pub estimated_tps: Option<f64>,
    /// Measured / estimated. Above 1.0 means the estimate was conservative.
    pub estimate_ratio: Option<f64>,
    /// What the prediction assumed, so a divergence can be traced.
    ///
    /// The estimate is only as good as its inputs: if the runtime loaded a
    /// different quantization than llmspec placed the model at, the two
    /// numbers describe different things and the ratio is meaningless.
    pub assumed: Option<Assumptions>,
}

/// The configuration the estimate was made for.
#[derive(Debug, Clone, Serialize)]
pub struct Assumptions {
    pub catalog_id: String,
    pub quantization: &'static str,
    pub context: u32,
    /// Weight bytes the speed model assumed are read per token.
    pub weights_gb: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub system: HardwareSummary,
    pub results: Vec<BenchResult>,
    /// Efficiency factor that would have made the estimates match what was
    /// measured, or `None` when nothing could be compared.
    pub suggested_efficiency: Option<f64>,
}

impl BenchReport {
    pub fn new(system: HardwareSummary, results: Vec<BenchResult>, current_efficiency: f64) -> Self {
        let suggested = suggested_efficiency(&results, current_efficiency);
        BenchReport {
            system,
            results,
            suggested_efficiency: suggested,
        }
    }
}

/// The efficiency factor that would reconcile estimate with measurement.
///
/// The speed model is linear in `efficiency`, so scaling the current value by
/// the median measured/estimated ratio lands the estimate on the measurement.
/// Calibrating from the median rather than a single run keeps one unlucky
/// benchmark from moving the setting.
fn suggested_efficiency(results: &[BenchResult], current: f64) -> Option<f64> {
    let mut ratios: Vec<f64> = results
        .iter()
        .filter_map(|r| r.estimate_ratio)
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect();
    if ratios.is_empty() {
        return None;
    }
    ratios.sort_by(f64::total_cmp);
    // The efficiency factor is a fraction of peak bandwidth, so it cannot
    // exceed 1.0 however fast the measurement was.
    Some((current * median(&ratios)).clamp(0.01, 1.0))
}

/// The parts of the machine a benchmark number is only meaningful alongside.
#[derive(Debug, Clone, Serialize)]
pub struct HardwareSummary {
    pub cpu: String,
    pub gpu: String,
    pub vram_gb: f64,
    pub ram_gb: f64,
    pub backend: &'static str,
}

impl HardwareSummary {
    pub fn from(hw: &Hardware) -> HardwareSummary {
        HardwareSummary {
            cpu: hw.cpu_brand.clone(),
            gpu: hw.primary_gpu_name(),
            vram_gb: (hw.total_vram_gb() * 10.0).round() / 10.0,
            ram_gb: (hw.total_ram_gb * 10.0).round() / 10.0,
            backend: hw.backend.label(),
        }
    }
}

// ---------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------

/// Pick the runtime to benchmark against: the forced one if given, otherwise
/// the first live one.
pub fn select_runtime(
    registry: &mut ProviderRegistry,
    forced: Option<RuntimeKind>,
) -> Result<DiscoveredRuntime, String> {
    let live = registry.discover();
    match forced {
        Some(kind) => live
            .into_iter()
            .find(|r| r.kind == kind)
            .ok_or_else(|| format!("{} is not running on this machine", kind.label())),
        None => live.into_iter().next().ok_or_else(|| {
            "no local runtime is running (tried Ollama, llama.cpp, LM Studio, vLLM, \
             Docker Model Runner)"
                .to_string()
        }),
    }
}

/// Benchmark one model reference on one runtime.
pub fn run_one(
    runtime: &Runtime,
    model_ref: &str,
    runs: usize,
    tokens: Option<u32>,
) -> Result<BenchResult, String> {
    let budget = tokens.unwrap_or(DEFAULT_TOKENS);
    let runs = runs.max(1);

    // The first request pays for loading the model into memory; timing it
    // would report disk speed rather than inference speed.
    let _ = runtime.generate(model_ref, BENCH_PROMPT, WARMUP_TOKENS)?;

    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        samples.push(runtime.generate(model_ref, BENCH_PROMPT, budget)?);
    }

    Ok(summarize(model_ref, runtime.kind, &samples))
}

/// Reduce a set of runs to one result.
fn summarize(model_ref: &str, kind: RuntimeKind, samples: &[Sample]) -> BenchResult {
    let mut rates: Vec<f64> = samples.iter().map(Sample::tokens_per_second).collect();
    rates.sort_by(f64::total_cmp);

    let ttfts: Vec<f64> = samples.iter().filter_map(|s| s.ttft_seconds).collect();
    let last = samples.last();

    BenchResult {
        model_ref: model_ref.to_string(),
        runtime: kind.label(),
        runs: samples.len(),
        tokens_per_second: median(&rates),
        tps_min: rates.first().copied().unwrap_or(0.0),
        tps_max: rates.last().copied().unwrap_or(0.0),
        ttft_seconds: if ttfts.is_empty() {
            None
        } else {
            let mut sorted = ttfts;
            sorted.sort_by(f64::total_cmp);
            Some(median(&sorted))
        },
        output_tokens: last.map(|s| s.output_tokens).unwrap_or(0),
        prompt_tokens: last.map(|s| s.prompt_tokens).unwrap_or(0),
        estimated_tps: None,
        estimate_ratio: None,
        assumed: None,
    }
}

/// Attach llmspec's own prediction so the two can be compared.
pub fn attach_estimate(result: &mut BenchResult, analysis: &FitResult, weights_gb: f64) {
    let estimated = analysis.tokens_per_second;
    result.estimated_tps = Some(estimated);
    if estimated > 0.0 {
        result.estimate_ratio = Some(result.tokens_per_second / estimated);
    }
    result.assumed = Some(Assumptions {
        catalog_id: analysis.model_id.clone(),
        quantization: analysis.quant.label(),
        context: analysis.context,
        weights_gb,
    });
}

/// Median of an already-sorted slice.
fn median(sorted: &[f64]) -> f64 {
    match sorted.len() {
        0 => 0.0,
        n if n % 2 == 1 => sorted[n / 2],
        n => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tokens: u32, seconds: f64, ttft: Option<f64>) -> Sample {
        Sample {
            output_tokens: tokens,
            prompt_tokens: 20,
            generate_seconds: seconds,
            ttft_seconds: ttft,
            wall_seconds: seconds,
        }
    }

    #[test]
    fn median_handles_both_parities() {
        assert_eq!(median(&[]), 0.0);
        assert_eq!(median(&[5.0]), 5.0);
        assert_eq!(median(&[1.0, 3.0]), 2.0);
        assert_eq!(median(&[1.0, 3.0, 100.0]), 3.0);
    }

    #[test]
    fn summary_reports_median_and_spread() {
        let samples = vec![
            sample(100, 5.0, Some(0.2)),  // 20 tok/s
            sample(100, 4.0, Some(0.1)),  // 25 tok/s
            sample(100, 2.0, Some(0.15)), // 50 tok/s
        ];
        let result = summarize("test:7b", RuntimeKind::Ollama, &samples);
        assert_eq!(result.runs, 3);
        assert!((result.tokens_per_second - 25.0).abs() < 1e-6);
        assert!((result.tps_min - 20.0).abs() < 1e-6);
        assert!((result.tps_max - 50.0).abs() < 1e-6);
        assert!((result.ttft_seconds.unwrap() - 0.15).abs() < 1e-6);
        assert_eq!(result.runtime, "Ollama");
    }

    #[test]
    fn summary_omits_ttft_when_no_run_reported_it() {
        let samples = vec![sample(64, 2.0, None)];
        let result = summarize("test", RuntimeKind::LmStudio, &samples);
        assert!(result.ttft_seconds.is_none());
    }

    /// An analysis whose predicted throughput is `tps`.
    fn analysis(tps: f64) -> FitResult {
        use crate::fit::{self, SpeedConfig};
        use crate::hardware::Hardware;
        use crate::models::{ModelDb, UseCase};

        let db = ModelDb::embedded();
        let model = db.find("Qwen/Qwen3-8B").unwrap();
        let mut hw = Hardware::detect();
        hw.apply_overrides(Some(24.0), Some(64.0), None);
        let mut result = fit::analyze(model, &hw, UseCase::General, &SpeedConfig::default());
        result.tokens_per_second = tps;
        result
    }

    #[test]
    fn estimate_ratio_compares_measured_against_predicted() {
        let mut result = summarize("test", RuntimeKind::Ollama, &[sample(100, 2.0, None)]);
        attach_estimate(&mut result, &analysis(25.0), 4.6);
        // 50 measured against 25 estimated: the model was twice as fast.
        assert!((result.estimate_ratio.unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn the_estimate_records_what_it_assumed() {
        // Without this a divergent ratio is unactionable: the user cannot
        // tell whether the model is wrong or the runtime simply loaded a
        // different quantization.
        let mut result = summarize("test", RuntimeKind::Ollama, &[sample(100, 2.0, None)]);
        let analysis = analysis(25.0);
        attach_estimate(&mut result, &analysis, 4.6);

        let assumed = result.assumed.expect("assumptions recorded");
        assert_eq!(assumed.catalog_id, "Qwen/Qwen3-8B");
        assert_eq!(assumed.quantization, analysis.quant.label());
        assert_eq!(assumed.context, analysis.context);
        assert!((assumed.weights_gb - 4.6).abs() < 1e-9);
    }

    #[test]
    fn estimate_ratio_is_skipped_when_prediction_is_zero() {
        let mut result = summarize("test", RuntimeKind::Ollama, &[sample(100, 2.0, None)]);
        attach_estimate(&mut result, &analysis(0.0), 4.6);
        assert!(result.estimate_ratio.is_none());
        assert_eq!(result.estimated_tps, Some(0.0));
        // The assumptions are still worth recording.
        assert!(result.assumed.is_some());
    }

    #[test]
    fn zero_duration_does_not_divide_by_zero() {
        let result = summarize("test", RuntimeKind::Ollama, &[sample(100, 0.0, None)]);
        assert_eq!(result.tokens_per_second, 0.0);
    }
}
