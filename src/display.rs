//! Classic CLI rendering: tables, detail views and JSON output.

use colored::{Color, Colorize};
use serde::Serialize;

use crate::bench::BenchReport;
use crate::doctor::{Report, Severity};
use crate::fit::{FitLevel, FitResult, HardwarePlan, RunMode};
use crate::hardware::Hardware;
use crate::models::Model;
use crate::providers::DiscoveredRuntime;

const COL_RANK: usize = 3;
const COL_NAME: usize = 30;
const COL_PROVIDER: usize = 12;
const COL_PARAMS: usize = 7;
const COL_QUANT: usize = 7;
const COL_MODE: usize = 8;
const COL_FIT: usize = 10;
const COL_MEM: usize = 6;
const COL_CTX: usize = 7;
const COL_TPS: usize = 8;
const COL_SCORE: usize = 6;
const COL_USE: usize = 10;

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

pub fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('~');
    out
}

fn pad(text: &str, width: usize) -> String {
    let text = truncate(text, width);
    let len = text.chars().count();
    format!("{}{}", text, " ".repeat(width.saturating_sub(len)))
}

fn rpad(text: &str, width: usize) -> String {
    let text = truncate(text, width);
    let len = text.chars().count();
    format!("{}{}", " ".repeat(width.saturating_sub(len)), text)
}

pub fn format_params(params_b: f64, active_b: Option<f64>) -> String {
    match active_b {
        Some(active) => format!("{params_b:.0}B/{active:.0}B"),
        None if params_b < 1.0 => format!("{:.0}M", params_b * 1000.0),
        None if params_b < 10.0 => format!("{params_b:.1}B"),
        None => format!("{params_b:.0}B"),
    }
}

pub fn format_context(tokens: u32) -> String {
    if tokens >= 1024 {
        format!("{}K", tokens / 1024)
    } else {
        tokens.to_string()
    }
}

/// A file or memory size, in the units a person would say out loud.
pub fn format_size_gb(gb: f64) -> String {
    if gb < 1.0 {
        format!("{:.0} MB", gb * 1024.0)
    } else if gb < 10.0 {
        format!("{gb:.1} GB")
    } else {
        format!("{gb:.0} GB")
    }
}

pub fn format_tps(tps: f64) -> String {
    if tps >= 100.0 {
        format!("{tps:.0}")
    } else if tps >= 10.0 {
        format!("{tps:.1}")
    } else {
        format!("{tps:.2}")
    }
}

pub fn fit_color(fit: FitLevel) -> Color {
    match fit {
        FitLevel::Perfect => Color::Green,
        FitLevel::Good => Color::Cyan,
        FitLevel::Marginal => Color::Yellow,
        FitLevel::TooTight => Color::Red,
    }
}

pub fn mode_color(mode: RunMode) -> Color {
    match mode {
        RunMode::Gpu => Color::Green,
        RunMode::Moe => Color::Magenta,
        RunMode::CpuGpu => Color::Yellow,
        RunMode::Cpu => Color::BrightBlack,
    }
}

// ---------------------------------------------------------------------------
// System summary
// ---------------------------------------------------------------------------

pub fn render_system(hw: &Hardware) -> String {
    let mut out = String::new();
    out.push_str(&format!("{}\n", "System".bold().underline()));
    out.push_str(&format!(
        "  CPU        {} ({} cores / {} threads, {})\n",
        hw.cpu_brand, hw.cpu_cores, hw.cpu_threads, hw.arch
    ));
    out.push_str(&format!(
        "  RAM        {:.1} GB total, {:.1} GB available\n",
        hw.total_ram_gb, hw.available_ram_gb
    ));
    if hw.gpus.is_empty() {
        out.push_str(&format!("  GPU        {}\n", "none detected".yellow()));
    } else {
        for (i, gpu) in hw.gpus.iter().enumerate() {
            let bandwidth = match gpu.bandwidth_gb_s {
                Some(bw) => format!("{bw:.0} GB/s"),
                None => "bandwidth unknown".to_string(),
            };
            let estimated = if gpu.vram_estimated {
                " (estimated)"
            } else {
                ""
            };
            out.push_str(&format!(
                "  GPU {}      {} [{}] — {:.1} GB VRAM{}, {}\n",
                i,
                gpu.name,
                gpu.vendor.label(),
                gpu.vram_gb,
                estimated,
                bandwidth
            ));
        }
    }
    out.push_str(&format!("  Backend    {}\n", hw.backend.label().bold()));
    if hw.simulated {
        out.push_str(&format!(
            "  {}\n",
            "SIM  hardware values are overridden".yellow()
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Fit table
// ---------------------------------------------------------------------------

pub fn render_table(results: &[FitResult]) -> String {
    let mut out = String::new();
    let header = format!(
        "{} {} {} {} {} {} {} {} {} {} {} {}",
        pad("#", COL_RANK),
        pad("Model", COL_NAME),
        pad("Provider", COL_PROVIDER),
        rpad("Params", COL_PARAMS),
        pad("Quant", COL_QUANT),
        pad("Mode", COL_MODE),
        pad("Fit", COL_FIT),
        rpad("Mem%", COL_MEM),
        rpad("Ctx", COL_CTX),
        rpad("tok/s", COL_TPS),
        rpad("Score", COL_SCORE),
        pad("Use Case", COL_USE),
    );
    out.push_str(&format!("{}\n", header.bold()));
    out.push_str(&format!(
        "{}\n",
        "-".repeat(header.chars().count()).bright_black()
    ));

    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} {} {} {}\n",
            pad(&(i + 1).to_string(), COL_RANK).bright_black(),
            pad(&r.name, COL_NAME),
            pad(&r.provider, COL_PROVIDER).bright_black(),
            rpad(&format_params(r.params_b, r.active_params_b), COL_PARAMS),
            pad(r.quant.label(), COL_QUANT),
            pad(r.mode.label(), COL_MODE).color(mode_color(r.mode)),
            pad(r.fit.label(), COL_FIT).color(fit_color(r.fit)),
            rpad(&format!("{:.0}%", r.mem_percent), COL_MEM),
            rpad(&format_context(r.context), COL_CTX),
            rpad(&format_tps(r.tokens_per_second), COL_TPS),
            rpad(&format!("{:.1}", r.scores.composite), COL_SCORE).bold(),
            pad(r.use_case.as_str(), COL_USE).bright_black(),
        ));
    }
    out
}

/// Plain database listing, without any hardware analysis.
pub fn render_model_list(models: &[Model]) -> String {
    let mut out = String::new();
    let header = format!(
        "{} {} {} {} {} {} {}",
        pad("Model", COL_NAME),
        pad("Provider", COL_PROVIDER),
        rpad("Params", COL_PARAMS),
        rpad("Ctx", COL_CTX),
        pad("Use Case", COL_USE),
        pad("License", 14),
        pad("GGUF", 5),
    );
    out.push_str(&format!("{}\n", header.bold()));
    out.push_str(&format!(
        "{}\n",
        "-".repeat(header.chars().count()).bright_black()
    ));
    for m in models {
        out.push_str(&format!(
            "{} {} {} {} {} {} {}\n",
            pad(&m.name, COL_NAME),
            pad(&m.provider, COL_PROVIDER).bright_black(),
            rpad(&format_params(m.params_b, m.active_params_b), COL_PARAMS),
            rpad(&format_context(m.context_length), COL_CTX),
            pad(m.use_case.as_str(), COL_USE).bright_black(),
            pad(&m.license, 14).bright_black(),
            pad(if m.gguf { "yes" } else { "-" }, 5),
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// Detail view
// ---------------------------------------------------------------------------

pub fn render_detail(result: &FitResult, model: &Model) -> String {
    let mut out = String::new();
    out.push_str(&format!("{}\n", result.name.bold().underline()));
    out.push_str(&format!("  {}\n\n", result.model_id.bright_black()));

    out.push_str(&format!("{}\n", "Model".bold()));
    out.push_str(&row("Provider", &result.provider));
    out.push_str(&row(
        "Parameters",
        &format_params(result.params_b, result.active_params_b),
    ));
    if let Some(active) = result.active_params_b {
        out.push_str(&row(
            "Architecture",
            &format!("MoE — {active:.1}B active of {:.1}B total", result.params_b),
        ));
    }
    out.push_str(&row("Max context", &format_context(result.max_context)));
    out.push_str(&row("Use case", result.use_case.as_str()));
    out.push_str(&row("License", &result.license));
    out.push_str(&row("Released", &result.released));
    if !model.capabilities.is_empty() {
        out.push_str(&row("Capabilities", &model.capabilities.join(", ")));
    }
    if let Some(tag) = &result.ollama {
        out.push_str(&row("Ollama tag", tag));
    }

    out.push_str(&format!("\n{}\n", "Fit on this machine".bold()));
    out.push_str(&row_colored(
        "Verdict",
        result.fit.label(),
        fit_color(result.fit),
    ));
    out.push_str(&row_colored(
        "Run mode",
        result.mode.label(),
        mode_color(result.mode),
    ));
    out.push_str(&row("Quantization", result.quant.label()));
    out.push_str(&row("Context used", &format_context(result.context)));
    out.push_str(&row(
        "Memory needed",
        &format!(
            "{:.1} GB total, {:.1} GB resident",
            result.required_gb, result.resident_gb
        ),
    ));
    out.push_str(&row(
        "Memory used",
        &format!("{:.0}% of the pool", result.mem_percent),
    ));
    out.push_str(&row(
        "Throughput",
        &format!(
            "~{} tok/s (estimated)",
            format_tps(result.tokens_per_second)
        ),
    ));

    out.push_str(&format!("\n{}\n", "Scores".bold()));
    out.push_str(&bar("Quality", result.scores.quality));
    out.push_str(&bar("Speed", result.scores.speed));
    out.push_str(&bar("Fit", result.scores.fit));
    out.push_str(&bar("Context", result.scores.context));
    out.push_str(&format!(
        "  {:<14} {}\n",
        "Composite",
        format!("{:.1}", result.scores.composite).bold()
    ));
    out
}

fn row(label: &str, value: &str) -> String {
    format!("  {label:<14} {value}\n")
}

fn row_colored(label: &str, value: &str, color: Color) -> String {
    format!("  {:<14} {}\n", label, value.color(color))
}

pub fn render_plan(plan: &HardwarePlan) -> String {
    let details = format!(
        "{} params, context={}",
        format_params(plan.params_b, None),
        format_context(plan.context_length)
    );
    let mut out = format!("{} @ {}\n", plan.model_name.bold(), details);
    out.push('\n');
    out.push_str(&row("Quantization", &format!("{:?}", plan.quantization)));
    out.push('\n');
    out.push_str(&row(
        "Min VRAM (GPU-only)",
        &format!("{:.1} GB", plan.min_vram_gb),
    ));
    out.push_str(&row(
        "Recommended VRAM",
        &format!("{:.1} GB", plan.recommended_vram_gb),
    ));
    out.push_str(&row(
        "Min RAM (CPU-only)",
        &format!("{:.1} GB", plan.min_ram_gb),
    ));
    out.push('\n');
    out.push_str(&row("Est. tok/s (GPU)", &format!("{:.1}", plan.tps_gpu)));
    out.push_str(&row("Est. tok/s (CPU)", &format!("{:.1}", plan.tps_cpu)));
    out.push('\n');
    out.push_str(&row("Viable modes", &plan.viable_modes.join(", ")));
    out
}

fn bar(label: &str, score: f64) -> String {
    let filled = ((score / 100.0) * 24.0).round().clamp(0.0, 24.0) as usize;
    let color = if score >= 75.0 {
        Color::Green
    } else if score >= 50.0 {
        Color::Cyan
    } else if score >= 25.0 {
        Color::Yellow
    } else {
        Color::Red
    };
    format!(
        "  {:<14} {} {:.1}\n",
        label,
        format!("{}{}", "#".repeat(filled), "·".repeat(24 - filled)).color(color),
        score
    )
}

// ---------------------------------------------------------------------------
// Doctor, runtimes and benchmarks
// ---------------------------------------------------------------------------

pub fn render_doctor(report: &Report) -> String {
    let mut out = format!(
        "{} {} on {} / {}\n\n",
        "llmspec".bold().underline(),
        report.version,
        report.os,
        report.arch
    );
    for check in &report.checks {
        let color = match check.severity {
            Severity::Ok => Color::Green,
            Severity::Warn => Color::Yellow,
            Severity::Info => Color::Cyan,
        };
        out.push_str(&format!(
            "  {} {} {}\n",
            pad(check.severity.marker(), 4).color(color),
            pad(check.name, 14).bold(),
            check.detail
        ));
        if let Some(hint) = &check.hint {
            out.push_str(&format!("       {:<14} {}\n", "", hint.bright_black()));
        }
    }
    out.push('\n');
    out.push_str(&match report.warnings() {
        0 => "  Everything llmspec needs was detected.\n"
            .green()
            .to_string(),
        1 => "  1 item needs attention.\n".yellow().to_string(),
        n => format!("  {n} items need attention.\n")
            .yellow()
            .to_string(),
    });
    out
}

pub fn render_runtimes(found: &[DiscoveredRuntime]) -> String {
    if found.is_empty() {
        return format!(
            "{}\n  {}\n",
            "No local runtime is responding.".yellow(),
            "Start one (e.g. `ollama serve`) and try again.".bright_black()
        );
    }
    let mut out = format!("{}\n", "Local runtimes".bold().underline());
    for runtime in found {
        let models = match runtime.disk_gb {
            Some(gb) => format!("{} models, {gb:.1} GB on disk", runtime.model_count),
            None => format!("{} models", runtime.model_count),
        };
        out.push_str(&format!(
            "  {} {} {}\n",
            pad(runtime.name, 22).bold(),
            pad(&runtime.base_url, 30).bright_black(),
            models,
        ));
    }
    out
}

pub fn render_bench(report: &BenchReport) -> String {
    let system = &report.system;
    let mut out = format!(
        "{}\n  {} · {} ({:.0} GB VRAM) · {:.0} GB RAM · {}\n\n",
        "Measured throughput".bold().underline(),
        system.cpu,
        system.gpu,
        system.vram_gb,
        system.ram_gb,
        system.backend
    );

    let header = format!(
        "{} {} {} {} {} {}",
        pad("Model", 28),
        pad("Runtime", 12),
        rpad("tok/s", 8),
        rpad("range", 14),
        rpad("TTFT", 8),
        rpad("vs est.", 9),
    );
    out.push_str(&format!("{}\n", header.bold()));
    out.push_str(&format!(
        "{}\n",
        "-".repeat(header.chars().count()).bright_black()
    ));

    for r in &report.results {
        // A ratio far from 1.0 means the bandwidth model mispredicted this
        // machine, which is the number worth looking at.
        let ratio = match r.estimate_ratio {
            Some(ratio) => {
                let color = if (0.75..=1.35).contains(&ratio) {
                    Color::Green
                } else {
                    Color::Yellow
                };
                rpad(&format!("{ratio:.2}x"), 9).color(color).to_string()
            }
            None => rpad("-", 9).bright_black().to_string(),
        };
        out.push_str(&format!(
            "{} {} {} {} {} {}\n",
            pad(&r.model_ref, 28),
            pad(r.runtime, 12).bright_black(),
            rpad(&format_tps(r.tokens_per_second), 8).bold(),
            rpad(
                &format!("{}-{}", format_tps(r.tps_min), format_tps(r.tps_max)),
                14
            )
            .bright_black(),
            rpad(
                &match r.ttft_seconds {
                    Some(t) => format!("{:.0}ms", t * 1000.0),
                    None => "-".to_string(),
                },
                8
            ),
            ratio,
        ));
    }

    if report.results.iter().any(|r| r.estimated_tps.is_some()) {
        out.push_str(&format!(
            "\n  {}\n",
            "vs est. is measured / estimated; 1.00x means the model predicted this machine exactly."
                .bright_black()
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct JsonReport<'a> {
    pub system: &'a Hardware,
    pub use_case: &'a str,
    pub count: usize,
    pub models: &'a [FitResult],
}

pub fn to_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_keeps_width() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("a-very-long-model-name", 10).chars().count(), 10);
        assert_eq!(pad("x", 5).chars().count(), 5);
        assert_eq!(rpad("x", 5).chars().count(), 5);
    }

    #[test]
    fn parameter_formatting() {
        assert_eq!(format_params(0.137, None), "137M");
        assert_eq!(format_params(7.62, None), "7.6B");
        assert_eq!(format_params(70.6, None), "71B");
        assert_eq!(format_params(46.7, Some(12.9)), "47B/13B");
    }

    #[test]
    fn context_formatting() {
        assert_eq!(format_context(131072), "128K");
        assert_eq!(format_context(8192), "8K");
        assert_eq!(format_context(512), "512");
    }
}
