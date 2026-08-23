//! Diagnostic report: what llmspec detected, how it detected it, and what it
//! had to guess.
//!
//! Every wrong recommendation traces back to a wrong input. `doctor` prints
//! those inputs with their provenance so a bug report can be answered without
//! a round trip.

use serde::Serialize;

use crate::hardware::{Backend, Hardware};
use crate::models::ModelDb;
use crate::providers::{ProviderRegistry, RuntimeKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Warn,
    Info,
}

impl Severity {
    pub fn marker(self) -> &'static str {
        match self {
            Severity::Ok => "ok",
            Severity::Warn => "warn",
            Severity::Info => "info",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub severity: Severity,
    pub detail: String,
    /// What to do about it, when there is something to do.
    pub hint: Option<String>,
}

impl Check {
    fn ok(name: &'static str, detail: impl Into<String>) -> Check {
        Check {
            name,
            severity: Severity::Ok,
            detail: detail.into(),
            hint: None,
        }
    }

    fn info(name: &'static str, detail: impl Into<String>) -> Check {
        Check {
            name,
            severity: Severity::Info,
            detail: detail.into(),
            hint: None,
        }
    }

    fn warn(name: &'static str, detail: impl Into<String>, hint: impl Into<String>) -> Check {
        Check {
            name,
            severity: Severity::Warn,
            detail: detail.into(),
            hint: Some(hint.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub version: &'static str,
    pub os: String,
    pub arch: String,
    pub checks: Vec<Check>,
    pub hardware: Hardware,
}

impl Report {
    /// True when nothing needs the user's attention.
    pub fn is_clean(&self) -> bool {
        self.checks.iter().all(|c| c.severity != Severity::Warn)
    }

    pub fn warnings(&self) -> usize {
        self.checks
            .iter()
            .filter(|c| c.severity == Severity::Warn)
            .count()
    }
}

/// Build the full diagnostic report.
pub fn run(hw: &Hardware, db: &ModelDb, registry: &mut ProviderRegistry) -> Report {
    let mut checks = Vec::new();

    checks.push(Check::ok(
        "CPU",
        format!(
            "{} — {} cores / {} threads ({})",
            hw.cpu_brand, hw.cpu_cores, hw.cpu_threads, hw.arch
        ),
    ));

    // RAM is the ceiling for CPU and hybrid placement, and the usable figure
    // is what the fit engine actually spends.
    checks.push(Check::ok(
        "RAM",
        format!(
            "{:.1} GB total, {:.1} GB available, {:.1} GB usable for weights",
            hw.total_ram_gb,
            hw.available_ram_gb,
            hw.usable_ram_gb()
        ),
    ));

    checks.extend(gpu_checks(hw));

    checks.push(match hw.backend {
        Backend::CpuX86 | Backend::CpuArm if hw.has_gpu() => Check::warn(
            "Backend",
            format!("{} despite a detected GPU", hw.backend.label()),
            "the GPU vendor could not be identified; pass --memory to size VRAM by hand",
        ),
        backend => Check::ok("Backend", backend.label()),
    });

    checks.push(runtime_check(registry));

    checks.push(Check::ok(
        "Model catalog",
        format!(
            "{} models, {} providers (schema v{}, source: {})",
            db.len(),
            db.provider_count(),
            db.schema_version,
            db.source
        ),
    ));

    if hw.simulated {
        checks.push(Check::info(
            "Overrides",
            "hardware values are simulated — results do not describe this machine",
        ));
    }

    Report {
        version: env!("CARGO_PKG_VERSION"),
        os: std::env::consts::OS.to_string(),
        arch: hw.arch.clone(),
        checks,
        hardware: hw.clone(),
    }
}

fn gpu_checks(hw: &Hardware) -> Vec<Check> {
    if hw.gpus.is_empty() {
        return vec![Check::warn(
            "GPU",
            "none detected",
            "install the vendor tools (nvidia-smi / rocm-smi) or pass --memory 8G to simulate one",
        )];
    }

    let mut checks = Vec::new();
    for (i, gpu) in hw.gpus.iter().enumerate() {
        let name: &'static str = if i == 0 { "GPU" } else { "GPU (extra)" };
        let bandwidth = match gpu.bandwidth_gb_s {
            Some(bw) => format!("{bw:.0} GB/s"),
            None => "bandwidth unknown".to_string(),
        };
        let detail = format!(
            "{} [{}] — {:.1} GB VRAM, {}",
            gpu.name,
            gpu.vendor.label(),
            gpu.vram_gb,
            bandwidth
        );
        // A guessed VRAM figure moves every fit verdict, so it is worth saying.
        if gpu.vram_estimated {
            checks.push(Check::warn(
                name,
                format!("{detail} (VRAM estimated from the model name)"),
                "pass --memory to give the exact figure",
            ));
        } else if gpu.bandwidth_gb_s.is_none() {
            checks.push(Check::warn(
                name,
                detail,
                "this GPU is not in the bandwidth table; speed falls back to a per-backend constant",
            ));
        } else {
            checks.push(Check::ok(name, detail));
        }
    }

    if hw.gpus.len() > 1 {
        checks.push(Check::info(
            "Multi-GPU",
            format!(
                "{} GPUs, {:.1} GB VRAM aggregated — assumes tensor parallelism across all of them",
                hw.gpus.len(),
                hw.total_vram_gb()
            ),
        ));
    }
    checks
}

fn runtime_check(registry: &mut ProviderRegistry) -> Check {
    let live = registry.discover();
    if live.is_empty() {
        let tried = RuntimeKind::ALL
            .iter()
            .map(|k| k.label())
            .collect::<Vec<_>>()
            .join(", ");
        return Check::warn(
            "Runtimes",
            format!("none responding (tried {tried})"),
            "start one — e.g. `ollama serve` — to enable `llmspec bench` and installed-model detection",
        );
    }
    let summary = live
        .iter()
        .map(|r| format!("{} at {} ({} models)", r.name, r.base_url, r.model_count))
        .collect::<Vec<_>>()
        .join("; ");
    Check::ok("Runtimes", summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{Backend, Gpu, Vendor};

    fn base_hw() -> Hardware {
        Hardware {
            cpu_brand: "Test CPU".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            arch: "x86_64".into(),
            total_ram_gb: 32.0,
            available_ram_gb: 20.0,
            gpus: Vec::new(),
            backend: Backend::CpuX86,
            simulated: false,
        }
    }

    fn gpu(name: &str, estimated: bool, bandwidth: Option<f64>) -> Gpu {
        Gpu {
            name: name.into(),
            vendor: Vendor::Nvidia,
            vram_gb: 24.0,
            bandwidth_gb_s: bandwidth,
            vram_estimated: estimated,
        }
    }

    fn find<'a>(report: &'a Report, name: &str) -> &'a Check {
        report
            .checks
            .iter()
            .find(|c| c.name == name)
            .expect("check present")
    }

    #[test]
    fn missing_gpu_is_a_warning_with_a_hint() {
        let mut registry = ProviderRegistry::new();
        let report = run(&base_hw(), &ModelDb::embedded(), &mut registry);
        let check = find(&report, "GPU");
        assert_eq!(check.severity, Severity::Warn);
        assert!(check.hint.as_ref().unwrap().contains("--memory"));
        assert!(!report.is_clean());
    }

    #[test]
    fn estimated_vram_is_flagged() {
        let mut hw = base_hw();
        hw.gpus.push(gpu("RTX 4090", true, Some(1008.0)));
        hw.backend = Backend::Cuda;
        let mut registry = ProviderRegistry::new();
        let report = run(&hw, &ModelDb::embedded(), &mut registry);
        let check = find(&report, "GPU");
        assert_eq!(check.severity, Severity::Warn);
        assert!(check.detail.contains("estimated"));
    }

    #[test]
    fn unknown_bandwidth_is_flagged_separately() {
        let mut hw = base_hw();
        hw.gpus.push(gpu("Some New GPU", false, None));
        hw.backend = Backend::Cuda;
        let mut registry = ProviderRegistry::new();
        let report = run(&hw, &ModelDb::embedded(), &mut registry);
        let check = find(&report, "GPU");
        assert_eq!(check.severity, Severity::Warn);
        assert!(check.hint.as_ref().unwrap().contains("bandwidth table"));
    }

    #[test]
    fn healthy_gpu_passes() {
        let mut hw = base_hw();
        hw.gpus.push(gpu("RTX 4090", false, Some(1008.0)));
        hw.backend = Backend::Cuda;
        let mut registry = ProviderRegistry::new();
        let report = run(&hw, &ModelDb::embedded(), &mut registry);
        assert_eq!(find(&report, "GPU").severity, Severity::Ok);
        assert_eq!(find(&report, "Backend").severity, Severity::Ok);
    }

    #[test]
    fn multi_gpu_adds_an_aggregation_note() {
        let mut hw = base_hw();
        hw.gpus.push(gpu("RTX 4090", false, Some(1008.0)));
        hw.gpus.push(gpu("RTX 4090", false, Some(1008.0)));
        hw.backend = Backend::Cuda;
        let mut registry = ProviderRegistry::new();
        let report = run(&hw, &ModelDb::embedded(), &mut registry);
        let check = find(&report, "Multi-GPU");
        assert_eq!(check.severity, Severity::Info);
        assert!(check.detail.contains("48.0 GB"));
    }

    #[test]
    fn cpu_backend_with_a_gpu_present_is_a_warning() {
        let mut hw = base_hw();
        hw.gpus.push(Gpu {
            vendor: Vendor::Unknown,
            ..gpu("Mystery GPU", false, Some(500.0))
        });
        // Backend deliberately left on CPU to model a failed vendor match.
        let mut registry = ProviderRegistry::new();
        let report = run(&hw, &ModelDb::embedded(), &mut registry);
        assert_eq!(find(&report, "Backend").severity, Severity::Warn);
    }

    #[test]
    fn simulated_hardware_is_announced() {
        let mut hw = base_hw();
        hw.simulated = true;
        let mut registry = ProviderRegistry::new();
        let report = run(&hw, &ModelDb::embedded(), &mut registry);
        assert_eq!(find(&report, "Overrides").severity, Severity::Info);
    }

    #[test]
    fn catalog_check_reports_real_counts() {
        let mut registry = ProviderRegistry::new();
        let db = ModelDb::embedded();
        let report = run(&base_hw(), &db, &mut registry);
        let check = find(&report, "Model catalog");
        assert!(check.detail.contains(&db.len().to_string()));
        assert!(report.warnings() >= 1);
    }
}
