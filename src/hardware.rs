//! System RAM / CPU / GPU detection and inference-backend selection.

use serde::Serialize;
use std::fmt;
use std::process::Command;

use sysinfo::System;

const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB_PER_GB: f64 = 1024.0;

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Backend {
    Cuda,
    Rocm,
    Sycl,
    Metal,
    CpuArm,
    CpuX86,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA",
            Backend::Rocm => "ROCm",
            Backend::Sycl => "SYCL",
            Backend::Metal => "Metal",
            Backend::CpuArm => "CPU (ARM)",
            Backend::CpuX86 => "CPU (x86)",
        }
    }

    /// Constant `K` of the fallback throughput model `K / params_b`, used when
    /// the GPU's real memory bandwidth is unknown.
    pub fn speed_constant(self) -> f64 {
        match self {
            Backend::Cuda => 220.0,
            Backend::Metal => 160.0,
            Backend::Rocm => 180.0,
            Backend::Sycl => 100.0,
            Backend::CpuArm => 90.0,
            Backend::CpuX86 => 70.0,
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl Vendor {
    pub fn label(self) -> &'static str {
        match self {
            Vendor::Nvidia => "NVIDIA",
            Vendor::Amd => "AMD",
            Vendor::Intel => "Intel",
            Vendor::Apple => "Apple",
            Vendor::Unknown => "Unknown",
        }
    }
}

// ---------------------------------------------------------------------------
// GPU bandwidth / VRAM reference table
// ---------------------------------------------------------------------------

/// One card in the reference table.
///
/// `fragment` is a matching key, not a name: it is compared case-insensitively
/// against the reported GPU name and the first hit wins, so more specific
/// fragments must come first. `display` is how the card is normally written,
/// for the output that has to name one rather than recognise one.
pub struct GpuSpec {
    fragment: &'static str,
    display: &'static str,
    pub bandwidth_gb_s: f64,
    /// Typical VRAM for the card, used when no driver reports the real figure.
    pub vram_gb: f64,
}

impl GpuSpec {
    pub fn display(&self) -> &'static str {
        self.display
    }
}

const fn gpu(
    fragment: &'static str,
    display: &'static str,
    bandwidth_gb_s: f64,
    vram_gb: f64,
) -> GpuSpec {
    GpuSpec {
        fragment,
        display,
        bandwidth_gb_s,
        vram_gb,
    }
}

const GPU_TABLE: &[GpuSpec] = &[
    // NVIDIA datacenter
    gpu("h200", "H200", 4800.0, 141.0),
    gpu("h100 pcie", "H100 PCIe", 2000.0, 80.0),
    gpu("h100", "H100", 3350.0, 80.0),
    gpu("a100 80", "A100 80GB", 2039.0, 80.0),
    gpu("a100", "A100", 1555.0, 40.0),
    gpu("l40s", "L40S", 864.0, 48.0),
    gpu("l40", "L40", 864.0, 48.0),
    gpu("l4", "L4", 300.0, 24.0),
    gpu("a40", "A40", 696.0, 48.0),
    gpu("a30", "A30", 933.0, 24.0),
    gpu("a10g", "A10G", 600.0, 24.0),
    gpu("a10", "A10", 600.0, 24.0),
    gpu("v100", "V100", 900.0, 32.0),
    gpu("t4", "T4", 320.0, 16.0),
    gpu("rtx 6000 ada", "RTX 6000 Ada", 960.0, 48.0),
    gpu("rtx a6000", "RTX A6000", 768.0, 48.0),
    gpu("rtx a5000", "RTX A5000", 768.0, 24.0),
    gpu("rtx a4000", "RTX A4000", 448.0, 16.0),
    // NVIDIA RTX 50 series
    gpu("5090", "RTX 5090", 1792.0, 32.0),
    gpu("5080", "RTX 5080", 960.0, 16.0),
    gpu("5070 ti", "RTX 5070 Ti", 896.0, 16.0),
    gpu("5070", "RTX 5070", 672.0, 12.0),
    gpu("5060 ti", "RTX 5060 Ti", 448.0, 16.0),
    gpu("5060", "RTX 5060", 448.0, 8.0),
    // NVIDIA RTX 40 series (laptop parts differ enough to list separately)
    gpu("4090 laptop", "RTX 4090 Laptop", 576.0, 16.0),
    gpu("4080 laptop", "RTX 4080 Laptop", 432.0, 12.0),
    gpu("4070 laptop", "RTX 4070 Laptop", 256.0, 8.0),
    gpu("4060 laptop", "RTX 4060 Laptop", 272.0, 8.0),
    gpu("4050 laptop", "RTX 4050 Laptop", 192.0, 6.0),
    gpu("4090", "RTX 4090", 1008.0, 24.0),
    gpu("4080 super", "RTX 4080 Super", 736.0, 16.0),
    gpu("4080", "RTX 4080", 717.0, 16.0),
    gpu("4070 ti super", "RTX 4070 Ti Super", 672.0, 16.0),
    gpu("4070 ti", "RTX 4070 Ti", 504.0, 12.0),
    gpu("4070 super", "RTX 4070 Super", 504.0, 12.0),
    gpu("4070", "RTX 4070", 504.0, 12.0),
    gpu("4060 ti", "RTX 4060 Ti", 288.0, 16.0),
    gpu("4060", "RTX 4060", 272.0, 8.0),
    // NVIDIA RTX 30 series
    gpu("3090 ti", "RTX 3090 Ti", 1008.0, 24.0),
    gpu("3090", "RTX 3090", 936.0, 24.0),
    gpu("3080 ti", "RTX 3080 Ti", 912.0, 12.0),
    gpu("3080", "RTX 3080", 760.0, 10.0),
    gpu("3070 ti", "RTX 3070 Ti", 608.0, 8.0),
    gpu("3070", "RTX 3070", 448.0, 8.0),
    gpu("3060 ti", "RTX 3060 Ti", 448.0, 8.0),
    gpu("3060", "RTX 3060", 360.0, 12.0),
    gpu("3050", "RTX 3050", 224.0, 8.0),
    // NVIDIA RTX 20 / GTX
    gpu("2080 ti", "RTX 2080 Ti", 616.0, 11.0),
    gpu("2080", "RTX 2080", 448.0, 8.0),
    gpu("2070", "RTX 2070", 448.0, 8.0),
    gpu("2060", "RTX 2060", 336.0, 6.0),
    gpu("1080 ti", "GTX 1080 Ti", 484.0, 11.0),
    gpu("1080", "GTX 1080", 320.0, 8.0),
    gpu("1070", "GTX 1070", 256.0, 8.0),
    gpu("1660", "GTX 1660", 192.0, 6.0),
    // AMD
    gpu("mi300x", "MI300X", 5300.0, 192.0),
    gpu("mi250x", "MI250X", 3276.0, 128.0),
    gpu("mi250", "MI250", 3276.0, 128.0),
    gpu("mi210", "MI210", 1638.0, 64.0),
    gpu("mi100", "MI100", 1229.0, 32.0),
    gpu("9070 xt", "RX 9070 XT", 645.0, 16.0),
    gpu("9070", "RX 9070", 645.0, 16.0),
    gpu("7900 xtx", "RX 7900 XTX", 960.0, 24.0),
    gpu("7900 xt", "RX 7900 XT", 800.0, 20.0),
    gpu("7800 xt", "RX 7800 XT", 624.0, 16.0),
    gpu("7700 xt", "RX 7700 XT", 432.0, 12.0),
    gpu("7600", "RX 7600", 288.0, 8.0),
    gpu("6950 xt", "RX 6950 XT", 576.0, 16.0),
    gpu("6900 xt", "RX 6900 XT", 512.0, 16.0),
    gpu("6800 xt", "RX 6800 XT", 512.0, 16.0),
    gpu("6700 xt", "RX 6700 XT", 384.0, 12.0),
    gpu("6600", "RX 6600", 224.0, 8.0),
    // Intel
    gpu("arc b580", "Arc B580", 456.0, 12.0),
    gpu("arc a770", "Arc A770", 560.0, 16.0),
    gpu("arc a750", "Arc A750", 512.0, 8.0),
    gpu("arc a380", "Arc A380", 186.0, 6.0),
    gpu("arc a580", "Arc A580", 512.0, 8.0),
    gpu("arc b570", "Arc B570", 380.0, 10.0),
    // Apple Silicon. The VRAM column is the base configuration; real machines
    // are sized from the unified memory pool at detection time.
    gpu("m4 max", "M4 Max", 546.0, 36.0),
    gpu("m4 pro", "M4 Pro", 273.0, 24.0),
    gpu("m4", "M4", 120.0, 16.0),
    gpu("m3 ultra", "M3 Ultra", 819.0, 96.0),
    gpu("m3 max", "M3 Max", 400.0, 36.0),
    gpu("m3 pro", "M3 Pro", 150.0, 18.0),
    gpu("m3", "M3", 100.0, 8.0),
    gpu("m2 ultra", "M2 Ultra", 800.0, 64.0),
    gpu("m2 max", "M2 Max", 400.0, 32.0),
    gpu("m2 pro", "M2 Pro", 200.0, 16.0),
    gpu("m2", "M2", 100.0, 8.0),
    gpu("m1 ultra", "M1 Ultra", 800.0, 64.0),
    gpu("m1 max", "M1 Max", 400.0, 32.0),
    gpu("m1 pro", "M1 Pro", 200.0, 16.0),
    gpu("m1", "M1", 68.0, 8.0),
];

/// Fraction of unified memory an Apple Silicon Mac will hand to the GPU.
///
/// macOS caps `iogpu.wired_limit_mb` at roughly 75% of physical RAM by
/// default; treating the whole pool as VRAM would promise placements the OS
/// refuses to make.
const APPLE_UNIFIED_VRAM_FRACTION: f64 = 0.75;

/// Reduce a reported GPU name to the form the table fragments are written in.
///
/// Vendors decorate the name differently in every source: `nvidia-smi` says
/// "NVIDIA GeForce RTX 4090", the Windows registry says
/// "Intel(R) Arc(TM) A770 Graphics", `lspci` adds bracketed codenames. Marks
/// and punctuation are dropped and runs of whitespace collapsed so that all of
/// them match the same fragment.
fn normalize_gpu_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let stripped = lower
        .replace("(r)", " ")
        .replace("(tm)", " ")
        .replace(['®', '™'], " ");
    let mut out = String::with_capacity(stripped.len());
    for ch in stripped.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if !out.ends_with(' ') {
            out.push(' ');
        }
    }
    out.trim().to_string()
}

/// True when `fragment`'s words appear consecutively in `name`.
///
/// Matching on whole words rather than raw substrings keeps short fragments
/// honest: `m4` must not claim a "Tesla M40", and `l4` must not claim an
/// "L40S".
fn matches_fragment(name: &str, fragment: &str) -> bool {
    let words: Vec<&str> = name.split(' ').collect();
    let wanted: Vec<&str> = fragment.split(' ').collect();
    if wanted.is_empty() || wanted.len() > words.len() {
        return false;
    }
    words.windows(wanted.len()).any(|window| window == wanted)
}

fn lookup_gpu(name: &str) -> Option<&'static GpuSpec> {
    let normalized = normalize_gpu_name(name);
    GPU_TABLE
        .iter()
        .find(|spec| matches_fragment(&normalized, spec.fragment))
}

/// Cards from the reference table that clear both bars, slowest first.
///
/// The order is deliberate: the first entry is the least card that does the
/// job, which is the one worth naming when someone asks what it would take to
/// hit a throughput target.
pub fn gpus_reaching(bandwidth_gb_s: f64, vram_gb: f64) -> Vec<&'static GpuSpec> {
    let mut hits: Vec<&GpuSpec> = GPU_TABLE
        .iter()
        .filter(|spec| spec.bandwidth_gb_s >= bandwidth_gb_s && spec.vram_gb >= vram_gb)
        .collect();
    hits.sort_by(|a, b| a.bandwidth_gb_s.total_cmp(&b.bandwidth_gb_s));
    hits
}

// ---------------------------------------------------------------------------
// Hardware model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Gpu {
    pub name: String,
    pub vendor: Vendor,
    pub vram_gb: f64,
    /// Memory bandwidth in GB/s when the model is recognised.
    pub bandwidth_gb_s: Option<f64>,
    /// True when VRAM was estimated from the model name instead of measured.
    pub vram_estimated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hardware {
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub arch: String,
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub gpus: Vec<Gpu>,
    pub backend: Backend,
    /// Measured main-memory bandwidth in GB/s, when it could be measured.
    ///
    /// The CPU-side counterpart of [`Gpu::bandwidth_gb_s`]: it sets the
    /// throughput ceiling for any weights that do not fit on the card.
    /// `None` means the fallback is in use.
    #[serde(default)]
    pub ram_bandwidth_gb_s: Option<f64>,
    /// True when any value was overridden via flags or the TUI simulator.
    pub simulated: bool,
}

impl Hardware {
    /// A machine that does not exist, for questions that are not about this one.
    ///
    /// Planning asks what a model would demand of *any* machine, so its answer
    /// must not shift with whatever card happens to be in the box that runs
    /// llmspec — and it must not pay for a `sysinfo` scan to say so. The CPU
    /// figures are placeholders: nothing in the planning path reads them, and
    /// the throughput model uses a fixed system-memory bandwidth for CPU runs.
    fn reference(gpus: Vec<Gpu>, backend: Backend) -> Hardware {
        Hardware {
            cpu_brand: "reference CPU".to_string(),
            cpu_cores: 8,
            cpu_threads: 16,
            arch: "reference".to_string(),
            total_ram_gb: 1024.0,
            available_ram_gb: 1024.0,
            gpus,
            backend,
            // Left unmeasured on purpose: a plan must read the same on every
            // machine, so the CPU figure is the shipped constant.
            ram_bandwidth_gb_s: None,
            simulated: true,
        }
    }

    /// A reference GPU with the given bandwidth and enough VRAM to hold
    /// anything, so a placement never fails for the wrong reason.
    pub fn reference_gpu(name: &str, bandwidth_gb_s: f64, vram_gb: f64) -> Hardware {
        Hardware::reference(
            vec![Gpu {
                name: name.to_string(),
                vendor: Vendor::Unknown,
                vram_gb,
                bandwidth_gb_s: Some(bandwidth_gb_s),
                vram_estimated: true,
            }],
            Backend::Cuda,
        )
    }

    /// A reference machine with no GPU at all.
    pub fn reference_cpu() -> Hardware {
        Hardware::reference(Vec::new(), Backend::CpuX86)
    }

    /// Total VRAM across all detected GPUs.
    pub fn total_vram_gb(&self) -> f64 {
        self.gpus.iter().map(|g| g.vram_gb).sum()
    }

    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty() && self.total_vram_gb() > 0.0
    }

    /// Main-memory bandwidth, measured if known and assumed otherwise.
    pub fn ram_bandwidth(&self) -> f64 {
        self.ram_bandwidth_gb_s
            .unwrap_or(CPU_MEM_BANDWIDTH_FALLBACK_GB_S)
    }

    /// True when [`Self::ram_bandwidth`] is a measurement rather than a guess.
    pub fn ram_bandwidth_measured(&self) -> bool {
        self.ram_bandwidth_gb_s.is_some()
    }

    /// Bandwidth of the primary (largest) GPU, when known.
    pub fn primary_bandwidth(&self) -> Option<f64> {
        self.gpus
            .iter()
            .max_by(|a, b| a.vram_gb.total_cmp(&b.vram_gb))
            .and_then(|g| g.bandwidth_gb_s)
    }

    pub fn primary_gpu_name(&self) -> String {
        match self.gpus.first() {
            Some(g) if self.gpus.len() == 1 => g.name.clone(),
            Some(g) => format!("{} x{}", g.name, self.gpus.len()),
            None => "none".to_string(),
        }
    }

    /// Memory a model may occupy in RAM, leaving room for the OS.
    pub fn usable_ram_gb(&self) -> f64 {
        (self.total_ram_gb * 0.85)
            .min(self.total_ram_gb - 2.0)
            .max(0.0)
    }

    /// Detect the real machine.
    pub fn detect() -> Hardware {
        let mut sys = System::new_all();
        sys.refresh_memory();

        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|b| !b.is_empty())
            .unwrap_or_else(|| "Unknown CPU".to_string());
        let cpu_threads = sys.cpus().len().max(1);
        let cpu_cores = System::physical_core_count().unwrap_or(cpu_threads);
        let arch = System::cpu_arch();

        let total_ram_gb = sys.total_memory() as f64 / BYTES_PER_GB;
        let available_ram_gb = sys.available_memory() as f64 / BYTES_PER_GB;

        // Apple Silicon first: on those machines the GPU is the same memory
        // the CPU uses, so probing discrete-GPU tools would find nothing and
        // wrongly report a CPU-only box.
        let mut gpus = detect_apple(&cpu_brand, total_ram_gb);
        if gpus.is_empty() {
            gpus = detect_nvidia();
        }
        if gpus.is_empty() {
            gpus.extend(detect_amd());
        }
        if gpus.is_empty() {
            gpus.extend(detect_intel());
        }

        let backend = backend_for(&gpus, &arch);

        Hardware {
            cpu_brand,
            cpu_cores,
            cpu_threads,
            arch,
            total_ram_gb,
            available_ram_gb,
            gpus,
            backend,
            // Measuring costs tens of milliseconds and the answer is cached
            // between runs, so detection leaves it to the caller to fill in.
            ram_bandwidth_gb_s: None,
            simulated: false,
        }
    }

    /// Apply CLI overrides (`--memory`, `--ram`, `--cpu-cores`).
    pub fn apply_overrides(
        &mut self,
        vram_gb: Option<f64>,
        ram_gb: Option<f64>,
        cpu_cores: Option<usize>,
    ) {
        if let Some(ram) = ram_gb {
            self.total_ram_gb = ram;
            self.available_ram_gb = ram;
            self.simulated = true;
        }
        if let Some(cores) = cpu_cores {
            self.cpu_cores = cores;
            self.cpu_threads = self.cpu_threads.max(cores);
            self.simulated = true;
        }
        if let Some(vram) = vram_gb {
            self.set_vram(vram);
            self.simulated = true;
        }
    }

    /// Force total VRAM to `vram_gb`, synthesising a GPU when none was found.
    pub fn set_vram(&mut self, vram_gb: f64) {
        if vram_gb <= 0.0 {
            self.gpus.clear();
        } else if let Some(first) = self.gpus.first_mut() {
            first.vram_gb = vram_gb;
            first.vram_estimated = true;
            self.gpus.truncate(1);
        } else {
            self.gpus.push(Gpu {
                name: "Synthetic GPU".to_string(),
                vendor: Vendor::Unknown,
                vram_gb,
                bandwidth_gb_s: None,
                vram_estimated: true,
            });
        }
        self.backend = backend_for(&self.gpus, &self.arch);
    }
}

fn backend_for(gpus: &[Gpu], arch: &str) -> Backend {
    match gpus.first().map(|g| g.vendor) {
        Some(Vendor::Nvidia) => Backend::Cuda,
        Some(Vendor::Amd) => Backend::Rocm,
        Some(Vendor::Intel) => Backend::Sycl,
        Some(Vendor::Apple) => Backend::Metal,
        Some(Vendor::Unknown) if !gpus.is_empty() => Backend::Cuda,
        _ => {
            if arch.contains("aarch64") || arch.contains("arm") {
                Backend::CpuArm
            } else {
                Backend::CpuX86
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Vendor-specific probes
// ---------------------------------------------------------------------------

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// `nvidia-smi` works on both Windows and Linux when the driver is installed.
fn detect_nvidia() -> Vec<Gpu> {
    let Some(output) = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
    ) else {
        return Vec::new();
    };

    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, ',');
            let name = parts.next()?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let reported_mib = parts.next().and_then(|v| v.trim().parse::<f64>().ok());
            let table = lookup_gpu(&name);
            let (vram_gb, estimated) = match reported_mib {
                Some(mib) if mib > 0.0 => (mib / MIB_PER_GB, false),
                // Driver present but VRAM unreadable: fall back to the table.
                _ => (table.map(|spec| spec.vram_gb).unwrap_or(0.0), true),
            };
            Some(Gpu {
                name,
                vendor: Vendor::Nvidia,
                vram_gb,
                bandwidth_gb_s: table.map(|spec| spec.bandwidth_gb_s),
                vram_estimated: estimated,
            })
        })
        .collect()
}

/// Apple Silicon: one integrated GPU sharing the system's unified memory.
///
/// `sysinfo` already reports the chip name as the CPU brand ("Apple M3 Pro"),
/// so no extra process is needed to identify the part — only to size the pool,
/// which `total_ram_gb` already carries.
fn detect_apple(cpu_brand: &str, total_ram_gb: f64) -> Vec<Gpu> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    let lower = cpu_brand.to_ascii_lowercase();
    if !lower.contains("apple") {
        // An Intel Mac has no unified-memory GPU worth reporting.
        return Vec::new();
    }
    let table = lookup_gpu(cpu_brand);
    vec![Gpu {
        name: cpu_brand.to_string(),
        vendor: Vendor::Apple,
        // Real capacity comes from the installed memory, not the table's
        // base-configuration figure.
        vram_gb: (total_ram_gb * APPLE_UNIFIED_VRAM_FRACTION).max(0.0),
        bandwidth_gb_s: table.map(|spec| spec.bandwidth_gb_s),
        vram_estimated: true,
    }]
}

/// AMD GPUs: `rocm-smi` on Linux, WMI on Windows.
fn detect_amd() -> Vec<Gpu> {
    if cfg!(target_os = "windows") {
        return detect_windows_gpus(Vendor::Amd);
    }
    if !cfg!(target_os = "linux") {
        return Vec::new();
    }
    let Some(output) = run(
        "rocm-smi",
        &["--showproductname", "--showmeminfo", "vram", "--csv"],
    ) else {
        return Vec::new();
    };

    let mut gpus = Vec::new();
    for line in output.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() < 2 || !fields[0].starts_with("card") {
            continue;
        }
        let name = fields
            .iter()
            .find(|f| {
                f.to_ascii_lowercase().contains("radeon")
                    || f.to_ascii_lowercase().contains("instinct")
            })
            .map(ToString::to_string)
            .unwrap_or_else(|| "AMD GPU".to_string());
        // VRAM is reported in bytes by `--showmeminfo vram`.
        let vram_gb = fields
            .iter()
            .filter_map(|f| f.parse::<f64>().ok())
            .find(|v| *v > 1e8)
            .map(|bytes| bytes / BYTES_PER_GB)
            .or_else(|| lookup_gpu(&name).map(|spec| spec.vram_gb))
            .unwrap_or(0.0);
        gpus.push(Gpu {
            bandwidth_gb_s: lookup_gpu(&name).map(|spec| spec.bandwidth_gb_s),
            vram_estimated: vram_gb == 0.0,
            name,
            vendor: Vendor::Amd,
            vram_gb,
        });
    }
    gpus
}

/// Intel Arc: `lspci` on Linux, WMI on Windows.
fn detect_intel() -> Vec<Gpu> {
    if cfg!(target_os = "windows") {
        return detect_windows_gpus(Vendor::Intel);
    }
    if !cfg!(target_os = "linux") {
        return Vec::new();
    }
    let Some(output) = run("lspci", &[]) else {
        return Vec::new();
    };
    output
        .lines()
        .filter(|l| {
            let lower = l.to_ascii_lowercase();
            lower.contains("vga") && lower.contains("intel") && lower.contains("arc")
        })
        .map(|line| {
            let name = line
                .split(':')
                .next_back()
                .unwrap_or("Intel Arc")
                .trim()
                .to_string();
            let table = lookup_gpu(&name);
            Gpu {
                vram_gb: table.map(|spec| spec.vram_gb).unwrap_or(0.0),
                bandwidth_gb_s: table.map(|spec| spec.bandwidth_gb_s),
                vram_estimated: true,
                name,
                vendor: Vendor::Intel,
            }
        })
        .collect()
}

/// Windows display adapters, read from the driver's registry key.
///
/// `Win32_VideoController.AdapterRAM` is a 32-bit field and silently saturates
/// at 4 GB, which would understate every modern card. The driver class key
/// carries the real figure as a 64-bit `qwMemorySize`.
fn detect_windows_gpus(vendor: Vendor) -> Vec<Gpu> {
    if !cfg!(target_os = "windows") {
        return Vec::new();
    }
    const CLASS_KEY: &str =
        r"HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
    let script = format!(
        "Get-ChildItem '{CLASS_KEY}' -ErrorAction SilentlyContinue | ForEach-Object {{ \
           $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue; \
           if ($p.DriverDesc) {{ \
             '{{0}}|{{1}}' -f $p.DriverDesc, $p.'HardwareInformation.qwMemorySize' }} }}"
    );
    let Some(output) = run(
        "powershell",
        &["-NoProfile", "-NonInteractive", "-Command", &script],
    ) else {
        return Vec::new();
    };

    output
        .lines()
        .filter_map(|line| parse_windows_adapter(line, vendor))
        .collect()
}

/// Parse one `DriverDesc|qwMemorySize` line, keeping only `vendor`'s adapters.
fn parse_windows_adapter(line: &str, vendor: Vendor) -> Option<Gpu> {
    let (name, size) = line.split_once('|')?;
    let name = name.trim();
    if name.is_empty() || !matches_vendor(name, vendor) {
        return None;
    }
    let table = lookup_gpu(name);
    let reported = size.trim().parse::<f64>().ok().filter(|b| *b > 0.0);
    let (vram_gb, estimated) = match reported {
        Some(bytes) => (bytes / BYTES_PER_GB, false),
        None => (table.map(|spec| spec.vram_gb).unwrap_or(0.0), true),
    };
    Some(Gpu {
        name: name.to_string(),
        vendor,
        vram_gb,
        bandwidth_gb_s: table.map(|spec| spec.bandwidth_gb_s),
        vram_estimated: estimated,
    })
}

fn matches_vendor(name: &str, vendor: Vendor) -> bool {
    let lower = name.to_ascii_lowercase();
    match vendor {
        Vendor::Amd => {
            lower.contains("amd") || lower.contains("radeon") || lower.contains("instinct")
        }
        Vendor::Intel => lower.contains("intel") || lower.contains("arc"),
        Vendor::Nvidia => lower.contains("nvidia") || lower.contains("geforce"),
        Vendor::Apple => lower.contains("apple"),
        Vendor::Unknown => true,
    }
}

// ---------------------------------------------------------------------------
// System memory bandwidth
// ---------------------------------------------------------------------------

/// System-memory bandwidth (GB/s) assumed when the machine cannot be measured.
///
/// Roughly a dual-channel DDR4-3200 desktop: low enough not to promise
/// throughput a slow machine cannot reach, high enough not to write off CPU
/// inference on a fast one.
pub const CPU_MEM_BANDWIDTH_FALLBACK_GB_S: f64 = 60.0;

/// Buffer size for the bandwidth probe.
///
/// It has to be comfortably larger than the last-level cache or the probe
/// measures the cache instead of main memory. 64 MiB clears every desktop and
/// laptop L3 in circulation, including the 128 MiB V-Cache parts once the
/// read is streaming rather than resident.
const PROBE_BYTES: usize = 64 * 1024 * 1024;

/// Timed passes over the buffer. The fastest is kept: a pass can only be
/// slowed by something else on the machine, never speeded up, so the best of
/// a few is closer to the hardware than their average.
const PROBE_PASSES: usize = 5;

/// Measure how fast this machine streams from main memory, in GB/s.
///
/// CPU-resident inference is bandwidth-bound the same way GPU inference is, so
/// this is the CPU-side counterpart of the GPU bandwidth table. It is measured
/// rather than assumed because the spread across machines is enormous — a
/// dual-channel DDR4 laptop and a DDR5 desktop differ by a factor of three,
/// and no CPU model string reliably predicts which one you have.
///
/// The probe reads; it never writes. A read-only stream is what token
/// generation actually does, and it avoids the write-allocate traffic that
/// would make a memcpy benchmark report a number inference never sees.
///
/// Returns `None` if the buffer cannot be allocated, or if the result is
/// implausible enough that a stored constant is the better answer.
pub fn measure_ram_bandwidth_gb_s() -> Option<f64> {
    use std::hint::black_box;
    use std::time::Instant;

    let len = PROBE_BYTES / std::mem::size_of::<u64>();
    let mut buffer: Vec<u64> = Vec::new();
    buffer.try_reserve_exact(len).ok()?;
    // Distinct values, so nothing downstream can be folded into a constant.
    buffer.extend((0..len).map(|i| i as u64));

    // An untimed pass first: the timed ones must measure memory traffic, not
    // the page faults that first touching a fresh allocation costs.
    black_box(sum(&buffer));

    let mut best = f64::INFINITY;
    for _ in 0..PROBE_PASSES {
        let start = Instant::now();
        black_box(sum(&buffer));
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 && elapsed < best {
            best = elapsed;
        }
    }
    if !best.is_finite() {
        return None;
    }

    let gb_s = PROBE_BYTES as f64 / BYTES_PER_GB / best;
    // A number outside this range means the probe measured something other
    // than memory — a cache, or a machine descheduling us mid-pass. Refusing
    // to answer beats reporting a figure the estimates would trust.
    (2.0..=2000.0).contains(&gb_s).then_some(gb_s)
}

/// Sequential read of the whole buffer.
///
/// Written as a plain fold so the compiler is free to vectorise and prefetch
/// it, which is exactly what the memory system does under inference too.
fn sum(buffer: &[u64]) -> u64 {
    buffer
        .iter()
        .fold(0u64, |acc, &value| acc.wrapping_add(value))
}

// ---------------------------------------------------------------------------
// Size parsing for CLI overrides
// ---------------------------------------------------------------------------

/// Parse `32G`, `128GB`, `16GiB`, `512M`, `1T` (case-insensitive) into GB.
pub fn parse_size_gb(input: &str) -> Result<f64, String> {
    let raw = input.trim();
    if raw.is_empty() {
        return Err("empty size".to_string());
    }
    let digits_end = raw
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(raw.len());
    let (number, suffix) = raw.split_at(digits_end);
    let value: f64 = number
        .parse()
        .map_err(|_| format!("invalid size '{raw}': expected a number like 32G"))?;
    if value < 0.0 {
        return Err(format!("invalid size '{raw}': must not be negative"));
    }

    let unit = suffix.trim().to_ascii_lowercase();
    let gb = match unit.as_str() {
        "" | "g" | "gb" | "gib" => value,
        "m" | "mb" | "mib" => value / 1024.0,
        "t" | "tb" | "tib" => value * 1024.0,
        "k" | "kb" | "kib" => value / (1024.0 * 1024.0),
        other => return Err(format!("unknown unit '{other}': use G, M or T")),
    };
    Ok(gb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sizes() {
        assert_eq!(parse_size_gb("32G").unwrap(), 32.0);
        assert_eq!(parse_size_gb("128gb").unwrap(), 128.0);
        assert_eq!(parse_size_gb("16GiB").unwrap(), 16.0);
        assert_eq!(parse_size_gb("512M").unwrap(), 0.5);
        assert_eq!(parse_size_gb("1T").unwrap(), 1024.0);
        assert_eq!(parse_size_gb("24").unwrap(), 24.0);
        assert!(parse_size_gb("abc").is_err());
        assert!(parse_size_gb("12X").is_err());
    }

    #[test]
    fn the_memory_probe_reports_a_plausible_figure() {
        let Some(measured) = measure_ram_bandwidth_gb_s() else {
            // A machine that cannot allocate the buffer is allowed to decline;
            // the fallback covers it. Nothing else about the test applies.
            return;
        };
        assert!(
            (2.0..=2000.0).contains(&measured),
            "{measured} GB/s is outside anything real hardware does"
        );
    }

    #[test]
    fn an_unmeasured_machine_falls_back_rather_than_reporting_zero() {
        let mut hw = Hardware::reference_cpu();
        assert!(!hw.ram_bandwidth_measured());
        assert_eq!(hw.ram_bandwidth(), CPU_MEM_BANDWIDTH_FALLBACK_GB_S);

        hw.ram_bandwidth_gb_s = Some(93.0);
        assert!(hw.ram_bandwidth_measured());
        assert_eq!(hw.ram_bandwidth(), 93.0);
    }

    #[test]
    fn gpu_table_prefers_specific_names() {
        let laptop = lookup_gpu("NVIDIA GeForce RTX 4090 Laptop GPU").unwrap();
        assert_eq!(laptop.display(), "RTX 4090 Laptop");
        assert_eq!(laptop.vram_gb, 16.0);
        assert_eq!(laptop.bandwidth_gb_s, 576.0);
        let desktop = lookup_gpu("NVIDIA GeForce RTX 4090").unwrap();
        assert_eq!(desktop.display(), "RTX 4090");
        assert_eq!(desktop.vram_gb, 24.0);
        assert_eq!(desktop.bandwidth_gb_s, 1008.0);
    }

    #[test]
    fn gpu_names_match_through_vendor_decorations() {
        assert_eq!(
            normalize_gpu_name("Intel(R) Arc(TM) A770 Graphics"),
            "intel arc a770 graphics"
        );
        assert_eq!(
            lookup_gpu("Intel(R) Arc(TM) A770 Graphics")
                .unwrap()
                .vram_gb,
            16.0
        );
        assert_eq!(lookup_gpu("NVIDIA H100 80GB HBM3").unwrap().vram_gb, 80.0);
        assert_eq!(lookup_gpu("AMD Radeon RX 9070 XT").unwrap().vram_gb, 16.0);
    }

    #[test]
    fn short_fragments_do_not_match_longer_model_numbers() {
        // "m4" must not claim a Tesla M40, and "l4" must not claim an L40S.
        assert!(lookup_gpu("Tesla M40").is_none());
        assert_eq!(lookup_gpu("NVIDIA L40S").unwrap().vram_gb, 48.0);
        assert_eq!(lookup_gpu("NVIDIA L4").unwrap().vram_gb, 24.0);
        assert_eq!(lookup_gpu("Apple M4 Max").unwrap().bandwidth_gb_s, 546.0);
        assert!(lookup_gpu("Completely Unknown Card").is_none());
    }

    #[test]
    fn every_card_in_the_table_has_a_printable_name() {
        for spec in GPU_TABLE {
            assert!(
                !spec.display.is_empty() && !spec.display.contains("  "),
                "{} has no usable display name",
                spec.fragment
            );
            assert!(
                spec.bandwidth_gb_s > 0.0 && spec.vram_gb > 0.0,
                "{}",
                spec.fragment
            );
        }
    }

    #[test]
    fn gpus_reaching_names_the_least_sufficient_card_first() {
        let hits = gpus_reaching(700.0, 24.0);
        assert!(
            hits.iter()
                .all(|s| s.bandwidth_gb_s >= 700.0 && s.vram_gb >= 24.0),
            "every hit must clear both bars"
        );
        assert!(
            hits.windows(2)
                .all(|w| w[0].bandwidth_gb_s <= w[1].bandwidth_gb_s),
            "hits must be ordered slowest first"
        );
        // A 24 GB card at 960 GB/s qualifies; a 16 GB one at the same speed
        // does not, and neither does a 20 GB one.
        let names: Vec<&str> = hits.iter().map(|s| s.display()).collect();
        assert!(names.contains(&"RX 7900 XTX"), "got {names:?}");
        assert!(!names.contains(&"RTX 5080"), "got {names:?}");
        assert!(!names.contains(&"RX 7900 XT"), "got {names:?}");
        assert!(gpus_reaching(99_000.0, 8.0).is_empty());
    }

    #[test]
    fn windows_adapter_lines_are_parsed() {
        let gpu = parse_windows_adapter("AMD Radeon RX 7900 XTX|25753026560", Vendor::Amd).unwrap();
        assert_eq!(gpu.vendor, Vendor::Amd);
        assert!((gpu.vram_gb - 24.0).abs() < 0.1, "got {}", gpu.vram_gb);
        assert!(!gpu.vram_estimated);
        assert_eq!(gpu.bandwidth_gb_s, Some(960.0));
    }

    #[test]
    fn windows_adapter_without_a_size_falls_back_to_the_table() {
        let gpu = parse_windows_adapter("Intel(R) Arc(TM) A770 Graphics|", Vendor::Intel).unwrap();
        assert!(gpu.vram_estimated);
        assert_eq!(gpu.vram_gb, 16.0);
    }

    #[test]
    fn windows_adapter_lines_are_filtered_by_vendor() {
        // A machine can list several adapters; only the asked-for vendor's
        // entries belong in the result.
        assert!(parse_windows_adapter("NVIDIA GeForce RTX 4090|100", Vendor::Amd).is_none());
        assert!(parse_windows_adapter("|1024", Vendor::Amd).is_none());
        assert!(parse_windows_adapter("no separator", Vendor::Amd).is_none());
    }

    #[test]
    fn apple_silicon_sizes_vram_from_unified_memory() {
        // Off macOS the probe is inert, so assert the sizing rule directly.
        let spec = lookup_gpu("Apple M3 Max").unwrap();
        assert_eq!(spec.bandwidth_gb_s, 400.0);
        assert!((128.0 * APPLE_UNIFIED_VRAM_FRACTION - 96.0).abs() < 1e-9);
        assert!(detect_apple("Intel Core i7", 32.0).is_empty());
    }

    #[test]
    fn apple_vendor_selects_the_metal_backend() {
        let gpus = vec![Gpu {
            name: "Apple M3 Pro".into(),
            vendor: Vendor::Apple,
            vram_gb: 27.0,
            bandwidth_gb_s: Some(150.0),
            vram_estimated: true,
        }];
        assert_eq!(backend_for(&gpus, "aarch64"), Backend::Metal);
    }

    #[test]
    fn synthetic_gpu_from_memory_override() {
        let mut hw = Hardware {
            cpu_brand: "test".into(),
            cpu_cores: 8,
            cpu_threads: 16,
            arch: "x86_64".into(),
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            gpus: Vec::new(),
            backend: Backend::CpuX86,
            ram_bandwidth_gb_s: None,
            simulated: false,
        };
        hw.apply_overrides(Some(24.0), None, None);
        assert!(hw.has_gpu());
        assert_eq!(hw.total_vram_gb(), 24.0);
        assert!(hw.simulated);
    }
}
