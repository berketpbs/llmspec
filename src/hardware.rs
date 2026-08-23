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

/// `(name fragment, memory bandwidth GB/s, typical VRAM GB)`.
///
/// Matched case-insensitively against the reported GPU name; the first hit
/// wins, so more specific fragments must come first.
const GPU_TABLE: &[(&str, f64, f64)] = &[
    // NVIDIA datacenter
    ("h200", 4800.0, 141.0),
    ("h100 pcie", 2000.0, 80.0),
    ("h100", 3350.0, 80.0),
    ("a100 80", 2039.0, 80.0),
    ("a100", 1555.0, 40.0),
    ("l40s", 864.0, 48.0),
    ("l40", 864.0, 48.0),
    ("l4", 300.0, 24.0),
    ("a40", 696.0, 48.0),
    ("a30", 933.0, 24.0),
    ("a10g", 600.0, 24.0),
    ("a10", 600.0, 24.0),
    ("v100", 900.0, 32.0),
    ("t4", 320.0, 16.0),
    ("rtx 6000 ada", 960.0, 48.0),
    ("rtx a6000", 768.0, 48.0),
    ("rtx a5000", 768.0, 24.0),
    ("rtx a4000", 448.0, 16.0),
    // NVIDIA RTX 50 series
    ("5090", 1792.0, 32.0),
    ("5080", 960.0, 16.0),
    ("5070 ti", 896.0, 16.0),
    ("5070", 672.0, 12.0),
    ("5060 ti", 448.0, 16.0),
    ("5060", 448.0, 8.0),
    // NVIDIA RTX 40 series (laptop parts differ enough to list separately)
    ("4090 laptop", 576.0, 16.0),
    ("4080 laptop", 432.0, 12.0),
    ("4070 laptop", 256.0, 8.0),
    ("4060 laptop", 272.0, 8.0),
    ("4050 laptop", 192.0, 6.0),
    ("4090", 1008.0, 24.0),
    ("4080 super", 736.0, 16.0),
    ("4080", 717.0, 16.0),
    ("4070 ti super", 672.0, 16.0),
    ("4070 ti", 504.0, 12.0),
    ("4070 super", 504.0, 12.0),
    ("4070", 504.0, 12.0),
    ("4060 ti", 288.0, 16.0),
    ("4060", 272.0, 8.0),
    // NVIDIA RTX 30 series
    ("3090 ti", 1008.0, 24.0),
    ("3090", 936.0, 24.0),
    ("3080 ti", 912.0, 12.0),
    ("3080", 760.0, 10.0),
    ("3070 ti", 608.0, 8.0),
    ("3070", 448.0, 8.0),
    ("3060 ti", 448.0, 8.0),
    ("3060", 360.0, 12.0),
    ("3050", 224.0, 8.0),
    // NVIDIA RTX 20 / GTX
    ("2080 ti", 616.0, 11.0),
    ("2080", 448.0, 8.0),
    ("2070", 448.0, 8.0),
    ("2060", 336.0, 6.0),
    ("1080 ti", 484.0, 11.0),
    ("1080", 320.0, 8.0),
    ("1070", 256.0, 8.0),
    ("1660", 192.0, 6.0),
    // AMD
    ("mi300x", 5300.0, 192.0),
    ("mi250x", 3276.0, 128.0),
    ("mi250", 3276.0, 128.0),
    ("mi210", 1638.0, 64.0),
    ("mi100", 1229.0, 32.0),
    ("9070 xt", 645.0, 16.0),
    ("9070", 645.0, 16.0),
    ("7900 xtx", 960.0, 24.0),
    ("7900 xt", 800.0, 20.0),
    ("7800 xt", 624.0, 16.0),
    ("7700 xt", 432.0, 12.0),
    ("7600", 288.0, 8.0),
    ("6950 xt", 576.0, 16.0),
    ("6900 xt", 512.0, 16.0),
    ("6800 xt", 512.0, 16.0),
    ("6700 xt", 384.0, 12.0),
    ("6600", 224.0, 8.0),
    // Intel
    ("arc b580", 456.0, 12.0),
    ("arc a770", 560.0, 16.0),
    ("arc a750", 512.0, 8.0),
    ("arc a380", 186.0, 6.0),
    ("arc a580", 512.0, 8.0),
    ("arc b570", 380.0, 10.0),
    // Apple Silicon. The VRAM column is the base configuration; real machines
    // are sized from the unified memory pool at detection time.
    ("m4 max", 546.0, 36.0),
    ("m4 pro", 273.0, 24.0),
    ("m4", 120.0, 16.0),
    ("m3 ultra", 819.0, 96.0),
    ("m3 max", 400.0, 36.0),
    ("m3 pro", 150.0, 18.0),
    ("m3", 100.0, 8.0),
    ("m2 ultra", 800.0, 64.0),
    ("m2 max", 400.0, 32.0),
    ("m2 pro", 200.0, 16.0),
    ("m2", 100.0, 8.0),
    ("m1 ultra", 800.0, 64.0),
    ("m1 max", 400.0, 32.0),
    ("m1 pro", 200.0, 16.0),
    ("m1", 68.0, 8.0),
];

/// Fraction of unified memory an Apple Silicon Mac will hand to the GPU.
///
/// macOS caps `iogpu.wired_limit_mb` at roughly 75% of physical RAM by
/// default; treating the whole pool as VRAM would promise placements the OS
/// refuses to make.
const APPLE_UNIFIED_VRAM_FRACTION: f64 = 0.75;

fn lookup_gpu(name: &str) -> Option<(f64, f64)> {
    let lower = name.to_ascii_lowercase();
    GPU_TABLE
        .iter()
        .find(|(fragment, _, _)| lower.contains(fragment))
        .map(|(_, bandwidth, vram)| (*bandwidth, *vram))
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
    /// True when any value was overridden via flags or the TUI simulator.
    pub simulated: bool,
}

impl Hardware {
    /// Total VRAM across all detected GPUs.
    pub fn total_vram_gb(&self) -> f64 {
        self.gpus.iter().map(|g| g.vram_gb).sum()
    }

    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty() && self.total_vram_gb() > 0.0
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
                _ => (table.map(|(_, vram)| vram).unwrap_or(0.0), true),
            };
            Some(Gpu {
                name,
                vendor: Vendor::Nvidia,
                vram_gb,
                bandwidth_gb_s: table.map(|(bw, _)| bw),
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
        bandwidth_gb_s: table.map(|(bw, _)| bw),
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
            .map(|s| s.to_string())
            .unwrap_or_else(|| "AMD GPU".to_string());
        // VRAM is reported in bytes by `--showmeminfo vram`.
        let vram_gb = fields
            .iter()
            .filter_map(|f| f.parse::<f64>().ok())
            .find(|v| *v > 1e8)
            .map(|bytes| bytes / BYTES_PER_GB)
            .or_else(|| lookup_gpu(&name).map(|(_, vram)| vram))
            .unwrap_or(0.0);
        gpus.push(Gpu {
            bandwidth_gb_s: lookup_gpu(&name).map(|(bw, _)| bw),
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
                vram_gb: table.map(|(_, vram)| vram).unwrap_or(0.0),
                bandwidth_gb_s: table.map(|(bw, _)| bw),
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
        None => (table.map(|(_, vram)| vram).unwrap_or(0.0), true),
    };
    Some(Gpu {
        name: name.to_string(),
        vendor,
        vram_gb,
        bandwidth_gb_s: table.map(|(bw, _)| bw),
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
    fn gpu_table_prefers_specific_names() {
        let (bw, vram) = lookup_gpu("NVIDIA GeForce RTX 4090 Laptop GPU").unwrap();
        assert_eq!(vram, 16.0);
        assert_eq!(bw, 576.0);
        let (bw, vram) = lookup_gpu("NVIDIA GeForce RTX 4090").unwrap();
        assert_eq!(vram, 24.0);
        assert_eq!(bw, 1008.0);
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
        let table = lookup_gpu("Apple M3 Max").unwrap();
        assert_eq!(table.0, 400.0);
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
            simulated: false,
        };
        hw.apply_overrides(Some(24.0), None, None);
        assert!(hw.has_gpu());
        assert_eq!(hw.total_vram_gb(), 24.0);
        assert!(hw.simulated);
    }
}
