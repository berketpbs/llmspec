//! Persisted user settings and user-supplied models.
//!
//! Two files live in the config directory:
//!
//! - `config.json` — TUI theme, default use case and speed tunables, so a
//!   session starts where the last one left off.
//! - `models.json` — extra models merged into the embedded catalog, for
//!   anything private or newer than the shipped database.
//!
//! Both are optional. A missing, unreadable or malformed file is never fatal:
//! llmspec falls back to its defaults rather than refusing to start.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::fit::SpeedConfig;
use crate::hardware::{CPU_MEM_BANDWIDTH_FALLBACK_GB_S, MeasuredThroughput};
use crate::models::{Model, UseCase};

const CONFIG_FILE: &str = "config.json";
const CUSTOM_MODELS_FILE: &str = "models.json";
const MEASUREMENTS_FILE: &str = "measurements.json";

/// Directory holding llmspec's configuration.
///
/// `LLMSPEC_CONFIG_DIR` wins, then the platform config directory, then a
/// dotfile in the home directory. Returns `None` when the environment gives
/// no usable location at all, in which case nothing is persisted.
pub fn config_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LLMSPEC_CONFIG_DIR")
        && !dir.trim().is_empty()
    {
        return Some(PathBuf::from(dir));
    }
    // %APPDATA%\llmspec on Windows, $XDG_CONFIG_HOME/llmspec elsewhere.
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from))
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        })?;
    Some(base.join("llmspec"))
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// How a theme preference is stored.
///
/// Themes were once written as a position in the TUI's theme list, which meant
/// that list could never be reordered without silently moving everyone's
/// theme. They are written by name now; the old form is still accepted so an
/// existing config keeps working, and `serde(untagged)` picks between them on
/// whether the JSON value is a string or a number.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ThemeRef {
    Name(String),
    Index(usize),
}

impl Default for ThemeRef {
    fn default() -> Self {
        ThemeRef::Name("default".to_string())
    }
}

/// An efficiency factor fitted to measurements taken on one machine.
///
/// The GPU and backend are recorded with it, and a calibration only applies
/// when they still match. A number fitted on a 4060 is not a property of the
/// speed model — it is a property of that card, that driver and that runtime,
/// and carrying it onto a different machine would silently make every
/// estimate wrong in a way nothing else would reveal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub efficiency: f64,
    /// How many measured/estimated pairs the fit came from. One benchmark is
    /// worth less than five, and the reader deserves to know which it is.
    pub samples: usize,
    /// Seconds since the Unix epoch. Stored as a number rather than a date
    /// string so it needs no date library to write or compare.
    pub measured_at: u64,
    pub gpu: String,
    pub backend: String,
}

impl Calibration {
    /// Whether this calibration was taken on the machine now in front of us.
    pub fn matches(&self, gpu: &str, backend: &str) -> bool {
        self.gpu == gpu && self.backend == backend
    }

    /// Whole days since the measurement, for anything that reports its age.
    pub fn age_days(&self) -> u64 {
        now_unix().saturating_sub(self.measured_at) / 86_400
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Version of the speed model the stored tunables and calibration belong to.
///
/// An efficiency factor means something only inside the formula it was fitted
/// to. When the formula changes, a stored 0.55 — or a calibration fitted by
/// `bench` — describes a model that no longer exists, and applying it would
/// silently skew every estimate. Bump this whenever the speed model changes
/// shape, and older values are dropped on load rather than misapplied.
///
/// - 1: bandwidth × efficiency × a penalty per run mode.
/// - 2: time-additive reads from each memory pool plus a per-token overhead.
pub const SPEED_MODEL_VERSION: u32 = 2;

fn legacy_speed_model() -> u32 {
    1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Speed model the `speed` and `calibration` values were written for.
    /// A file that predates the field was written for version 1.
    #[serde(default = "legacy_speed_model")]
    pub speed_model: u32,
    /// Name of the TUI theme, or a bare index from an older config.
    pub theme: ThemeRef,
    /// Use case the TUI and CLI rank for when none is given on the command line.
    pub use_case: UseCase,
    /// Persisted speed tunables from the TUI's advanced-config panel.
    pub speed: PersistedSpeed,
    /// Efficiency factor fitted to real measurements from `bench --calibrate`.
    ///
    /// Absent until someone measures. See [`Calibration`] for why it records
    /// the machine it was taken on.
    #[serde(default)]
    pub calibration: Option<Calibration>,
    /// Measured system-memory bandwidth in GB/s, cached after the first run.
    ///
    /// This is a property of the machine rather than a preference, but it
    /// lives here for the same reason a cache does: measuring costs tens of
    /// milliseconds, the answer does not change between runs, and paying for
    /// it on every invocation would be visible in a tool that otherwise
    /// answers in under a second.
    #[serde(default)]
    pub ram_bandwidth_gb_s: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            speed_model: SPEED_MODEL_VERSION,
            theme: ThemeRef::default(),
            use_case: UseCase::General,
            calibration: None,
            speed: PersistedSpeed::default(),
            ram_bandwidth_gb_s: None,
        }
    }
}

/// Mirrors the tunable half of [`SpeedConfig`]. `context_cap` is deliberately
/// excluded: it belongs to one invocation, not to the user's preferences.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedSpeed {
    pub efficiency: f64,
    pub cpu_efficiency: f64,
    pub gpu_factor: f64,
    pub cpu_offload_factor: f64,
    pub moe_offload_factor: f64,
    pub tensor_parallel_factor: f64,
    pub cpu_only_factor: f64,
}

impl Default for PersistedSpeed {
    fn default() -> Self {
        PersistedSpeed::from(&SpeedConfig::default())
    }
}

impl PersistedSpeed {
    pub fn from(cfg: &SpeedConfig) -> PersistedSpeed {
        PersistedSpeed {
            efficiency: cfg.efficiency,
            cpu_efficiency: cfg.cpu_efficiency,
            gpu_factor: cfg.gpu_factor,
            cpu_offload_factor: cfg.cpu_offload_factor,
            moe_offload_factor: cfg.moe_offload_factor,
            tensor_parallel_factor: cfg.tensor_parallel_factor,
            cpu_only_factor: cfg.cpu_only_factor,
        }
    }

    /// Apply the stored factors on top of a config, keeping its context cap
    /// and KV-cache format, which belong to the invocation.
    pub fn apply_to(&self, cfg: &SpeedConfig) -> SpeedConfig {
        SpeedConfig {
            efficiency: self.efficiency,
            cpu_efficiency: self.cpu_efficiency,
            gpu_factor: self.gpu_factor,
            cpu_offload_factor: self.cpu_offload_factor,
            moe_offload_factor: self.moe_offload_factor,
            tensor_parallel_factor: self.tensor_parallel_factor,
            cpu_only_factor: self.cpu_only_factor,
            context_cap: cfg.context_cap,
            kv_quant: cfg.kv_quant,
            calibrated: cfg.calibrated,
        }
    }
}

impl Config {
    /// This machine's memory bandwidth, measured once and remembered.
    ///
    /// The first call on a machine runs the probe and writes the result back
    /// to `config.json`; every later call reads it. A machine that cannot be
    /// measured, or cannot persist anything, falls back to the shipped
    /// constant rather than paying for a failing probe on every run.
    pub fn ram_bandwidth(&mut self) -> f64 {
        if let Some(measured) = self.ram_bandwidth_gb_s {
            return measured;
        }
        let Some(measured) = crate::hardware::measure_ram_bandwidth_gb_s() else {
            return CPU_MEM_BANDWIDTH_FALLBACK_GB_S;
        };
        self.ram_bandwidth_gb_s = Some(measured);
        // Best effort: an unwritable config directory costs a re-measure next
        // run, which is not worth interrupting the user over.
        let _ = self.save();
        measured
    }

    /// Load the stored config, falling back to defaults on any problem.
    pub fn load() -> Config {
        config_dir()
            .map(|dir| dir.join(CONFIG_FILE))
            .and_then(|path| Config::load_from(&path).ok())
            .unwrap_or_default()
    }

    pub fn load_from(path: &Path) -> Result<Config, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let config: Config =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(config.migrated())
    }

    /// Drop speed settings written for an older speed model.
    ///
    /// Everything else in the file — theme, use case, the measured bandwidth
    /// — is a fact or a preference that survives a formula change, and is kept.
    pub fn migrated(mut self) -> Config {
        if self.speed_model != SPEED_MODEL_VERSION {
            self.speed = PersistedSpeed::default();
            self.calibration = None;
            self.speed_model = SPEED_MODEL_VERSION;
        }
        self
    }

    /// Write the config, creating the directory if needed.
    pub fn save(&self) -> Result<PathBuf, String> {
        let dir = config_dir().ok_or_else(|| "no config directory available".to_string())?;
        let path = dir.join(CONFIG_FILE);
        self.save_to(&path)?;
        Ok(path)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// Measured throughput
// ---------------------------------------------------------------------------

/// One `bench` result, with the machine it was measured on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredMeasurement {
    pub gpu: String,
    pub backend: String,
    pub runtime: String,
    pub context: u32,
    #[serde(flatten)]
    pub throughput: MeasuredThroughput,
}

/// Every throughput `bench` has measured, across machines.
///
/// Kept in its own file rather than `config.json`: it is data rather than a
/// preference, it grows with use, and a user resetting their settings should
/// not lose their measurements with them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MeasurementStore {
    #[serde(default)]
    pub runs: Vec<StoredMeasurement>,
}

impl MeasurementStore {
    pub fn load() -> MeasurementStore {
        config_dir()
            .map(|dir| dir.join(MEASUREMENTS_FILE))
            .and_then(|path| MeasurementStore::load_from(&path).ok())
            .unwrap_or_default()
    }

    pub fn load_from(path: &Path) -> Result<MeasurementStore, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn save(&self) -> Result<PathBuf, String> {
        let dir = config_dir().ok_or_else(|| "no config directory available".to_string())?;
        let path = dir.join(MEASUREMENTS_FILE);
        self.save_to(&path)?;
        Ok(path)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Add a measurement, replacing any earlier one of the same placement on
    /// the same machine and runtime. A re-run is a better answer to the same
    /// question, not a second answer.
    pub fn record(&mut self, run: StoredMeasurement) {
        self.runs.retain(|old| {
            !(old.gpu == run.gpu
                && old.backend == run.backend
                && old.runtime == run.runtime
                && old.throughput.model_id == run.throughput.model_id
                && old.throughput.quant == run.throughput.quant
                && old.throughput.run_mode == run.throughput.run_mode)
        });
        self.runs.push(run);
    }

    /// The measurements taken on a machine with this GPU and backend.
    pub fn for_machine(&self, gpu: &str, backend: &str) -> Vec<MeasuredThroughput> {
        self.runs
            .iter()
            .filter(|run| run.gpu == gpu && run.backend == backend)
            .map(|run| run.throughput.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Custom models
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CustomModels {
    #[serde(default)]
    models: Vec<Model>,
}

/// Load user-supplied models, if the file exists.
///
/// Accepts either `{"models": [...]}` or a bare `[...]`, since both spellings
/// are natural to write by hand.
pub fn load_custom_models() -> Vec<Model> {
    let Some(path) = config_dir().map(|dir| dir.join(CUSTOM_MODELS_FILE)) else {
        return Vec::new();
    };
    match load_custom_models_from(&path) {
        Ok(models) => models,
        Err(e) => {
            // A typo in a hand-written file should say so, not vanish.
            if path.exists() {
                eprintln!("warning: ignoring custom models — {e}");
            }
            Vec::new()
        }
    }
}

pub fn load_custom_models_from(path: &Path) -> Result<Vec<Model>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_custom_models(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn parse_custom_models(text: &str) -> Result<Vec<Model>, String> {
    if text.trim_start().starts_with('[') {
        serde_json::from_str::<Vec<Model>>(text).map_err(|e| e.to_string())
    } else {
        serde_json::from_str::<CustomModels>(text)
            .map(|c| c.models)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = env::temp_dir();
        path.push(format!("llmspec-test-{}-{name}", std::process::id()));
        path
    }

    #[test]
    fn defaults_round_trip_through_json() {
        let path = temp_path("config.json");
        let config = Config {
            speed_model: SPEED_MODEL_VERSION,
            theme: ThemeRef::Name("solarized".to_string()),
            use_case: UseCase::Coding,
            calibration: None,
            speed: PersistedSpeed {
                efficiency: 0.7,
                ..PersistedSpeed::default()
            },
            ram_bandwidth_gb_s: Some(94.5),
        };
        config.save_to(&path).unwrap();
        let loaded = Config::load_from(&path).unwrap();
        assert_eq!(loaded, config);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn partial_config_files_fill_in_defaults() {
        let config: Config = serde_json::from_str(r#"{"theme": "gruvbox"}"#).unwrap();
        assert_eq!(config.theme, ThemeRef::Name("gruvbox".to_string()));
        assert_eq!(config.use_case, UseCase::General);
        assert_eq!(config.speed, PersistedSpeed::default());
    }

    #[test]
    fn a_calibration_survives_a_round_trip() {
        let path = temp_path("calibrated.json");
        let config = Config {
            calibration: Some(Calibration {
                efficiency: 0.62,
                samples: 4,
                measured_at: 1_756_000_000,
                gpu: "NVIDIA GeForce RTX 4060 Laptop GPU".to_string(),
                backend: "CUDA".to_string(),
            }),
            ..Config::default()
        };
        config.save_to(&path).unwrap();
        assert_eq!(Config::load_from(&path).unwrap(), config);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_config_written_before_calibration_existed_still_parses() {
        // Every field llmspec has ever added has to stay optional: a config
        // that predates it must not stop the tool from starting.
        let config: Config = serde_json::from_str(r#"{"use_case": "coding"}"#).unwrap();
        assert_eq!(config.calibration, None);
        assert_eq!(config.use_case, UseCase::Coding);
    }

    #[test]
    fn speed_settings_from_an_older_speed_model_are_dropped_on_load() {
        // A 0.55 efficiency and a calibration fitted to the old formula would
        // skew every estimate the new one makes, so they are discarded — but
        // the theme and the measured bandwidth are facts that survive.
        let path = temp_path("legacy-speed.json");
        fs::write(
            &path,
            r#"{
                "theme": "nord",
                "speed": {"efficiency": 0.55, "cpu_only_factor": 0.3},
                "calibration": {"efficiency": 0.81, "samples": 1, "measured_at": 0,
                                "gpu": "RTX 4060", "backend": "CUDA"},
                "ram_bandwidth_gb_s": 44.0
            }"#,
        )
        .unwrap();
        let loaded = Config::load_from(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(loaded.speed, PersistedSpeed::default());
        assert_eq!(loaded.calibration, None);
        assert_eq!(loaded.speed_model, SPEED_MODEL_VERSION);
        assert_eq!(loaded.theme, ThemeRef::Name("nord".to_string()));
        assert_eq!(loaded.ram_bandwidth_gb_s, Some(44.0));
    }

    #[test]
    fn speed_settings_written_for_the_current_model_are_kept() {
        let config = Config {
            speed: PersistedSpeed {
                efficiency: 0.66,
                ..PersistedSpeed::default()
            },
            ..Config::default()
        };
        let path = temp_path("current-speed.json");
        config.save_to(&path).unwrap();
        let loaded = Config::load_from(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert!((loaded.speed.efficiency - 0.66).abs() < 1e-9);
    }

    fn run(gpu: &str, quant: &str, tps: f64, at: u64) -> StoredMeasurement {
        StoredMeasurement {
            gpu: gpu.to_string(),
            backend: "CUDA".to_string(),
            runtime: "Ollama".to_string(),
            context: 4096,
            throughput: MeasuredThroughput {
                model_id: "Qwen/Qwen2.5-7B-Instruct".to_string(),
                quant: quant.to_string(),
                run_mode: "GPU".to_string(),
                tokens_per_second: tps,
                measured_at: at,
            },
        }
    }

    #[test]
    fn a_rerun_replaces_the_measurement_it_repeats() {
        let mut store = MeasurementStore::default();
        store.record(run("RTX 4060", "Q4_K_M", 48.0, 1));
        store.record(run("RTX 4060", "Q4_K_M", 51.1, 2));
        store.record(run("RTX 4060", "Q8_0", 30.0, 3));
        store.record(run("RTX 4090", "Q4_K_M", 140.0, 4));
        assert_eq!(store.runs.len(), 3);

        let here = store.for_machine("RTX 4060", "CUDA");
        assert_eq!(here.len(), 2);
        assert!(
            here.iter()
                .any(|m| m.quant == "Q4_K_M" && m.tokens_per_second == 51.1)
        );
        // Another machine's numbers never leak into this one's.
        assert!(here.iter().all(|m| m.tokens_per_second != 140.0));
    }

    #[test]
    fn measurements_survive_a_round_trip() {
        let path = temp_path("measurements.json");
        let mut store = MeasurementStore::default();
        store.record(run("RTX 4060", "Q4_K_M", 51.1, 1));
        store.save_to(&path).unwrap();
        let loaded = MeasurementStore::load_from(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded, store);
    }

    #[test]
    fn a_calibration_belongs_to_the_machine_it_was_measured_on() {
        let calibration = Calibration {
            efficiency: 0.62,
            samples: 2,
            measured_at: 0,
            gpu: "RTX 4060".to_string(),
            backend: "CUDA".to_string(),
        };
        assert!(calibration.matches("RTX 4060", "CUDA"));
        assert!(!calibration.matches("RTX 4090", "CUDA"), "a different card");
        assert!(
            !calibration.matches("RTX 4060", "Metal"),
            "a different backend"
        );
    }

    #[test]
    fn a_config_written_before_themes_had_names_still_parses() {
        // Themes used to be stored as a position in the theme list. Those
        // files are still out there, and dropping the number would reset the
        // user to the default theme without saying so.
        let config: Config = serde_json::from_str(r#"{"theme": 7}"#).unwrap();
        assert_eq!(config.theme, ThemeRef::Index(7));
    }

    #[test]
    fn themes_are_written_back_as_plain_names() {
        // The file is meant to be hand-editable, so the value has to be the
        // name and not a tagged enum.
        let config = Config {
            theme: ThemeRef::Name("kanagawa".to_string()),
            ..Config::default()
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(
            json.contains(r#""theme":"kanagawa""#),
            "unexpected encoding: {json}"
        );
    }

    #[test]
    fn missing_file_is_not_an_error_for_load() {
        // `load_from` is explicit about failure; `load` swallows it and
        // returns defaults, which is what every caller wants.
        assert!(Config::load_from(&temp_path("absent.json")).is_err());
        assert!(load_custom_models_from(&temp_path("absent.json")).is_err());
    }

    #[test]
    fn speed_factors_survive_a_round_trip_and_keep_the_context_cap() {
        let cfg = SpeedConfig {
            efficiency: 0.42,
            cpu_only_factor: 0.11,
            context_cap: Some(8192),
            ..SpeedConfig::default()
        };
        let restored = PersistedSpeed::from(&cfg).apply_to(&SpeedConfig {
            context_cap: Some(4096),
            ..SpeedConfig::default()
        });
        assert!((restored.efficiency - 0.42).abs() < 1e-9);
        assert!((restored.cpu_only_factor - 0.11).abs() < 1e-9);
        // The cap comes from the invocation, not the stored preferences.
        assert_eq!(restored.context_cap, Some(4096));
    }

    #[test]
    fn custom_models_accept_both_shapes() {
        let entry = r#"{
            "id": "local/my-model",
            "name": "My Model",
            "provider": "Local",
            "params_b": 7.0,
            "context_length": 8192,
            "use_case": "general"
        }"#;
        let wrapped = parse_custom_models(&format!(r#"{{"models":[{entry}]}}"#)).unwrap();
        let bare = parse_custom_models(&format!("[{entry}]")).unwrap();
        assert_eq!(wrapped.len(), 1);
        assert_eq!(bare.len(), 1);
        assert_eq!(wrapped[0].id, "local/my-model");
        // Optional fields take their defaults.
        assert_eq!(bare[0].quality_tier, 3);
        assert!(!bare[0].gguf);
    }

    #[test]
    fn malformed_custom_models_report_an_error() {
        assert!(parse_custom_models(r#"{"models":[{"id":"broken"}]}"#).is_err());
        assert!(parse_custom_models("not json").is_err());
    }

    #[test]
    fn a_config_directory_is_always_resolvable() {
        // Every supported platform sets at least one of APPDATA, XDG_CONFIG_HOME
        // or HOME, so llmspec always has somewhere to persist to.
        let dir = config_dir().expect("a config directory");
        assert!(dir.ends_with("llmspec") || env::var_os("LLMSPEC_CONFIG_DIR").is_some());
    }
}
