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
use crate::hardware::CPU_MEM_BANDWIDTH_FALLBACK_GB_S;
use crate::models::{Model, UseCase};

const CONFIG_FILE: &str = "config.json";
const CUSTOM_MODELS_FILE: &str = "models.json";

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Index into the TUI theme list.
    pub theme: usize,
    /// Use case the TUI and CLI rank for when none is given on the command line.
    pub use_case: UseCase,
    /// Persisted speed tunables from the TUI's advanced-config panel.
    pub speed: PersistedSpeed,
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
            theme: 0,
            use_case: UseCase::General,
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
            gpu_factor: cfg.gpu_factor,
            cpu_offload_factor: cfg.cpu_offload_factor,
            moe_offload_factor: cfg.moe_offload_factor,
            tensor_parallel_factor: cfg.tensor_parallel_factor,
            cpu_only_factor: cfg.cpu_only_factor,
        }
    }

    /// Apply the stored factors on top of a config, keeping its context cap.
    pub fn apply_to(&self, cfg: &SpeedConfig) -> SpeedConfig {
        SpeedConfig {
            efficiency: self.efficiency,
            gpu_factor: self.gpu_factor,
            cpu_offload_factor: self.cpu_offload_factor,
            moe_offload_factor: self.moe_offload_factor,
            tensor_parallel_factor: self.tensor_parallel_factor,
            cpu_only_factor: self.cpu_only_factor,
            context_cap: cfg.context_cap,
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
    pub fn ram_bandwidth(&mut self, threads: usize) -> f64 {
        if let Some(measured) = self.ram_bandwidth_gb_s {
            return measured;
        }
        let Some(measured) = crate::hardware::measure_ram_bandwidth_gb_s(threads) else {
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
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
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
            theme: 3,
            use_case: UseCase::Coding,
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
        let config: Config = serde_json::from_str(r#"{"theme": 7}"#).unwrap();
        assert_eq!(config.theme, 7);
        assert_eq!(config.use_case, UseCase::General);
        assert_eq!(config.speed, PersistedSpeed::default());
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
