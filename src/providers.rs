//! Runtime provider integration: Ollama, llama.cpp, etc.
//!
//! Providers offer local model hosting and download capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

const DISCOVERY_TTL: u64 = 300;
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Ollama client.
#[derive(Debug, Clone)]
pub struct Ollama {
    pub base_url: String,
}

impl Ollama {
    /// Default instance: OLLAMA_HOST env var if set, else localhost:11434.
    pub fn default() -> Self {
        let base_url = env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
        Ollama { base_url }
    }

    /// Custom Ollama instance.
    #[allow(dead_code)]
    pub fn with_url(url: impl Into<String>) -> Self {
        Ollama {
            base_url: url.into(),
        }
    }

    /// True if Ollama answered /api/tags successfully.
    /// This is a network call (no separate cheap probe).
    pub fn is_alive(&self) -> bool {
        self.list_models().is_ok()
    }

    /// List installed models from /api/tags.
    pub fn list_models(&self) -> Result<Vec<InstalledModel>, String> {
        let url = format!("{}/api/tags", self.base_url.trim_end_matches('/'));
        let mut resp = ureq::get(&url).call().map_err(|e| format!("GET {url}: {e}"))?;
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("reading /api/tags: {e}"))?;
        parse_tags(&text)
    }

    /// Pull a model blocking (stream=false). Warning: can take minutes for
    /// large models; run on a background thread in UI contexts.
    /// Known issue: remote OLLAMA_HOST hangs due to no timeout; use locally.
    pub fn pull(&self, tag: &str) -> Result<(), String> {
        let url = format!("{}/api/pull", self.base_url.trim_end_matches('/'));
        let body = PullRequest { name: tag, stream: false };
        ureq::post(&url)
            .send_json(body)
            .map_err(|e| format!("POST {url} ({tag}): {e}"))?;
        Ok(())
    }

    /// Delete an installed model (future Download Manager support).
    #[allow(dead_code)]
    pub fn delete(&self, tag: &str) -> Result<(), String> {
        let url = format!("{}/api/delete", self.base_url.trim_end_matches('/'));
        let body = DeleteRequest { name: tag };
        ureq::delete(&url)
            .force_send_body()
            .send_json(body)
            .map_err(|e| format!("DELETE {url} ({tag}): {e}"))?;
        Ok(())
    }
}

#[derive(Serialize)]
struct PullRequest<'a> {
    name: &'a str,
    stream: bool,
}

#[derive(Serialize)]
#[allow(dead_code)]
struct DeleteRequest<'a> {
    name: &'a str,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagEntry>,
}

#[derive(Deserialize)]
struct TagEntry {
    name: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    modified_at: String,
}

/// Parse Ollama /api/tags JSON response (pure, testable without network).
fn parse_tags(text: &str) -> Result<Vec<InstalledModel>, String> {
    let parsed: TagsResponse =
        serde_json::from_str(text).map_err(|e| format!("parsing /api/tags: {e}"))?;
    Ok(parsed
        .models
        .into_iter()
        .map(|t| InstalledModel {
            name: t.name,
            provider: "ollama".to_string(),
            size_bytes: t.size,
            modified_at: t.modified_at,
        })
        .collect())
}

/// An installed model from any provider.
#[derive(Debug, Clone, Serialize)]
pub struct InstalledModel {
    pub name: String,
    pub provider: String,
    pub size_bytes: u64,
    pub modified_at: String,
}

impl InstalledModel {
    pub fn size_gb(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Runtime provider discovery and model management.
#[derive(Debug)]
pub struct ProviderRegistry {
    cache: HashMap<String, (Vec<InstalledModel>, u64)>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        ProviderRegistry {
            cache: HashMap::new(),
        }
    }

    /// Discover available providers (quick liveness check).
    pub fn discover(&self) -> Vec<Provider> {
        let mut providers = Vec::new();
        if Ollama::default().is_alive() {
            providers.push(Provider::Ollama(Ollama::default()));
        }
        providers
    }

    /// List all models from all alive providers (cached, TTL 300s).
    pub fn list_all_models(&mut self) -> Result<Vec<InstalledModel>, String> {
        self.list_all_models_impl(false)
    }

    /// List all models, bypassing the cache (used by 'r' refresh key).
    pub fn refresh_all_models(&mut self) -> Result<Vec<InstalledModel>, String> {
        self.list_all_models_impl(true)
    }

    fn list_all_models_impl(&mut self, force: bool) -> Result<Vec<InstalledModel>, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut all = Vec::new();
        let cache_key = "ollama";

        let cached = if !force {
            self.cache.get(cache_key).and_then(|(models, ts)| {
                if now.saturating_sub(*ts) < DISCOVERY_TTL {
                    Some(models.clone())
                } else {
                    None
                }
            })
        } else {
            None
        };

        let ollama_models = if let Some(cached) = cached {
            cached
        } else {
            let ollama = Ollama::default();
            match ollama.list_models() {
                Ok(models) => {
                    self.cache.insert(cache_key.to_string(), (models.clone(), now));
                    models
                }
                Err(_) => Vec::new(),
            }
        };
        all.extend(ollama_models);

        Ok(all)
    }

    /// Check whether an exact Ollama tag is installed (no fuzzy matching).
    #[allow(dead_code)]
    pub fn is_installed(&mut self, ollama_tag: &str) -> Result<bool, String> {
        let models = self.list_all_models()?;
        Ok(models.iter().any(|m| m.name == ollama_tag))
    }

    /// Pull a model from Ollama (runs blocking; use on a background thread).
    #[allow(dead_code)]
    pub fn pull_from_ollama(&self, tag: &str) -> Result<(), String> {
        Ollama::default().pull(tag)
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Provider {
    Ollama(Ollama),
}

impl Provider {
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Ollama(_) => "Ollama",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_starts_empty() {
        let registry = ProviderRegistry::new();
        assert_eq!(registry.cache.len(), 0);
    }

    #[test]
    fn discovers_providers_if_alive() {
        let registry = ProviderRegistry::new();
        let providers = registry.discover();
        // Ollama is probably absent in CI — just verify this doesn't panic.
        // When running locally with Ollama on port 11434, providers.len() > 0.
        println!("Discovered {} provider(s)", providers.len());
    }

    #[test]
    fn provider_name() {
        let ollama = Provider::Ollama(Ollama::default());
        assert_eq!(ollama.name(), "Ollama");
    }

    #[test]
    fn installed_model_size_gb() {
        let model = InstalledModel {
            name: "test".to_string(),
            provider: "ollama".to_string(),
            size_bytes: 1024 * 1024 * 1024,
            modified_at: "2025-01-01T00:00:00Z".to_string(),
        };
        assert!((model.size_gb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn parse_tags_handles_valid_response() {
        let json = r#"{"models":[{"name":"llama3.2:1b","size":2019393189,"modified_at":"2025-01-15T10:30:00Z"}]}"#;
        let result = parse_tags(json).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llama3.2:1b");
        assert_eq!(result[0].size_bytes, 2019393189);
        assert_eq!(result[0].provider, "ollama");
    }

    #[test]
    fn parse_tags_handles_empty_models_array() {
        let json = r#"{"models":[]}"#;
        let result = parse_tags(json).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn parse_tags_defaults_missing_optional_fields() {
        let json = r#"{"models":[{"name":"test"}]}"#;
        let result = parse_tags(json).unwrap();
        assert_eq!(result[0].name, "test");
        assert_eq!(result[0].size_bytes, 0);
        assert_eq!(result[0].modified_at, "");
    }

    #[test]
    fn parse_tags_rejects_malformed_json() {
        let json = r#"{"models":[{"name":"test""#;
        let result = parse_tags(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("parsing"));
    }

    #[test]
    fn parse_tags_handles_multiple_entries() {
        let json = r#"{"models":[{"name":"llama3.2:1b","size":1000},{"name":"mistral:7b","size":2000}]}"#;
        let result = parse_tags(json).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "llama3.2:1b");
        assert_eq!(result[1].name, "mistral:7b");
    }
}
