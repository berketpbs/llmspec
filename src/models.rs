//! Model database, quantization hierarchy and memory estimation.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The model database is embedded at build time.
const EMBEDDED_DB: &str = include_str!("../data/models.json");

/// Bytes per element of a fp16 KV cache entry.
const KV_ELEMENT_BYTES: f64 = 2.0;

/// Fallback KV-cache cost in GB per (token x billion params), derived from
/// Llama-3.1-8B geometry. Only used when a model has no geometry fields.
const KV_FALLBACK_GB_PER_TOKEN_PER_B: f64 = 1.63e-5;

/// Fixed runtime overhead (CUDA context, compute buffers, allocator slack).
const BASE_OVERHEAD_GB: f64 = 0.40;

/// Additional overhead proportional to the resident weights.
const OVERHEAD_WEIGHT_FRACTION: f64 = 0.05;

const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------
// Use cases
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UseCase {
    General,
    Coding,
    Reasoning,
    Chat,
    Multimodal,
    Embedding,
}

impl UseCase {
    pub const ALL: [UseCase; 6] = [
        UseCase::General,
        UseCase::Coding,
        UseCase::Reasoning,
        UseCase::Chat,
        UseCase::Multimodal,
        UseCase::Embedding,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            UseCase::General => "general",
            UseCase::Coding => "coding",
            UseCase::Reasoning => "reasoning",
            UseCase::Chat => "chat",
            UseCase::Multimodal => "multimodal",
            UseCase::Embedding => "embedding",
        }
    }

    pub fn parse(s: &str) -> Option<UseCase> {
        match s.trim().to_ascii_lowercase().as_str() {
            "general" | "all-round" => Some(UseCase::General),
            "coding" | "code" => Some(UseCase::Coding),
            "reasoning" | "reason" => Some(UseCase::Reasoning),
            "chat" => Some(UseCase::Chat),
            "multimodal" | "vision" | "vlm" => Some(UseCase::Multimodal),
            "embedding" | "embed" => Some(UseCase::Embedding),
            _ => None,
        }
    }

    /// Context length (tokens) that fully satisfies this use case.
    pub fn target_context(self) -> f64 {
        match self {
            UseCase::General => 32_768.0,
            // 32k covers a large working set of code; asking for more would
            // punish code-tuned models, which mostly ship 32k native windows.
            UseCase::Coding => 32_768.0,
            UseCase::Reasoning => 32_768.0,
            UseCase::Chat => 8_192.0,
            UseCase::Multimodal => 32_768.0,
            UseCase::Embedding => 8_192.0,
        }
    }
}

impl fmt::Display for UseCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quant {
    Q8_0,
    Q6K,
    Q5KM,
    Q4KM,
    Q3KM,
    Q2K,
}

impl Quant {
    /// Ordered from highest quality to most compressed. The fit engine walks
    /// this list top-down and keeps the first level that fits.
    pub const HIERARCHY: [Quant; 6] = [
        Quant::Q8_0,
        Quant::Q6K,
        Quant::Q5KM,
        Quant::Q4KM,
        Quant::Q3KM,
        Quant::Q2K,
    ];

    /// Effective bits per weight, matching llama.cpp GGUF sizes.
    pub fn bits_per_weight(self) -> f64 {
        match self {
            Quant::Q8_0 => 8.50,
            Quant::Q6K => 6.56,
            Quant::Q5KM => 5.67,
            Quant::Q4KM => 4.83,
            Quant::Q3KM => 3.91,
            Quant::Q2K => 3.35,
        }
    }

    /// Retained output quality relative to the fp16 model (0..1).
    ///
    /// Shaped after measured llama.cpp perplexity loss: everything down to
    /// Q4_K_M is close to lossless, while Q3_K_M degrades noticeably and Q2_K
    /// is a last resort rather than a trade.
    pub fn quality_factor(self) -> f64 {
        match self {
            Quant::Q8_0 => 1.000,
            Quant::Q6K => 0.990,
            Quant::Q5KM => 0.975,
            Quant::Q4KM => 0.950,
            Quant::Q3KM => 0.800,
            Quant::Q2K => 0.580,
        }
    }

    /// Throughput multiplier used by the fallback (non-bandwidth) speed model.
    pub fn speed_multiplier(self) -> f64 {
        match self {
            Quant::Q8_0 => 0.75,
            Quant::Q6K => 0.85,
            Quant::Q5KM => 0.92,
            Quant::Q4KM => 1.00,
            Quant::Q3KM => 1.08,
            Quant::Q2K => 1.15,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Quant::Q8_0 => "Q8_0",
            Quant::Q6K => "Q6_K",
            Quant::Q5KM => "Q5_K_M",
            Quant::Q4KM => "Q4_K_M",
            Quant::Q3KM => "Q3_K_M",
            Quant::Q2K => "Q2_K",
        }
    }

    pub fn parse(s: &str) -> Option<Quant> {
        let key: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        match key.as_str() {
            "q80" | "q8" => Some(Quant::Q8_0),
            "q6k" | "q6" => Some(Quant::Q6K),
            "q5km" | "q5" | "q5k" => Some(Quant::Q5KM),
            "q4km" | "q4" | "q4k" => Some(Quant::Q4KM),
            "q3km" | "q3" | "q3k" => Some(Quant::Q3KM),
            "q2k" | "q2" => Some(Quant::Q2K),
            _ => None,
        }
    }
}

impl fmt::Display for Quant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// HuggingFace repo id, e.g. `meta-llama/Llama-3.1-8B-Instruct`.
    pub id: String,
    pub name: String,
    pub provider: String,
    /// Total parameter count in billions.
    pub params_b: f64,
    /// Active parameters per token for MoE models; `None` for dense models.
    #[serde(default)]
    pub active_params_b: Option<f64>,
    pub context_length: u32,
    pub use_case: UseCase,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub released: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Family reputation, 1 (weak) to 5 (frontier-class for its size).
    #[serde(default = "default_tier")]
    pub quality_tier: u8,
    /// Whether downloadable GGUF builds are known to exist.
    #[serde(default)]
    pub gguf: bool,
    /// Matching Ollama tag, when one exists.
    #[serde(default)]
    pub ollama: Option<String>,

    // Geometry, used for exact KV-cache sizing. Optional: scraped entries may
    // lack it, in which case a parameter-count heuristic is used instead.
    #[serde(default)]
    pub layers: Option<u32>,
    #[serde(default)]
    pub hidden_size: Option<u32>,
    #[serde(default)]
    pub kv_heads: Option<u32>,
    #[serde(default)]
    pub head_dim: Option<u32>,
}

fn default_tier() -> u8 {
    3
}

impl Model {
    pub fn is_moe(&self) -> bool {
        self.active_params_b.is_some()
    }

    /// Parameters actually read per token (equals `params_b` for dense models).
    pub fn active_params(&self) -> f64 {
        self.active_params_b.unwrap_or(self.params_b)
    }

    /// Size of the full weight tensor set at a given quantization.
    pub fn weights_gb(&self, quant: Quant) -> f64 {
        self.params_b * 1e9 * quant.bits_per_weight() / 8.0 / BYTES_PER_GB
    }

    /// Size of only the per-token active weights (MoE expert offloading).
    pub fn active_weights_gb(&self, quant: Quant) -> f64 {
        self.active_params() * 1e9 * quant.bits_per_weight() / 8.0 / BYTES_PER_GB
    }

    /// KV cache size at `context` tokens, fp16.
    pub fn kv_cache_gb(&self, context: u32) -> f64 {
        match (self.layers, self.kv_heads, self.head_dim) {
            (Some(layers), Some(kv_heads), Some(head_dim)) => {
                2.0 * f64::from(layers)
                    * f64::from(kv_heads)
                    * f64::from(head_dim)
                    * f64::from(context)
                    * KV_ELEMENT_BYTES
                    / BYTES_PER_GB
            }
            _ => f64::from(context) * self.params_b * KV_FALLBACK_GB_PER_TOKEN_PER_B,
        }
    }

    pub fn overhead_gb(&self, resident_weights_gb: f64) -> f64 {
        BASE_OVERHEAD_GB + resident_weights_gb * OVERHEAD_WEIGHT_FRACTION
    }

    /// Total memory needed to run the whole model at `quant` with `context`.
    pub fn total_memory_gb(&self, quant: Quant, context: u32) -> f64 {
        let weights = self.weights_gb(quant);
        weights + self.kv_cache_gb(context) + self.overhead_gb(weights)
    }

    /// Memory that must be resident on the accelerator when experts are
    /// offloaded to system RAM. Dense models fall back to the full footprint.
    pub fn moe_resident_gb(&self, quant: Quant, context: u32) -> f64 {
        let active = self.active_weights_gb(quant);
        active + self.kv_cache_gb(context) + self.overhead_gb(active)
    }

    /// Substring match over name, id, provider, use case and parameter size.
    pub fn matches(&self, query: &str) -> bool {
        let q = query.trim().to_ascii_lowercase();
        if q.is_empty() {
            return true;
        }
        let params = format!("{:.0}b", self.params_b);
        q.split_whitespace().all(|term| {
            self.name.to_ascii_lowercase().contains(term)
                || self.id.to_ascii_lowercase().contains(term)
                || self.provider.to_ascii_lowercase().contains(term)
                || self.use_case.as_str().contains(term)
                || params.contains(term)
                || self
                    .capabilities
                    .iter()
                    .any(|c| c.to_ascii_lowercase().contains(term))
        })
    }
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ModelDb {
    #[serde(default)]
    pub schema_version: u32,
    #[serde(default)]
    pub source: String,
    pub models: Vec<Model>,
}

impl ModelDb {
    /// Load the database embedded at build time.
    pub fn embedded() -> ModelDb {
        serde_json::from_str(EMBEDDED_DB).expect("embedded model database is valid JSON")
    }

    /// The embedded database plus any models the user has added locally.
    pub fn load() -> ModelDb {
        let mut db = ModelDb::embedded();
        db.merge(crate::config::load_custom_models());
        db
    }

    /// Add models, letting a user entry replace an embedded one with the same
    /// id — that is how a stale shipped record gets corrected locally.
    pub fn merge(&mut self, extra: Vec<Model>) {
        for model in extra {
            match self.models.iter_mut().find(|m| m.id == model.id) {
                Some(existing) => *existing = model,
                None => self.models.push(model),
            }
        }
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Number of distinct model publishers in the catalog.
    pub fn provider_count(&self) -> usize {
        let mut providers: Vec<&str> = self.models.iter().map(|m| m.provider.as_str()).collect();
        providers.sort_unstable();
        providers.dedup();
        providers.len()
    }

    /// Look a model up by id, name, runtime tag, or free text.
    ///
    /// Exact matches win over fuzzy ones so that a precise query is never
    /// answered with a near miss.
    pub fn find(&self, query: &str) -> Option<&Model> {
        let q = query.trim().to_ascii_lowercase();
        self.models
            .iter()
            .find(|m| {
                m.id.to_ascii_lowercase() == q
                    || m.name.to_ascii_lowercase() == q
                    || m.ollama.as_deref().map(str::to_ascii_lowercase) == Some(q.clone())
            })
            .or_else(|| self.find_for_runtime(&q))
            .or_else(|| self.models.iter().find(|m| m.matches(&q)))
    }

    /// Resolve a name a local runtime used back to its catalog entry.
    ///
    /// Runtimes rename models: Ollama calls it `qwen3:8b`, LM Studio
    /// `qwen3-8b`, llama.cpp `Qwen3-8B-Q4_K_M.gguf`. Benchmarks arrive under
    /// those names and have to be matched to a catalog entry before the
    /// measurement can be compared with the estimate.
    pub fn find_for_runtime(&self, reference: &str) -> Option<&Model> {
        let wanted = crate::providers::normalize_model_name(reference);
        if wanted.is_empty() {
            return None;
        }
        self.models.iter().find(|m| {
            m.ollama
                .as_deref()
                .is_some_and(|tag| crate::providers::normalize_model_name(tag) == wanted)
                || crate::providers::normalize_model_name(&m.id) == wanted
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> ModelDb {
        ModelDb::embedded()
    }

    #[test]
    fn embedded_database_loads() {
        let db = db();
        assert!(
            db.len() >= 20,
            "seed database should hold at least 20 models"
        );
        assert!(db.models.iter().all(|m| m.params_b > 0.0));
    }

    #[test]
    fn llama_8b_q4_weights_are_realistic() {
        let db = db();
        let m = db.find("meta-llama/Llama-3.1-8B-Instruct").unwrap();
        let gb = m.weights_gb(Quant::Q4KM);
        // The published Q4_K_M GGUF is ~4.6 GiB.
        assert!((gb - 4.6).abs() < 0.3, "got {gb:.2} GB");
    }

    #[test]
    fn kv_cache_matches_hand_calculation() {
        let db = db();
        let m = db.find("meta-llama/Llama-3.1-8B-Instruct").unwrap();
        // 2 * 32 layers * 8 kv heads * 128 head dim * 8192 tokens * 2 bytes = 1 GiB
        assert!((m.kv_cache_gb(8192) - 1.0).abs() < 0.01);
    }

    #[test]
    fn moe_resident_footprint_is_much_smaller() {
        let db = db();
        let m = db.find("mistralai/Mixtral-8x7B-Instruct-v0.1").unwrap();
        assert!(m.is_moe());
        let full = m.total_memory_gb(Quant::Q4KM, 8192);
        let resident = m.moe_resident_gb(Quant::Q4KM, 8192);
        assert!(
            resident < full / 2.0,
            "full {full:.1} vs resident {resident:.1}"
        );
    }

    #[test]
    fn every_model_id_is_unique() {
        let db = db();
        let mut ids: Vec<&str> = db.models.iter().map(|m| m.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "the catalog has a duplicate id");
    }

    #[test]
    fn no_two_models_claim_the_same_runtime_tag() {
        // A shared tag means `d` downloads a different model than the one on
        // screen, so this is a correctness constraint rather than tidiness.
        let db = db();
        let mut tags: Vec<&str> = db
            .models
            .iter()
            .filter_map(|m| m.ollama.as_deref())
            .collect();
        tags.sort_unstable();
        let mut seen = None;
        for tag in tags {
            assert_ne!(Some(tag), seen, "two models claim the Ollama tag '{tag}'");
            seen = Some(tag);
        }
    }

    #[test]
    fn catalog_entries_are_internally_consistent() {
        for model in db().models {
            let id = &model.id;
            assert!(model.params_b > 0.0, "{id} has no parameter count");
            // Sentence-embedding models legitimately cap at a few hundred
            // tokens; anything shorter than that is a typo.
            assert!(
                model.context_length >= 128,
                "{id} has an implausible context length"
            );
            assert!(
                (1..=5).contains(&model.quality_tier),
                "{id} has an out-of-range quality tier"
            );
            if let Some(active) = model.active_params_b {
                assert!(
                    active > 0.0 && active < model.params_b,
                    "{id}: MoE active parameters must be a fraction of the total"
                );
            }
            // Partial geometry would be silently ignored by the KV estimate,
            // so it is either all present or all absent.
            let geometry = [
                model.layers.is_some(),
                model.kv_heads.is_some(),
                model.head_dim.is_some(),
            ];
            assert!(
                geometry.iter().all(|&p| p) || !geometry.iter().any(|&p| p),
                "{id} has partial KV geometry, which would be ignored"
            );
        }
    }

    #[test]
    fn provider_count_deduplicates() {
        let db = db();
        assert!(db.provider_count() > 1);
        assert!(db.provider_count() <= db.len());
    }

    #[test]
    fn merging_adds_new_models_and_replaces_matching_ids() {
        let mut db = db();
        let before = db.len();
        let existing_id = db.models[0].id.clone();

        let replacement = Model {
            name: "Renamed By User".to_string(),
            ..db.models[0].clone()
        };
        let addition = Model {
            id: "local/private-model".to_string(),
            name: "Private Model".to_string(),
            ..db.models[0].clone()
        };
        db.merge(vec![replacement, addition]);

        assert_eq!(db.len(), before + 1, "only the new id should grow the db");
        assert_eq!(db.find(&existing_id).unwrap().name, "Renamed By User");
        assert!(db.find("local/private-model").is_some());
    }

    #[test]
    fn quant_hierarchy_is_monotonic() {
        let mut prev = f64::MAX;
        for q in Quant::HIERARCHY {
            assert!(q.bits_per_weight() < prev);
            prev = q.bits_per_weight();
        }
    }
}
