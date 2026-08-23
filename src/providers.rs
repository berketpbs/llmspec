//! Local runtime discovery and model management.
//!
//! llmspec talks to whatever inference server is already running on the
//! machine. Ollama has its own API; everything else in the list speaks the
//! OpenAI-compatible `/v1` surface, so one client covers llama.cpp's
//! `llama-server`, LM Studio, vLLM and Docker Model Runner.
//!
//! Discovery is a liveness probe against each runtime's default port. Nothing
//! leaves the machine: every endpoint here is loopback unless the user points
//! an env var somewhere else.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a discovery result stays fresh.
const DISCOVERY_TTL: u64 = 300;

/// Budget for the "is anything listening" check that precedes every request.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(300);

/// Probes must not stall the TUI when a port is open but unresponsive.
const PROBE_TIMEOUT: Duration = Duration::from_millis(1500);

/// Generation calls are allowed much longer — a cold model load is slow.
const GENERATE_TIMEOUT: Duration = Duration::from_secs(300);

/// Pulls take as long as they take; the cap only stops a wedged connection.
const PULL_TIMEOUT: Duration = Duration::from_secs(3600);

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .new_agent()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Runtime kinds
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeKind {
    Ollama,
    LlamaCpp,
    LmStudio,
    Vllm,
    DockerModelRunner,
    Mlx,
}

impl RuntimeKind {
    pub const ALL: [RuntimeKind; 6] = [
        RuntimeKind::Ollama,
        RuntimeKind::LlamaCpp,
        RuntimeKind::LmStudio,
        RuntimeKind::Vllm,
        RuntimeKind::DockerModelRunner,
        RuntimeKind::Mlx,
    ];

    pub fn label(self) -> &'static str {
        match self {
            RuntimeKind::Ollama => "Ollama",
            RuntimeKind::LlamaCpp => "llama.cpp",
            RuntimeKind::LmStudio => "LM Studio",
            RuntimeKind::Vllm => "vLLM",
            RuntimeKind::DockerModelRunner => "Docker Model Runner",
            RuntimeKind::Mlx => "MLX",
        }
    }

    /// Short lowercase identifier used by `--force-runtime` and JSON output.
    pub fn slug(self) -> &'static str {
        match self {
            RuntimeKind::Ollama => "ollama",
            RuntimeKind::LlamaCpp => "llamacpp",
            RuntimeKind::LmStudio => "lmstudio",
            RuntimeKind::Vllm => "vllm",
            RuntimeKind::DockerModelRunner => "docker",
            RuntimeKind::Mlx => "mlx",
        }
    }

    pub fn parse(s: &str) -> Option<RuntimeKind> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace(['_', '-', '.', ' '], "")
            .as_str()
        {
            "ollama" => Some(RuntimeKind::Ollama),
            "llamacpp" | "llama" | "llamaserver" | "ggml" => Some(RuntimeKind::LlamaCpp),
            "lmstudio" | "lms" => Some(RuntimeKind::LmStudio),
            "vllm" => Some(RuntimeKind::Vllm),
            "docker" | "dmr" | "dockermodelrunner" => Some(RuntimeKind::DockerModelRunner),
            "mlx" | "mlxlm" => Some(RuntimeKind::Mlx),
            _ => None,
        }
    }

    /// Default base URL, overridable per runtime by an environment variable.
    ///
    /// The defaults name `127.0.0.1` rather than `localhost` on purpose: on
    /// Windows `localhost` resolves to `::1` first, and probing a port nothing
    /// is bound to then waits out the whole connect timeout instead of being
    /// refused immediately.
    pub fn base_url(self) -> String {
        let (var, default) = match self {
            RuntimeKind::Ollama => ("OLLAMA_HOST", "http://127.0.0.1:11434"),
            RuntimeKind::LlamaCpp => ("LLAMA_CPP_HOST", "http://127.0.0.1:8080"),
            RuntimeKind::LmStudio => ("LMSTUDIO_HOST", "http://127.0.0.1:1234"),
            RuntimeKind::Vllm => ("VLLM_HOST", "http://127.0.0.1:8000"),
            RuntimeKind::DockerModelRunner => ("DOCKER_MODEL_HOST", "http://127.0.0.1:12434"),
            RuntimeKind::Mlx => ("MLX_HOST", "http://127.0.0.1:8080"),
        };
        let raw = env::var(var).unwrap_or_else(|_| default.to_string());
        normalize_url(&raw)
    }

    /// Path that lists the runtime's local models.
    fn list_path(self) -> &'static str {
        match self {
            RuntimeKind::Ollama => "/api/tags",
            RuntimeKind::DockerModelRunner => "/engines/v1/models",
            _ => "/v1/models",
        }
    }

    /// Path used for a chat-completion request.
    fn chat_path(self) -> &'static str {
        match self {
            RuntimeKind::Ollama => "/api/generate",
            RuntimeKind::DockerModelRunner => "/engines/v1/chat/completions",
            _ => "/v1/chat/completions",
        }
    }

    /// Relative single-stream throughput against a llama.cpp baseline of 1.0.
    ///
    /// MLX reads weights straight out of unified memory with kernels tuned for
    /// Apple Silicon; vLLM's paged attention mostly pays off under batching,
    /// so single-stream gains are modest. Ollama wraps llama.cpp and is a wash.
    pub fn speed_factor(self) -> f64 {
        match self {
            RuntimeKind::LlamaCpp | RuntimeKind::Ollama | RuntimeKind::DockerModelRunner => 1.0,
            RuntimeKind::LmStudio => 0.98,
            RuntimeKind::Mlx => 1.15,
            RuntimeKind::Vllm => 1.10,
        }
    }

    /// Weight format the runtime loads. GGUF runtimes can only run models with
    /// published GGUF builds; MLX and vLLM read the original safetensors.
    pub fn needs_gguf(self) -> bool {
        !matches!(self, RuntimeKind::Mlx | RuntimeKind::Vllm)
    }

    /// Command that downloads a model for this runtime.
    pub fn install_command(self, model_ref: &str) -> String {
        match self {
            RuntimeKind::Ollama => format!("ollama pull {model_ref}"),
            RuntimeKind::DockerModelRunner => format!("docker model pull {model_ref}"),
            RuntimeKind::Mlx => format!("mlx_lm.convert --hf-path {model_ref}"),
            RuntimeKind::LlamaCpp => format!("llama-server -hf {model_ref}"),
            RuntimeKind::LmStudio => format!("lms get {model_ref}"),
            RuntimeKind::Vllm => format!("vllm serve {model_ref}"),
        }
    }

    /// Command that starts an interactive session with an installed model.
    ///
    /// For the server-shaped runtimes this is the same command that installs
    /// it — they fetch on first use and then serve — so the two differ only
    /// where the runtime separates the steps.
    pub fn run_command(self, model_ref: &str) -> String {
        match self {
            RuntimeKind::Ollama => format!("ollama run {model_ref}"),
            RuntimeKind::DockerModelRunner => format!("docker model run {model_ref}"),
            RuntimeKind::Mlx => format!("mlx_lm.generate --model {model_ref} --prompt 'hello'"),
            RuntimeKind::LlamaCpp => format!("llama-server -hf {model_ref}"),
            RuntimeKind::LmStudio => format!("lms load {model_ref}"),
            RuntimeKind::Vllm => format!("vllm serve {model_ref}"),
        }
    }

    /// Whether this runtime identifies models by a registry tag of its own
    /// rather than by the upstream repository id.
    pub fn uses_own_registry(self) -> bool {
        matches!(self, RuntimeKind::Ollama | RuntimeKind::DockerModelRunner)
    }
}

impl fmt::Display for RuntimeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

fn normalize_url(raw: &str) -> String {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        trimmed.to_string()
    } else {
        // `OLLAMA_HOST=127.0.0.1:11434` is a documented form; add the scheme.
        format!("http://{trimmed}")
    }
}

// ---------------------------------------------------------------------------
// Installed models
// ---------------------------------------------------------------------------

/// A model a local runtime reports as available.
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

/// Quantization and format markers that appear on a file name but are not
/// part of the model's identity. Longest first, so `q4_k_m` is stripped whole
/// rather than leaving a stray `_m`.
const FORMAT_SUFFIXES: &[&str] = &[
    "q2-k", "q3-k-s", "q3-k-m", "q3-k-l", "q4-k-s", "q4-k-m", "q5-k-s", "q5-k-m", "q4-0", "q4-1",
    "q5-0", "q5-1", "q6-k", "q8-0", "iq2-xxs", "iq3-xxs", "iq4-nl", "iq4-xs", "bf16", "fp16",
    "fp32", "f16", "f32", "gguf", "mlx", "gptq", "awq", "int4", "int8", "2bit", "3bit", "4bit",
    "5bit", "6bit", "8bit",
];

/// Reduce a model name to a form comparable across runtimes.
///
/// The same weights are called `Qwen/Qwen2.5-7B-Instruct` upstream,
/// `qwen2.5-7b-instruct` by LM Studio and
/// `Qwen2.5-7B-Instruct-Q4_K_M.gguf` by llama.cpp. Dropping the publisher
/// prefix, the file extension and the quantization marker makes those three
/// the same string.
fn normalize_model_name(name: &str) -> String {
    // Everything before the last slash is the publisher, which only one of
    // the runtimes reports.
    let tail = name.rsplit('/').next().unwrap_or(name);
    let lowered = tail.to_ascii_lowercase();

    let mut normalized = String::with_capacity(lowered.len());
    for ch in lowered.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch);
        } else if !normalized.ends_with('-') {
            normalized.push('-');
        }
    }

    // Strip trailing format markers repeatedly: `...-instruct-q4-k-m-gguf`
    // carries two of them.
    let mut trimmed = normalized.trim_matches('-').to_string();
    loop {
        let before = trimmed.len();
        for suffix in FORMAT_SUFFIXES {
            if let Some(stem) = trimmed.strip_suffix(suffix)
                && stem.ends_with('-')
            {
                trimmed.truncate(stem.len() - 1);
                break;
            }
        }
        if trimmed.len() == before {
            return trimmed;
        }
    }
}

/// The models local runtimes report as already downloaded.
///
/// Matching is deliberately conservative. A wrong "already installed" tick
/// sends the user looking for a file that is not there, while a missed one
/// only costs them a redundant `ollama pull` that returns immediately.
#[derive(Debug, Default, Clone)]
pub struct InstalledIndex {
    /// Names exactly as the runtimes reported them, lowercased.
    raw: HashSet<String>,
    /// The same names reduced by [`normalize_model_name`].
    normalized: HashSet<String>,
}

impl InstalledIndex {
    pub fn from_models(models: Vec<InstalledModel>) -> InstalledIndex {
        let mut index = InstalledIndex::default();
        for model in models {
            index.insert(&model.name);
        }
        index
    }

    pub fn insert(&mut self, name: &str) {
        self.raw.insert(name.to_ascii_lowercase());
        self.normalized.insert(normalize_model_name(name));
    }

    /// Number of distinct models known to be installed.
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// True when either the runtime-specific tag or the upstream id is present.
    pub fn contains(&self, ollama_tag: Option<&str>, model_id: &str) -> bool {
        if let Some(tag) = ollama_tag
            && self.raw.contains(&tag.to_ascii_lowercase())
        {
            return true;
        }
        self.normalized.contains(&normalize_model_name(model_id))
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A client for one local runtime.
#[derive(Debug, Clone)]
pub struct Runtime {
    pub kind: RuntimeKind,
    pub base_url: String,
}

impl Runtime {
    pub fn new(kind: RuntimeKind) -> Runtime {
        Runtime {
            kind,
            base_url: kind.base_url(),
        }
    }

    pub fn with_url(kind: RuntimeKind, url: &str) -> Runtime {
        Runtime {
            kind,
            base_url: normalize_url(url),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// True when something is listening on the runtime's port.
    ///
    /// Discovery probes six runtimes, five of which are usually absent. A TCP
    /// connect answers "nothing there" in microseconds, where an HTTP request
    /// to a dead port can wait out the full timeout.
    fn port_is_open(&self) -> bool {
        let Some(authority) = self
            .base_url
            .split_once("://")
            .map(|(_, rest)| rest.split('/').next().unwrap_or(rest))
        else {
            return true;
        };
        // A host with no port, or one that needs real DNS, is left to the HTTP
        // client rather than guessed at here.
        let Ok(addrs) = authority.to_socket_addrs() else {
            return true;
        };
        addrs
            .into_iter()
            .any(|addr| TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).is_ok())
    }

    /// Models the runtime has locally.
    pub fn list_models(&self) -> Result<Vec<InstalledModel>, String> {
        if !self.port_is_open() {
            return Err(format!("nothing listening at {}", self.base_url));
        }
        let url = self.url(self.kind.list_path());
        let mut resp = agent(PROBE_TIMEOUT)
            .get(&url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let text = resp
            .body_mut()
            .read_to_string()
            .map_err(|e| format!("reading {url}: {e}"))?;
        match self.kind {
            RuntimeKind::Ollama => parse_ollama_tags(&text),
            _ => parse_openai_models(&text, self.kind.slug()),
        }
    }

    /// Download a model. Only runtimes with a pull API support this.
    pub fn pull(&self, model_ref: &str) -> Result<(), String> {
        match self.kind {
            RuntimeKind::Ollama => {
                let url = self.url("/api/pull");
                agent(PULL_TIMEOUT)
                    .post(&url)
                    .send_json(PullRequest {
                        name: model_ref,
                        stream: false,
                    })
                    .map_err(|e| format!("POST {url} ({model_ref}): {e}"))?;
                Ok(())
            }
            other => Err(format!(
                "{} has no pull API — run: {}",
                other.label(),
                other.install_command(model_ref)
            )),
        }
    }

    /// Generate `max_tokens` from `prompt` and report what the run cost.
    ///
    /// Ollama returns its own token counters, which are exact. The
    /// OpenAI-compatible runtimes report a token count in `usage` but no
    /// timing breakdown, so time-to-first-token is only available from a
    /// streaming response and is left unset for them.
    pub fn generate(
        &self,
        model_ref: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<Sample, String> {
        let url = self.url(self.kind.chat_path());
        let http = agent(GENERATE_TIMEOUT);
        let started = std::time::Instant::now();

        let text = match self.kind {
            RuntimeKind::Ollama => {
                let body = OllamaGenerate {
                    model: model_ref,
                    prompt,
                    stream: false,
                    options: OllamaOptions {
                        num_predict: max_tokens,
                    },
                };
                http.post(&url)
                    .send_json(body)
                    .map_err(|e| format!("POST {url}: {e}"))?
                    .body_mut()
                    .read_to_string()
                    .map_err(|e| format!("reading {url}: {e}"))?
            }
            _ => {
                let body = ChatRequest {
                    model: model_ref,
                    max_tokens,
                    stream: false,
                    messages: vec![ChatMessage {
                        role: "user",
                        content: prompt,
                    }],
                };
                http.post(&url)
                    .send_json(body)
                    .map_err(|e| format!("POST {url}: {e}"))?
                    .body_mut()
                    .read_to_string()
                    .map_err(|e| format!("reading {url}: {e}"))?
            }
        };

        let elapsed = started.elapsed().as_secs_f64();
        let mut sample = match self.kind {
            RuntimeKind::Ollama => parse_ollama_sample(&text)?,
            _ => parse_openai_sample(&text, elapsed)?,
        };
        sample.wall_seconds = elapsed;
        Ok(sample)
    }
}

/// One measured generation run.
#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    /// Tokens the model produced.
    pub output_tokens: u32,
    /// Tokens in the prompt, when the runtime reports them.
    pub prompt_tokens: u32,
    /// Seconds spent generating (excludes prompt processing when known).
    pub generate_seconds: f64,
    /// Seconds until the first token, when the runtime reports prompt timing.
    pub ttft_seconds: Option<f64>,
    /// Total wall-clock time for the request.
    pub wall_seconds: f64,
}

impl Sample {
    pub fn tokens_per_second(&self) -> f64 {
        if self.generate_seconds > 0.0 {
            f64::from(self.output_tokens) / self.generate_seconds
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PullRequest<'a> {
    name: &'a str,
    stream: bool,
}

#[derive(Serialize)]
struct OllamaOptions {
    num_predict: u32,
}

#[derive(Serialize)]
struct OllamaGenerate<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    stream: bool,
    messages: Vec<ChatMessage<'a>>,
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

#[derive(Deserialize)]
struct OpenAiModels {
    #[serde(default)]
    data: Vec<OpenAiModel>,
}

#[derive(Deserialize)]
struct OpenAiModel {
    id: String,
    #[serde(default)]
    created: Option<i64>,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    eval_duration: u64,
    #[serde(default)]
    prompt_eval_count: u32,
    #[serde(default)]
    prompt_eval_duration: u64,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    usage: Option<Usage>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens: u32,
}

// ---------------------------------------------------------------------------
// Parsing (pure, testable without a network)
// ---------------------------------------------------------------------------

fn parse_ollama_tags(text: &str) -> Result<Vec<InstalledModel>, String> {
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

fn parse_openai_models(text: &str, provider: &str) -> Result<Vec<InstalledModel>, String> {
    let parsed: OpenAiModels =
        serde_json::from_str(text).map_err(|e| format!("parsing /v1/models: {e}"))?;
    Ok(parsed
        .data
        .into_iter()
        .map(|m| InstalledModel {
            name: m.id,
            provider: provider.to_string(),
            // The OpenAI model listing carries no size field.
            size_bytes: 0,
            modified_at: m.created.map(|c| c.to_string()).unwrap_or_default(),
        })
        .collect())
}

/// Ollama reports durations in nanoseconds.
fn parse_ollama_sample(text: &str) -> Result<Sample, String> {
    let parsed: OllamaGenerateResponse =
        serde_json::from_str(text).map_err(|e| format!("parsing /api/generate: {e}"))?;
    if let Some(err) = parsed.error {
        return Err(format!("ollama: {err}"));
    }
    const NANOS: f64 = 1e9;
    Ok(Sample {
        output_tokens: parsed.eval_count,
        prompt_tokens: parsed.prompt_eval_count,
        generate_seconds: parsed.eval_duration as f64 / NANOS,
        ttft_seconds: if parsed.prompt_eval_duration > 0 {
            Some(parsed.prompt_eval_duration as f64 / NANOS)
        } else {
            None
        },
        wall_seconds: 0.0,
    })
}

fn parse_openai_sample(text: &str, elapsed: f64) -> Result<Sample, String> {
    let parsed: ChatResponse =
        serde_json::from_str(text).map_err(|e| format!("parsing chat completion: {e}"))?;
    if let Some(err) = parsed.error {
        return Err(format!("runtime error: {err}"));
    }
    let usage = parsed
        .usage
        .ok_or_else(|| "response carried no usage counters".to_string())?;
    Ok(Sample {
        output_tokens: usage.completion_tokens,
        prompt_tokens: usage.prompt_tokens,
        // Without streaming there is no prompt/generate split; the whole
        // request is attributed to generation, which slightly understates
        // tok/s on long prompts.
        generate_seconds: elapsed,
        ttft_seconds: None,
        wall_seconds: elapsed,
    })
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// A runtime found listening on this machine.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveredRuntime {
    pub kind: RuntimeKind,
    pub name: &'static str,
    pub base_url: String,
    pub model_count: usize,
    /// Disk used by the runtime's models, when it reports sizes. The
    /// OpenAI-compatible listing has no size field, so this stays `None` for
    /// everything except Ollama.
    pub disk_gb: Option<f64>,
}

/// Discovers runtimes and caches what they report.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    cache: HashMap<RuntimeKind, (Vec<InstalledModel>, u64)>,
    discovered: Option<(Vec<DiscoveredRuntime>, u64)>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        ProviderRegistry::default()
    }

    /// Probe every known runtime. Cached for `DISCOVERY_TTL`.
    pub fn discover(&mut self) -> Vec<DiscoveredRuntime> {
        let now = now_secs();
        if let Some((found, ts)) = &self.discovered
            && now.saturating_sub(*ts) < DISCOVERY_TTL
        {
            return found.clone();
        }

        let mut found = Vec::new();
        let mut seen_urls: Vec<String> = Vec::new();
        for kind in RuntimeKind::ALL {
            let runtime = Runtime::new(kind);
            // llama.cpp and MLX share a default port; whichever answers first
            // claims it rather than being reported twice.
            let endpoint = format!("{}{}", runtime.base_url, kind.list_path());
            if seen_urls.contains(&endpoint) {
                continue;
            }
            let Ok(models) = runtime.list_models() else {
                continue;
            };
            seen_urls.push(endpoint);
            let disk_gb: f64 = models.iter().map(InstalledModel::size_gb).sum();
            found.push(DiscoveredRuntime {
                kind,
                name: kind.label(),
                base_url: runtime.base_url.clone(),
                model_count: models.len(),
                disk_gb: (disk_gb > 0.0).then_some(disk_gb),
            });
            self.cache.insert(kind, (models, now));
        }

        self.discovered = Some((found.clone(), now));
        found
    }

    /// Installed models across every live runtime.
    ///
    /// Reuses whatever [`ProviderRegistry::discover`] already fetched, so the
    /// common "discover then list" pairing costs one probe rather than two.
    /// A caller that wants fresh data builds a new registry.
    pub fn list_all_models(&mut self) -> Result<Vec<InstalledModel>, String> {
        let now = now_secs();
        let mut all = Vec::new();
        for kind in RuntimeKind::ALL {
            let cached = self.cache.get(&kind).and_then(|(models, ts)| {
                (now.saturating_sub(*ts) < DISCOVERY_TTL).then(|| models.clone())
            });
            let models = match cached {
                Some(models) => models,
                None => match Runtime::new(kind).list_models() {
                    Ok(models) => {
                        self.cache.insert(kind, (models.clone(), now));
                        models
                    }
                    // A runtime that is not running is not an error.
                    Err(_) => continue,
                },
            };
            all.extend(models);
        }
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_starts_empty() {
        let registry = ProviderRegistry::new();
        assert!(registry.cache.is_empty());
        assert!(registry.discovered.is_none());
    }

    #[test]
    fn discovery_does_not_panic_without_runtimes() {
        let mut registry = ProviderRegistry::new();
        let found = registry.discover();
        // Nothing is guaranteed to run in CI — only that probing is safe.
        println!("discovered {} runtime(s)", found.len());
    }

    #[test]
    fn runtime_slugs_round_trip() {
        for kind in RuntimeKind::ALL {
            assert_eq!(RuntimeKind::parse(kind.slug()), Some(kind));
            assert_eq!(RuntimeKind::parse(kind.label()), Some(kind));
        }
    }

    #[test]
    fn runtime_aliases_parse() {
        assert_eq!(RuntimeKind::parse("llama.cpp"), Some(RuntimeKind::LlamaCpp));
        assert_eq!(RuntimeKind::parse("LM-Studio"), Some(RuntimeKind::LmStudio));
        assert_eq!(
            RuntimeKind::parse("dmr"),
            Some(RuntimeKind::DockerModelRunner)
        );
        assert_eq!(RuntimeKind::parse("nonesuch"), None);
    }

    #[test]
    fn gguf_runtimes_are_marked() {
        assert!(RuntimeKind::Ollama.needs_gguf());
        assert!(RuntimeKind::LlamaCpp.needs_gguf());
        assert!(!RuntimeKind::Mlx.needs_gguf());
        assert!(!RuntimeKind::Vllm.needs_gguf());
    }

    #[test]
    fn urls_get_a_scheme_and_lose_trailing_slashes() {
        assert_eq!(normalize_url("127.0.0.1:11434"), "http://127.0.0.1:11434");
        assert_eq!(
            normalize_url("http://localhost:8080/"),
            "http://localhost:8080"
        );
        assert_eq!(
            normalize_url("https://gpu.internal:9000"),
            "https://gpu.internal:9000"
        );
    }

    #[test]
    fn parse_tags_handles_valid_response() {
        let json = r#"{"models":[{"name":"llama3.2:1b","size":2019393189,"modified_at":"2025-01-15T10:30:00Z"}]}"#;
        let result = parse_ollama_tags(json).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, "llama3.2:1b");
        assert_eq!(result[0].size_bytes, 2019393189);
        assert_eq!(result[0].provider, "ollama");
    }

    #[test]
    fn parse_tags_handles_empty_models_array() {
        assert!(parse_ollama_tags(r#"{"models":[]}"#).unwrap().is_empty());
    }

    #[test]
    fn parse_tags_defaults_missing_optional_fields() {
        let result = parse_ollama_tags(r#"{"models":[{"name":"test"}]}"#).unwrap();
        assert_eq!(result[0].name, "test");
        assert_eq!(result[0].size_bytes, 0);
        assert_eq!(result[0].modified_at, "");
    }

    #[test]
    fn parse_tags_rejects_malformed_json() {
        let err = parse_ollama_tags(r#"{"models":[{"name":"test""#).unwrap_err();
        assert!(err.contains("parsing"));
    }

    #[test]
    fn parse_tags_handles_multiple_entries() {
        let json =
            r#"{"models":[{"name":"llama3.2:1b","size":1000},{"name":"mistral:7b","size":2000}]}"#;
        let result = parse_ollama_tags(json).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[1].name, "mistral:7b");
    }

    #[test]
    fn parse_openai_model_listing() {
        let json = r#"{"object":"list","data":[{"id":"qwen2.5-7b-instruct","object":"model","created":1737000000},{"id":"phi-4"}]}"#;
        let result = parse_openai_models(json, "lmstudio").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "qwen2.5-7b-instruct");
        assert_eq!(result[0].provider, "lmstudio");
        assert_eq!(result[1].modified_at, "");
    }

    #[test]
    fn ollama_sample_converts_nanoseconds() {
        let json = r#"{"eval_count":120,"eval_duration":4000000000,"prompt_eval_count":18,"prompt_eval_duration":250000000}"#;
        let sample = parse_ollama_sample(json).unwrap();
        assert_eq!(sample.output_tokens, 120);
        assert!((sample.generate_seconds - 4.0).abs() < 1e-9);
        assert!((sample.ttft_seconds.unwrap() - 0.25).abs() < 1e-9);
        assert!((sample.tokens_per_second() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn ollama_sample_surfaces_errors() {
        let err = parse_ollama_sample(r#"{"error":"model not found"}"#).unwrap_err();
        assert!(err.contains("model not found"));
    }

    #[test]
    fn openai_sample_uses_wall_time() {
        let json = r#"{"usage":{"completion_tokens":64,"prompt_tokens":12}}"#;
        let sample = parse_openai_sample(json, 2.0).unwrap();
        assert_eq!(sample.output_tokens, 64);
        assert_eq!(sample.prompt_tokens, 12);
        assert!((sample.tokens_per_second() - 32.0).abs() < 1e-6);
        assert!(sample.ttft_seconds.is_none());
    }

    #[test]
    fn openai_sample_without_usage_is_an_error() {
        assert!(parse_openai_sample(r#"{"choices":[]}"#, 1.0).is_err());
    }

    #[test]
    fn normalization_unifies_the_runtimes_spellings_of_one_model() {
        // Upstream repo id, LM Studio's listing, and a llama.cpp GGUF file
        // name all describe the same weights.
        let expected = "qwen2-5-7b-instruct";
        for spelling in [
            "Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5-7b-instruct",
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "qwen2.5-7b-instruct-GGUF",
            "lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit",
        ] {
            assert_eq!(
                normalize_model_name(spelling),
                expected,
                "failed on {spelling}"
            );
        }
    }

    #[test]
    fn normalization_keeps_distinct_models_distinct() {
        // The dangerous failure is a false positive: telling the user a model
        // is on disk when only a relative of it is.
        assert_ne!(
            normalize_model_name("phi-4"),
            normalize_model_name("phi-4-mini")
        );
        assert_ne!(
            normalize_model_name("Qwen/Qwen3-4B"),
            normalize_model_name("Qwen/Qwen3-8B")
        );
        assert_ne!(
            normalize_model_name("meta-llama/Llama-3.2-1B-Instruct"),
            normalize_model_name("meta-llama/Llama-3.2-3B-Instruct")
        );
    }

    #[test]
    fn stacked_format_suffixes_are_all_stripped() {
        assert_eq!(normalize_model_name("Model-Q4_K_M-GGUF"), "model");
        assert_eq!(normalize_model_name("Model.f16.gguf"), "model");
        // A suffix that is the whole name is not a model name; leave it be
        // rather than reducing it to nothing.
        assert_eq!(normalize_model_name("gguf"), "gguf");
    }

    #[test]
    fn the_index_matches_by_tag_and_by_id() {
        let index = InstalledIndex::from_models(vec![
            InstalledModel {
                name: "qwen2.5:7b".into(),
                provider: "ollama".into(),
                size_bytes: 0,
                modified_at: String::new(),
            },
            InstalledModel {
                name: "phi-4-mini-instruct".into(),
                provider: "lmstudio".into(),
                size_bytes: 0,
                modified_at: String::new(),
            },
        ]);

        // Ollama is matched on its own tag, exactly.
        assert!(index.contains(Some("qwen2.5:7b"), "Qwen/Qwen2.5-7B-Instruct"));
        assert!(index.contains(Some("QWEN2.5:7B"), "irrelevant"));
        // LM Studio is matched on the normalized upstream id.
        assert!(index.contains(None, "microsoft/Phi-4-mini-instruct"));
        // And a different model is not.
        assert!(!index.contains(Some("phi4:14b"), "microsoft/phi-4"));
        assert!(!index.contains(None, "Qwen/Qwen3-8B"));
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
    }

    #[test]
    fn an_empty_index_matches_nothing() {
        let index = InstalledIndex::default();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(!index.contains(Some("anything:7b"), "any/model"));
    }

    #[test]
    fn installed_model_size_gb() {
        let model = InstalledModel {
            name: "test".to_string(),
            provider: "ollama".to_string(),
            size_bytes: 1024 * 1024 * 1024,
            modified_at: String::new(),
        };
        assert!((model.size_gb() - 1.0).abs() < 0.01);
    }

    #[test]
    fn install_and_run_commands_are_runtime_specific() {
        assert_eq!(
            RuntimeKind::Ollama.install_command("llama3.2:1b"),
            "ollama pull llama3.2:1b"
        );
        assert_eq!(
            RuntimeKind::Ollama.run_command("llama3.2:1b"),
            "ollama run llama3.2:1b"
        );
        assert!(
            RuntimeKind::Vllm
                .install_command("Qwen/Qwen2.5-7B-Instruct")
                .starts_with("vllm serve")
        );
        // Every runtime must produce a non-empty command naming the model.
        for kind in RuntimeKind::ALL {
            for command in [
                kind.install_command("some/model"),
                kind.run_command("some/model"),
            ] {
                assert!(command.contains("some/model"), "{}: {command}", kind.slug());
            }
        }
    }

    #[test]
    fn registry_backed_runtimes_are_marked() {
        // These name models by their own tag; the rest use the upstream repo id.
        assert!(RuntimeKind::Ollama.uses_own_registry());
        assert!(RuntimeKind::DockerModelRunner.uses_own_registry());
        assert!(!RuntimeKind::Vllm.uses_own_registry());
        assert!(!RuntimeKind::LlamaCpp.uses_own_registry());
    }
}
