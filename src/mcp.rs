//! An MCP server over stdio, exposing the fit engine as tools.
//!
//! The analysis is the same one `serve` puts behind HTTP; this puts it where
//! an assistant can reach it. Transport is newline-delimited JSON-RPC on
//! stdin/stdout, so **nothing else may ever be written to stdout** while this
//! runs — diagnostics go to stderr, which the spec reserves for exactly that.
//!
//! The protocol has two eras and real clients are still spread across both:
//!
//! - **Modern** (revision `2026-07-28` and later) is stateless. Every request
//!   carries its protocol version in `_meta`, and `server/discover` reports
//!   what the server speaks.
//! - **Legacy** (`2025-11-25` and earlier) opens with an `initialize`
//!   handshake that negotiates one version for the life of the connection.
//!
//! This server is dual-era: it answers both, and picks its behaviour from how
//! the client opens. That is the one combination the specification's
//! compatibility matrix marks as working against clients of either era.

use std::io::{BufRead, Write};

use serde_json::{Value, json};

use crate::display;
use crate::fit::{self, FitLevel, FitResult, SpeedConfig};
use crate::hardware::Hardware;
use crate::models::{Lookup, ModelDb, Quant, UseCase};
use crate::providers::ProviderRegistry;
use crate::verify;

/// The modern revision this server speaks.
const MODERN_VERSION: &str = "2026-07-28";

/// Legacy revisions accepted in an `initialize` handshake, newest first.
///
/// A legacy client gets its own version echoed back when it names one of
/// these, and the newest otherwise — the negotiation rule that revision
/// requires. Adding an entry is how support for another old client is
/// declared; nothing else reads this list.
const LEGACY_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const SERVER_NAME: &str = "llmspec";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Shown to the model. Worth the words: without them an assistant tends to
/// call `fit` for questions `plan` answers, and reports estimates as measured.
const INSTRUCTIONS: &str = "\
llmspec ranks local LLMs against the hardware it is running on.

Use `system` for what this machine has, `fit` for what it can run, `model` for
one model's full analysis, `search` to look a model up by name, `runtimes` for
what is installed and running locally, and `plan` for the hardware a model
would need — including which GPUs reach a target tokens/sec.

Throughput figures are estimates from memory bandwidth, not measurements. Say
so when reporting them.";

/// JSON-RPC and MCP error codes used below.
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;

/// Which era the connected client opened with.
///
/// It decides one thing: whether results carry `resultType`, which the modern
/// revision added and the legacy one does not define.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Era {
    Unknown,
    Modern,
    Legacy,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// The tools this server exposes.
///
/// Name, description and schema all hang off this one enum so `tools/list`
/// and `tools/call` cannot drift apart — advertising a tool that does not
/// dispatch is the failure mode worth designing out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tool {
    System,
    Runtimes,
    Fit,
    Model,
    Search,
    Plan,
    Verify,
}

impl Tool {
    const ALL: [Tool; 7] = [
        Tool::System,
        Tool::Runtimes,
        Tool::Fit,
        Tool::Model,
        Tool::Search,
        Tool::Plan,
        Tool::Verify,
    ];

    fn name(self) -> &'static str {
        match self {
            Tool::System => "system",
            Tool::Runtimes => "runtimes",
            Tool::Fit => "fit",
            Tool::Model => "model",
            Tool::Search => "search",
            Tool::Plan => "plan",
            Tool::Verify => "verify",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Tool::System => "Detected hardware",
            Tool::Runtimes => "Local runtimes",
            Tool::Fit => "Rank models for this machine",
            Tool::Model => "Analyse one model",
            Tool::Search => "Search the catalog",
            Tool::Plan => "Plan hardware for a model",
            Tool::Verify => "Check a model file on disk",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Tool::System => {
                "GPUs, VRAM, system RAM, CPU cores and the inference backend detected on \
                 this machine, with the memory bandwidth every throughput estimate is \
                 derived from."
            }
            Tool::Runtimes => {
                "Inference runtimes currently running locally (Ollama, llama.cpp, LM \
                 Studio, vLLM and others) and the models each one already has installed."
            }
            Tool::Fit => {
                "Rank the catalog by how well each model fits this machine. Returns the \
                 chosen quantization, placement, download size, context, estimated \
                 tokens/sec and an overall score. Answers \"what can I run\"."
            }
            Tool::Model => {
                "Everything known about one model plus its full analysis on this machine. \
                 Answers \"can I run this specific model, and how fast\"."
            }
            Tool::Search => {
                "Find models in the catalog by name, provider, parameter size or use case. \
                 Catalog lookup only — no hardware analysis."
            }
            Tool::Plan => {
                "What hardware a model needs, independent of this machine: VRAM for \
                 weights and KV cache at a given context and quantization, and — with \
                 target_tokens_per_sec — the memory bandwidth required and which GPUs \
                 clear it. Answers \"what would I have to buy\"."
            }
            Tool::Verify => {
                "Read a .gguf or .safetensors file's header and report whether it is                  structurally intact — the check for a download that stopped short.                  Reports the architecture, parameter count and quantization mix it                  actually contains. Reads the header only, not the weights."
            }
        }
    }

    fn schema(self) -> Value {
        match self {
            Tool::System | Tool::Runtimes => json!({
                "type": "object",
                "additionalProperties": false
            }),
            Tool::Fit => json!({
                "type": "object",
                "properties": {
                    "use_case": {
                        "type": "string",
                        "description": "Weight the scores for this use case. Defaults to \
                                        the configured one.",
                        "enum": ["general", "coding", "reasoning", "chat", "multimodal", "embedding"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum models to return (default 10).",
                        "minimum": 1
                    },
                    "provider": {
                        "type": "string",
                        "description": "Only models from this provider (substring match)."
                    },
                    "min_fit": {
                        "type": "string",
                        "description": "Minimum fit level.",
                        "enum": ["perfect", "good", "marginal", "too_tight"]
                    },
                    "min_tokens_per_sec": {
                        "type": "number",
                        "description": "Only models estimated to reach at least this speed."
                    },
                    "max_size_gb": {
                        "type": "number",
                        "description": "Only models whose download is at most this size."
                    },
                    "include_unrunnable": {
                        "type": "boolean",
                        "description": "Include models this machine cannot load (default false)."
                    }
                },
                "additionalProperties": false
            }),
            Tool::Model => json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Model name or id, e.g. \"Llama-3.1-8B\" or \"qwen2.5 7b\"."
                    },
                    "use_case": {
                        "type": "string",
                        "description": "Weight the scores for this use case.",
                        "enum": ["general", "coding", "reasoning", "chat", "multimodal", "embedding"]
                    }
                },
                "required": ["name"],
                "additionalProperties": false
            }),
            Tool::Search => json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free text matched against name, provider, size and use case."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum models to return (default 20).",
                        "minimum": 1
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
            Tool::Plan => json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name or id."
                    },
                    "context": {
                        "type": "integer",
                        "description": "Context length in tokens. Defaults to the model's maximum.",
                        "minimum": 1
                    },
                    "quant": {
                        "type": "string",
                        "description": "Quantization to plan for (default q4_k_m)."
                    },
                    "target_tokens_per_sec": {
                        "type": "number",
                        "description": "Also report the bandwidth and the GPUs needed to reach this speed."
                    }
                },
                "required": ["model"],
                "additionalProperties": false
            }),
            Tool::Verify => json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to a .gguf or .safetensors file on this machine."
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    fn from_name(name: &str) -> Option<Tool> {
        Tool::ALL.iter().copied().find(|t| t.name() == name)
    }

    fn definition(self) -> Value {
        json!({
            "name": self.name(),
            "title": self.title(),
            "description": self.description(),
            "inputSchema": self.schema()
        })
    }
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

pub struct Mcp {
    hw: Hardware,
    db: ModelDb,
    cfg: SpeedConfig,
    default_use_case: UseCase,
    registry: ProviderRegistry,
    era: Era,
}

impl Mcp {
    pub fn new(hw: Hardware, db: ModelDb, cfg: SpeedConfig, default_use_case: UseCase) -> Mcp {
        Mcp {
            hw,
            db,
            cfg,
            default_use_case,
            registry: ProviderRegistry::new(),
            era: Era::Unknown,
        }
    }

    /// Serve until the input stream ends.
    ///
    /// Closing stdin is the portable shutdown signal, and the only one on
    /// Windows, so end-of-input is a clean exit rather than an error.
    pub fn run(&mut self, input: &mut impl BufRead, output: &mut impl Write) -> Result<(), String> {
        let mut line = String::new();
        loop {
            line.clear();
            match input.read_line(&mut line) {
                Ok(0) => return Ok(()),
                Ok(_) => {}
                Err(e) => return Err(format!("cannot read from stdin: {e}")),
            }
            if line.trim().is_empty() {
                continue;
            }
            if let Some(response) = self.handle(line.trim()) {
                writeln!(output, "{response}")
                    .map_err(|e| format!("cannot write to stdout: {e}"))?;
                output
                    .flush()
                    .map_err(|e| format!("cannot flush stdout: {e}"))?;
            }
        }
    }

    /// Handle one message. `None` means the message was a notification, which
    /// JSON-RPC forbids answering.
    fn handle(&mut self, line: &str) -> Option<String> {
        let message: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(e) => {
                return Some(render(error_response(
                    Value::Null,
                    PARSE_ERROR,
                    &format!("invalid JSON: {e}"),
                    None,
                )));
            }
        };

        let Some(method) = message.get("method").and_then(Value::as_str) else {
            return Some(render(error_response(
                message.get("id").cloned().unwrap_or(Value::Null),
                INVALID_REQUEST,
                "request has no method",
                None,
            )));
        };

        // No id means a notification. We take no action on any of them, but
        // answering one would be a protocol violation.
        let id = message.get("id").cloned()?;
        let params = message.get("params").cloned().unwrap_or(Value::Null);

        Some(render(self.dispatch(method, &params, id)))
    }

    fn dispatch(&mut self, method: &str, params: &Value, id: Value) -> Value {
        // A request carrying protocol metadata is a modern one, whatever it
        // asks for. Recording that before the version check means an
        // unsupported-version error is still shaped for the era that sent it.
        if let Some(requested) = requested_version(params) {
            self.era = Era::Modern;
            if requested != MODERN_VERSION {
                return error_response(
                    id,
                    UNSUPPORTED_PROTOCOL_VERSION,
                    "Unsupported protocol version",
                    Some(json!({
                        "supported": [MODERN_VERSION],
                        "requested": requested
                    })),
                );
            }
        }

        match method {
            "server/discover" => {
                if self.era == Era::Unknown {
                    self.era = Era::Modern;
                }
                success(id, self.discover())
            }
            "initialize" => {
                self.era = Era::Legacy;
                success(id, self.initialize(params))
            }
            "ping" => success(id, self.complete(json!({}))),
            "tools/list" => success(
                id,
                self.complete(json!({
                    "tools": Tool::ALL.iter().map(|t| t.definition()).collect::<Vec<_>>()
                })),
            ),
            "tools/call" => self.tools_call(params, id),
            other => error_response(
                id,
                METHOD_NOT_FOUND,
                &format!("unknown method '{other}'"),
                None,
            ),
        }
    }

    /// The modern handshake-free announcement.
    fn discover(&self) -> Value {
        self.complete(json!({
            "supportedVersions": [MODERN_VERSION],
            "capabilities": { "tools": {} },
            "_meta": {
                "io.modelcontextprotocol/serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            },
            "instructions": INSTRUCTIONS
        }))
    }

    /// The legacy handshake. Echo the client's version when we know it,
    /// otherwise name our newest — an unknown version is not an error here,
    /// it just means the client decides whether it can live with the answer.
    fn initialize(&self, params: &Value) -> Value {
        let requested = params.get("protocolVersion").and_then(Value::as_str);
        let agreed = match requested {
            Some(v) if LEGACY_VERSIONS.contains(&v) => v,
            _ => LEGACY_VERSIONS[0],
        };
        json!({
            "protocolVersion": agreed,
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": {
                "name": SERVER_NAME,
                "title": "llmspec",
                "version": SERVER_VERSION
            },
            "instructions": INSTRUCTIONS
        })
    }

    fn tools_call(&mut self, params: &Value, id: Value) -> Value {
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return error_response(id, INVALID_PARAMS, "tools/call needs a tool name", None);
        };
        // An unknown tool is a protocol error: the model cannot fix it by
        // retrying with different arguments, which is the line the spec draws
        // between the two error channels.
        let Some(tool) = Tool::from_name(name) else {
            return error_response(id, INVALID_PARAMS, &format!("unknown tool: {name}"), None);
        };
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));

        // Everything past here is a tool execution error: bad arguments and
        // missing models are things the model can correct on a retry, so they
        // travel in the result rather than as a JSON-RPC error.
        let outcome = match tool {
            Tool::System => self.tool_system(),
            Tool::Runtimes => self.tool_runtimes(),
            Tool::Fit => self.tool_fit(&arguments),
            Tool::Model => self.tool_model(&arguments),
            Tool::Search => self.tool_search(&arguments),
            Tool::Plan => self.tool_plan(&arguments),
            Tool::Verify => self.tool_verify(&arguments),
        };

        success(id, self.tool_result(outcome))
    }

    fn tool_result(&self, outcome: Result<Value, String>) -> Value {
        match outcome {
            Ok(structured) => {
                // The serialized JSON goes in a text block as well: a client
                // that ignores structuredContent still sees the answer.
                let text = serde_json::to_string_pretty(&structured)
                    .unwrap_or_else(|e| format!("could not serialize result: {e}"));
                self.complete(json!({
                    "content": [{ "type": "text", "text": text }],
                    "structuredContent": structured,
                    "isError": false
                }))
            }
            Err(message) => self.complete(json!({
                "content": [{ "type": "text", "text": message }],
                "isError": true
            })),
        }
    }

    /// Stamp `resultType` on modern results only — the legacy revision does
    /// not define the field.
    fn complete(&self, mut result: Value) -> Value {
        if self.era == Era::Modern
            && let Some(object) = result.as_object_mut()
        {
            object.insert("resultType".to_string(), json!("complete"));
        }
        result
    }

    // -- tools --------------------------------------------------------------

    fn tool_system(&self) -> Result<Value, String> {
        to_value(&self.hw)
    }

    fn tool_runtimes(&mut self) -> Result<Value, String> {
        to_value(&self.registry.discover())
    }

    fn tool_fit(&self, args: &Value) -> Result<Value, String> {
        let target = self.use_case(args)?;
        let mut results = fit::analyze_all(&self.db.models, &self.hw, target, &self.cfg);

        if let Some(provider) = arg_str(args, "provider") {
            let needle = provider.to_lowercase();
            results.retain(|r| r.provider.to_lowercase().contains(&needle));
        }
        if let Some(raw) = arg_str(args, "min_fit") {
            let level = FitLevel::parse(raw).ok_or_else(|| {
                format!(
                    "unknown fit level '{raw}'. Try one of: {}",
                    FitLevel::hint()
                )
            })?;
            results.retain(|r| r.fit >= level);
        }
        if let Some(floor) = arg_f64(args, "min_tokens_per_sec")? {
            results.retain(|r| r.tokens_per_second >= floor);
        }
        if let Some(ceiling) = arg_f64(args, "max_size_gb")? {
            results.retain(|r| r.download_gb <= ceiling);
        }
        if !args
            .get("include_unrunnable")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            results.retain(FitResult::is_runnable);
        }
        results.truncate(arg_usize(args, "limit")?.unwrap_or(10));

        to_value(&display::JsonReport {
            system: &self.hw,
            use_case: target.as_str(),
            count: results.len(),
            models: &results,
        })
    }

    fn tool_model(&self, args: &Value) -> Result<Value, String> {
        let name = require_str(args, "name")?;
        let target = self.use_case(args)?;
        let model = self.lookup(name)?;
        to_value(&fit::analyze(model, &self.hw, target, &self.cfg))
    }

    /// One model by the name the assistant passed.
    ///
    /// An ambiguous name is an error listing the candidates rather than the
    /// first match: an assistant handed the wrong model's numbers would relay
    /// them with full confidence, while a list of ids is something it can
    /// retry from.
    fn lookup(&self, name: &str) -> Result<&crate::models::Model, String> {
        match self.db.resolve(name) {
            Lookup::NotFound => Err(format!(
                "no model matches '{name}'. Try the search tool first."
            )),
            found => found.into_result(name),
        }
    }

    fn tool_search(&self, args: &Value) -> Result<Value, String> {
        let query = require_str(args, "query")?;
        let mut matched: Vec<_> = self.db.models.iter().filter(|m| m.matches(query)).collect();
        if matched.is_empty() {
            return Err(format!("no model matches '{query}'"));
        }
        matched.truncate(arg_usize(args, "limit")?.unwrap_or(20));
        to_value(&json!({
            "count": matched.len(),
            "models": to_value(&matched)?
        }))
    }

    fn tool_plan(&self, args: &Value) -> Result<Value, String> {
        let name = require_str(args, "model")?;
        let model = self.lookup(name)?;
        let quant = match arg_str(args, "quant") {
            Some(raw) => Quant::parse(raw).ok_or_else(|| {
                format!(
                    "unknown quantization '{raw}'. Try one of: {}",
                    Quant::hint()
                )
            })?,
            None => Quant::Q4KM,
        };
        let context = match arg_usize(args, "context")? {
            Some(n) => u32::try_from(n).map_err(|_| format!("context {n} is out of range"))?,
            None => model.context_length,
        };
        let target_tps = arg_f64(args, "target_tokens_per_sec")?;
        to_value(&fit::plan(model, quant, context, &self.cfg, target_tps))
    }

    fn tool_verify(&self, args: &Value) -> Result<Value, String> {
        let path = require_str(args, "path")?;
        let report = verify::verify(std::path::Path::new(path))?;
        to_value(&report)
    }

    fn use_case(&self, args: &Value) -> Result<UseCase, String> {
        match arg_str(args, "use_case") {
            Some(raw) => UseCase::parse(raw).ok_or_else(|| {
                format!("unknown use case '{raw}'. Try one of: {}", UseCase::hint())
            }),
            None => Ok(self.default_use_case),
        }
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC helpers
// ---------------------------------------------------------------------------

fn success(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error_response(id: Value, code: i64, message: &str, data: Option<Value>) -> Value {
    let mut error = json!({ "code": code, "message": message });
    if let Some(data) = data
        && let Some(object) = error.as_object_mut()
    {
        object.insert("data".to_string(), data);
    }
    json!({ "jsonrpc": "2.0", "id": id, "error": error })
}

/// One message per line, and never an embedded newline — the stdio framing
/// depends on it, so this is compact JSON rather than pretty.
fn render(message: Value) -> String {
    serde_json::to_string(&message).unwrap_or_else(|e| {
        format!(r#"{{"jsonrpc":"2.0","id":null,"error":{{"code":{PARSE_ERROR},"message":"could not serialize response: {e}"}}}}"#)
    })
}

/// The protocol version a modern request declares, if it declares one.
fn requested_version(params: &Value) -> Option<&str> {
    params
        .get("_meta")?
        .get("io.modelcontextprotocol/protocolVersion")?
        .as_str()
}

fn to_value<T: serde::Serialize>(value: &T) -> Result<Value, String> {
    serde_json::to_value(value).map_err(|e| format!("could not serialize result: {e}"))
}

fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)?.as_str().filter(|s| !s.trim().is_empty())
}

fn require_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    arg_str(args, key).ok_or_else(|| format!("the '{key}' argument is required"))
}

/// Numbers, tolerating the string spelling. Some clients hand every argument
/// over as a string, and rejecting `"8"` for a number would be a needless
/// round trip for the model to recover from.
fn arg_f64(args: &Value, key: &str) -> Result<Option<f64>, String> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(Value::String(s)) => s
            .trim()
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("'{key}' must be a number, got '{s}'")),
        Some(other) => Err(format!("'{key}' must be a number, got {other}")),
    }
}

fn arg_usize(args: &Value, key: &str) -> Result<Option<usize>, String> {
    match arg_f64(args, key)? {
        None => Ok(None),
        Some(n) if n >= 1.0 && n.fract() == 0.0 => Ok(Some(n as usize)),
        Some(n) => Err(format!("'{key}' must be a positive whole number, got {n}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelDb;

    fn server() -> Mcp {
        Mcp::new(
            Hardware::reference_gpu("Test GPU", 900.0, 24.0),
            ModelDb::embedded(),
            SpeedConfig::default(),
            UseCase::General,
        )
    }

    /// Send one message and parse whatever came back.
    fn ask(server: &mut Mcp, message: Value) -> Value {
        let line = serde_json::to_string(&message).unwrap();
        let raw = server.handle(&line).expect("expected a response");
        serde_json::from_str(&raw).unwrap()
    }

    fn call(server: &mut Mcp, tool: &str, arguments: Value) -> Value {
        ask(
            server,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": tool, "arguments": arguments }
            }),
        )
    }

    #[test]
    fn a_legacy_client_gets_its_own_version_echoed_back() {
        let mut s = server();
        let response = ask(
            &mut s,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "protocolVersion": "2025-06-18", "capabilities": {} }
            }),
        );
        assert_eq!(response["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(response["result"]["serverInfo"]["name"], SERVER_NAME);
        assert!(response["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn a_legacy_client_naming_an_unknown_version_is_offered_our_newest() {
        // The legacy rule is to answer with a version we do support, not to
        // fail: the client decides whether it can live with the answer.
        let mut s = server();
        let response = ask(
            &mut s,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "protocolVersion": "1.0.0" }
            }),
        );
        assert_eq!(response["result"]["protocolVersion"], LEGACY_VERSIONS[0]);
    }

    #[test]
    fn discover_reports_the_modern_version_and_the_tools_capability() {
        let mut s = server();
        let response = ask(
            &mut s,
            json!({
                "jsonrpc": "2.0", "id": "d1", "method": "server/discover",
                "params": { "_meta": {
                    "io.modelcontextprotocol/protocolVersion": MODERN_VERSION
                }}
            }),
        );
        let result = &response["result"];
        assert_eq!(result["supportedVersions"][0], MODERN_VERSION);
        assert_eq!(result["resultType"], "complete");
        assert!(result["capabilities"]["tools"].is_object());
        assert_eq!(
            result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
            SERVER_NAME
        );
    }

    #[test]
    fn an_unsupported_modern_version_is_refused_with_the_list_we_do_support() {
        let mut s = server();
        let response = ask(
            &mut s,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "tools/list",
                "params": { "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "1900-01-01"
                }}
            }),
        );
        assert_eq!(response["error"]["code"], UNSUPPORTED_PROTOCOL_VERSION);
        assert_eq!(response["error"]["data"]["supported"][0], MODERN_VERSION);
        assert_eq!(response["error"]["data"]["requested"], "1900-01-01");
    }

    #[test]
    fn result_type_is_stamped_for_modern_clients_and_withheld_from_legacy_ones() {
        // The field is a modern addition; a legacy client has no definition
        // for it, so the same handler has to shape its result differently.
        let mut modern = server();
        ask(
            &mut modern,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "server/discover",
                "params": { "_meta": {
                    "io.modelcontextprotocol/protocolVersion": MODERN_VERSION
                }}
            }),
        );
        let listed = ask(
            &mut modern,
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
        );
        assert_eq!(listed["result"]["resultType"], "complete");

        let mut legacy = server();
        ask(
            &mut legacy,
            json!({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": { "protocolVersion": "2025-11-25" }
            }),
        );
        let listed = ask(
            &mut legacy,
            json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" }),
        );
        assert!(
            listed["result"].get("resultType").is_none(),
            "legacy clients must not be sent resultType"
        );
    }

    #[test]
    fn notifications_are_never_answered() {
        let mut s = server();
        assert!(
            s.handle(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
                .is_none()
        );
        assert!(
            s.handle(r#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{}}"#)
                .is_none()
        );
    }

    #[test]
    fn every_advertised_tool_dispatches() {
        // The guarantee the Tool enum exists to make: nothing can be listed
        // that tools/call does not know how to run.
        let mut s = server();
        let listed = ask(
            &mut s,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }),
        );
        let tools = listed["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), Tool::ALL.len());
        for tool in tools {
            let name = tool["name"].as_str().unwrap();
            assert!(
                Tool::from_name(name).is_some(),
                "{name} is listed but cannot be resolved"
            );
            assert!(
                !tool["description"].as_str().unwrap().is_empty(),
                "{name} has no description"
            );
            assert_eq!(
                tool["inputSchema"]["type"], "object",
                "{name}'s schema is not an object schema"
            );
        }
    }

    #[test]
    fn malformed_json_is_a_parse_error_rather_than_a_crash() {
        let mut s = server();
        let raw = s.handle("{not json").unwrap();
        let response: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(response["error"]["code"], PARSE_ERROR);
        assert!(response["id"].is_null());
    }

    #[test]
    fn an_unknown_method_is_a_method_not_found_error() {
        let mut s = server();
        let response = ask(
            &mut s,
            json!({ "jsonrpc": "2.0", "id": 7, "method": "resources/list" }),
        );
        assert_eq!(response["error"]["code"], METHOD_NOT_FOUND);
        assert_eq!(response["id"], 7);
    }

    #[test]
    fn an_unknown_tool_is_a_protocol_error_not_a_tool_error() {
        // The model cannot recover from this by adjusting arguments, which is
        // the line the spec draws between the two error channels.
        let mut s = server();
        let response = call(&mut s, "no_such_tool", json!({}));
        assert_eq!(response["error"]["code"], INVALID_PARAMS);
    }

    #[test]
    fn a_bad_argument_is_a_tool_error_the_model_can_retry() {
        let mut s = server();
        let response = call(&mut s, "fit", json!({ "use_case": "sorcery" }));
        assert!(
            response.get("error").is_none(),
            "must not be a protocol error"
        );
        assert_eq!(response["result"]["isError"], true);
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("sorcery"),
            "the message should name the input"
        );
        assert!(
            text.contains("general"),
            "the message should list the choices"
        );
    }

    #[test]
    fn an_ambiguous_model_name_lists_candidates_instead_of_guessing() {
        let mut s = server();
        let response = call(&mut s, "model", json!({ "name": "llama" }));
        assert_eq!(response["result"]["isError"], true);
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("matches"), "{text}");
        assert!(text.contains("meta-llama/"), "{text}");

        // An exact id still answers.
        let response = call(
            &mut s,
            "plan",
            json!({ "model": "meta-llama/Llama-3.1-8B-Instruct" }),
        );
        assert!(response["result"]["isError"] != true, "{response}");
    }

    #[test]
    fn a_missing_required_argument_names_the_argument() {
        let mut s = server();
        let response = call(&mut s, "model", json!({}));
        assert_eq!(response["result"]["isError"], true);
        assert!(
            response["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("'name'")
        );
    }

    #[test]
    fn fit_ranks_the_catalog_and_honours_limit() {
        let mut s = server();
        let response = call(&mut s, "fit", json!({ "limit": 3 }));
        assert_eq!(response["result"]["isError"], false);
        let report = &response["result"]["structuredContent"];
        assert_eq!(report["count"], 3);
        assert_eq!(report["models"].as_array().unwrap().len(), 3);
        assert!(report["system"].is_object());
    }

    #[test]
    fn fit_filters_compose() {
        let mut s = server();
        let response = call(
            &mut s,
            "fit",
            json!({ "limit": 50, "min_fit": "perfect", "max_size_gb": 6.0 }),
        );
        let models = response["result"]["structuredContent"]["models"]
            .as_array()
            .unwrap()
            .clone();
        for model in &models {
            assert_eq!(model["fit"], "Perfect", "min_fit was not applied");
            assert!(
                model["download_gb"].as_f64().unwrap() <= 6.0,
                "max_size_gb was not applied"
            );
        }
    }

    #[test]
    fn numeric_arguments_may_arrive_as_strings() {
        // Some clients stringify every argument; making the model retry over
        // that is a round trip for nothing.
        let mut s = server();
        let response = call(&mut s, "fit", json!({ "limit": "2" }));
        assert_eq!(response["result"]["isError"], false);
        assert_eq!(response["result"]["structuredContent"]["count"], 2);
    }

    #[test]
    fn a_tool_result_carries_the_same_answer_as_text_and_as_structure() {
        // A client that ignores structuredContent still has to see the answer.
        let mut s = server();
        let response = call(&mut s, "system", json!({}));
        let result = &response["result"];
        let text = result["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed, result["structuredContent"]);
    }

    #[test]
    fn search_finds_a_model_and_plan_sizes_it() {
        let mut s = server();
        let found = call(&mut s, "search", json!({ "query": "qwen", "limit": 5 }));
        assert_eq!(found["result"]["isError"], false);
        let name = found["result"]["structuredContent"]["models"][0]["name"]
            .as_str()
            .unwrap()
            .to_string();

        let planned = call(
            &mut s,
            "plan",
            json!({ "model": name, "target_tokens_per_sec": 30.0 }),
        );
        assert_eq!(planned["result"]["isError"], false);
        assert!(planned["result"]["structuredContent"].is_object());
    }

    #[test]
    fn a_model_that_is_not_in_the_catalog_points_at_the_search_tool() {
        let mut s = server();
        let response = call(&mut s, "model", json!({ "name": "definitely-not-a-model" }));
        assert_eq!(response["result"]["isError"], true);
        assert!(
            response["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("search")
        );
    }

    #[test]
    fn the_run_loop_answers_line_by_line_and_stops_at_end_of_input() {
        let mut s = server();
        let input = concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}"#,
            "\n",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            "\n",
            "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
            "\n"
        );
        let mut output = Vec::new();
        s.run(&mut input.as_bytes(), &mut output).unwrap();

        let text = String::from_utf8(output).unwrap();
        let lines: Vec<_> = text.lines().collect();
        // Two requests, one notification, one blank line: two responses.
        assert_eq!(lines.len(), 2, "got: {text}");
        for line in &lines {
            let parsed: Value = serde_json::from_str(line).expect("each line is one JSON message");
            assert_eq!(parsed["jsonrpc"], "2.0");
        }
        assert_eq!(
            serde_json::from_str::<Value>(lines[1]).unwrap()["result"]["tools"]
                .as_array()
                .unwrap()
                .len(),
            Tool::ALL.len()
        );
    }
}
