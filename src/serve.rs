//! A small read-only HTTP API over the fit engine.
//!
//! The analysis is pure and takes milliseconds, so the server is a plain
//! blocking accept loop: one connection at a time, no runtime, no extra
//! dependency. It binds loopback by default because it reports what hardware
//! the machine has, which is not something to expose by accident.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use crate::display;
use crate::fit::{self, FitLevel, FitResult, RunMode, SpeedConfig};
use crate::hardware::Hardware;
use crate::models::{ModelDb, Quant, UseCase};
use crate::providers::ProviderRegistry;

/// Cap on the request line plus headers. Anything larger is a client bug.
const MAX_HEADER_BYTES: usize = 8 * 1024;

/// A slow or wedged client must not hold the single-threaded loop.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

pub struct Server {
    hw: Hardware,
    db: ModelDb,
    cfg: SpeedConfig,
    default_use_case: UseCase,
    registry: ProviderRegistry,
}

impl Server {
    pub fn new(hw: Hardware, db: ModelDb, cfg: SpeedConfig, default_use_case: UseCase) -> Server {
        Server {
            hw,
            db,
            cfg,
            default_use_case,
            registry: ProviderRegistry::new(),
        }
    }

    /// Bind and serve until the process is interrupted.
    pub fn listen(&mut self, host: &str, port: u16) -> Result<(), String> {
        let addr = format!("{host}:{port}");
        let listener = TcpListener::bind(&addr).map_err(|e| format!("cannot bind {addr}: {e}"))?;
        let bound = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or(addr.clone());

        eprintln!("llmspec API listening on http://{bound}");
        for route in ROUTES {
            eprintln!("  {route}");
        }

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    // One bad connection must not end the server.
                    if let Err(e) = self.handle(stream) {
                        eprintln!("request failed: {e}");
                    }
                }
                Err(e) => eprintln!("accept failed: {e}"),
            }
        }
        Ok(())
    }

    fn handle(&mut self, mut stream: TcpStream) -> Result<(), String> {
        let _ = stream.set_read_timeout(Some(CLIENT_TIMEOUT));
        let _ = stream.set_write_timeout(Some(CLIENT_TIMEOUT));

        let request = match read_request(&stream) {
            Ok(request) => request,
            Err(e) => return write_response(&mut stream, 400, &error_json(&e)),
        };

        if request.method != "GET" {
            return write_response(
                &mut stream,
                405,
                &error_json("only GET is supported by this API"),
            );
        }

        let (status, body) = self.route(&request);
        write_response(&mut stream, status, &body)
    }

    fn route(&mut self, request: &Request) -> (u16, String) {
        match request.path.as_str() {
            "/" | "/health" => (
                200,
                display::to_json(&Health {
                    status: "ok",
                    version: env!("CARGO_PKG_VERSION"),
                    models: self.db.len(),
                    routes: ROUTES,
                }),
            ),
            "/system" => (200, display::to_json(&self.hw)),
            "/runtimes" => (200, display::to_json(&self.registry.discover())),
            "/catalog" => (200, display::to_json(&self.db.models)),
            "/models" | "/models/top" => self.models(request),
            path if path.starts_with("/models/") => {
                self.one_model(path.trim_start_matches("/models/"), request)
            }
            other => (404, error_json(&format!("no route {other}"))),
        }
    }

    fn models(&mut self, request: &Request) -> (u16, String) {
        let target = match request.use_case(self.default_use_case) {
            Ok(target) => target,
            Err(e) => return (400, error_json(&e)),
        };
        let cfg = SpeedConfig {
            context_cap: request.max_context().or(self.cfg.context_cap),
            ..self.cfg
        };

        // Filter the catalog before analysing: search and provider narrow the
        // input set, everything else judges the result.
        let candidates: Vec<_> = self
            .db
            .models
            .iter()
            .filter(|m| match request.get("search") {
                Some(q) => m.matches(q),
                None => true,
            })
            .filter(|m| match request.get("provider") {
                Some(p) => m
                    .provider
                    .to_ascii_lowercase()
                    .contains(&p.to_ascii_lowercase()),
                None => true,
            })
            .cloned()
            .collect();

        let mut results = fit::analyze_all(&candidates, &self.hw, target, &cfg);

        if let Some(raw) = request.get("quant") {
            match Quant::parse(raw) {
                Some(q) => results.retain(|r| r.quant == q),
                None => return (400, error_json(&format!("unknown quantization '{raw}'"))),
            }
        }
        if let Some(raw) = request.get("mode") {
            match RunMode::parse(raw) {
                Some(m) => results.retain(|r| r.mode == m),
                None => return (400, error_json(&format!("unknown run mode '{raw}'"))),
            }
        }
        if let Some(raw) = request.get("min_fit") {
            match FitLevel::parse(raw) {
                Some(f) => results.retain(|r| r.fit >= f),
                None => return (400, error_json(&format!("unknown fit level '{raw}'"))),
            }
        }
        if request.flag("perfect") {
            results.retain(|r| r.fit == FitLevel::Perfect);
        }
        // `/models/top` and any request that did not ask for unrunnable models
        // returns only what this machine can actually load.
        if request.path == "/models/top" || !request.flag("include_too_tight") {
            results.retain(FitResult::is_runnable);
        }

        if let Some(n) = request.limit() {
            results.truncate(n);
        } else if request.path == "/models/top" {
            results.truncate(5);
        }

        (
            200,
            display::to_json(&display::JsonReport {
                system: &self.hw,
                use_case: target.as_str(),
                count: results.len(),
                models: &results,
            }),
        )
    }

    fn one_model(&mut self, id: &str, request: &Request) -> (u16, String) {
        let query = percent_decode(id);
        let Some(model) = self.db.find(&query) else {
            return (404, error_json(&format!("no model matches '{query}'")));
        };
        let target = match request.use_case(self.default_use_case) {
            Ok(target) => target,
            Err(e) => return (400, error_json(&e)),
        };
        let cfg = SpeedConfig {
            context_cap: request.max_context().or(self.cfg.context_cap),
            ..self.cfg
        };
        (
            200,
            display::to_json(&fit::analyze(model, &self.hw, target, &cfg)),
        )
    }
}

const ROUTES: &[&str] = &[
    "GET /health",
    "GET /system",
    "GET /runtimes",
    "GET /catalog",
    "GET /models?limit&use_case&provider&search&quant&mode&min_fit&perfect&include_too_tight&max_context",
    "GET /models/top",
    "GET /models/{id}",
];

#[derive(serde::Serialize)]
struct Health {
    status: &'static str,
    version: &'static str,
    models: usize,
    routes: &'static [&'static str],
}

// ---------------------------------------------------------------------------
// Request parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: HashMap<String, String>,
}

impl Request {
    fn get(&self, key: &str) -> Option<&str> {
        self.query
            .get(key)
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    }

    /// A parameter that is true when present, unless explicitly `false`/`0`.
    fn flag(&self, key: &str) -> bool {
        match self.query.get(key) {
            None => false,
            Some(v) => !matches!(v.as_str(), "false" | "0" | "no"),
        }
    }

    fn limit(&self) -> Option<usize> {
        self.get("limit")
            .or_else(|| self.get("n"))
            .and_then(|v| v.parse().ok())
    }

    fn max_context(&self) -> Option<u32> {
        self.get("max_context").and_then(|v| v.parse().ok())
    }

    fn use_case(&self, fallback: UseCase) -> Result<UseCase, String> {
        match self.get("use_case") {
            None => Ok(fallback),
            Some(raw) => UseCase::parse(raw).ok_or_else(|| format!("unknown use case '{raw}'")),
        }
    }
}

/// Parse the request line, then drain headers so the socket is left clean.
fn read_request(stream: &TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    let mut consumed = reader
        .by_ref()
        .take(MAX_HEADER_BYTES as u64)
        .read_line(&mut line)
        .map_err(|e| format!("reading request line: {e}"))?;
    if consumed == 0 {
        return Err("empty request".to_string());
    }

    let request = parse_request_line(&line)?;

    // Headers are not used, but must be read for the response to be delivered
    // reliably; stop at the blank line or the byte cap, whichever comes first.
    while consumed < MAX_HEADER_BYTES {
        let mut header = String::new();
        let n = reader
            .by_ref()
            .take((MAX_HEADER_BYTES - consumed) as u64)
            .read_line(&mut header)
            .map_err(|e| format!("reading headers: {e}"))?;
        consumed += n;
        if n == 0 || header.trim().is_empty() {
            break;
        }
    }

    Ok(request)
}

fn parse_request_line(line: &str) -> Result<Request, String> {
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "malformed request line".to_string())?
        .to_ascii_uppercase();
    let target = parts
        .next()
        .ok_or_else(|| "malformed request line: no target".to_string())?;

    let (raw_path, raw_query) = match target.split_once('?') {
        Some((path, query)) => (path, Some(query)),
        None => (target, None),
    };

    let mut query = HashMap::new();
    if let Some(raw) = raw_query {
        for pair in raw.split('&').filter(|p| !p.is_empty()) {
            let (key, value) = match pair.split_once('=') {
                Some((k, v)) => (k, v),
                None => (pair, ""),
            };
            query.insert(percent_decode(key), percent_decode(value));
        }
    }

    // Keep a trailing slash from creating a second spelling of every route.
    let path = raw_path.trim_end_matches('/');
    Ok(Request {
        method,
        path: if path.is_empty() {
            "/".to_string()
        } else {
            percent_decode(path)
        },
        query,
    })
}

/// Minimal percent-decoding, plus `+` for spaces in query values.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                match u8::from_str_radix(&input[i + 1..i + 3], 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    // Not a valid escape: keep the literal '%'.
                    Err(_) => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

fn write_response(stream: &mut TcpStream, status: u16, body: &str) -> Result<(), String> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(head.as_bytes())
        .and_then(|()| stream.write_all(body.as_bytes()))
        .and_then(|()| stream.flush())
        .map_err(|e| format!("writing response: {e}"))
}

fn error_json(message: &str) -> String {
    display::to_json(&serde_json::json!({ "error": message }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Request {
        parse_request_line(line).unwrap()
    }

    #[test]
    fn parses_a_bare_path() {
        let r = parse("GET /system HTTP/1.1\r\n");
        assert_eq!(r.method, "GET");
        assert_eq!(r.path, "/system");
        assert!(r.query.is_empty());
    }

    #[test]
    fn parses_query_parameters() {
        let r = parse("GET /models?limit=5&use_case=coding HTTP/1.1\r\n");
        assert_eq!(r.path, "/models");
        assert_eq!(r.limit(), Some(5));
        assert_eq!(r.use_case(UseCase::General).unwrap(), UseCase::Coding);
    }

    #[test]
    fn valueless_parameters_are_flags() {
        let r = parse("GET /models?perfect HTTP/1.1\r\n");
        assert!(r.flag("perfect"));
        assert!(!r.flag("include_too_tight"));
        let off = parse("GET /models?perfect=false HTTP/1.1\r\n");
        assert!(!off.flag("perfect"));
    }

    #[test]
    fn trailing_slashes_collapse() {
        assert_eq!(parse("GET / HTTP/1.1\r\n").path, "/");
        assert_eq!(parse("GET /models/ HTTP/1.1\r\n").path, "/models");
    }

    #[test]
    fn percent_escapes_are_decoded() {
        let r = parse("GET /models/Qwen%2FQwen2.5-7B-Instruct HTTP/1.1\r\n");
        assert_eq!(r.path, "/models/Qwen/Qwen2.5-7B-Instruct");
        let q = parse("GET /models?search=llama+8b HTTP/1.1\r\n");
        assert_eq!(q.get("search"), Some("llama 8b"));
    }

    #[test]
    fn malformed_escapes_are_left_alone() {
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
    }

    #[test]
    fn method_is_normalised_and_missing_target_rejected() {
        assert_eq!(parse("get /health HTTP/1.1\r\n").method, "GET");
        assert!(parse_request_line("GET").is_err());
        assert!(parse_request_line("").is_err());
    }

    #[test]
    fn unknown_use_case_is_rejected() {
        let r = parse("GET /models?use_case=telepathy HTTP/1.1\r\n");
        assert!(r.use_case(UseCase::General).is_err());
    }

    #[test]
    fn empty_values_read_as_absent() {
        let r = parse("GET /models?provider=&limit=3 HTTP/1.1\r\n");
        assert_eq!(r.get("provider"), None);
        assert_eq!(r.limit(), Some(3));
    }

    fn test_server() -> Server {
        let mut hw = Hardware::detect();
        hw.apply_overrides(Some(24.0), Some(64.0), None);
        Server::new(
            hw,
            ModelDb::embedded(),
            SpeedConfig::default(),
            UseCase::General,
        )
    }

    #[test]
    fn health_route_reports_the_catalog_size() {
        let mut server = test_server();
        let (status, body) = server.route(&parse("GET /health HTTP/1.1\r\n"));
        assert_eq!(status, 200);
        assert!(body.contains("\"status\": \"ok\""));
        assert!(body.contains(&format!("\"models\": {}", ModelDb::embedded().len())));
    }

    #[test]
    fn top_route_defaults_to_five_runnable_models() {
        let mut server = test_server();
        let (status, body) = server.route(&parse("GET /models/top HTTP/1.1\r\n"));
        assert_eq!(status, 200);
        assert!(body.contains("\"count\": 5"));
        assert!(!body.contains("Too Tight"));
    }

    #[test]
    fn model_route_returns_one_analysis() {
        let mut server = test_server();
        let (status, body) = server.route(&parse(
            "GET /models/Qwen%2FQwen2.5-7B-Instruct HTTP/1.1\r\n",
        ));
        assert_eq!(status, 200);
        assert!(body.contains("Qwen2.5 7B"));
    }

    #[test]
    fn unknown_model_is_a_404() {
        let mut server = test_server();
        let (status, body) = server.route(&parse("GET /models/not-a-model HTTP/1.1\r\n"));
        assert_eq!(status, 404);
        assert!(body.contains("error"));
    }

    #[test]
    fn unknown_route_is_a_404() {
        let mut server = test_server();
        let (status, _) = server.route(&parse("GET /nope HTTP/1.1\r\n"));
        assert_eq!(status, 404);
    }

    #[test]
    fn bad_filter_values_are_400s() {
        let mut server = test_server();
        assert_eq!(
            server.route(&parse("GET /models?quant=q9 HTTP/1.1\r\n")).0,
            400
        );
        assert_eq!(
            server
                .route(&parse("GET /models?mode=quantum HTTP/1.1\r\n"))
                .0,
            400
        );
        assert_eq!(
            server
                .route(&parse("GET /models?min_fit=flawless HTTP/1.1\r\n"))
                .0,
            400
        );
    }

    #[test]
    fn search_and_limit_narrow_the_result_set() {
        let mut server = test_server();
        let (status, body) = server.route(&parse("GET /models?search=qwen&limit=3 HTTP/1.1\r\n"));
        assert_eq!(status, 200);
        assert!(body.contains("\"count\": 3"));
    }
}
