//! llmspec — find the LLMs that actually run well on your hardware.

mod bench;
mod config;
mod display;
mod doctor;
mod fit;
mod hardware;
mod models;
mod providers;
mod serve;
mod tui_app;
mod tui_events;
mod tui_form;
mod tui_theme;
mod tui_ui;

use clap::{Parser, Subcommand};
use colored::Colorize;

use crate::config::Config;
use crate::fit::{FitLevel, FitResult, RunMode, SpeedConfig};
use crate::hardware::{Hardware, parse_size_gb};
use crate::models::{ModelDb, Quant, UseCase};
use crate::providers::{ProviderRegistry, Runtime, RuntimeKind};

#[derive(Parser, Debug)]
#[command(
    name = "llmspec",
    version,
    about = "Find the LLMs that actually run well on your hardware",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Print the classic table instead of starting the TUI
    #[arg(long, global = true)]
    cli: bool,

    /// Machine-readable output
    #[arg(long, global = true)]
    json: bool,

    /// Override detected VRAM, e.g. 24G (creates a synthetic GPU if none found)
    #[arg(long, global = true, value_name = "SIZE")]
    memory: Option<String>,

    /// Override detected system RAM, e.g. 128G
    #[arg(long, global = true, value_name = "SIZE")]
    ram: Option<String>,

    /// Override detected CPU core count
    #[arg(long, global = true, value_name = "N")]
    cpu_cores: Option<usize>,

    /// Cap the context length used for memory estimation
    #[arg(long, global = true, value_name = "TOKENS")]
    max_context: Option<u32>,

    /// Target use case: general, coding, reasoning, chat, multimodal, embedding
    #[arg(long, short = 'u', global = true, value_name = "USE_CASE")]
    use_case: Option<String>,

    /// Score for a specific runtime: ollama, llamacpp, lmstudio, vllm, docker, mlx
    #[arg(long, global = true, value_name = "RUNTIME")]
    force_runtime: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Show the detected hardware
    System,

    /// List every model in the database (no hardware analysis)
    List,

    /// Search models by name, provider, size or use case
    Search {
        /// Free-text query, e.g. "llama 8b"
        query: Vec<String>,
        /// Maximum number of rows
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Show everything known about one model
    Info {
        /// Model id or name, e.g. "Mistral-7B"
        model: Vec<String>,
    },

    /// Rank models by how well they fit this machine
    Fit {
        /// Only show perfect fits
        #[arg(long)]
        perfect: bool,
        /// Minimum fit level: perfect, good, marginal, too_tight
        #[arg(long, value_name = "LEVEL")]
        min_fit: Option<String>,
        /// Only show models that can actually run
        #[arg(long)]
        runnable: bool,
        /// Filter by provider (substring match)
        #[arg(long, value_name = "NAME")]
        provider: Option<String>,
        /// Only show models placed at this quantization, e.g. q4_k_m
        #[arg(long, value_name = "QUANT")]
        quant: Option<String>,
        /// Only show models placed in this run mode: gpu, moe, cpu+gpu, cpu
        #[arg(long, value_name = "MODE")]
        mode: Option<String>,
        /// Only show models estimated to reach at least this many tokens/sec
        #[arg(long, value_name = "TPS")]
        min_tps: Option<f64>,
        /// Only show models whose download is at most this size, e.g. 8G
        #[arg(long, value_name = "SIZE")]
        max_size: Option<String>,
        /// Only show models that hold at least this much context
        #[arg(long, value_name = "TOKENS")]
        min_context: Option<u32>,
        /// Maximum number of rows
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Top recommendations for this machine (JSON by default)
    Recommend {
        /// How many models to return
        #[arg(long, default_value_t = 5)]
        limit: usize,
        /// Human-readable table instead of JSON
        #[arg(long)]
        table: bool,
    },

    /// Diagnostic report: what was detected, and what had to be guessed
    Doctor,

    /// List the local inference runtimes that are currently running
    Runtimes,

    /// Measure real tokens/sec against a running runtime
    Bench {
        /// Model reference as the runtime names it, e.g. "qwen2.5:7b".
        /// Defaults to every model the runtime reports.
        model: Vec<String>,
        /// Benchmark every model the runtime has installed
        #[arg(long)]
        all: bool,
        /// Timed runs per model (the first, untimed, run loads the model)
        #[arg(long, default_value_t = 3, value_name = "N")]
        runs: usize,
        /// Tokens to generate per run
        #[arg(long, value_name = "N")]
        tokens: Option<u32>,
    },

    /// Serve the fit analysis over a read-only HTTP API
    Serve {
        /// Address to bind. Loopback by default: the API reports this machine's hardware
        #[arg(long, default_value = "127.0.0.1", value_name = "HOST")]
        host: String,
        /// Port to listen on
        #[arg(long, default_value_t = 8228, value_name = "PORT")]
        port: u16,
    },

    /// Plan hardware requirements for a model configuration
    Plan {
        /// Model name or id, e.g. "Llama-3.1-8B"
        model: Vec<String>,
        /// Context length in tokens (default: model's max)
        #[arg(long, value_name = "TOKENS")]
        context: Option<u32>,
        /// Quantization level (default: q4_k_m)
        #[arg(long, value_name = "QUANT")]
        quant: Option<String>,
        /// Also report the bandwidth and cards needed to reach this many tokens/sec
        #[arg(long, value_name = "TPS")]
        target_tps: Option<f64>,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("{} {err}", "error:".red().bold());
        std::process::exit(1);
    }
}

/// Everything the commands share, resolved once from the flags and the
/// stored configuration.
///
/// The handlers below take this rather than a dozen loose arguments, and the
/// two that own the process for the rest of its life — `serve` and the TUI —
/// take it by value.
struct Session {
    hw: Hardware,
    db: ModelDb,
    /// Use case the scores are weighted for.
    target: UseCase,
    cfg: SpeedConfig,
    /// `--force-runtime`, which narrows the catalog and shifts the estimate.
    runtime: Option<RuntimeKind>,
}

impl Session {
    fn build(cli: &Cli) -> Result<Session, String> {
        let mut stored = Config::load();
        // An explicit `--use-case` beats the stored preference, which beats
        // the built-in default.
        let target = match &cli.use_case {
            Some(raw) => {
                UseCase::parse(raw).ok_or_else(|| unknown("use case", raw, &UseCase::hint()))?
            }
            None => stored.use_case,
        };
        let runtime = resolve_runtime(cli)?;
        let mut hw = build_hardware(cli)?;
        // Measured once per machine and cached. Every estimate for weights
        // that spill into RAM is only as good as this figure.
        hw.ram_bandwidth_gb_s = Some(stored.ram_bandwidth(hw.cpu_cores));
        Ok(Session {
            hw,
            db: ModelDb::load(),
            target,
            cfg: SpeedConfig {
                context_cap: resolve_context_cap(cli),
                // A forced runtime shifts the throughput estimate: MLX and
                // vLLM read the same weights faster than a GGUF loader does.
                gpu_factor: stored.speed.gpu_factor
                    * runtime.map_or(1.0, RuntimeKind::speed_factor),
                ..stored.speed.apply_to(&SpeedConfig::default())
            },
            runtime,
        })
    }

    /// Rank the models a forced runtime could actually load.
    fn analyze(&self, models: &[models::Model]) -> Vec<FitResult> {
        fit::analyze_all(models, &self.hw, self.target, &self.cfg)
    }

    /// Look one model up by the words the user typed.
    fn find(&self, words: &[String]) -> Result<&models::Model, String> {
        let query = words.join(" ");
        self.db
            .find(&query)
            .ok_or_else(|| format!("no model matches '{query}'"))
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let session = Session::build(&cli)?;
    match &cli.command {
        Some(Command::System) => cmd_system(&cli, &session),
        Some(Command::List) => cmd_list(&cli, &session),
        Some(Command::Search { query, limit }) => cmd_search(&cli, &session, query, *limit),
        Some(Command::Info { model }) => cmd_info(&cli, &session, model),
        Some(Command::Fit { .. }) => cmd_fit(&cli, &session),
        Some(Command::Recommend { limit, table }) => cmd_recommend(&cli, &session, *limit, *table),
        Some(Command::Doctor) => cmd_doctor(&cli, &session),
        Some(Command::Runtimes) => cmd_runtimes(&cli),
        Some(Command::Bench {
            model,
            all,
            runs,
            tokens,
        }) => cmd_bench(&cli, &session, model, *all, *runs, *tokens),
        Some(Command::Plan {
            model,
            context,
            quant,
            target_tps,
        }) => cmd_plan(
            &cli,
            &session,
            model,
            *context,
            quant.as_deref(),
            *target_tps,
        ),
        Some(Command::Serve { host, port }) => cmd_serve(session, host, *port),
        None => cmd_default(&cli, session),
    }
}

fn cmd_system(cli: &Cli, session: &Session) -> Result<(), String> {
    if cli.json {
        println!("{}", display::to_json(&session.hw));
    } else {
        print!("{}", display::render_system(&session.hw));
    }
    Ok(())
}

fn cmd_list(cli: &Cli, session: &Session) -> Result<(), String> {
    let db = &session.db;
    if cli.json {
        println!("{}", display::to_json(&db.models));
    } else {
        print!("{}", display::render_model_list(&db.models));
        println!(
            "\n{} models in database (schema v{}, source: {})",
            db.len(),
            db.schema_version,
            db.source
        );
    }
    Ok(())
}

fn cmd_search(
    cli: &Cli,
    session: &Session,
    query: &[String],
    limit: Option<usize>,
) -> Result<(), String> {
    let query = query.join(" ");
    let matched: Vec<_> = session
        .db
        .models
        .iter()
        .filter(|m| m.matches(&query))
        .cloned()
        .collect();
    if matched.is_empty() {
        return Err(format!("no model matches '{query}'"));
    }
    let mut results = session.analyze(&matched);
    truncate_results(&mut results, limit);
    emit_results(cli, session, &results);
    Ok(())
}

fn cmd_info(cli: &Cli, session: &Session, model: &[String]) -> Result<(), String> {
    let found = session.find(model)?;
    let result = fit::analyze(found, &session.hw, session.target, &session.cfg);
    if cli.json {
        println!("{}", display::to_json(&result));
    } else {
        let runtime = suggested_runtime(session.runtime, &result);
        print!("{}", display::render_detail(&result, found, Some(runtime)));
    }
    Ok(())
}

fn cmd_fit(cli: &Cli, session: &Session) -> Result<(), String> {
    let Some(Command::Fit {
        perfect,
        min_fit,
        runnable,
        provider,
        quant,
        mode,
        min_tps,
        max_size,
        min_context,
        limit,
    }) = &cli.command
    else {
        unreachable!("cmd_fit is only reached from the Fit arm");
    };

    let mut results = session.analyze(&catalog_for(&session.db, session.runtime));
    if let Some(name) = provider {
        let needle = name.to_ascii_lowercase();
        results.retain(|r| r.provider.to_ascii_lowercase().contains(&needle));
    }
    if let Some(raw) = quant {
        let wanted = parse_quant(raw)?;
        results.retain(|r| r.quant == wanted);
    }
    if let Some(raw) = mode {
        let wanted =
            RunMode::parse(raw).ok_or_else(|| unknown("run mode", raw, &RunMode::hint()))?;
        results.retain(|r| r.mode == wanted);
    }
    // Practical thresholds: how fast it has to be, how much disk it may take,
    // and how much context it has to hold.
    if let Some(floor) = min_tps {
        results.retain(|r| r.tokens_per_second >= *floor);
    }
    if let Some(raw) = max_size {
        let ceiling = parse_size_gb(raw)?;
        results.retain(|r| r.download_gb <= ceiling);
    }
    if let Some(floor) = min_context {
        results.retain(|r| r.context >= *floor);
    }
    if let Some(floor) = resolve_min_fit(*perfect, *runnable, min_fit.as_deref())? {
        results.retain(|r| r.fit >= floor);
    }
    truncate_results(&mut results, *limit);
    emit_results(cli, session, &results);
    Ok(())
}

fn cmd_recommend(cli: &Cli, session: &Session, limit: usize, table: bool) -> Result<(), String> {
    let mut results = session.analyze(&catalog_for(&session.db, session.runtime));
    results.retain(FitResult::is_runnable);
    truncate_results(&mut results, Some(limit));
    // `recommend` defaults to JSON; `--table` opts back into text.
    if table && !cli.json {
        print!("{}", display::render_table(&results));
    } else {
        print!("{}", report_json(session, &results));
        println!();
    }
    Ok(())
}

fn cmd_doctor(cli: &Cli, session: &Session) -> Result<(), String> {
    let mut registry = ProviderRegistry::new();
    let report = doctor::run(&session.hw, &session.db, &mut registry);
    if cli.json {
        println!("{}", display::to_json(&report));
    } else {
        print!("{}", display::render_doctor(&report));
    }
    // A clean report exits 0; warnings are worth a non-zero status so a CI
    // check can gate on detection actually having worked.
    if report.is_clean() {
        Ok(())
    } else {
        std::process::exit(2);
    }
}

fn cmd_runtimes(cli: &Cli) -> Result<(), String> {
    let found = ProviderRegistry::new().discover();
    if cli.json {
        println!("{}", display::to_json(&found));
    } else {
        print!("{}", display::render_runtimes(&found));
    }
    Ok(())
}

fn cmd_bench(
    cli: &Cli,
    session: &Session,
    model: &[String],
    all: bool,
    runs: usize,
    tokens: Option<u32>,
) -> Result<(), String> {
    let mut registry = ProviderRegistry::new();
    let discovered = bench::select_runtime(&mut registry, session.runtime)?;
    let client = Runtime::with_url(discovered.kind, &discovered.base_url);

    let targets = bench_targets(&client, &model.join(" "), all)?;
    let mut results = Vec::new();
    for model_ref in &targets {
        if !cli.json {
            eprintln!(
                "benchmarking {model_ref} on {} ({runs} runs)...",
                discovered.name
            );
        }
        let mut result = bench::run_one(&client, model_ref, runs, tokens)?;
        // Compare against what llmspec would have predicted, when the
        // runtime's name for the model resolves to a catalog entry.
        if let Some(found) = session.db.find_for_runtime(model_ref) {
            let analysis = fit::analyze(found, &session.hw, session.target, &session.cfg);
            let weights_gb = found.weights_gb(analysis.quant);
            bench::attach_estimate(&mut result, &analysis, weights_gb);
        }
        results.push(result);
    }

    let report = bench::BenchReport::new(
        bench::HardwareSummary::from(&session.hw),
        results,
        session.cfg.efficiency,
    );
    if cli.json {
        println!("{}", display::to_json(&report));
    } else {
        print!("{}", display::render_bench(&report));
    }
    Ok(())
}

fn cmd_plan(
    cli: &Cli,
    session: &Session,
    model: &[String],
    context: Option<u32>,
    quant: Option<&str>,
    target_tps: Option<f64>,
) -> Result<(), String> {
    let found = session.find(model)?;
    let context = context.unwrap_or(found.context_length);
    let quant = match quant {
        Some(raw) => parse_quant(raw)?,
        None => Quant::Q4KM,
    };
    let plan = fit::plan(found, quant, context, &session.cfg, target_tps);
    if cli.json {
        println!("{}", display::to_json(&plan));
    } else {
        print!("{}", display::render_plan(&plan));
    }
    Ok(())
}

fn cmd_serve(session: Session, host: &str, port: u16) -> Result<(), String> {
    let Session {
        hw,
        db,
        cfg,
        target,
        ..
    } = session;
    serve::Server::new(hw, db, cfg, target).listen(host, port)
}

/// No subcommand: the TUI, unless output was asked for in text or JSON.
fn cmd_default(cli: &Cli, session: Session) -> Result<(), String> {
    if cli.cli || cli.json {
        let results = session.analyze(&catalog_for(&session.db, session.runtime));
        emit_results(cli, &session, &results);
        return Ok(());
    }
    let Session {
        hw,
        db,
        target,
        cfg,
        ..
    } = session;
    let mut app = tui_app::App::new(hw, db, target, cfg);
    tui_events::run(&mut app).map_err(|e| format!("terminal error: {e}"))
}

/// The one shape every "you typed something I don't know" error takes.
///
/// The accepted spellings come from the enums themselves, so a new variant
/// cannot leave a stale list behind in an error message.
fn unknown(what: &str, given: &str, accepted: &str) -> String {
    format!("unknown {what} '{given}' (try: {accepted})")
}

fn parse_quant(raw: &str) -> Result<Quant, String> {
    Quant::parse(raw).ok_or_else(|| unknown("quantization", raw, &Quant::hint()))
}

fn build_hardware(cli: &Cli) -> Result<Hardware, String> {
    let mut hw = Hardware::detect();
    let vram = cli.memory.as_deref().map(parse_size_gb).transpose()?;
    let ram = cli.ram.as_deref().map(parse_size_gb).transpose()?;
    hw.apply_overrides(vram, ram, cli.cpu_cores);
    Ok(hw)
}

fn resolve_runtime(cli: &Cli) -> Result<Option<RuntimeKind>, String> {
    match &cli.force_runtime {
        None => Ok(None),
        Some(raw) => RuntimeKind::parse(raw)
            .map(Some)
            .ok_or_else(|| unknown("runtime", raw, &RuntimeKind::hint())),
    }
}

/// Which runtime's commands `info` should suggest.
///
/// An explicit `--force-runtime` wins. Otherwise the first runtime actually
/// running on the machine, and failing that the one most likely to be
/// installed for this model: Ollama when it has a tag, llama.cpp otherwise.
fn suggested_runtime(forced: Option<RuntimeKind>, result: &FitResult) -> RuntimeKind {
    if let Some(kind) = forced {
        return kind;
    }
    if let Some(live) = ProviderRegistry::new().discover().first() {
        return live.kind;
    }
    if result.ollama.is_some() {
        RuntimeKind::Ollama
    } else {
        RuntimeKind::LlamaCpp
    }
}

/// The models a runtime can actually load.
///
/// GGUF loaders cannot run a model with no GGUF build, so listing one as a
/// perfect fit would be a lie. Without a forced runtime the whole catalog is
/// in play and availability is left to the `--runnable` / `a` filters.
fn catalog_for(db: &ModelDb, runtime: Option<RuntimeKind>) -> Vec<models::Model> {
    match runtime {
        Some(kind) if kind.needs_gguf() => db.models.iter().filter(|m| m.gguf).cloned().collect(),
        _ => db.models.clone(),
    }
}

/// Resolve which model references to benchmark.
fn bench_targets(client: &Runtime, query: &str, all: bool) -> Result<Vec<String>, String> {
    if !query.is_empty() && !all {
        return Ok(vec![query.to_string()]);
    }
    let installed = client.list_models()?;
    if installed.is_empty() {
        return Err(format!(
            "{} reports no installed models — name one explicitly",
            client.kind.label()
        ));
    }
    let mut names: Vec<String> = installed.into_iter().map(|m| m.name).collect();
    if all {
        return Ok(names);
    }
    // No model named and `--all` not given: benchmark the first one, which is
    // enough to answer "is my machine as fast as llmspec thinks".
    names.truncate(1);
    Ok(names)
}

/// `--max-context`, falling back to `OLLAMA_CONTEXT_LENGTH`.
fn resolve_context_cap(cli: &Cli) -> Option<u32> {
    cli.max_context.or_else(|| {
        std::env::var("OLLAMA_CONTEXT_LENGTH")
            .ok()
            .and_then(|v| v.trim().parse().ok())
    })
}

fn resolve_min_fit(
    perfect: bool,
    runnable: bool,
    min_fit: Option<&str>,
) -> Result<Option<FitLevel>, String> {
    if perfect {
        return Ok(Some(FitLevel::Perfect));
    }
    if let Some(raw) = min_fit {
        return FitLevel::parse(raw)
            .map(Some)
            .ok_or_else(|| unknown("fit level", raw, &FitLevel::hint()));
    }
    if runnable {
        return Ok(Some(FitLevel::Marginal));
    }
    Ok(None)
}

fn truncate_results(results: &mut Vec<FitResult>, limit: Option<usize>) {
    if let Some(n) = limit {
        results.truncate(n);
    }
}

fn emit_results(cli: &Cli, session: &Session, results: &[FitResult]) {
    if cli.json {
        print!("{}", report_json(session, results));
        println!();
    } else {
        print!("{}", display::render_table(results));
    }
}

fn report_json(session: &Session, results: &[FitResult]) -> String {
    display::to_json(&display::JsonReport {
        system: &session.hw,
        use_case: session.target.as_str(),
        count: results.len(),
        models: results,
    })
}
