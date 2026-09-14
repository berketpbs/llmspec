//! llmspec — find the LLMs that actually run well on your hardware.

#[cfg(test)]
mod accuracy;
mod bench;
mod config;
mod display;
mod doctor;
mod fit;
mod hardware;
mod mcp;
mod models;
mod providers;
mod serve;
mod tui_app;
mod tui_events;
mod tui_form;
mod tui_theme;
mod tui_ui;
mod verify;

use clap::{Parser, Subcommand};
use colored::Colorize;

use crate::config::{Calibration, Config, MeasurementStore, StoredMeasurement};
use crate::fit::{FitLevel, FitResult, RunMode, SpeedConfig};
use crate::hardware::{Hardware, MeasuredThroughput, parse_size_gb};
use crate::models::{KvQuant, ModelDb, Quant, UseCase};
use crate::providers::{InstalledModel, ProviderRegistry, Runtime, RuntimeKind};

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

    /// Simulate a GPU by model name, e.g. "RTX 3090" or "M4 Pro" — brings its
    /// VRAM and memory bandwidth; see `llmspec gpus`
    #[arg(long, global = true, value_name = "NAME")]
    gpu: Option<String>,

    /// Number of simulated GPUs (with --gpu)
    #[arg(
        long,
        global = true,
        value_name = "N",
        default_value_t = 1,
        requires = "gpu"
    )]
    gpu_count: usize,

    /// How the runtime stores the KV cache: f16, q8_0, q4_0
    #[arg(long, global = true, value_name = "TYPE")]
    kv_quant: Option<String>,

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
        /// Fit the efficiency factor to what was measured and save it
        #[arg(long)]
        calibrate: bool,
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

    /// Serve the fit analysis to an assistant over MCP (stdio)
    Mcp,

    /// Check a model file on disk for damage or truncation
    Verify {
        /// Path to a .gguf or .safetensors file
        file: std::path::PathBuf,
    },

    /// The GPUs llmspec knows the memory bandwidth of, for use with --gpu
    Gpus {
        /// Only cards whose name contains this, e.g. "4090" or "m4"
        filter: Vec<String>,
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
        let kv_quant = match &cli.kv_quant {
            Some(raw) => KvQuant::parse(raw)
                .ok_or_else(|| unknown("KV cache type", raw, &KvQuant::hint()))?,
            None => KvQuant::default(),
        };
        let mut hw = build_hardware(cli)?;
        // Measured once per machine and cached. Every estimate for weights
        // that spill into RAM is only as good as this figure.
        hw.ram_bandwidth_gb_s = Some(stored.ram_bandwidth());
        // What `bench` measured here outranks any estimate — but only on the
        // machine it was measured on, never on a simulated one.
        if !hw.simulated {
            hw.measured =
                MeasurementStore::load().for_machine(&hw.primary_gpu_name(), hw.backend.label());
        }
        // A calibration fitted by `bench --calibrate` replaces the shipped
        // efficiency factor, but only on the machine it was measured on: the
        // number describes that card and runtime, not the speed model.
        let calibrated = stored
            .calibration
            .as_ref()
            .filter(|c| !hw.simulated && c.matches(&hw.primary_gpu_name(), hw.backend.label()))
            .map(|c| c.efficiency);
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
                efficiency: calibrated.unwrap_or(stored.speed.efficiency),
                calibrated: calibrated.is_some(),
                kv_quant,
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
    // `verify` answers a question about a file, not about this machine, so it
    // skips the hardware probe the rest of the commands need.
    if let Some(Command::Verify { file }) = &cli.command {
        return cmd_verify(&cli, file);
    }
    // Neither does listing the GPU table.
    if let Some(Command::Gpus { filter }) = &cli.command {
        return cmd_gpus(&cli, filter);
    }
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
            calibrate,
        }) => cmd_bench(&cli, &session, model, *all, *runs, *tokens, *calibrate),
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
        Some(Command::Gpus { filter }) => cmd_gpus(&cli, filter),
        Some(Command::Mcp) => cmd_mcp(session),
        // Answered above, before the session was built. Repeating the call
        // here keeps the match exhaustive without a panicking arm.
        Some(Command::Verify { file }) => cmd_verify(&cli, file),
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
    calibrate: bool,
) -> Result<(), String> {
    let mut registry = ProviderRegistry::new();
    let discovered = bench::select_runtime(&mut registry, session.runtime)?;
    let client = Runtime::with_url(discovered.kind, &discovered.base_url);

    let targets = bench_targets(&client, &model.join(" "), all)?;
    let mut results = Vec::new();
    for target in &targets {
        let model_ref = &target.reference;
        if !cli.json {
            eprintln!(
                "benchmarking {model_ref} on {} ({runs} runs)...",
                discovered.name
            );
        }
        let mut result = bench::run_one(&client, model_ref, runs, tokens)?;
        // Compare against what llmspec would have predicted, when the
        // runtime's name for the model resolves to a catalog entry.
        if let Some(found) = session
            .db
            .find_for_runtime_sized(model_ref, target.params_b)
        {
            let (assumed, estimated) = bench_estimate(session, found, target);
            bench::attach_estimate(&mut result, assumed, estimated);
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
    record_measurements(session, discovered.kind, &report);
    if calibrate {
        save_calibration(session, &report)?;
    }
    Ok(())
}

/// What llmspec predicts for the run a benchmark just measured.
///
/// When the runtime says which quantization it loaded, the estimate is made
/// for exactly those bytes at the context the benchmark asked for; the ratio
/// then measures the speed model and nothing else. Otherwise llmspec's own
/// placement stands in, and the output says so.
fn bench_estimate(
    session: &Session,
    model: &models::Model,
    target: &BenchTarget,
) -> (bench::Assumptions, f64) {
    let reported = target.quantization.as_deref().and_then(Quant::parse);
    if let Some(quant) = reported
        && let Some((mode, tps)) = fit::estimate_at_quant(
            model,
            &session.hw,
            quant,
            providers::BENCH_CONTEXT,
            &session.cfg,
        )
    {
        let assumed = bench::Assumptions {
            catalog_id: model.id.clone(),
            quantization: quant.label(),
            quantization_source: "runtime",
            context: providers::BENCH_CONTEXT.min(model.context_length),
            run_mode: mode,
            weights_gb: model.weights_gb(quant),
        };
        return (assumed, tps);
    }
    let analysis = fit::analyze(model, &session.hw, session.target, &session.cfg);
    let assumed = bench::Assumptions {
        catalog_id: model.id.clone(),
        quantization: analysis.quant.label(),
        quantization_source: "placement",
        context: analysis.context,
        run_mode: analysis.mode,
        weights_gb: model.weights_gb(analysis.quant),
    };
    (assumed, analysis.estimate.formula_tps)
}

/// Keep what was measured, so it replaces the estimate from now on.
///
/// Only runs matched to a catalog entry at a known quantization are kept —
/// a measurement is only reusable if it is clear which placement it
/// describes — and never from a simulated machine, whose flags describe
/// hardware the benchmark did not run on.
fn record_measurements(session: &Session, runtime: RuntimeKind, report: &bench::BenchReport) {
    if session.hw.simulated {
        return;
    }
    let mut store = MeasurementStore::load();
    let mut kept = 0;
    for result in &report.results {
        let Some(assumed) = &result.assumed else {
            continue;
        };
        if assumed.quantization_source != "runtime" || result.tokens_per_second <= 0.0 {
            continue;
        }
        store.record(StoredMeasurement {
            gpu: session.hw.primary_gpu_name(),
            backend: session.hw.backend.label().to_string(),
            runtime: runtime.label().to_string(),
            context: assumed.context,
            throughput: MeasuredThroughput {
                model_id: assumed.catalog_id.clone(),
                quant: assumed.quantization.to_string(),
                run_mode: assumed.run_mode.label().to_string(),
                tokens_per_second: result.tokens_per_second,
                measured_at: crate::config::now_unix(),
            },
        });
        kept += 1;
    }
    if kept == 0 {
        return;
    }
    // Losing a measurement costs a re-run, not a wrong answer, so a failed
    // write is reported but does not fail the benchmark.
    match store.save() {
        Ok(path) => eprintln!(
            "saved {kept} measurement{} to {} — they replace the estimate for {} from now on",
            if kept == 1 { "" } else { "s" },
            path.display(),
            if kept == 1 {
                "that model"
            } else {
                "those models"
            }
        ),
        Err(e) => eprintln!("could not save measurements: {e}"),
    }
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

/// List the bandwidth table, so `--gpu` can be given a name it will accept.
fn cmd_gpus(cli: &Cli, filter: &[String]) -> Result<(), String> {
    let needle = filter.join(" ").to_ascii_lowercase();
    let cards = hardware::known_gpus(&needle);
    if cards.is_empty() {
        return Err(format!("no known GPU matches '{needle}'"));
    }
    if cli.json {
        println!("{}", display::to_json(&cards));
    } else {
        print!("{}", display::render_gpus(&cards));
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

/// Fit the efficiency factor to what `bench` just measured and store it.
///
/// The speed model is linear in `efficiency`, so this is the one knob that
/// reconciles the estimate with a measurement. It is written only when asked
/// for: a benchmark of one model on a busy machine should not silently move
/// every number llmspec reports.
fn save_calibration(session: &Session, report: &bench::BenchReport) -> Result<(), String> {
    let Some(efficiency) = report.suggested_efficiency else {
        return Err(
            "nothing to calibrate from: none of the benchmarked models matched a \
             catalog entry, so there is no estimate to compare against"
                .to_string(),
        );
    };
    let samples = report
        .results
        .iter()
        .filter(|r| r.estimate_ratio.is_some())
        .count();

    let mut config = Config::load();
    let previous = config.calibration.as_ref().map(|c| c.efficiency);
    config.calibration = Some(Calibration {
        efficiency,
        samples,
        measured_at: crate::config::now_unix(),
        gpu: session.hw.primary_gpu_name(),
        backend: session.hw.backend.label().to_string(),
    });
    config.save()?;

    let from = previous.unwrap_or(session.cfg.efficiency);
    eprintln!(
        "calibrated: efficiency {from:.2} → {efficiency:.2}, from {samples} measurement{} on {}",
        if samples == 1 { "" } else { "s" },
        session.hw.primary_gpu_name()
    );
    Ok(())
}

/// Report on a model file, exiting non-zero when it is damaged.
///
/// The exit code is the point for scripts: this is the check worth running
/// after a download and before a long job that would only fail later.
fn cmd_verify(cli: &Cli, file: &std::path::Path) -> Result<(), String> {
    let report = verify::verify(file)?;
    if cli.json {
        println!("{}", display::to_json(&report));
    } else {
        print!("{}", display::render_verify(&report));
    }
    if report.is_intact() {
        Ok(())
    } else {
        Err(format!("{} is not intact", file.display()))
    }
}

/// Speak MCP on stdin/stdout until the client closes the stream.
///
/// stdout belongs to the protocol here, so the startup line goes to stderr —
/// clients capture it as a log, and anything else on stdout would be read as
/// a malformed message.
fn cmd_mcp(session: Session) -> Result<(), String> {
    let Session {
        hw,
        db,
        cfg,
        target,
        ..
    } = session;
    eprintln!(
        "llmspec MCP server on stdio ({} models, {})",
        db.len(),
        hw.backend.label()
    );
    let mut server = mcp::Mcp::new(hw, db, cfg, target);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    server.run(&mut stdin.lock(), &mut stdout.lock())
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
    // The named card first, so `--memory` can still resize it: a 3090 with
    // 20 GB is a question someone might reasonably ask of a used listing.
    if let Some(name) = &cli.gpu {
        // RAM is read before the card is placed, since a simulated Mac sizes
        // its VRAM from the unified pool.
        if let Some(raw) = &cli.ram {
            let ram = parse_size_gb(raw)?;
            hw.total_ram_gb = ram;
            hw.available_ram_gb = ram;
        }
        hw.simulate_gpu(name, cli.gpu_count)?;
    }
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

/// One model to benchmark, with whatever the runtime knows about it.
///
/// The parameter count matters because a tag need not carry one: `:latest` is
/// an alias, and without the reported size the measurement cannot be lined up
/// against the estimate for the model it actually names.
struct BenchTarget {
    reference: String,
    params_b: Option<f64>,
    /// The quantization the runtime reports holding the weights at.
    quantization: Option<String>,
}

impl From<InstalledModel> for BenchTarget {
    fn from(model: InstalledModel) -> BenchTarget {
        BenchTarget {
            reference: model.name,
            params_b: model.params_b,
            quantization: model.quantization,
        }
    }
}

/// Resolve which models to benchmark.
fn bench_targets(client: &Runtime, query: &str, all: bool) -> Result<Vec<BenchTarget>, String> {
    // A named model is benchmarked whether or not the runtime lists it, but
    // the listing is still worth consulting for the size it reports.
    if !query.is_empty() && !all {
        let listed = client.list_models().ok().and_then(|installed| {
            installed
                .into_iter()
                .find(|m| m.name.eq_ignore_ascii_case(query))
        });
        return Ok(vec![BenchTarget {
            reference: query.to_string(),
            params_b: listed.as_ref().and_then(|m| m.params_b),
            quantization: listed
                .and_then(|m| m.quantization)
                .or_else(|| providers::quantization_in_name(query)),
        }]);
    }

    let installed = client.list_models()?;
    if installed.is_empty() {
        return Err(format!(
            "{} reports no installed models — name one explicitly",
            client.kind.label()
        ));
    }
    let mut targets: Vec<BenchTarget> = installed.into_iter().map(BenchTarget::from).collect();
    if !all {
        // No model named and `--all` not given: benchmark the first one, which
        // is enough to answer "is my machine as fast as llmspec thinks".
        targets.truncate(1);
    }
    Ok(targets)
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
