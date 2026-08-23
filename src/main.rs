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
        /// Target tokens/sec (for planning reverse)
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

fn run(cli: Cli) -> Result<(), String> {
    let hw = build_hardware(&cli)?;
    let db = ModelDb::embedded();
    let target = match &cli.use_case {
        Some(raw) => UseCase::parse(raw)
            .ok_or_else(|| format!("unknown use case '{raw}' (try: general, coding, reasoning, chat, multimodal, embedding)"))?,
        None => UseCase::General,
    };
    let cfg = SpeedConfig {
        context_cap: resolve_context_cap(&cli),
        ..SpeedConfig::default()
    };

    match &cli.command {
        Some(Command::System) => {
            if cli.json {
                println!("{}", display::to_json(&hw));
            } else {
                print!("{}", display::render_system(&hw));
            }
        }

        Some(Command::List) => {
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
        }

        Some(Command::Search { query, limit }) => {
            let q = query.join(" ");
            let matched: Vec<_> = db
                .models
                .iter()
                .filter(|m| m.matches(&q))
                .cloned()
                .collect();
            if matched.is_empty() {
                return Err(format!("no model matches '{q}'"));
            }
            let mut results = fit::analyze_all(&matched, &hw, target, &cfg);
            truncate_results(&mut results, *limit);
            emit_results(&cli, &hw, target, &results);
        }

        Some(Command::Info { model }) => {
            let q = model.join(" ");
            let found = db
                .find(&q)
                .ok_or_else(|| format!("no model matches '{q}'"))?;
            let result = fit::analyze(found, &hw, target, &cfg);
            if cli.json {
                println!("{}", display::to_json(&result));
            } else {
                print!("{}", display::render_detail(&result, found));
            }
        }

        Some(Command::Fit {
            perfect,
            min_fit,
            runnable,
            provider,
            quant,
            mode,
            limit,
        }) => {
            let mut results = fit::analyze_all(&db.models, &hw, target, &cfg);
            if let Some(name) = provider {
                let needle = name.to_ascii_lowercase();
                results.retain(|r| r.provider.to_ascii_lowercase().contains(&needle));
            }
            if let Some(raw) = quant {
                let wanted = Quant::parse(raw)
                    .ok_or_else(|| format!("unknown quantization '{raw}' (try: q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k)"))?;
                results.retain(|r| r.quant == wanted);
            }
            if let Some(raw) = mode {
                let wanted = RunMode::parse(raw).ok_or_else(|| {
                    format!("unknown run mode '{raw}' (try: gpu, moe, cpu+gpu, cpu)")
                })?;
                results.retain(|r| r.mode == wanted);
            }
            let floor = resolve_min_fit(*perfect, *runnable, min_fit.as_deref())?;
            if let Some(floor) = floor {
                results.retain(|r| r.fit >= floor);
            }
            truncate_results(&mut results, *limit);
            emit_results(&cli, &hw, target, &results);
        }

        Some(Command::Recommend { limit, table }) => {
            let mut results = fit::analyze_all(&db.models, &hw, target, &cfg);
            results.retain(FitResult::is_runnable);
            truncate_results(&mut results, Some(*limit));
            // `recommend` defaults to JSON; `--table` opts back into text.
            if *table && !cli.json {
                print!("{}", display::render_table(&results));
            } else {
                print!("{}", report_json(&hw, target, &results));
                println!();
            }
        }

        Some(Command::Plan {
            model,
            context,
            quant,
            target_tps: _,
        }) => {
            let q = model.join(" ");
            let found = db
                .find(&q)
                .ok_or_else(|| format!("no model matches '{q}'"))?;
            let ctx = context.unwrap_or(found.context_length);
            let quant_level = match quant.as_deref() {
                Some(raw) => Quant::parse(raw)
                    .ok_or_else(|| format!("unknown quantization '{raw}' (try: q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k)"))?,
                None => Quant::Q4KM,
            };
            let plan = fit::plan(found, quant_level, ctx, &cfg);
            if cli.json {
                println!("{}", display::to_json(&plan));
            } else {
                print!("{}", display::render_plan(&plan));
            }
        }

        None => {
            if cli.cli || cli.json {
                let results = fit::analyze_all(&db.models, &hw, target, &cfg);
                emit_results(&cli, &hw, target, &results);
            } else {
                let mut app = tui_app::App::new(hw, db, target, cfg);
                tui_events::run(&mut app).map_err(|e| format!("terminal error: {e}"))?;
            }
        }
    }

    Ok(())
}

fn build_hardware(cli: &Cli) -> Result<Hardware, String> {
    let mut hw = Hardware::detect();
    let vram = cli.memory.as_deref().map(parse_size_gb).transpose()?;
    let ram = cli.ram.as_deref().map(parse_size_gb).transpose()?;
    hw.apply_overrides(vram, ram, cli.cpu_cores);
    Ok(hw)
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
        return FitLevel::parse(raw).map(Some).ok_or_else(|| {
            format!("unknown fit level '{raw}' (try: perfect, good, marginal, too_tight)")
        });
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

fn emit_results(cli: &Cli, hw: &Hardware, target: UseCase, results: &[FitResult]) {
    if cli.json {
        print!("{}", report_json(hw, target, results));
        println!();
    } else {
        print!("{}", display::render_table(results));
    }
}

fn report_json(hw: &Hardware, target: UseCase, results: &[FitResult]) -> String {
    display::to_json(&display::JsonReport {
        system: hw,
        use_case: target.as_str(),
        count: results.len(),
        models: results,
    })
}
