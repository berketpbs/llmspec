# llmspec

**Hundreds of models & providers. One command to find what runs on your hardware.**

A terminal tool that right-sizes LLM models to your system's RAM, CPU, and GPU.
Detects your hardware, scores each model across quality, speed, fit, and context
dimensions, and tells you which ones will actually run well on your machine.

Ships with an interactive TUI (default) and a classic CLI mode. Supports
multi-GPU setups, MoE architectures, dynamic quantization selection, speed
estimation from memory bandwidth, and Ollama integration.

---

## Install

### From source

```sh
cargo install --path .
```

Or build manually:

```sh
cargo build --release
# Binary: target/release/llmspec
```

The model database is embedded at build time — the binary has no runtime data
dependency.

---

## Quick start

```sh
# Interactive TUI (default)
llmspec

# Detected hardware
llmspec system

# Top 5 models for your machine
llmspec fit -n 5

# Coding-optimized recommendations as JSON
llmspec recommend --json --use-case coding --limit 5
```

### Example output

```
$ llmspec system
System
  CPU        AMD Ryzen 7 7840HS w/ Radeon 780M Graphics (8 cores / 16 threads, x86_64)
  RAM        15.3 GB total, 7.2 GB available
  GPU 0      NVIDIA GeForce RTX 4060 Laptop GPU [NVIDIA] — 8.0 GB VRAM, 272 GB/s
  Backend    CUDA

$ llmspec fit -n 5
#   Model                   Provider      Params Quant   Mode   Fit       Mem%  Ctx  tok/s Score
1   Qwen2.5 7B Instruct     Alibaba Qwen    7.6B Q4_K_M  GPU    Perfect    83%  32K   34.9  76.8
2   Qwen3 30B A3B           Alibaba Qwen  30B/3B Q3_K_M  MoE    Good       62%  32K   79.7  75.3
3   Qwen2.5-VL 7B Instruct  Alibaba Qwen    8.3B Q4_K_M  GPU    Good       88%  32K   32.1  73.6
4   DeepSeek R1 Distill 7B  DeepSeek        7.6B Q4_K_M  GPU    Perfect    83%  32K   34.9  74.0
5   Mistral 7B Instruct     Mistral AI      7.3B Q4_K_M  GPU    Perfect    79%  32K   36.3  72.1
```

---

## Usage

### TUI (interactive)

Run with no arguments to launch the interactive interface:

```sh
llmspec
```

| Key | Action |
|---|---|
| `j` / `k`, `↑` / `↓` | Move between models |
| `g` / `G` | Jump to first / last |
| `PgUp` / `PgDn`, `Ctrl-U` / `Ctrl-D` | Scroll by 10 |
| `/` | Search by name, provider, size or use case |
| `f` | Cycle fit filter: All → Runnable → Perfect → Good → Marginal |
| `a` | Cycle availability: All → GGUF Available |
| `s` | Cycle sort: Score → Params → Speed → Mem% → Ctx → Date → Use Case |
| `u` | Cycle target use case and re-rank |
| `d` | Download selected model via Ollama |
| `r` | Refresh installed models from Ollama |
| `Enter` | Toggle the detail panel |
| `h` / `?` | Help |
| `q` / `Esc` | Quit |

### CLI (classic)

```sh
llmspec --cli                           # all models, ranked for this machine
llmspec system                          # detected hardware
llmspec list                            # every model in the database
llmspec search "llama 8b"               # search and rank
llmspec info "Mistral-7B"               # full detail for one model
llmspec fit --perfect -n 5              # perfect fits only, top 5
llmspec fit --mode moe                  # MoE offload placements only
llmspec fit --quant q4_k_m              # models placed at Q4_K_M
llmspec recommend --limit 5             # top recommendations as JSON
llmspec recommend --json --use-case coding --limit 3
```

### Global flags

| Flag | Description |
|---|---|
| `--json` | Machine-readable output (default for `recommend`) |
| `-u`, `--use-case` | `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding` |
| `--memory SIZE` | Override VRAM (e.g. `24G`). Creates a synthetic GPU if none detected |
| `--ram SIZE` | Override system RAM (e.g. `128G`) |
| `--cpu-cores N` | Override core count |
| `--max-context N` | Cap the context length used for memory estimation |
| `--cli` | Force table output instead of the TUI |

Sizes accept `G`/`GB`/`GiB`, `M`/`MB`/`MiB`, `T`/`TB`/`TiB` (case-insensitive).

Simulate different hardware with a single command:

```sh
llmspec fit --memory 24G --ram 64G -u coding -n 10
```

---

## How it works

### Four-dimensional scoring

Each model is scored 0–100 on four dimensions, then combined with
use-case-dependent weights into a composite score:

| Dimension | What it measures |
|---|---|
| **Quality** | Parameter count, family reputation, quantization loss, use-case affinity |
| **Speed** | Estimated tok/s from memory bandwidth and model size |
| **Fit** | Memory-use efficiency — the sweet spot is 50–80% of available pool |
| **Context** | Context window capacity against the use-case target |

### Dynamic quantization

Instead of assuming a fixed quantization, llmspec walks the hierarchy from
Q8_0 (highest quality) down to Q2_K (most compressed) and picks the
highest-quality level that fits. If nothing fits at full context, it retries
at progressively shorter context lengths.

### Speed estimation

LLM inference is memory-bandwidth bound: each token requires reading the
active weights once. For recognized GPUs (~80 in the table), the real
bandwidth drives the estimate:

```
tok/s ≈ bandwidth_GB/s ÷ model_bytes × efficiency
```

For unrecognized GPUs, per-backend constants are used (CUDA: 220, ROCm: 180,
CPU x86: 70, etc.).

### Run modes

| Mode | Description |
|---|---|
| **GPU** | Full model in VRAM — fastest |
| **MoE** | Active experts in VRAM, inactive in RAM |
| **CPU+GPU** | Partial GPU offload, weights spill to RAM |
| **CPU** | Entirely in system RAM |

### Fit levels

| Level | Meaning |
|---|---|
| **Perfect** | Recommended memory met on GPU, with headroom |
| **Good** | Fits comfortably; ceiling for MoE and CPU+GPU |
| **Marginal** | Tight squeeze, or CPU-only (always caps here) |
| **Too Tight** | Does not fit anywhere — always sorted last |

### MoE support

Mixture-of-experts models are detected from their active parameter count.
With expert offloading, only the active experts stay resident — Mixtral 8x7B
needs roughly the VRAM of a 13B model rather than a 47B one.

---

## Model database

95 models from 20+ providers, embedded at build time:

| Provider | Models |
|---|---|
| **Meta** | Llama 3.1, 3.2, 3.3, Llama 4 Scout/Maverick, Vision |
| **Alibaba Qwen** | Qwen2.5, Qwen3 (0.6B–235B), QwQ, Coder, VL |
| **Google** | Gemma 2, Gemma 3, CodeGemma |
| **Microsoft** | Phi-3, Phi-3.5, Phi-4 |
| **Mistral AI** | Mistral 7B, NeMo, Small, Large, Codestral, Mixtral, Pixtral |
| **DeepSeek** | V3, R1, R1 Distills, Coder V2 |
| **IBM** | Granite 3.3 |
| **Cohere** | Command R/R+, Aya Expanse |
| **NVIDIA** | Nemotron, Minitron |
| **Others** | StarCoder2, Yi, OLMo 2, Falcon 3, InternLM, Solar Pro, Hermes 3 |
| **Embedding** | nomic-embed, BGE-M3, BGE Large, MiniLM, Arctic Embed, Qwen3 Embed |

---

## Hardware detection

| Platform | Method |
|---|---|
| **Windows** | RAM/CPU native; NVIDIA via `nvidia-smi` |
| **Linux** | RAM/CPU native; NVIDIA via `nvidia-smi`, AMD via `rocm-smi`, Intel Arc via `lspci` |

~80 GPUs have known memory bandwidth (NVIDIA consumer + datacenter, AMD, Intel,
Apple). If auto-detection fails, override with `--memory` / `--ram` /
`--cpu-cores`.

---

## Runtime providers

### Ollama

llmspec detects installed Ollama models and can download new ones:

- Auto-detects Ollama at `http://localhost:11434` (or `OLLAMA_HOST`)
- `GET /api/tags` lists installed models (shown with ✓ in the TUI)
- `d` in the TUI triggers `POST /api/pull` with background download
- `r` refreshes the installed model list

---

## Project layout

```
src/
  main.rs         CLI argument parsing, entry point, TUI launch
  hardware.rs     RAM/CPU/GPU detection, backend selection, size parsing
  models.rs       Model database, quantization hierarchy, memory estimation
  fit.rs          Scoring, speed estimation, placement, MoE offloading
  providers.rs    Runtime provider integration (Ollama)
  display.rs      CLI tables, detail view, JSON output
  tui_app.rs      TUI state: filters, sorting, navigation
  tui_ui.rs       TUI rendering (ratatui)
  tui_events.rs   TUI event loop and key handling
data/
  models.json     Model database (95 models, embedded at build time)
docs/
  REQUIREMENTS.md Full product requirements
```

---

## Tests

```sh
cargo test
```

52 tests covering:
- Memory math against hand calculations
- Placement and ranking behavior
- MoE offloading correctness
- Size parsing (`24G`, `128GiB`, `4T`)
- Key handling and mode transitions
- TUI rendering (including cramped terminals and empty result sets)
- Ollama API response parsing

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Run `cargo test` and `cargo clippy`
4. Open a pull request

---

## License

MIT
