# llmspec

Find the LLMs that actually run well on **your** hardware.

`llmspec` detects your RAM, CPU and GPU, then ranks a database of open-weight
models by how well each one would really run — picking the best quantization
and context length that fits, working out the run mode (GPU, MoE offload,
CPU+GPU or CPU) and estimating throughput from memory bandwidth.

Targets Windows and Linux.

```
$ llmspec system
System
  CPU        AMD Ryzen 7 7840HS w/ Radeon 780M Graphics (8 cores / 16 threads, x86_64)
  RAM        15.3 GB total, 7.2 GB available
  GPU 0      NVIDIA GeForce RTX 4060 Laptop GPU [NVIDIA] — 8.0 GB VRAM, 272 GB/s
  Backend    CUDA

$ llmspec fit -n 4
#   Model                   Provider      Params Quant   Mode   Fit       Mem%  Ctx  tok/s Score
1   Qwen2.5 7B Instruct     Alibaba Qwen    7.6B Q4_K_M  GPU    Perfect    83%  32K   34.9  76.8
2   Qwen3 30B A3B           Alibaba Qwen  30B/3B Q3_K_M  MoE    Good       62%  32K   79.7  75.3
3   Qwen2.5-VL 7B Instruct  Alibaba Qwen    8.3B Q4_K_M  GPU    Good       88%  32K   32.1  73.6
4   DeepSeek R1 Distill 7B  DeepSeek        7.6B Q4_K_M  GPU    Perfect    83%  32K   34.9  74.0
```

## Build

```sh
cargo build --release
```

The binary lands in `target/release/llmspec`. The model database is embedded at
build time, so the binary has no runtime data dependency.

## Usage

Run with no arguments for the interactive TUI:

```sh
llmspec
```

| Key | Action |
|---|---|
| `j` / `k`, `↑` / `↓` | Move between models |
| `g` / `G`, `PgUp` / `PgDn` | Jump to first / last, scroll by 10 |
| `/` | Search by name, provider, size or use case |
| `Ctrl-U` | Clear the search |
| `f` | Cycle fit filter: All, Runnable, Perfect, Good, Marginal |
| `a` | Cycle availability filter: All, GGUF Avail |
| `s` | Cycle sort column |
| `u` | Cycle target use case and re-rank |
| `Enter` | Toggle the detail panel |
| `h` / `?` | Help |
| `q` / `Esc` | Quit |

Or use the classic CLI:

```sh
llmspec --cli                       # every model, ranked for this machine
llmspec system                      # detected hardware
llmspec list                        # the model database, no analysis
llmspec search "llama 8b"           # search and rank
llmspec info "Mistral-7B"           # full detail for one model
llmspec fit --perfect -n 5          # only perfect fits, top 5
llmspec fit --mode moe              # only models placed in MoE offload
llmspec fit --quant q4_k_m          # only models placed at Q4_K_M
llmspec recommend --limit 5         # top recommendations as JSON
```

### Global flags

| Flag | Meaning |
|---|---|
| `--json` | Machine-readable output (default for `recommend`) |
| `-u`, `--use-case` | `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding` |
| `--memory SIZE` | Override detected VRAM, e.g. `24G`. Creates a synthetic GPU if none was found |
| `--ram SIZE` | Override detected system RAM, e.g. `128G` |
| `--cpu-cores N` | Override detected core count |
| `--max-context N` | Cap the context length used for memory estimation |
| `--cli` | Force the table instead of the TUI |

Sizes accept `G`/`GB`/`GiB`, `M`/`MB`/`MiB`, `T`/`TB`/`TiB`, case-insensitive.
If `--max-context` is not given, `OLLAMA_CONTEXT_LENGTH` is used when set.

Simulating other hardware is just a flag away:

```sh
llmspec fit --memory 24G --ram 64G -u coding -n 10
```

## How the ranking works

Each model is scored on four dimensions from 0 to 100, then combined with
use-case-dependent weights.

| Dimension | What it measures |
|---|---|
| **Quality** | Parameter count, family reputation, quantization loss, use-case affinity |
| **Speed** | Estimated tok/s, saturating at roughly twice reading speed |
| **Fit** | Memory-use efficiency — filling the pool is good, overflowing it is not |
| **Context** | Context window against what the use case actually needs |

**Placement.** Rather than following a fixed priority order, every viable
combination of run mode, context length and quantization is scored, and the
highest composite wins. The trade-off between a longer context, a better
quantization and a faster run mode is exactly what the four dimensions already
express, so the placement optimises the same objective as the ranking.

**Speed.** LLM inference is memory-bandwidth bound: each token requires reading
the active weights once. For recognised GPUs the estimate is
`bandwidth / bytes_read_per_token × efficiency`; for unknown ones a per-backend
constant stands in. Weights that spill into system RAM are charged the much
slower system memory bandwidth.

**Memory.** Weights are sized from the real GGUF bits-per-weight of each
quantization level, and the KV cache from the model's actual layer/head
geometry. On long-context models the cache dwarfs the weights — Qwen3-4B at its
full 256k window needs far more cache than Q4 weights — which is why the
placement steps the context down rather than declaring the model unrunnable.

**MoE.** Mixture-of-experts models are detected from their active parameter
count. With expert offloading only the active experts stay resident, so
Mixtral 8x7B needs roughly the VRAM of a 13B model rather than a 47B one.

**Fit levels.** `Perfect` needs GPU residency with headroom. `Good` is the
ceiling for MoE offload and CPU+GPU. `Marginal` covers a tight squeeze, and is
the ceiling for CPU-only however much RAM there is. `Too Tight` means it does
not fit anywhere, and always sorts last.

## Project layout

```
src/
  main.rs         CLI argument parsing, entry point, TUI launch
  hardware.rs     RAM/CPU/GPU detection, backend selection, size parsing
  models.rs       Model database, quantization hierarchy, memory estimation
  fit.rs          Scoring, speed estimation, placement, MoE offloading
  display.rs      CLI tables, detail view, JSON output
  tui_app.rs      TUI state: filters, sorting, navigation
  tui_ui.rs       TUI rendering (ratatui)
  tui_events.rs   TUI event loop and key handling
data/
  models.json     Model database, embedded at build time
docs/
  REQUIREMENTS.md Full product requirements
```

## Hardware detection

| Platform | Method |
|---|---|
| Windows | RAM/CPU native; NVIDIA via `nvidia-smi` |
| Linux | RAM/CPU native; NVIDIA via `nvidia-smi`, AMD via `rocm-smi`, Intel Arc via `lspci` |

Around 80 GPUs have known memory bandwidth and VRAM, used for both the speed
model and as a fallback when a driver reports VRAM incorrectly. AMD and Intel
detection on Windows is not implemented yet — use `--memory` there.

## Status

Working today: hardware detection, the seed model database, quantization and
memory estimation, the scoring engine, the full CLI, and the TUI with search,
filtering, sorting and the detail panel.

Not built yet: the HuggingFace scraper, plan mode, hardware simulation and
themes in the TUI, Ollama and llama.cpp integration, the REST API, and the
download manager. See `docs/REQUIREMENTS.md`.

## Tests

```sh
cargo test
```

Covers memory maths against hand calculations, placement and ranking
behaviour, size parsing, key handling, and TUI rendering against a test
backend (including a cramped terminal and an empty result set).
