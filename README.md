<div align="center">

# llmspec

**240 models. 56 providers. One command to find what actually runs on your machine.**

[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

---

Downloading a 40 GB model to find out it crawls at 0.3 tokens a second is a
bad afternoon. llmspec reads your hardware, works out what every model in its
catalog would cost to run on it, and tells you which ones are worth your
bandwidth — before you spend it.

```
$ llmspec fit -n 5
#   Model                        Provider      Params Quant   Mode  Fit           Size  Mem%  Ctx  tok/s Score
1   Qwen3 30B A3B Instruct 2507  Alibaba Qwen  30B/3B Q3_K_M  MoE   Good        13 GB   62%  32K   79.7  77.3
2   GPT-OSS 20B                  OpenAI         21B/4B Q6_K   MoE   Good        16 GB   79%  64K   43.5  77.2
3   Qwen2.5 7B Instruct          Alibaba Qwen    7.6B Q4_K_M  GPU   Perfect    4.3 GB   83%  32K   34.9  76.8
4   Qwen3 30B A3B                Alibaba Qwen  30B/3B Q3_K_M  MoE   Good        13 GB   62%  32K   79.7  75.3
5   GLM-4 9B Chat                Zhipu AI        9.4B Q3_K_M  GPU   Marginal   4.3 GB   92%  32K   35.0  75.2
```

## Contents

- [Install](#install) · [Quick start](#quick-start) · [The interactive interface](#the-interactive-interface)
- [Commands](#commands) · [Filtering](#filtering) · [Benchmarking](#benchmarking) · [HTTP API](#http-api)
- [Configuration](#configuration) · [Adding your own models](#adding-your-own-models)
- [How it decides](#how-it-decides) · [Supported hardware](#supported-hardware) · [Runtimes](#runtimes)

---

## Install

```sh
cargo install --path .
```

Or build it directly:

```sh
cargo build --release      # target/release/llmspec
```

Needs Rust 1.85 or newer. The model catalog is compiled into the binary, so
there is nothing else to install and nothing to fetch at runtime.

---

## Quick start

```sh
llmspec                      # interactive interface
llmspec fit -n 10            # the ten best models for this machine
llmspec doctor               # what was detected, and what was guessed
llmspec bench                # measure real tokens/sec
```

Four questions llmspec exists to answer:

**"What should I download?"**

```sh
llmspec fit --perfect -n 5
```

**"What fits in 8 GB and still runs at a readable speed?"**

```sh
llmspec fit --memory 8G --min-tps 15 --max-size 6G
```

**"Can my machine run this specific model?"**

```sh
llmspec info "Qwen2.5-Coder-32B"
```

**"What would I need to buy to run it?"**

```sh
llmspec plan "Llama-3.3-70B" --quant q4_k_m --context 32768
```

---

## The interactive interface

Running `llmspec` with no arguments opens the full interface: every model,
ranked, filterable, with a detail panel that tells you the exact command to
run the one you picked.

```
┌ llmspec ─────────────────────────────────────────────────────────────────┐
│ CPU        AMD Ryzen 7 7840HS (8 cores / 16 threads)                      │
│ RAM        15.3 GB total, 6.3 GB available                                │
│ GPU        NVIDIA GeForce RTX 4060 Laptop GPU — 8.0 GB VRAM, 272 GB/s     │
│ Backend    CUDA   use case general   runtimes Ollama                      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Keys

| Key | Action |
|---|---|
| `j` `k`, `↑` `↓` | Move between models |
| `g` `G`, `Home` `End` | Jump to first / last |
| `PgUp` `PgDn`, `Ctrl-U` `Ctrl-D` | Scroll by ten |
| `/` | Search by name, provider, size or capability |
| `f` | Fit filter: all → runnable → perfect → good → marginal |
| `a` | Show: all → GGUF builds → already installed |
| `s` | Sort column |
| `u` | Target use case, and re-rank for it |
| `Enter` | Detail panel: memory, context, and the command to run it |
| `p` | Hardware plan: what this model would need from any machine |
| `m` then `c` | Mark a model, then compare it with the selected one |
| `d` | Download the selected model through Ollama |
| `r` | Re-probe local runtimes and installed models |
| `S` | Simulate different VRAM, RAM or core count |
| `A` | Edit the speed model's tunables |
| `t` | Cycle the colour theme |
| `h` `?` | Help |
| `q` `Esc` | Quit |

Downloads and runtime probes run in the background — the interface never
blocks on the network. Ten themes are included; the choice is remembered.

---

## Commands

| Command | What it does |
|---|---|
| `llmspec` | Interactive interface |
| `llmspec fit` | Every model ranked for this machine |
| `llmspec recommend` | Top picks as JSON, for scripts and agents |
| `llmspec info <model>` | One model in full, with the command to run it |
| `llmspec plan <model>` | What hardware this model would need |
| `llmspec search <query>` | Search the catalog and rank the matches |
| `llmspec list` | The catalog, with no hardware analysis |
| `llmspec system` | Detected hardware |
| `llmspec doctor` | Diagnostic report; exits non-zero on warnings |
| `llmspec runtimes` | Local inference servers that are running |
| `llmspec bench` | Measure real tokens/sec against a running runtime |
| `llmspec serve` | Read-only HTTP API |

### Global flags

| Flag | Description |
|---|---|
| `--json` | Machine-readable output (the default for `recommend`) |
| `-u`, `--use-case` | `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding` |
| `--force-runtime` | Score for one runtime: `ollama`, `llamacpp`, `lmstudio`, `vllm`, `docker`, `mlx` |
| `--memory SIZE` | Override VRAM, e.g. `24G` — creates a synthetic GPU if none is found |
| `--ram SIZE` | Override system RAM, e.g. `128G` |
| `--cpu-cores N` | Override the core count |
| `--max-context N` | Cap the context used for memory estimation |
| `--cli` | Force table output instead of the interface |

Sizes accept `G`/`GB`/`GiB`, `M`/`MB`/`MiB`, `T`/`TB`/`TiB`, case-insensitive.

The overrides make llmspec a shopping tool as much as a diagnostic one:

```sh
llmspec fit --memory 24G --ram 64G -u coding -n 10     # if I bought a 3090
llmspec fit --memory 0 --ram 64G                        # CPU-only server
```

`--force-runtime` does two things: it shifts the throughput estimate to that
runtime's characteristics, and it hides models the runtime cannot load — a
GGUF loader never sees a model with no GGUF build.

---

## Filtering

`fit` narrows on the things people actually decide by:

```sh
llmspec fit --min-tps 20              # must be fast enough to read along
llmspec fit --max-size 8G             # must fit the disk budget
llmspec fit --min-context 32768       # must hold a real working set
llmspec fit --perfect                 # must fit VRAM with headroom
llmspec fit --mode gpu                # no CPU offload
llmspec fit --quant q4_k_m            # placed at a specific quantization
llmspec fit --provider mistral        # one publisher
```

They compose:

```sh
llmspec fit -u coding --min-tps 25 --max-size 10G --min-context 32768 -n 5
```

> *"A coding model that runs at 25+ tokens a second, downloads in under 10 GB,
> and holds 32k of context."*

---

## Benchmarking

Every other number llmspec prints is an estimate. `bench` is the ground truth:
it asks a running runtime to generate tokens and reports what came back.

```sh
llmspec bench                    # the first installed model on the first live runtime
llmspec bench qwen3:8b           # a specific model
llmspec bench --all --runs 5     # everything installed, five timed runs each
llmspec bench --json             # machine-readable
```

```
$ llmspec bench qwen3:8b
Measured throughput
  AMD Ryzen 7 7840HS · NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) · 15 GB RAM · CUDA

Model                        Runtime         tok/s          range     TTFT   vs est.
------------------------------------------------------------------------------------
qwen3:8b                     Ollama           57.5      57.4-57.6     18ms     1.77x

What the estimate assumed
  Qwen/Qwen3-8B at Q4_K_M, 16K context, 4.6 GB of weights read per token
  If the ratio is far from 1.00x, check these against what the runtime actually
  loaded; the remaining gap is the speed model itself, which is deliberately
  conservative.

  To match these measurements, set the efficiency factor to 0.97
  (press A in the TUI; the value is saved for next time)
```

The first run is untimed — it pays for loading the model, and timing it would
report disk speed rather than inference speed. Results are the median of the
timed runs.

`vs est.` is measured over estimated. The estimate is conservative by design,
so a ratio above 1.0 is normal. What matters is that llmspec tells you exactly
what it assumed and hands you the one number that reconciles the two, instead
of leaving you to guess which knob to turn.

---

## HTTP API

`llmspec serve` exposes the analysis as read-only JSON. It binds loopback by
default, because it reports what hardware the machine has.

```sh
llmspec serve --host 127.0.0.1 --port 8228
```

| Route | Returns |
|---|---|
| `GET /health` | Version, catalog size, route list |
| `GET /system` | Detected hardware |
| `GET /runtimes` | Local runtimes that are running |
| `GET /catalog` | The model database, unanalysed |
| `GET /models` | Ranked fit analysis |
| `GET /models/top` | The five best runnable models |
| `GET /models/{id}` | One model's analysis |

`/models` accepts `limit`, `use_case`, `provider`, `search`, `quant`, `mode`,
`min_fit`, `perfect`, `include_too_tight`, `max_context`, `min_tps`,
`max_size_gb` and `min_context`.

```sh
curl "http://127.0.0.1:8228/models?use_case=coding&min_tps=20&limit=3"
curl "http://127.0.0.1:8228/models/Qwen%2FQwen2.5-7B-Instruct"
```

Built on `std::net` — serving adds no dependency.

---

## Configuration

llmspec remembers the theme, the target use case and the speed tunables, so a
session starts where the last one ended. Files live in:

| Platform | Location |
|---|---|
| Windows | `%APPDATA%\llmspec\` |
| Linux, macOS | `$XDG_CONFIG_HOME/llmspec/` or `~/.config/llmspec/` |
| Anywhere | `$LLMSPEC_CONFIG_DIR` overrides both |

`config.json` holds the settings. A malformed one falls back to defaults
rather than stopping llmspec from starting.

```json
{
  "theme": 3,
  "use_case": "coding",
  "speed": { "efficiency": 0.72, "gpu_factor": 1.0 }
}
```

---

## Adding your own models

Anything newer than the build, or private, goes in `models.json` in the same
directory. It is merged into the catalog at startup.

```json
{
  "models": [
    {
      "id": "internal/support-bot-7b",
      "name": "Support Bot 7B",
      "provider": "Internal",
      "params_b": 7.6,
      "context_length": 32768,
      "use_case": "chat",
      "gguf": true,
      "layers": 28, "hidden_size": 3584, "kv_heads": 4, "head_dim": 128
    }
  ]
}
```

| Field | Required | Notes |
|---|---|---|
| `id`, `name`, `provider` | yes | `id` is the upstream repository path |
| `params_b` | yes | Total parameters in billions |
| `context_length` | yes | Native maximum |
| `use_case` | yes | One of the six |
| `active_params_b` | no | Set it to declare a MoE model |
| `ollama` | no | Runtime tag, if one exists |
| `quality_tier` | no | 1–5 family reputation, default 3 |
| `layers`, `hidden_size`, `kv_heads`, `head_dim` | no | All four or none |

An entry whose `id` matches a shipped one replaces it — that is how a stale
record gets corrected locally.

With the geometry, the KV cache is sized exactly. Without it, llmspec falls
back to a parameter-count heuristic, which is less precise on models that use
multi-head rather than grouped-query attention.

---

## How it decides

Each model is scored 0–100 on four dimensions, weighted by use case:

| Dimension | What it measures |
|---|---|
| **Quality** | Parameter count, family reputation, quantization loss, use-case fit |
| **Speed** | Estimated tokens/sec from memory bandwidth and weight size |
| **Fit** | Memory efficiency — the plateau runs from 50% to 95% of the pool |
| **Context** | Context held against what the use case needs |

Reasoning leans on quality (0.55); chat leans on speed (0.35).

**Quantization is chosen, not assumed.** llmspec walks Q8_0 down to Q2_K and
picks the best-scoring level that fits, retrying at shorter contexts when the
full window will not.

**Mixture-of-experts is modelled properly.** Only the active experts need to
be resident, so Mixtral 8x7B needs about the VRAM of a 13B model, not a 47B
one.

**A model that fits but crawls is not a recommendation.** Scores scale towards
zero below 3 tokens a second, and a model that does not fit anywhere scores 0
and sorts last.

| Fit level | Meaning |
|---|---|
| **Perfect** | Fits VRAM with 15% headroom |
| **Good** | Fits, or offloading cleanly |
| **Marginal** | Over 90% of VRAM, or CPU-only |
| **Too Tight** | Does not fit anywhere |

The full derivation — memory arithmetic, the bandwidth model, every weight and
threshold, and the limitations — is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Supported hardware

| Platform | Detection |
|---|---|
| **Windows** | CPU/RAM native; NVIDIA via `nvidia-smi`; AMD and Intel from the display driver's registry key |
| **Linux** | CPU/RAM native; NVIDIA via `nvidia-smi`; AMD via `rocm-smi`; Intel Arc via `lspci` |
| **macOS** | Apple Silicon detected from the chip; VRAM sized from unified memory |

Around 90 GPUs have a known memory bandwidth — NVIDIA consumer and datacenter,
AMD Radeon and Instinct, Intel Arc, and Apple M1 through M4. Unrecognised
cards fall back to a per-backend constant, and `llmspec doctor` says so rather
than quietly guessing.

Multi-GPU VRAM is summed, with a tensor-parallelism factor applied.

Two details worth knowing:

- On Windows, VRAM comes from the driver's 64-bit `qwMemorySize`, not
  `Win32_VideoController.AdapterRAM` — that field is 32-bit and reports 4 GB
  for a 24 GB card.
- On Apple Silicon, 75% of unified memory is treated as usable VRAM, matching
  the default `iogpu.wired_limit_mb` cap.

If detection is wrong, `--memory`, `--ram` and `--cpu-cores` override it.

---

## Runtimes

llmspec talks to whatever inference server is already running.

| Runtime | Default endpoint | Environment override |
|---|---|---|
| Ollama | `http://127.0.0.1:11434` | `OLLAMA_HOST` |
| llama.cpp (`llama-server`) | `http://127.0.0.1:8080` | `LLAMA_CPP_HOST` |
| LM Studio | `http://127.0.0.1:1234` | `LMSTUDIO_HOST` |
| vLLM | `http://127.0.0.1:8000` | `VLLM_HOST` |
| Docker Model Runner | `http://127.0.0.1:12434` | `DOCKER_MODEL_HOST` |
| MLX | `http://127.0.0.1:8080` | `MLX_HOST` |

Ollama uses its own API; the rest speak the OpenAI-compatible `/v1` surface.
Discovery does a TCP connect check first, so probing five absent runtimes
costs microseconds rather than five timeouts.

llmspec recognises a model across runtimes even when they disagree about its
name — `Qwen/Qwen2.5-7B-Instruct`, `qwen2.5:7b`, `qwen2.5-7b-instruct` and
`Qwen2.5-7B-Instruct-Q4_K_M.gguf` all resolve to the same catalog entry. The
matching is deliberately conservative: `phi-4` and `phi-4-mini` stay distinct,
because a wrong "already installed" tick is worse than a missed one.

Ollama is the only runtime with a download API, so `d` in the interface works
there. For the others, the detail panel prints the command to run yourself.

**Nothing leaves your machine.** Every endpoint is loopback unless you point
an environment variable elsewhere, and llmspec makes no other network calls.

---

## The catalog

240 models from 56 providers, embedded at build time: 39 mixture-of-experts
architectures, 34 vision and multimodal models, 20 embedding and reranking
models, from 23M to 1T parameters.

Meta · Alibaba Qwen · OpenAI · Google · Microsoft · Mistral AI · DeepSeek ·
Zhipu AI · IBM · Cohere · NVIDIA · Moonshot · MiniMax · xAI · AI21 ·
Databricks · Tencent · Baidu · ByteDance · LG AI · Allen AI · TII ·
Hugging Face · OpenBMB · Liquid AI · Arcee · Nous Research · Perplexity ·
Shanghai AI Lab · BAAI · Nomic · Jina · and more.

The catalog is generated by `scripts/add_models.py`, which merges entries by
`id` so re-running it corrects a record rather than duplicating it.

---

## Development

```sh
cargo test          # 191 tests
cargo clippy --all-targets
cargo fmt
```

The test suite covers the memory arithmetic against hand calculations, GPU
name matching including the collisions it is designed to avoid, placement and
ranking under every sort order, runtime response parsing for both API shapes,
HTTP routing and error statuses, config round-tripping, key handling, and TUI
rendering — including cramped terminals, empty result sets and every theme.

Dependencies: `clap`, `colored`, `crossterm`, `ratatui`, `serde`,
`serde_json`, `sysinfo`, `ureq`. No unsafe code.

---

## License

MIT
