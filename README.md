# llmspec

**240 models. 56 providers. One command to find what runs on your hardware.**

A terminal tool that right-sizes LLM models to your system's RAM, CPU, and GPU.
Detects your hardware, scores each model across quality, speed, fit, and context
dimensions, and tells you which ones will actually run well on your machine.

Ships with an interactive TUI (default) and a classic CLI mode. Supports
multi-GPU setups, Apple Silicon unified memory, MoE architectures, dynamic
quantization selection, speed estimation from memory bandwidth, real
benchmarking against a running runtime, and a read-only HTTP API.

Local runtimes: **Ollama**, **llama.cpp**, **LM Studio**, **vLLM**,
**Docker Model Runner**, **MLX**.

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
| `r` | Refresh installed models from every live runtime |
| `t` | Cycle theme (persisted between sessions) |
| `m` / `c` | Mark a model / compare against the marked one |
| `p` | Hardware requirements for the selected model |
| `S` | Simulate different VRAM / RAM / core count |
| `A` | Advanced config: speed tunables (persisted) |
| `Enter` | Toggle the detail panel |
| `h` / `?` | Help |
| `q` / `Esc` | Quit |

### CLI (classic)

```sh
llmspec --cli                           # all models, ranked for this machine
llmspec system                          # detected hardware
llmspec doctor                          # diagnostic report (exits 2 on warnings)
llmspec list                            # every model in the database
llmspec search "llama 8b"               # search and rank
llmspec info "Mistral-7B"               # full detail for one model
llmspec fit --perfect -n 5              # perfect fits only, top 5
llmspec fit --mode moe                  # MoE offload placements only
llmspec fit --quant q4_k_m              # models placed at Q4_K_M
llmspec recommend --limit 5             # top recommendations as JSON
llmspec recommend --json --use-case coding --limit 3
llmspec plan "Llama-3.1-70B" --quant q4_k_m --context 32768
llmspec runtimes                        # local inference servers that are running
llmspec bench qwen2.5:7b                # measure real tok/s
llmspec serve --port 8228               # read-only HTTP API
```

### Global flags

| Flag | Description |
|---|---|
| `--json` | Machine-readable output (default for `recommend`) |
| `-u`, `--use-case` | `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding` |
| `--force-runtime` | Score for one runtime: `ollama`, `llamacpp`, `lmstudio`, `vllm`, `docker`, `mlx` |
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

`--force-runtime` does two things: it shifts the throughput estimate to that
runtime's characteristics, and it drops models the runtime cannot load — a
GGUF loader never sees a model with no GGUF build.

---

## Benchmarking

Every other number llmspec prints is an estimate from a bandwidth model.
`bench` is the ground truth — it asks a running runtime to generate tokens and
reports what actually came back:

```sh
llmspec bench                    # first installed model on the first live runtime
llmspec bench qwen2.5:7b         # one model
llmspec bench --all --runs 5     # every installed model, five timed runs each
llmspec bench --json             # machine-readable
```

```
$ llmspec bench qwen2.5:7b
Measured throughput
  AMD Ryzen 7 7840HS · NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) · 15 GB RAM · CUDA

Model                        Runtime         tok/s          range     TTFT   vs est.
--------------------------------------------------------------------------------------
qwen2.5:7b                   Ollama           32.4      31.8-33.1     91ms     0.93x
```

The first run is untimed: it pays for loading the model, and timing it would
report disk speed rather than inference speed. `vs est.` is measured over
estimated — a figure far from `1.00x` means the bandwidth model mispredicted
this machine.

---

## HTTP API

`llmspec serve` exposes the fit engine over a read-only JSON API. It binds
loopback by default, because it reports what hardware the machine has.

```sh
llmspec serve --host 127.0.0.1 --port 8228
```

| Route | Returns |
|---|---|
| `GET /health` | Version and catalog size |
| `GET /system` | Detected hardware |
| `GET /runtimes` | Local runtimes that are running |
| `GET /catalog` | The model database, unanalysed |
| `GET /models` | Ranked fit analysis |
| `GET /models/top` | Top 5 runnable models |
| `GET /models/{id}` | One model's analysis |

`/models` accepts `limit`, `use_case`, `provider`, `search`, `quant`, `mode`,
`min_fit`, `perfect`, `include_too_tight` and `max_context`.

```sh
curl "http://127.0.0.1:8228/models?use_case=coding&limit=3&min_fit=good"
curl "http://127.0.0.1:8228/models/Qwen%2FQwen2.5-7B-Instruct"
```

---

## Configuration

llmspec persists the TUI theme, the target use case and the speed tunables so
a session starts where the last one left off. Two optional files live in the
config directory — `%APPDATA%\llmspec` on Windows, `$XDG_CONFIG_HOME/llmspec`
(or `~/.config/llmspec`) elsewhere, or wherever `LLMSPEC_CONFIG_DIR` points:

| File | Purpose |
|---|---|
| `config.json` | Theme, default use case, speed factors |
| `models.json` | Extra models merged into the embedded catalog |

A user entry with the same `id` as a shipped one replaces it, which is how a
stale record gets corrected locally. Both files are optional, and a malformed
one falls back to defaults rather than stopping llmspec from starting.

```json
{
  "models": [
    {
      "id": "local/my-finetune",
      "name": "My Finetune 7B",
      "provider": "Internal",
      "params_b": 7.6,
      "context_length": 32768,
      "use_case": "coding",
      "gguf": true,
      "layers": 28, "hidden_size": 3584, "kv_heads": 4, "head_dim": 128
    }
  ]
}
```

Geometry (`layers`, `hidden_size`, `kv_heads`, `head_dim`) is optional. With
it the KV cache is sized exactly; without it llmspec falls back to a
parameter-count heuristic.

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
active weights once. For recognized GPUs (~85 in the table), the real
bandwidth drives the estimate:

```
tok/s ≈ bandwidth_GB/s ÷ model_bytes × efficiency
```

For unrecognized GPUs, per-backend constants are used (CUDA: 220, ROCm: 180,
CPU x86: 70, etc.). `--force-runtime` scales the result by that runtime's
single-stream characteristics — MLX reads unified memory with Apple-tuned
kernels, vLLM's paged attention gains little outside batching.

Run `llmspec bench` to check the estimate against a real generation on your
own machine; the tunables behind it are editable in the TUI (`A`) and
persisted.

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

240 models from 56 providers, embedded at build time — 39 MoE architectures,
20 embedding models, 34 vision/multimodal models, from 23M to 1T parameters:

| Provider | Models |
|---|---|
| **Meta** | Llama 3.1/3.2/3.3 (1B–405B), Llama 4 Scout/Maverick, Vision, Guard, Code Llama |
| **Alibaba Qwen** | Qwen2.5, Qwen3 (0.6B–235B), Qwen3-Next, QwQ, Coder, VL, Math, Embedding |
| **OpenAI** | GPT-OSS 20B / 120B |
| **Google** | Gemma 2, Gemma 3 (270M–27B), Gemma 3n, CodeGemma, MedGemma, EmbeddingGemma |
| **Microsoft** | Phi-2, Phi-3, Phi-3.5, Phi-4, Phi-4 Reasoning, Phi-4 Multimodal |
| **Mistral AI** | Mistral, Ministral, NeMo, Small 3, Large, Codestral, Devstral, Magistral, Mixtral, Pixtral |
| **DeepSeek** | V2 Lite, V3, R1, R1 Distills, Coder, Coder V2, Prover |
| **Zhipu AI** | GLM-4, GLM-4 32B, GLM-Z1, GLM-4.5 Air, GLM-4V |
| **IBM** | Granite 3.3, Granite 4.0, Granite Vision, Granite Code, Granite Embedding |
| **Cohere** | Command A, Command R/R+, Command R7B, Aya Expanse, Aya Vision |
| **NVIDIA** | Nemotron Nano/Super, Llama-Nemotron, Minitron, NVLM |
| **Moonshot / MiniMax / xAI** | Kimi K2, Kimi-VL, Moonlight, MiniMax Text/M1, Grok-1, Grok-2 |
| **Others** | StarCoder2, Yi, OLMo 2, Tulu 3, Molmo, Falcon 3/H1/Mamba, InternLM, InternVL, EXAONE, ERNIE, Hunyuan, Seed-OSS, Jamba, DBRX, SmolLM2/3, MiniCPM, LFM2, Zamba2, RWKV |
| **Embedding** | nomic-embed, BGE-M3/Large/Reranker, E5, GTE, Jina v3, mxbai, Stella, Arctic, MiniLM |

The catalog is generated from `scripts/add_models.py`, which merges entries by
`id` — re-running it corrects a shipped record rather than duplicating it.

---

## Hardware detection

| Platform | Method |
|---|---|
| **Windows** | RAM/CPU native; NVIDIA via `nvidia-smi`; AMD and Intel from the display driver's registry key |
| **Linux** | RAM/CPU native; NVIDIA via `nvidia-smi`, AMD via `rocm-smi`, Intel Arc via `lspci` |
| **macOS** | Apple Silicon detected from the chip name; VRAM sized from unified memory |

On Windows the VRAM figure comes from the driver's 64-bit `qwMemorySize`
rather than `Win32_VideoController.AdapterRAM`, which is a 32-bit field that
silently saturates at 4 GB. On Apple Silicon 75% of unified memory is treated
as usable VRAM, matching the default `iogpu.wired_limit_mb` cap.

~85 GPUs have known memory bandwidth (NVIDIA consumer + datacenter, AMD, Intel,
Apple). Names are matched on word boundaries after stripping vendor decorations,
so `Intel(R) Arc(TM) A770 Graphics` resolves and `m4` never claims a Tesla M40.
If auto-detection fails, override with `--memory` / `--ram` / `--cpu-cores`.

---

## Runtime providers

llmspec talks to whatever inference server is already running. Ollama has its
own API; the rest speak the OpenAI-compatible `/v1` surface.

| Runtime | Default endpoint | Env override |
|---|---|---|
| Ollama | `http://127.0.0.1:11434` | `OLLAMA_HOST` |
| llama.cpp (`llama-server`) | `http://127.0.0.1:8080` | `LLAMA_CPP_HOST` |
| LM Studio | `http://127.0.0.1:1234` | `LMSTUDIO_HOST` |
| vLLM | `http://127.0.0.1:8000` | `VLLM_HOST` |
| Docker Model Runner | `http://127.0.0.1:12434` | `DOCKER_MODEL_HOST` |
| MLX | `http://127.0.0.1:8080` | `MLX_HOST` |

Discovery does a TCP connect check before any HTTP request, so probing five
absent runtimes costs microseconds instead of five timeouts. Nothing leaves the
machine: every default endpoint is loopback.

Ollama is the only runtime with a download API — `d` in the TUI triggers a
background `POST /api/pull`, and `r` refreshes the installed list. For the
others, `llmspec info` prints the install command to run yourself.

---

## Project layout

```
src/
  main.rs         CLI argument parsing, entry point, TUI launch
  hardware.rs     RAM/CPU/GPU detection, backend selection, size parsing
  models.rs       Model database, quantization hierarchy, memory estimation
  fit.rs          Scoring, speed estimation, placement, MoE offloading
  providers.rs    Local runtime discovery, model listing, generation
  bench.rs        Real throughput measurement and estimate comparison
  doctor.rs       Diagnostic report over detection and its guesses
  serve.rs        Read-only HTTP API (std::net, no extra dependency)
  config.rs       Persisted settings and user-supplied models
  display.rs      CLI tables, detail view, JSON output
  tui_app.rs      TUI state: filters, sorting, navigation
  tui_ui.rs       TUI rendering (ratatui)
  tui_events.rs   TUI event loop and key handling
data/
  models.json     Model database (240 models, embedded at build time)
scripts/
  add_models.py   Catalog generator; merges entries by id
docs/
  REQUIREMENTS.md Full product requirements
```

---

## Tests

```sh
cargo test
```

107 tests covering:
- Memory math against hand calculations
- Placement and ranking behavior
- MoE offloading correctness
- GPU name matching, including vendor decorations and short-fragment collisions
- Windows adapter parsing and Apple unified-memory sizing
- Size parsing (`24G`, `128GiB`, `4T`)
- Runtime response parsing for both the Ollama and OpenAI-compatible shapes
- Benchmark summarisation (median, spread, estimate ratio)
- HTTP request parsing, routing and error statuses
- Config round-tripping and custom-model merging
- Key handling and mode transitions
- TUI rendering (including cramped terminals and empty result sets)

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Run `cargo test` and `cargo clippy`
4. Open a pull request

---

## License

MIT
