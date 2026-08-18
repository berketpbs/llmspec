# llmspec — Product Requirements

A terminal tool that works out which LLMs will genuinely run well on the user's
hardware (RAM, CPU, GPU/VRAM). Written in Rust, targeting Windows and Linux,
offering both an interactive TUI and a classic CLI.

- **Project / crate / binary name:** `llmspec`
- **Environment variable prefix:** `LLMSPEC_`
- **Config directory:** Linux `~/.config/llmspec/`, Windows `%APPDATA%\llmspec\`

---

## 1. Product definition

Scores and ranks hundreds of models across quality, speed, fit and context.
The goal is to answer "can I actually run this model, and how well?" with a
number and a justification.

---

## 2. Distribution (future)

- Windows: `scoop install llmspec`
- macOS/Linux: Homebrew (`brew install llmspec`), MacPorts, `curl | sh` quick-install script
- `cargo install llmspec` / crates.io package
- `uv tool install llmspec` / `uvx` (Python wrapper — optional, low priority)
- Docker/Podman image (emits JSON, queryable with `jq`)
- From source: `cargo build --release`

---

## 3. Hardware detection (`hardware` module)

- **RAM/CPU**: total and available RAM, core count, via the `sysinfo` crate.
- **NVIDIA GPU**: via `nvidia-smi` (present on both Windows and Linux).
  Multi-GPU supported — VRAM is summed. If the command fails, estimate VRAM
  from the GPU model name (fallback).
- **AMD GPU**: `rocm-smi` (Linux). Windows needs a different approach
  (e.g. a WMI/DXGI query) — still to be researched.
- **Apple Silicon**: `system_profiler` — out of scope (we target Windows/Linux),
  but a placeholder may stay in the code structure.
- **Intel Arc**: sysfs (`mem_info_vram_total`, discrete) / `lspci` (integrated), Linux.
- **Backend selection**: a CUDA / ROCm / SYCL / CPU (ARM) / CPU (x86) label is
  assigned automatically from the result and used in the speed estimate.
- **Windows specifics**: NVIDIA detection works through `nvidia-smi` when
  installed; RAM/CPU detection is native and reliable.
- **Override flags**: `--memory=32G`, `--ram=128G`, `--cpu-cores=16` — for when
  auto-detection fails, or to simulate different target hardware. Accepted
  units: G/GB/GiB, M/MB/MiB, T/TB/TiB (case-insensitive). If no GPU was
  detected, `--memory` synthesises one.

---

## 4. Model database

- Source: a Python scraper (`scripts/scrape_hf_models.py`, stdlib-only, no pip
  dependencies) pulls from the HuggingFace REST API, writes `data/hf_models.json`,
  and the file is embedded into the binary **at build time** (`include_str!`).
- Hundreds of models across dozens of providers: Meta Llama, Mistral, Qwen,
  Google Gemma, Microsoft Phi, DeepSeek, IBM Granite, Allen Institute OLMo,
  xAI Grok, Cohere, BigCode, 01.ai, Upstage, TII Falcon, Zhipu GLM,
  Moonshot Kimi, Baidu ERNIE and others.
- Categories: general purpose, coding (CodeLlama, StarCoder2, Qwen2.5/3-Coder),
  reasoning (DeepSeek-R1), multimodal/vision (Llama 3.2 Vision, Qwen2.5-VL),
  chat, enterprise, embedding (nomic-embed, bge).
- **MoE detection**: derived from the model config (`num_local_experts`,
  `num_experts_per_tok`) or from known architecture mappings. Example: Mixtral
  8x7B has 46.7B total parameters but only ~12.9B active per token, dropping the
  VRAM requirement from 23.9 GB to ~6.6 GB with expert offloading.
- GGUF source enrichment: downloadable GGUF links from providers such as unsloth
  and bartowski, with a 7-day TTL cache (`data/gguf_sources_cache.json`).
  Skippable with `--no-gguf-sources`.
- Updating: `make update-models` or `./scripts/update_models.sh` — backs up the
  existing data, validates the JSON, rebuilds the binary.

---

## 5. Quantization and memory estimation

- Instead of assuming a fixed quantization, choose **dynamically**: walk the
  hierarchy from Q8_0 (highest quality) down to Q2_K (most compressed) and take
  the highest-quality level that fits available memory.
- If nothing fits at full context, retry at **half context**.
- VRAM is the primary constraint for GPU inference; system RAM is the fallback
  for CPU-only execution.

---

## 6. Multi-dimensional scoring

Every model is scored 0–100 on four dimensions:

| Dimension | What it measures |
|---|---|
| **Quality** | Parameter count, family reputation, quantization loss, task affinity |
| **Speed** | Estimated tok/s from backend, parameter count and quantization |
| **Fit** | Memory-use efficiency (sweet spot: 50–80% of available memory) |
| **Context** | Context window capacity against the use-case target |

- The dimensions combine into a **weighted composite score**. Weights vary by
  use case (General, Coding, Reasoning, Chat, Multimodal, Embedding). For
  example: Chat raises the Speed weight to 0.35; Reasoning raises Quality to 0.55.
- Models sort by composite score; unrunnable models ("Too Tight") always last.

---

## 7. Speed estimation

- LLM inference is **memory-bandwidth-bound**: each token requires reading the
  whole of the model's active weights from VRAM once.
- When the GPU model is recognised, its real memory bandwidth is used:
  `(bandwidth_GB_s / model_size_GB) × efficiency_factor`
- The efficiency factor (default `0.55`) and the per-mode speed multipliers must
  be **user-adjustable** (Advanced Config).
- A bandwidth table covering ~80 GPUs (NVIDIA consumer + datacenter, AMD, Apple).
- For unrecognised GPUs, per-backend constants
  (fallback: `K / params_b × quant_speed_multiplier`):

| Backend | Constant |
|---|---|
| CUDA | 220 |
| Metal | 160 |
| ROCm | 180 |
| SYCL | 100 |
| CPU (ARM) | 90 |
| CPU (x86) | 70 |
| NPU (Ascend) | 390 |

---

## 8. Fit analysis

**Run modes:**

- **GPU** — the model fits in VRAM; fast inference
- **MoE** — expert offloading: active experts in VRAM, inactive ones in RAM
- **CPU+GPU** — VRAM is insufficient, partial GPU offload with spill to RAM
- **CPU** — no GPU; the model loads entirely into system RAM

**Fit levels:**

- **Perfect** — recommended memory met on the GPU; requires a GPU
- **Good** — fits comfortably; the ceiling for MoE offload and CPU+GPU
- **Marginal** — a tight squeeze, or CPU-only (CPU-only always caps here)
- **Too Tight** — no room anywhere, in VRAM or system RAM

---

## 9. TUI — `llmspec` with no arguments

System specs (CPU, RAM, GPU name, VRAM, backend) at the top. Models listed in a
scrollable table sorted by composite score. Each row: score, estimated tok/s,
best quantization for the hardware, run mode, memory use, use-case category.

### Normal mode key bindings

| Key | Action |
|---|---|
| `Up`/`Down` or `j`/`k` | Move between models |
| `/` | Search mode (partial match on name, provider, parameters, use case) |
| `Esc`/`Enter` | Leave search mode |
| `Ctrl-U` | Clear the search |
| `f` | Cycle fit filter: All, Runnable, Perfect, Good, Marginal |
| `a` | Cycle availability filter: All, GGUF Avail, Installed |
| `s` | Cycle sort column: Score, Params, Mem%, Ctx, Date, Use Case |
| `v` | Enter Visual mode (multi-model selection) |
| `V` | Enter Select mode (column-based filtering) |
| `t` | Cycle colour theme (saved automatically) |
| `p` | Plan mode for the selected model (hardware planning) |
| `P` | Provider filter popup |
| `U` | Use-case filter popup |
| `C` | Capability filter popup |
| `L` | License filter popup |
| `R` | Runtime/backend filter popup (llama.cpp, MLX, vLLM) |
| `S` | Hardware simulation popup (RAM/VRAM/CPU override) |
| `A` | Advanced configuration popup (efficiency, mode factors) |
| `b` | Community leaderboard view |
| `I` | Inference bench view (quality scoring against local models) |
| `h` | Help popup (all key bindings) |
| `m` | Mark the selected model for comparison |
| `c` | Open the comparison view (marked vs selected) |
| `x` | Clear the comparison mark |
| `i` | Toggle installed-first sorting |
| `d` | Download the selected model (provider picker if several apply) |
| `D` | Open the Download Manager |
| `r` | Refresh installed models from runtime providers |
| `Enter` | Toggle the detail view for the selected model |
| `PgUp`/`PgDn` | Scroll by 10 |
| `g`/`G` | Jump to first/last |
| `q` | Quit |

### Vim-style modes

- **Normal mode**: default; all keys above are active.
- **Visual mode (`v`)**: `v` anchors, `j`/`k` extends a contiguous row range.
  `c` opens the multi-comparison view, `m` marks for two-model comparison,
  `Esc`/`v` exits. In the multi-comparison table rows are attributes (Score,
  tok/s, Fit, Mem%, Params, Mode, Context, Quant) and columns are models; best
  values are highlighted; `h`/`l` scrolls horizontally.
- **Select mode (`V`)**: `h`/`l` moves between column headers, `Enter`/`Space`
  triggers that column's action:

  | Column | Filter action |
  |---|---|
  | Inst | Cycle the availability filter |
  | Model | Enter search mode |
  | Provider | Provider popup |
  | Params | Parameter-size range popup (<3B, 3-7B, 7-14B, 14-30B, 30-70B, 70B+) |
  | Score/tok/s/Mem%/Ctx/Date | Sort by that column |
  | Quant | Quantization popup |
  | Mode | Run-mode popup (GPU, MoE, CPU+GPU, CPU) |
  | Fit | Cycle the fit filter |
  | Use Case | Use-case popup |

  Row navigation still works in Select mode (`j`/`k`, arrows, `Ctrl-U`,
  `Ctrl-D`, `PageUp`/`PageDown`, `Home`/`End`).

### Plan mode (`p`)

The inverse of normal fit analysis: not "does this model fit my hardware?" but
"how much hardware would this model configuration need?"

| Key | Action |
|---|---|
| `Tab`/`j`/`k` | Move between editable fields (Context, Quant, Target TPS) |
| `Left`/`Right` | Move the cursor within a field |
| Typing | Edit the active field |
| `Backspace`/`Delete` | Delete a character |
| `Ctrl-U` | Clear the field |
| `Esc`/`q` | Leave Plan mode |

Shows: minimum/recommended VRAM, RAM and CPU cores; viable execution paths
(GPU, CPU offload, CPU-only); and the upgrade gaps needed to reach a better fit.

### Hardware simulation (`S`)

Overrides RAM/VRAM/CPU core count to show which models would fit on different
target hardware. All scores, fit levels and speed estimates recompute instantly.

| Key | Action |
|---|---|
| `Tab`/`j`/`k` | Move between the RAM/VRAM/CPU fields |
| Typing digits | Edit the selected field |
| `Enter` | Apply the simulation |
| `Ctrl-R` | Reset to the real detected hardware |
| `Esc` | Cancel and close |

While a simulation is active, a `SIM` badge appears in the system bar and the
status bar.

### Advanced configuration (`A`)

Tunes the speed and score calculation parameters (tok/s estimates can run
optimistic on some models). Changes apply instantly.

| Field | Description | Default |
|---|---|---|
| Efficiency | Global efficiency factor for bandwidth-based TPS | 0.55 |
| GPU factor | Speed multiplier for pure GPU inference | 1.0 |
| CPU Offload | Speed multiplier when weights spill into system RAM | 0.5 |
| MoE Offload | Speed multiplier for MoE expert swapping | 0.8 |
| Tensor Par | Speed multiplier for tensor-parallel inference | 0.9 |
| CPU Only | Speed multiplier for CPU-only execution | 0.3 |
| Context cap | Maximum context length used for memory estimation | auto |

Keys: `Tab`/`j`/`k` (move between fields), digits/`.` to type, `Left`/`Right`,
`Backspace`/`Delete`, `Ctrl-U` (clear), `Enter` (apply), `Esc`/`q` (close
without applying).

### Download Manager (`D`)

Full-screen view with three sections:

- **Active Download** — progress bar, model name, status message
- **Config** — GGUF model directory (editable, persisted)
- **History** — past downloads, newest first: model name, provider, status,
  date. Failed downloads can be removed from history; successful ones can be
  deleted from the provider.

| Key | Action |
|---|---|
| `Tab`/`Shift-Tab` | Cycle focus: Active → Config → History |
| `j`/`k`/arrows | Navigate the history list |
| `x` | Delete the selected model (asks for confirmation) |
| `y`/`n` | Confirm/cancel the deletion |
| `e` | Edit the download directory |
| `Enter` | Confirm the directory edit |
| `Esc`/`D`/`q` | Close, back to the model table |

For failed downloads (e.g. a 404) `x` removes the history entry. For successful
ones the model is deleted from the provider (supported for Ollama and llama.cpp).

### Community leaderboard (`b`)

**Real-world performance data** instead of theoretical estimates — measured
tok/s, TTFT (time to first token) and peak VRAM use from other users on the same
hardware. Requires an external community benchmark database (we would have to
stand up our own backend, or skip this in the first release — a large dependency).

Columns: Model (HF ID), Engine (llama.cpp/vLLM/Ollama/MLX), Quant, tok/s,
Total t/s, TTFT, VRAM, Ctx, User (verified users marked with `*`).

| Key | Action |
|---|---|
| `j`/`k`/arrows | Navigate results |
| `H` | Open the hardware picker (browse any GPU) |
| `r` | Refresh from the API |
| `b`/`q`/`Esc` | Close |

`H` allows picking one of 27 popular GPUs/chips (from RTX 5090 down to
CPU-only, Apple M1–M4, AMD RX/MI, NVIDIA datacenter) and loading that
hardware's benchmarks. "My Hardware (auto-detect)" returns to your own system.

**API key setup**: public benchmarks need no auth. Full access is granted with
an API key supplied through an environment variable (`LLMSPEC_API_KEY`) or a
CLI flag.

### Inference bench (`I`)

Runs a **live inference benchmark** against locally running providers (Ollama,
vLLM, MLX) — sending real requests and measuring TTFT, TPS and total latency.
Unlike the community leaderboard, this measures your real hardware with your
real models.

| Key | Action |
|---|---|
| `I` | Open the bench (auto-detects the provider and runs) |
| `I` (again) | Re-run from within the bench view |
| `j`/`k`/arrows | Navigate model results |
| `Enter` | Detail view for the selected model |
| `r` | Switch to the routing matrix view |
| `q`/`Esc` | Close the bench view |

Results are cached in `~/.config/llmspec/bench-cache.json` and load instantly on
subsequent runs.

CLI equivalents:

```
llmspec bench                                   # auto-detect + benchmark
llmspec bench --all                             # every discovered model
llmspec bench --provider ollama llama3.2        # a specific model
llmspec bench --provider ollama --url http://my-server:11434 llama3.2
llmspec bench --json                            # JSON, for scripting
llmspec bench --quality                         # role-based quality scoring
llmspec bench --quality --routing               # routing matrix
```

Environment variables: `OLLAMA_HOST` (default `http://localhost:11434`),
`VLLM_PORT` (default `8000`).

### Themes (`t`)

Cycles through 10 built-in colour themes. The choice is saved automatically to
`~/.config/llmspec/theme` and restored on the next launch. Examples: Default,
Dracula, Solarized, Nord, Monokai, Gruvbox, Catppuccin
(Latte/Frappé/Macchiato/Mocha).

### Web dashboard

When run in non-JSON mode, a web dashboard starts automatically in the
background on `0.0.0.0:8787`, reachable from any browser on the same network at
`http://<machine-ip>:8787`.

Environment variables: `LLMSPEC_DASHBOARD_HOST` (default `0.0.0.0`),
`LLMSPEC_DASHBOARD_PORT` (default `8787`). Disable with `--no-dashboard`.

---

## 10. CLI mode

`--cli` or any subcommand produces classic table output:

```
llmspec --cli                                   # all models, sorted by fit
llmspec fit --perfect -n 5                      # perfect fits only, top 5
llmspec system                                  # detected system specs
llmspec list                                    # every model in the database
llmspec search "llama 8b"                       # search by name/provider/size
llmspec info "Mistral-7B"                       # detailed view of one model
llmspec recommend --json --limit 5              # top 5 recommendations (JSON)
llmspec recommend --json --use-case coding --limit 3
llmspec recommend --force-runtime llamacpp      # bypass automatic MLX selection
llmspec plan "Qwen/Qwen3-4B" --context 8192
llmspec plan "..." --context 8192 --quant q4_k_m
llmspec plan "..." --context 8192 --target-tps 25 --json
llmspec serve --host 0.0.0.0 --port 8787        # REST API mode
```

### JSON output

`--json` can be added to any subcommand for machine-readable output. For
`recommend`, JSON is the default format. The `plan` JSON contains: the request
(`context`, `quantization`, `target_tps`), estimated minimum/recommended
hardware, per-path viability (`gpu`, `cpu_offload`, `cpu_only`) and the upgrade
gaps.

### Context-length capping

`--max-context` limits the context length used in memory estimation, without
changing the model's advertised maximum:

```
llmspec --max-context 4096 --cli
llmspec --max-context 8192 fit --perfect -n 5
```

If unset, the `OLLAMA_CONTEXT_LENGTH` environment variable is used when present.

---

## 11. REST API (`llmspec serve`)

Serves the same fit/scoring data as the TUI and CLI over HTTP, for cluster
schedulers and aggregators:

```
GET  /health                                    # liveness
GET  /api/v1/system                             # node hardware information
GET  /api/v1/models?min_fit=marginal&runtime=llamacpp&sort=score&limit=20
GET  /api/v1/models/top?limit=5&min_fit=good&use_case=coding
GET  /api/v1/models/{search}?runtime=any        # name/provider text search
```

Supported query parameters: `limit`/`n`, `perfect` (true/false), `min_fit`
(perfect|good|marginal|too_tight), `runtime` (any|mlx|llamacpp), `use_case`
(general|coding|reasoning|chat|multimodal|embedding), `provider` (substring
filter), `search` (free text), `sort` (score|tps|params|mem|ctx|date|use_case),
`include_too_tight` (default false on `/top`, true on `/models`), `max_context`,
`force_runtime` (mlx|llamacpp|vllm).

---

## 12. Runtime provider integrations

When several compatible providers exist, `d` in the TUI opens a provider picker.

### Ollama

- Requirement: Ollama installed and running (`ollama serve` or the desktop app)
- Default endpoint: `http://localhost:11434`, auto-detected
- Remote Ollama: via the `OLLAMA_HOST` environment variable
  (e.g. `http://192.168.1.100:11434`)
- Behaviour: at startup `GET /api/tags` lists installed models (marked with a
  green ✓ in the TUI); `d` triggers `POST /api/pull` with an animated progress
  indicator in the row
- Model name mapping: an **exact** mapping table between HF names
  (`Qwen/Qwen2.5-Coder-14B-Instruct`) and Ollama names (`qwen2.5-coder:14b`)
  must be maintained — fuzzy matching can point at the wrong model

### llama.cpp

- Requirement: `llama-cli`/`llama-server` on PATH; network access to HuggingFace
- Behaviour: maps HF models to known GGUF repos (with a heuristic fallback),
  downloads GGUF files into a local cache, marks models "installed" when
  matching files exist
- Environment variables: `LLAMA_CPP_PATH` (binary directory, checked before
  PATH), `LLAMA_SERVER_PORT` (default `8080`)

### Docker Model Runner

- Requirement: Docker Desktop with Model Runner enabled; default endpoint
  `http://localhost:12434`
- Behaviour: `GET /engines` for the model list; Ollama-style tag mapping
  (`ai/<tag>` format); `d` runs `docker model pull`
- Remote: `DOCKER_MODEL_RUNNER_HOST` environment variable

### LM Studio

- Requirement: LM Studio running with its local server enabled; default
  endpoint `http://127.0.0.1:1234`
- Behaviour: `GET /v1/models` for the list; `d` triggers
  `POST /api/v1/models/download`; progress polled via
  `GET /api/v1/models/download-status`; HF names are accepted directly (no
  mapping needed)
- Remote: `LMSTUDIO_HOST` environment variable

### MLX (Apple Silicon — optional/low priority for us)

MLX downloads map to `mlx-community/*` HF repos rather than the original
publisher.

---

## 13. Platform support

| Platform | Status | Method |
|---|---|---|
| **Windows** | Full support target | RAM/CPU native; NVIDIA GPU via `nvidia-smi` when installed |
| **Linux** | Full support target | GPU detection: `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), sysfs/`lspci` (Intel Arc), `npu-smi` (Ascend) |

GPU detection table:

| Vendor | Detection method | VRAM reporting |
|---|---|---|
| NVIDIA | `nvidia-smi` | Exact dedicated VRAM |
| AMD | `rocm-smi` (Linux) / separate solution needed on Windows | Detected (VRAM may be unknown) |
| Intel Arc (discrete) | sysfs | Exact dedicated VRAM |
| Intel Arc (integrated) | `lspci` | Shared system memory |

If auto-detection fails or reports the wrong value, override with
`--memory`/`--ram`/`--cpu-cores`.

---

## 14. Project layout

```
src/
  main.rs         -- CLI argument parsing, entry point, TUI launch
  hardware.rs     -- System RAM/CPU/GPU detection (multi-GPU, backend selection)
  models.rs       -- Model database, quantization hierarchy, dynamic quant selection
  fit.rs          -- Multi-dimensional scoring (Q/S/F/C), speed estimation, MoE offloading
  providers.rs    -- Runtime provider integration (Ollama, llama.cpp, Docker Model
                     Runner, LM Studio), install detection, pull/download
  display.rs      -- Classic CLI table rendering + JSON output
  tui_app.rs      -- TUI application state, filters, navigation
  tui_ui.rs       -- TUI rendering (ratatui)
  tui_events.rs   -- TUI keyboard event handling (crossterm)
data/
  models.json     -- Model database (embedded at build time)
scripts/
  scrape_hf_models.py       -- HuggingFace API scraper
  update_models.sh          -- Automated database update
```

---

## 15. Dependencies (Rust crates)

| Crate | Purpose |
|---|---|
| `clap` | CLI argument parsing (with derive macros) |
| `sysinfo` | Cross-platform RAM/CPU detection |
| `serde`/`serde_json` | Model database JSON (de)serialization |
| `colored` | Coloured CLI output |
| `ureq` | HTTP client for runtime/provider API integration |
| `ratatui` | Terminal UI framework |
| `crossterm` | Terminal input/output backend for ratatui |

---

## 16. Adding a model

1. Add the model's HF repo ID to the scraper's target list
2. If the model is gated (requires HF auth), add a fallback entry with the
   parameter count and context length
3. Run the automated update script
4. Validate the updated model list
5. Update the documentation

---

## 17. Deferrable for v1

- Apple Silicon / MLX support (we target Windows + Linux)
- Community leaderboard (needs an external benchmark service/backend — large
  additional work)
- Agent-framework skill integration (optional, later stage)
- Docker Model Runner and LM Studio integrations — after Ollama and llama.cpp

---

## 18. v1 development order

1. Hardware detection (Windows + Linux; NVIDIA and RAM/CPU first, AMD later)
2. Small hand-written model database (10-20 models) → then the HF scraper
3. Quantization + memory estimation + fit/scoring logic
4. Classic CLI table output (`--cli`, `list`, `search`, `info`, `recommend`, `system`)
5. TUI skeleton: system info on top, model table, basic navigation
   (`j`/`k`, `/` search, `f` filter, `q` quit)
6. TUI expansion: Plan mode, hardware simulation, themes
7. Ollama integration (installed-model detection + downloads)
8. REST API (`serve` command)
9. Download Manager, Advanced Config, llama.cpp integration
10. (Optional) Inference Bench, community leaderboard, other providers

---

## Implementation notes

Decisions taken while building steps 1–5 that depart from, or sharpen, the
above:

- **Placement by score, not by priority order.** Section 5 describes trying
  full context then half context. In practice the KV cache dominates on
  long-context models, so the implementation evaluates every viable
  (run mode, context, quantization) combination and takes the highest composite
  score. The context ladder is full, half, then 64k/32k/16k/8k/4k.
- **Usability floor.** A placement that fits but produces a token every few
  seconds is scaled towards zero, so it cannot out-rank a model that actually
  runs interactively.
- **`tabled` was dropped** in favour of a small hand-written table renderer:
  the CLI needs coloured cells, and ANSI-aware width handling was not worth the
  dependency.
- **Seed database** lives at `data/models.json` with ~27 hand-written entries
  carrying exact layer/head geometry for KV-cache sizing. The scraper must
  preserve that schema.
