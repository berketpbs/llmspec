# llmspec vs llmfit

llmspec is an independent Rust implementation of the same idea as
[llmfit](https://github.com/AlexsJones/llmfit): detect the machine, score every
model in a catalog against it, and say which ones will actually run.

This document records where the two line up and where they do not, so the gap
is a decision rather than a surprise. Figures are as of 2026-08-23.

---

## Where the two agree

The core is the same model, arrived at independently and now deliberately
aligned:

| Aspect | Both |
|---|---|
| Scoring | Four dimensions — quality, speed, fit, context — combined with per-use-case weights |
| Use-case weights | Reasoning leans on quality (0.55), chat on speed (0.35) |
| Speed model | `bandwidth ÷ bytes-per-token × 0.55`, with per-backend constants as fallback |
| Backend constants | CUDA 220, ROCm 180, Metal 160, SYCL 100, CPU ARM 90, CPU x86 70 |
| Fit levels | Perfect / Good / Marginal / Too Tight, with GPU-only Perfect |
| Quantization | Dynamic walk from Q8_0 down, retrying at shorter contexts |
| MoE | Active-expert residency, offloading the rest to system RAM |
| Interface | TUI by default, `--cli` for the classic table, `--json` everywhere |

---

## Feature comparison

| Feature | llmspec | llmfit |
|---|---|---|
| Models in catalog | 240 | ~497 |
| Model providers | 56 | ~133 |
| Hardware detection | NVIDIA, AMD, Intel, Apple Silicon | same |
| Windows AMD/Intel VRAM | driver registry (64-bit) | yes |
| Multi-GPU aggregation | yes | yes |
| Runtimes | Ollama, llama.cpp, LM Studio, vLLM, Docker Model Runner, MLX | same |
| `fit` / `recommend` / `info` / `plan` / `search` / `list` | yes | yes |
| `doctor` | yes | yes |
| `bench` | yes | yes |
| `serve` (HTTP API) | yes | yes |
| `--force-runtime` | yes | yes |
| Config persistence | theme, use case, speed factors | yes |
| User-supplied models | `models.json` in the config dir | yes |
| Community benchmark leaderboard | **no** | yes (`bench --share`, PR submission) |
| Benchmark-calibrated speed baselines | **no** — pure bandwidth model | yes (`baselines.json`, community data) |
| Task benchmarks / quality scores | **no** — quality is a size × tier heuristic | yes (`use_case_benchmarks.json`) |
| MCP server | **no** | yes |
| Web UI | **no** | yes (`llmfit-web`) |
| Desktop app | **no** | yes (`llmfit-desktop`) |
| Python package | **no** | yes (`pip install llmfit`) |
| Packaged installs (brew, scoop, Docker, Nix) | **no** — `cargo install` only | yes |
| Localised READMEs | **no** | ja, zh |

---

## Deliberate gaps

These are not oversights; each is a scope call.

**Community benchmark leaderboard.** llmfit collects measured throughput from
users and submits it as a GitHub PR, then uses the aggregate to calibrate its
speed estimates. llmspec measures throughput locally (`llmspec bench`) and
shows the ratio against its own prediction, but nothing is uploaded and no
account or network write is involved. Adding submission means owning a data
pipeline and a review burden.

**Benchmark-derived quality scores.** llmfit scores quality partly from task
benchmark data. llmspec derives it from effective parameter count, a hand-set
family tier and the quantization penalty. That is cheaper to keep current but
less discriminating between models of the same size.

**Distribution surface.** llmfit ships a Python wrapper, a web UI, a desktop
app, container images and several package managers. llmspec is one Rust binary,
installed from source.

---

## Where llmspec is ahead

**Probe cost.** Discovery does a 300 ms TCP connect check before any HTTP
request, so probing five absent runtimes costs microseconds rather than five
HTTP timeouts. Defaults name `127.0.0.1` rather than `localhost`, which on
Windows avoids an IPv6-first resolution that waits out the full connect
timeout.

**GPU name matching.** Table lookups strip vendor decorations and match on
word boundaries, so `Intel(R) Arc(TM) A770 Graphics` resolves, `m4` does not
claim a Tesla M40, and `l4` does not claim an L40S.

**Dependencies.** The HTTP API is built on `std::net`, so `serve` adds no
dependency. Total: clap, colored, crossterm, ratatui, serde, serde_json,
sysinfo, ureq.

---

## Honest summary

llmspec matches llmfit on the analysis engine and on every command that
analyses local hardware. It is roughly half the catalog size, and it does not
have the community data layer, the alternative front-ends or the packaging
breadth. For "what runs on this machine, and how fast", the two answer the
same question the same way.
