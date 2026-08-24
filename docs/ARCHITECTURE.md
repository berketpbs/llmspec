# How llmspec works

llmspec answers one question — *will this model run well on this machine?* —
by measuring the machine, estimating what each model would cost to run on it,
and scoring the result. This document describes the estimate: what it is built
from, where it is exact, and where it is a judgement call.

---

## 1. Measuring the machine

| What | How |
|---|---|
| CPU, RAM | `sysinfo` |
| NVIDIA VRAM | `nvidia-smi --query-gpu=name,memory.total` |
| AMD VRAM | `rocm-smi` on Linux; the display driver's registry key on Windows |
| Intel VRAM | `lspci` on Linux; the display driver's registry key on Windows |
| Apple Silicon | chip name from the CPU brand; VRAM sized from unified memory |
| Bandwidth | a table of ~90 GPUs, matched on the reported name |

Two details are easy to get wrong and are handled explicitly:

**Windows VRAM.** `Win32_VideoController.AdapterRAM` is a 32-bit field that
saturates at 4 GB, so a 24 GB card reports 4 GB. llmspec reads
`HardwareInformation.qwMemorySize` from the driver's class key instead, which
is 64-bit and correct.

**Apple unified memory.** There is no separate VRAM pool. macOS caps what the
GPU may wire at roughly 75% of physical memory by default
(`iogpu.wired_limit_mb`), so llmspec treats 75% of RAM as VRAM rather than the
whole pool, which would promise placements the OS refuses to make.

GPU names are matched after stripping vendor decorations (`(R)`, `(TM)`, `®`)
and on word boundaries, so `Intel(R) Arc(TM) A770 Graphics` resolves, and the
fragment `m4` does not claim a Tesla M40.

When detection fails, `--memory`, `--ram` and `--cpu-cores` override it, and
`llmspec doctor` reports every value together with whether it was measured or
guessed.

---

## 2. What a model costs to run

Three separate numbers, often confused:

```
download  = weights
resident  = weights + KV cache + runtime overhead     (must be in VRAM)
total     = the same, wherever it lives
```

### Weights

```
weights_bytes = params × bits_per_weight ÷ 8
```

`bits_per_weight` is the effective figure for llama.cpp's mixed quantization
schemes, not the nominal bit width:

| Quantization | Bits/weight | Quality retained | Notes |
|---|---|---|---|
| Q8_0 | 8.50 | 1.000 | effectively lossless |
| Q6_K | 6.56 | 0.990 | |
| Q5_K_M | 5.67 | 0.975 | |
| Q4_K_M | 4.83 | 0.950 | the usual default |
| Q3_K_M | 3.91 | 0.800 | noticeably degraded |
| Q2_K | 3.35 | 0.580 | a last resort, not a trade |

The quality column is shaped after measured perplexity loss: everything down
to Q4_K_M is close to lossless; below it the curve falls away sharply.

### KV cache

With the model's geometry known:

```
kv_bytes = 2 × layers × kv_heads × head_dim × context × 2 bytes
```

The leading 2 is the key and value tensors; the trailing 2 is fp16. Without
geometry, llmspec falls back to a per-parameter constant derived from
Llama-3.1-8B. That fallback is less exact — a model with multi-head attention
rather than grouped-query attention has a much larger cache than the constant
assumes — so the catalog carries real geometry wherever it is known (186 of
240 entries, checked against each model's `config.json` on HuggingFace), and
the schema rejects partial geometry, which would be silently ignored.

The fallback is keyed off the *active* parameter count, not the total. A KV
cache is sized by the attention layers, and a mixture-of-experts model's idle
experts contribute none: reading `params_b` sized DeepSeek-V3's cache at 90 GB
against a real 1.9 GB. For a dense model the two counts are the same, so only
MoE entries are affected.

Two architecture families are deliberately left without geometry rather than
given approximate numbers. Encoder-only embedding models (BERT, XLM-R, MPNet)
and recurrent ones (RWKV) have no autoregressive KV cache at all, so a
`kv_heads` copied from their attention head count would invent a cost that
does not exist. Multi-head latent attention (DeepSeek V2/V3) reports a
`num_key_value_heads` that its compressed cache does not actually pay for.
In both cases the parameter heuristic is the more honest estimate.

The KV cache is easy to underestimate. Qwen3-4B advertises a 262k context; the
cache alone at that length is around 36 GB against 2.4 GB of Q4 weights. Most
models are context-bound, not weight-bound.

### Overhead

```
overhead = 0.4 GB + 5% of resident weights
```

Covers the CUDA/Metal context, compute buffers and allocator slack.

---

## 3. Placement

For each model llmspec enumerates every viable combination of

- **run mode** — GPU, MoE offload, CPU+GPU, CPU
- **context** — full, half, then 64k / 32k / 16k / 8k / 4k
- **quantization** — Q8_0 down to Q2_K

scores each one, and keeps the highest-scoring combination.

This is one rule rather than three. A fixed priority order would, for
instance, take Q2_K at full context over Q4_K_M at half — a trade the score
model says is not worth making. Letting the placement optimise the same
objective the ranking does keeps the two consistent.

### Run modes

| Mode | What happens | Fit ceiling |
|---|---|---|
| **GPU** | everything resident in VRAM | Perfect |
| **MoE** | active experts in VRAM, the rest streams from RAM | Good |
| **CPU+GPU** | dense weights partly offloaded to RAM | Good |
| **CPU** | weights live in system RAM | Marginal |

The ceilings are deliberate. Offloading works, but it is never as clean as a
model that simply fits, and CPU-only inference is usable rather than good
however much RAM the machine has.

### Mixture-of-experts

A MoE model reads only its active experts per token, so only those need to be
resident:

```
resident = active_weights + kv_cache + overhead
```

Mixtral 8x7B needs roughly the VRAM of a 13B model rather than a 47B one. The
whole model still has to fit somewhere, so the total is checked against
VRAM + RAM.

### Fit levels

| Level | Meaning |
|---|---|
| **Perfect** | fits VRAM with 15% headroom |
| **Good** | fits, or is offloading cleanly |
| **Marginal** | over 90% of VRAM, or CPU-only |
| **Too Tight** | does not fit anywhere — always ranked last, score 0 |

---

## 4. Speed

Token generation is memory-bandwidth bound: every token requires reading the
active weights once. So:

```
tok/s ≈ bandwidth_GB/s ÷ weights_GB × efficiency
```

with `efficiency` defaulting to **0.55**, covering kernel overhead, attention
over the KV cache, and the gap between rated and achieved bandwidth.

For an unrecognised GPU there is no bandwidth figure, so a per-backend
constant stands in:

| Backend | K, in `K ÷ params_b × quant_multiplier` |
|---|---|
| CUDA | 220 |
| ROCm | 180 |
| Metal | 160 |
| SYCL | 100 |
| CPU (ARM) | 90 |
| CPU (x86) | 70 |

Mode factors then apply: 1.0 on GPU, 0.8 for MoE offload, 0.5 for CPU+GPU
spill, 0.3 for CPU-only, and 0.9 for multi-GPU tensor parallelism.

### The estimate is deliberately conservative

0.55 is a floor that holds across hardware rather than a fit to any one
machine. On a modern CUDA card with a small model, real throughput is often
well above it.

That is what `llmspec bench` is for. It measures actual tokens per second
against a running runtime, prints the ratio against the estimate, states what
the estimate assumed — model, quantization, context, bytes read per token —
and suggests the efficiency factor that would reconcile the two. That value is
editable in the TUI (`A`) and persisted, so the estimate can be calibrated to
the machine it is running on instead of being globally re-tuned on one
person's benchmark.

A ratio far from 1.00x has three usual causes, in order of likelihood:

1. the runtime loaded a different quantization than llmspec placed the model at
2. the GPU's bandwidth table entry is wrong or missing
3. the efficiency factor is not right for this hardware

The first two are visible in the benchmark output; the third is the knob.

---

## 5. Scoring

Four dimensions, each 0–100.

### Quality

```
effective_params = params                      (dense)
                 = √(total × active)           (MoE)

size  = 100 × ln(1 + effective_params) / ln(1 + 70)
score = size × family × quantization_quality × use_case_affinity
```

Logarithmic in size, because the step from 3B to 7B matters far more than
from 60B to 70B. For MoE the geometric mean of total and active parameters
tracks observed quality better than either number alone.

`family` is a reputation multiplier on a 0.80–1.00 scale. It comes from
`data/benchmarks.json` when a family entry matches the model id — public
HumanEval, GPQA and arena-style results, mapped onto that scale — and
otherwise from the catalog's hand-set 1–5 `quality_tier`. A measured number is
simply a finer reading of the quantity the tier estimates, so the two are
interchangeable.

Size stays the backbone, and the benchmark adjusts it rather than replacing
it. The two were briefly averaged instead, at 70% benchmark to 30% size, and
that is worth recording as a mistake: the benchmark numbers are per-family and
carry no size information at all. The `qwen3` entry alone covers fifteen
catalog models from 0.6B to 235B — a 392-fold spread sharing one score — so at
70% of the weight it ranked Qwen3-0.6B above Qwen2.5-7B, and a 270M Gemma
inherited the 27B model's evaluation. What a benchmark genuinely measures is
how one family compares with another at equal size, which is exactly what a
multiplier expresses and an average destroys.

`use_case_affinity` discounts a model asked to do something it was not tuned
for: a generalist standing in for a code model costs 30%, the reverse only
15%, and an embedding model is not interchangeable with a generative one at
all (0.30). It applies to the whole score, benchmark-derived or not — when it
applied to only part, Qwen3-Embedding-0.6B took the qwen3 family's chat number
at full credit and ranked thirteenth for general use.

This dimension still ranks families rather than individual checkpoints. It
puts a 7B coder above a 7B generalist for coding, and a 70B above a 7B; it
will not tell you which of two well-regarded 7B models writes better Python.

### Speed

```
score = 100 × min(tok/s ÷ 40, 1) ^ 0.5
```

Concave, and capped at 40 tok/s — roughly twice reading speed. Past that point
extra throughput buys nothing, and continuing to award points for it would
trade away quantization quality for speed nobody can use.

### Fit

Memory-efficiency, peaking on a plateau from 50% to 95% of the pool. Below
50% the hardware is being left idle — a larger model or a higher quantization
would have fit — so the score tapers. Above 95% it falls away sharply.

The plateau is wide on purpose. A narrower one would quietly prefer a smaller
quantization over a better one purely for using less memory.

### Context

```
score = 100                              if context ≥ target
      = 100 × √(context ÷ target)        otherwise
```

Square-root, because the marginal value of context falls off: 16k against a
32k target is worth more than half the points, not a quarter.

### Composite

```
composite = (Q·wq + S·ws + F·wf + C·wc) × usability
```

| Use case | Quality | Speed | Fit | Context |
|---|---|---|---|---|
| General | 0.50 | 0.25 | 0.10 | 0.15 |
| Coding | 0.45 | 0.20 | 0.05 | 0.30 |
| Reasoning | 0.65 | 0.15 | 0.05 | 0.15 |
| Chat | 0.40 | 0.35 | 0.10 | 0.15 |
| Multimodal | 0.50 | 0.25 | 0.10 | 0.15 |
| Embedding | 0.45 | 0.35 | 0.10 | 0.10 |

`usability` scales the whole thing towards zero below 3 tok/s. A placement
that technically fits but produces a token every few seconds is not an answer,
however good its quality and context scores are. A model that does not fit
scores 0 outright.

---

## 6. Talking to runtimes

llmspec uses whatever inference server is already running. Ollama has its own
API; the rest speak the OpenAI-compatible `/v1` surface.

Discovery does a 300 ms TCP connect before any HTTP request, so probing five
absent runtimes costs microseconds instead of five HTTP timeouts. Defaults
name `127.0.0.1` rather than `localhost`, because on Windows `localhost`
resolves to `::1` first and probing a port nothing is bound to then waits out
the full connect timeout.

### Matching model names

The same weights are called `Qwen/Qwen2.5-7B-Instruct` upstream,
`qwen2.5:7b` by Ollama, `qwen2.5-7b-instruct` by LM Studio and
`Qwen2.5-7B-Instruct-Q4_K_M.gguf` by llama.cpp. llmspec normalises all of
those to one key by dropping the publisher prefix, the file extension and any
quantization marker.

Matching is deliberately conservative — `phi-4` and `phi-4-mini` stay
distinct. A wrong "already installed" tick sends someone looking for a file
that is not there; a missed one costs a redundant `ollama pull` that returns
immediately.

---

## 7. Module layout

| Module | Responsibility |
|---|---|
| `hardware` | detection, backend selection, the GPU bandwidth table |
| `models` | catalog, quantization hierarchy, memory arithmetic |
| `fit` | placement, speed estimation, scoring, hardware planning |
| `providers` | runtime discovery, model listing, generation |
| `bench` | measurement, comparison against the estimate, calibration |
| `doctor` | diagnostic report over detection and its guesses |
| `serve` | read-only HTTP API (`std::net`, no extra dependency) |
| `config` | persisted settings and user-supplied models |
| `display` | CLI tables, detail views, JSON |
| `tui_app` | interactive state: filters, sorting, background work |
| `tui_ui` | rendering |
| `tui_events` | the event loop and key handling |
| `tui_theme` | colour palettes |
| `tui_form` | the reusable editable-field popup |

Dependencies point one way: `hardware` and `models` know nothing about
anything above them; `fit` depends only on those two; the interface layers
depend on everything below and on nothing beside them.

---

## 8. Known limitations

**Quality scores are heuristic.** Size, family reputation and quantization
loss — not task benchmarks. Good at ordering across sizes and specialities,
weak at separating peers.

**Speed is an estimate until measured.** The bandwidth model ignores prompt
processing entirely, so time-to-first-token on a long prompt is not predicted
at all. `bench` reports it where the runtime does.

**The GPU table is finite.** An unrecognised card falls back to a per-backend
constant, which is much coarser. `doctor` says so when it happens.

**Multi-GPU assumes clean tensor parallelism.** VRAM is summed and a 0.9
factor applied. Real behaviour depends on the interconnect and the runtime,
and a model that does not shard cleanly will do worse than predicted.

**The catalog is a snapshot.** It ships embedded in the binary. Models newer
than the build are added through `models.json` in the config directory.
