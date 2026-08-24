# Roadmap

What llmspec does today, and what it deliberately does not.

The behaviour is documented in [the README](../README.md); the reasoning
behind the numbers is in [ARCHITECTURE.md](ARCHITECTURE.md). This file is
only about scope.

---

## Done

**Analysis**

- Four-dimensional scoring with per-use-case weights
- Placement search over run mode × context × quantization
- Mixture-of-experts expert offloading
- Exact KV-cache sizing from model geometry, with a heuristic fallback
- Hardware planning — what a given model would need from any machine
- Throughput targets — the bandwidth and the cards a wanted tok/s implies

**Hardware**

- NVIDIA, AMD, Intel and Apple Silicon detection
- Windows AMD/Intel VRAM from the driver's 64-bit registry value
- Apple unified memory sized against the OS wiring cap
- Multi-GPU VRAM aggregation
- Overrides for every detected value, and simulation from the interface

**Runtimes**

- Ollama, llama.cpp, LM Studio, vLLM, Docker Model Runner, MLX
- Discovery with a TCP pre-check, so absent runtimes cost microseconds
- Cross-runtime model-name matching
- Background downloads through Ollama

**Interfaces**

- Interactive TUI with ten themes, comparison, planning and simulation
- Classic CLI for every command, with JSON everywhere
- Read-only HTTP API
- `doctor` diagnostics, exiting non-zero on warnings
- `bench` measurement, with estimate comparison and efficiency calibration

**Configuration**

- Persisted theme, use case and speed tunables
- User-supplied models merged into the catalog

---

## Not planned

Each of these is a decision, not an oversight.

**A community benchmark leaderboard.** Uploading measurements would mean
owning a data pipeline, a submission review process, and a privacy story for
hardware fingerprints. `bench` measures locally and calibrates locally; that
covers the same need for the person running it without any of that.

**Benchmark-derived quality scores.** Quality is a heuristic over size, family
reputation and quantization loss. Task-benchmark data would sharpen it, but it
goes stale quickly and would have to be re-collected for every new model. The
current model is honest about being an ordering, not a verdict.

**Alternative front-ends.** A web UI, a desktop app and a Python wrapper would
each need their own release, packaging and support. One binary that also
speaks JSON and HTTP covers scripting and integration.

**Training or fine-tuning estimates.** Inference memory and training memory
are different problems — optimiser state, gradients, activation checkpointing.
Answering the second badly would undermine trust in the first.

---

## Open

Ordered by how much they would improve the answer llmspec gives.

**Catalog coverage.** 240 models is a good spread but not exhaustive, and it
is a snapshot at build time. The generator (`scripts/add_models.py`) makes
additions cheap; the gap is breadth, not tooling.

**KV-cache quantization.** Runtimes increasingly store the cache at Q8 or Q4,
which roughly halves or quarters the context cost. llmspec always assumes
fp16, so it understates how much context actually fits. Modelling this would
change placement on long-context models more than any other single change.

**Prompt-processing speed.** The bandwidth model covers generation only, so
time-to-first-token on a long prompt is not predicted. `bench` reports it
where the runtime does, but nothing estimates it.

**Wider GPU coverage.** Around 85 cards have a known bandwidth. Anything else
falls back to a per-backend constant, which is much coarser. `doctor` reports
when this happens.

**Packaged installs.** Source builds only. Homebrew, Scoop and a container
image would each need release automation and signing.

---

## Adding a model

1. Add the entry to `scripts/add_models.py`
2. Run `python scripts/add_models.py` from the repository root
3. Run `cargo test` — the catalog tests check for duplicate ids, duplicate
   runtime tags, partial geometry and implausible values
4. Rebuild; the catalog is embedded at compile time

Geometry (`layers`, `hidden_size`, `kv_heads`, `head_dim`) is optional but
worth having: with it the KV cache is exact, without it llmspec falls back to
a heuristic derived from one model's proportions. Supply all four fields or
none — partial geometry is rejected, because it would be silently ignored.

For anything private or newer than the build, use `models.json` in the config
directory instead; it needs no rebuild.
