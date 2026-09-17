#!/usr/bin/env python3
"""Write data/models.json from the records in this file.

Run from the repository root:

    python scripts/add_models.py            # rewrite the catalog
    python scripts/add_models.py --check    # fail if the catalog has drifted

This file is the catalog's only source. Every record in data/models.json comes
from an entry here, so a correction is made here and never in the JSON: a hand
edit there is overwritten by the next run, and `--check`, which CI runs, fails
until it is moved into this file. Deleting an entry here removes the record.

Entries are keyed by `id`. A record keeps its position in the catalog across
runs, because runtime-tag lookups take the first match and reordering tied
records would quietly change which one that is.

Geometry (`layers`, `hidden_size`, `kv_heads`, `head_dim`) is optional and is
only filled in where the model's config is known. Without it llmspec sizes the
KV cache from a parameter-count heuristic, which is less exact but never wrong
in the way an invented number would be.
"""

import argparse
import json
import sys
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "models.json"


def m(
    id,
    name,
    provider,
    params_b,
    context_length,
    use_case,
    license="",
    released="",
    capabilities=(),
    quality_tier=3,
    gguf=True,
    ollama=None,
    active_params_b=None,
    layers=None,
    hidden_size=None,
    kv_heads=None,
    head_dim=None,
):
    entry = {
        "id": id,
        "name": name,
        "provider": provider,
        "params_b": params_b,
        "context_length": context_length,
        "use_case": use_case,
        "license": license,
        "released": released,
        "capabilities": list(capabilities),
        "quality_tier": quality_tier,
        "gguf": gguf,
    }
    if ollama:
        entry["ollama"] = ollama
    if active_params_b is not None:
        entry["active_params_b"] = active_params_b
    for key, value in (
        ("layers", layers),
        ("hidden_size", hidden_size),
        ("kv_heads", kv_heads),
        ("head_dim", head_dim),
    ):
        if value is not None:
            entry[key] = value
    return entry


MODELS = [
    # -- OpenAI open-weight -------------------------------------------------
    m("openai/gpt-oss-20b", "GPT-OSS 20B", "OpenAI", 20.9, 131072, "reasoning",
      "apache-2.0", "2025-08", ["tools", "reasoning"], 4, True, "gpt-oss:20b",
      active_params_b=3.6, layers=24, hidden_size=2880, kv_heads=8, head_dim=64),
    m("openai/gpt-oss-120b", "GPT-OSS 120B", "OpenAI", 116.8, 131072, "reasoning",
      "apache-2.0", "2025-08", ["tools", "reasoning"], 5, True, "gpt-oss:120b",
      active_params_b=5.1, layers=36, hidden_size=2880, kv_heads=8, head_dim=64),

    # -- Meta ---------------------------------------------------------------
    m("meta-llama/Llama-3.1-405B-Instruct", "Llama 3.1 405B Instruct", "Meta",
      405.0, 131072, "general", "llama3.1", "2024-07", ["tools"], 5, True,
      "llama3.1:405b", layers=126, hidden_size=16384, kv_heads=8, head_dim=128),
    m("meta-llama/Llama-Guard-3-8B", "Llama Guard 3 8B", "Meta", 8.03, 131072,
      "general", "llama3.1", "2024-07", ["safety"], 3, True, "llama-guard3:8b",
      layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
    m("codellama/CodeLlama-7b-Instruct-hf", "Code Llama 7B Instruct", "Meta",
      6.74, 16384, "coding", "llama2", "2023-08", [], 2, True, "codellama:7b",
      layers=32, hidden_size=4096, kv_heads=32, head_dim=128),
    m("codellama/CodeLlama-13b-Instruct-hf", "Code Llama 13B Instruct", "Meta",
      13.0, 16384, "coding", "llama2", "2023-08", [], 2, True, "codellama:13b",
      layers=40, hidden_size=5120, kv_heads=40, head_dim=128),
    m("codellama/CodeLlama-34b-Instruct-hf", "Code Llama 34B Instruct", "Meta",
      33.7, 16384, "coding", "llama2", "2023-08", [], 3, True, "codellama:34b",
      layers=48, hidden_size=8192, kv_heads=8, head_dim=128),
    m("codellama/CodeLlama-70b-Instruct-hf", "Code Llama 70B Instruct", "Meta",
      69.0, 4096, "coding", "llama2", "2024-01", [], 3, True, "codellama:70b",
      layers=80, hidden_size=8192, kv_heads=8, head_dim=128),

    # -- Alibaba Qwen -------------------------------------------------------
    # No Ollama tag: `qwen3:30b-a3b` belongs to the base Qwen3-30B-A3B, and
    # claiming it here would download a different model than the one shown.
    m("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3 30B A3B Instruct 2507",
      "Alibaba Qwen", 30.5, 262144, "general", "apache-2.0", "2025-07",
      ["tools"], 4, True, None, active_params_b=3.3, layers=48,
      hidden_size=2048, kv_heads=4, head_dim=128),
    m("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen3 Coder 30B A3B",
      "Alibaba Qwen", 30.5, 262144, "coding", "apache-2.0", "2025-07",
      ["tools"], 4, True, "qwen3-coder:30b", active_params_b=3.3, layers=48,
      hidden_size=2048, kv_heads=4, head_dim=128),
    m("Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen3 235B A22B Instruct 2507",
      "Alibaba Qwen", 235.0, 262144, "reasoning", "apache-2.0", "2025-07",
      ["tools", "reasoning"], 5, True, None, active_params_b=22.0, layers=94,
      hidden_size=4096, kv_heads=4, head_dim=128),
    m("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen3 Next 80B A3B", "Alibaba Qwen",
      80.0, 262144, "general", "apache-2.0", "2025-09", ["tools"], 5, True,
      None, active_params_b=3.0, layers=48, hidden_size=2048, kv_heads=2,
      head_dim=256),
    m("Qwen/Qwen2.5-Math-7B-Instruct", "Qwen2.5 Math 7B", "Alibaba Qwen", 7.62,
      4096, "reasoning", "apache-2.0", "2024-09", ["math"], 3, True, None,
      layers=28, hidden_size=3584, kv_heads=4, head_dim=128),
    m("Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen2.5 Coder 3B", "Alibaba Qwen",
      3.09, 32768, "coding", "qwen-research", "2024-11", [], 3, True,
      "qwen2.5-coder:3b", layers=36, hidden_size=2048, kv_heads=2, head_dim=128),
    m("Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen2.5 Coder 0.5B", "Alibaba Qwen",
      0.49, 32768, "coding", "apache-2.0", "2024-11", [], 2, True,
      "qwen2.5-coder:0.5b", layers=24, hidden_size=896, kv_heads=2, head_dim=64),
    m("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL 2B", "Alibaba Qwen", 2.21, 32768,
      "multimodal", "apache-2.0", "2024-08", ["vision"], 3, True, None,
      layers=28, hidden_size=1536, kv_heads=2, head_dim=128),
    m("Qwen/Qwen3-Embedding-4B", "Qwen3 Embedding 4B", "Alibaba Qwen", 4.02,
      32768, "embedding", "apache-2.0", "2025-06", ["embedding"], 4, True,
      None, layers=36, hidden_size=2560, kv_heads=8, head_dim=128),
    m("Qwen/Qwen3-Embedding-8B", "Qwen3 Embedding 8B", "Alibaba Qwen", 7.57,
      32768, "embedding", "apache-2.0", "2025-06", ["embedding"], 4, True,
      None, layers=36, hidden_size=4096, kv_heads=8, head_dim=128),
    m("Qwen/QwQ-32B-Preview", "QwQ 32B Preview", "Alibaba Qwen", 32.5, 32768,
      "reasoning", "apache-2.0", "2024-11", ["reasoning"], 3, True, None,
      layers=64, hidden_size=5120, kv_heads=8, head_dim=128),

    # -- DeepSeek -----------------------------------------------------------
    m("deepseek-ai/DeepSeek-V2-Lite-Chat", "DeepSeek V2 Lite Chat", "DeepSeek",
      15.7, 163840, "general", "deepseek", "2024-05", [], 3, True, None,
      active_params_b=2.4),
    m("deepseek-ai/deepseek-coder-6.7b-instruct", "DeepSeek Coder 6.7B",
      "DeepSeek", 6.74, 16384, "coding", "deepseek", "2023-11", [], 3, True,
      "deepseek-coder:6.7b", layers=32, hidden_size=4096, kv_heads=32, head_dim=128),
    m("deepseek-ai/deepseek-coder-33b-instruct", "DeepSeek Coder 33B",
      "DeepSeek", 33.3, 16384, "coding", "deepseek", "2023-11", [], 3, True,
      "deepseek-coder:33b", layers=62, hidden_size=7168, kv_heads=8,
      head_dim=128),
    m("deepseek-ai/DeepSeek-V3-0324", "DeepSeek V3 0324", "DeepSeek", 685.0,
      163840, "general", "mit", "2025-03", ["tools"], 5, True, None,
      active_params_b=37.0),
    m("deepseek-ai/DeepSeek-R1-0528", "DeepSeek R1 0528", "DeepSeek", 685.0,
      163840, "reasoning", "mit", "2025-05", ["reasoning"], 5, True, None,
      active_params_b=37.0),
    # `deepseek-r1:8b` has pointed at more than one build over time; it is the
    # 0528 Qwen3 distill now, and the Llama distill pins its own quantized tag.
    m("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "DeepSeek R1 0528 Qwen3 8B",
      "DeepSeek", 8.19, 131072, "reasoning", "mit", "2025-05", ["reasoning"],
      4, True, "deepseek-r1:8b", layers=36, hidden_size=4096, kv_heads=8,
      head_dim=128),
    m("deepseek-ai/DeepSeek-Prover-V2-7B", "DeepSeek Prover V2 7B", "DeepSeek",
      6.91, 65536, "reasoning", "mit", "2025-04", ["math"], 3, True, None,
      layers=30, hidden_size=4096, kv_heads=32, head_dim=128),

    # -- Mistral AI ---------------------------------------------------------
    # No Ollama tag: `ministral:8b` was withdrawn, and `ministral-3:8b` is the
    # 2512 model rather than this 2410 one.
    m("mistralai/Ministral-8B-Instruct-2410", "Ministral 8B", "Mistral AI",
      8.02, 131072, "general", "mrl", "2024-10", ["tools"], 4, True,
      None, layers=36, hidden_size=4096, kv_heads=8, head_dim=128),
    # `mistral-small:24b` is the 2501 build; the 3.2 refresh already in the
    # catalog keeps its own tag.
    m("mistralai/Mistral-Small-24B-Instruct-2501", "Mistral Small 3 24B",
      "Mistral AI", 23.6, 32768, "general", "apache-2.0", "2025-01",
      ["tools"], 4, True, None,
      layers=40, hidden_size=5120, kv_heads=8, head_dim=128),
    m("mistralai/Devstral-Small-2507", "Devstral Small 1.1", "Mistral AI",
      23.6, 131072, "coding", "apache-2.0", "2025-07", ["tools"], 4, True,
      "devstral:24b", layers=40, hidden_size=5120, kv_heads=8, head_dim=128),
    m("mistralai/Magistral-Small-2509", "Magistral Small 1.2", "Mistral AI",
      23.6, 131072, "reasoning", "apache-2.0", "2025-09", ["reasoning"], 4,
      True, "magistral:24b", layers=40, hidden_size=5120, kv_heads=8,
      head_dim=128),
    m("mistralai/Mathstral-7B-v0.1", "Mathstral 7B", "Mistral AI", 7.25, 32768,
      "reasoning", "apache-2.0", "2024-07", ["math"], 3, True, None,
      layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
    m("mistralai/Mamba-Codestral-7B-v0.1", "Codestral Mamba 7B", "Mistral AI",
      7.29, 262144, "coding", "apache-2.0", "2024-07", [], 3, True, None),

    # -- Microsoft ----------------------------------------------------------
    m("microsoft/Phi-4-reasoning-plus", "Phi-4 Reasoning Plus", "Microsoft",
      14.7, 32768, "reasoning", "mit", "2025-04", ["reasoning"], 4, True,
      "phi4-reasoning:14b", layers=40, hidden_size=5120, kv_heads=10, head_dim=128),
    m("microsoft/Phi-4-multimodal-instruct", "Phi-4 Multimodal", "Microsoft",
      5.57, 131072, "multimodal", "mit", "2025-02", ["vision", "audio"], 4,
      True, None, layers=32, hidden_size=3072, kv_heads=8, head_dim=128),
    m("microsoft/Phi-3.5-vision-instruct", "Phi-3.5 Vision", "Microsoft", 4.15,
      131072, "multimodal", "mit", "2024-08", ["vision"], 3, True, None,
      layers=32, hidden_size=3072, kv_heads=32, head_dim=96),
    m("microsoft/phi-2", "Phi-2", "Microsoft", 2.78, 2048, "general", "mit",
      "2023-12", [], 2, True, "phi:2.7b",
      layers=32, hidden_size=2560, kv_heads=32, head_dim=80),

    # -- Google -------------------------------------------------------------
    m("google/gemma-3-270m-it", "Gemma 3 270M", "Google", 0.27, 32768,
      "general", "gemma", "2025-08", [], 2, True, "gemma3:270m"),
    m("google/gemma-3n-E2B-it", "Gemma 3n E2B", "Google", 5.44, 32768,
      "multimodal", "gemma", "2025-06", ["vision", "audio"], 3, True,
      "gemma3n:e2b", active_params_b=2.0),
    m("google/gemma-3n-E4B-it", "Gemma 3n E4B", "Google", 7.85, 32768,
      "multimodal", "gemma", "2025-06", ["vision", "audio"], 4, True,
      "gemma3n:e4b", active_params_b=4.0),
    m("google/embeddinggemma-300m", "EmbeddingGemma 300M", "Google", 0.31,
      2048, "embedding", "gemma", "2025-09", ["embedding"], 4, True,
      "embeddinggemma:300m"),
    m("google/medgemma-4b-it", "MedGemma 4B", "Google", 4.3, 131072,
      "multimodal", "gemma", "2025-05", ["vision", "medical"], 3, True, None),
    m("google/shieldgemma-2b", "ShieldGemma 2B", "Google", 2.61, 8192,
      "general", "gemma", "2024-10", ["safety"], 3, True, "shieldgemma:2b"),

    # -- Zhipu AI / THUDM ---------------------------------------------------
    m("THUDM/glm-4-9b-chat", "GLM-4 9B Chat", "Zhipu AI", 9.4, 131072,
      "general", "glm-4", "2024-06", ["tools"], 4, True, "glm4:9b", layers=40,
      hidden_size=4096, kv_heads=2, head_dim=128),
    m("THUDM/GLM-4-32B-0414", "GLM-4 32B 0414", "Zhipu AI", 32.6, 32768,
      "general", "mit", "2025-04", ["tools"], 4, True, None),
    m("THUDM/GLM-Z1-32B-0414", "GLM-Z1 32B", "Zhipu AI", 32.6, 32768,
      "reasoning", "mit", "2025-04", ["reasoning"], 4, True, None, layers=61,
      hidden_size=6144, kv_heads=2, head_dim=128),
    m("zai-org/GLM-4.5-Air", "GLM-4.5 Air", "Zhipu AI", 106.0, 131072,
      "reasoning", "mit", "2025-07", ["tools", "reasoning"], 5, True, None,
      active_params_b=12.0, layers=46, hidden_size=4096, kv_heads=8,
      head_dim=128),
    m("THUDM/glm-4v-9b", "GLM-4V 9B", "Zhipu AI", 13.9, 8192, "multimodal",
      "glm-4", "2024-06", ["vision"], 3, True, None),

    # -- Moonshot AI --------------------------------------------------------
    m("moonshotai/Kimi-K2-Instruct", "Kimi K2 Instruct", "Moonshot AI", 1000.0,
      131072, "general", "modified-mit", "2025-07", ["tools"], 5, True, None,
      active_params_b=32.0, layers=61, hidden_size=7168, kv_heads=64,
      head_dim=112),
    m("moonshotai/Moonlight-16B-A3B-Instruct", "Moonlight 16B A3B",
      "Moonshot AI", 15.3, 8192, "general", "mit", "2025-02", [], 3, True,
      None, active_params_b=2.24),

    # -- MiniMax ------------------------------------------------------------
    m("MiniMaxAI/MiniMax-Text-01", "MiniMax Text 01", "MiniMax", 456.0,
      1000000, "general", "minimax", "2025-01", ["tools"], 4, False, None,
      active_params_b=45.9, layers=80, hidden_size=6144, kv_heads=8,
      head_dim=128),
    m("MiniMaxAI/MiniMax-M1-80k", "MiniMax M1 80k", "MiniMax", 456.0, 1000000,
      "reasoning", "apache-2.0", "2025-06", ["reasoning"], 5, False, None,
      active_params_b=45.9, layers=80, hidden_size=6144, kv_heads=8,
      head_dim=128),

    # -- xAI ----------------------------------------------------------------
    m("xai-org/grok-1", "Grok-1", "xAI", 314.0, 8192, "general",
      "apache-2.0", "2024-03", [], 3, False, None, active_params_b=86.0),
    m("xai-org/grok-2", "Grok-2", "xAI", 270.0, 131072, "general",
      "grok-2-community", "2025-08", [], 4, False, None, active_params_b=115.0,
      layers=64, hidden_size=8192, kv_heads=8, head_dim=128),

    # -- AI21 ---------------------------------------------------------------
    m("ai21labs/AI21-Jamba-Mini-1.6", "Jamba Mini 1.6", "AI21 Labs", 51.6,
      262144, "general", "jamba-open", "2025-03", ["tools"], 3, True, None,
      active_params_b=12.0),
    m("ai21labs/AI21-Jamba-Large-1.6", "Jamba Large 1.6", "AI21 Labs", 398.0,
      262144, "general", "jamba-open", "2025-03", ["tools"], 4, False, None,
      active_params_b=94.0),

    # -- Databricks ---------------------------------------------------------
    m("databricks/dbrx-instruct", "DBRX Instruct", "Databricks", 132.0, 32768,
      "general", "databricks-open", "2024-03", [], 3, True, "dbrx:132b",
      active_params_b=36.0),

    # -- Tencent ------------------------------------------------------------
    m("tencent/Hunyuan-A13B-Instruct", "Hunyuan A13B", "Tencent", 80.0, 262144,
      "general", "tencent-hunyuan", "2025-06", ["tools"], 4, True, None,
      active_params_b=13.0, layers=32, hidden_size=4096, kv_heads=8,
      head_dim=128),

    # -- Baidu --------------------------------------------------------------
    m("baidu/ERNIE-4.5-21B-A3B-PT", "ERNIE 4.5 21B A3B", "Baidu", 21.0, 131072,
      "general", "apache-2.0", "2025-06", ["tools"], 4, True, None,
      active_params_b=3.0, layers=28, hidden_size=2560, kv_heads=4,
      head_dim=128),
    m("baidu/ERNIE-4.5-0.3B-PT", "ERNIE 4.5 0.3B", "Baidu", 0.36, 131072,
      "general", "apache-2.0", "2025-06", [], 2, True, None, layers=18,
      hidden_size=1024, kv_heads=2, head_dim=128),

    # -- ByteDance ----------------------------------------------------------
    m("ByteDance-Seed/Seed-OSS-36B-Instruct", "Seed-OSS 36B", "ByteDance",
      36.0, 524288, "reasoning", "apache-2.0", "2025-08", ["reasoning"], 4,
      True, None, layers=64, hidden_size=5120, kv_heads=8, head_dim=128),

    # -- LG AI Research -----------------------------------------------------
    m("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "EXAONE 3.5 2.4B", "LG AI",
      2.41, 32768, "general", "exaone", "2024-12", [], 3, True, "exaone3.5:2.4b"),
    m("LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct", "EXAONE 3.5 7.8B", "LG AI",
      7.82, 32768, "general", "exaone", "2024-12", [], 3, True, "exaone3.5:7.8b"),
    m("LGAI-EXAONE/EXAONE-3.5-32B-Instruct", "EXAONE 3.5 32B", "LG AI", 32.0,
      32768, "general", "exaone", "2024-12", [], 4, True, "exaone3.5:32b"),
    m("LGAI-EXAONE/EXAONE-Deep-32B", "EXAONE Deep 32B", "LG AI", 32.0, 32768,
      "reasoning", "exaone", "2025-03", ["reasoning"], 4, True, "exaone-deep:32b"),

    # -- NVIDIA -------------------------------------------------------------
    m("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "Nemotron Nano 8B", "NVIDIA",
      8.03, 131072, "reasoning", "nvidia-open", "2025-03", ["reasoning"], 4,
      True, None, layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
    m("nvidia/Llama-3_3-Nemotron-Super-49B-v1", "Nemotron Super 49B", "NVIDIA",
      49.0, 131072, "reasoning", "nvidia-open", "2025-03", ["reasoning"], 4,
      True, None),
    m("nvidia/NVLM-D-72B", "NVLM-D 72B", "NVIDIA", 79.4, 32768, "multimodal",
      "cc-by-nc-4.0", "2024-09", ["vision"], 4, False, None),

    # -- OpenBMB / MiniCPM --------------------------------------------------
    m("openbmb/MiniCPM3-4B", "MiniCPM3 4B", "OpenBMB", 4.07, 32768, "general",
      "apache-2.0", "2024-09", ["tools"], 3, True, None, layers=62,
      hidden_size=2560, kv_heads=40, head_dim=64),
    m("openbmb/MiniCPM-V-2_6", "MiniCPM-V 2.6", "OpenBMB", 8.1, 32768,
      "multimodal", "apache-2.0", "2024-08", ["vision"], 4, True, "minicpm-v:8b"),
    m("openbmb/MiniCPM-o-2_6", "MiniCPM-o 2.6", "OpenBMB", 8.67, 32768,
      "multimodal", "apache-2.0", "2025-01", ["vision", "audio"], 4, True,
      None, layers=28, hidden_size=3584, kv_heads=4, head_dim=128),

    # -- Allen AI -----------------------------------------------------------
    m("allenai/OLMo-2-0325-32B-Instruct", "OLMo 2 32B Instruct", "Allen AI",
      32.2, 4096, "general", "apache-2.0", "2025-03", [], 4, True, None,
      layers=64, hidden_size=5120, kv_heads=8, head_dim=128),
    m("allenai/Llama-3.1-Tulu-3-8B", "Tulu 3 8B", "Allen AI", 8.03, 131072,
      "general", "llama3.1", "2024-11", [], 3, True, "tulu3:8b", layers=32,
      hidden_size=4096, kv_heads=8, head_dim=128),
    m("allenai/Llama-3.1-Tulu-3-70B", "Tulu 3 70B", "Allen AI", 70.6, 131072,
      "general", "llama3.1", "2024-11", [], 4, True, "tulu3:70b", layers=80,
      hidden_size=8192, kv_heads=8, head_dim=128),
    m("allenai/Molmo-7B-D-0924", "Molmo 7B-D", "Allen AI", 8.02, 4096,
      "multimodal", "apache-2.0", "2024-09", ["vision"], 3, False, None,
      layers=28, hidden_size=3584, kv_heads=4, head_dim=128),

    # -- HuggingFace TB -----------------------------------------------------
    m("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2 135M", "Hugging Face",
      0.135, 8192, "general", "apache-2.0", "2024-11", [], 2, True,
      "smollm2:135m", layers=30, hidden_size=576, kv_heads=3, head_dim=64),
    m("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M", "Hugging Face",
      0.362, 8192, "general", "apache-2.0", "2024-11", [], 2, True,
      "smollm2:360m", layers=32, hidden_size=960, kv_heads=5, head_dim=64),
    m("HuggingFaceTB/SmolLM3-3B", "SmolLM3 3B", "Hugging Face", 3.08, 131072,
      "general", "apache-2.0", "2025-07", ["tools", "reasoning"], 3, True,
      None, layers=36, hidden_size=2048, kv_heads=4, head_dim=128),
    m("HuggingFaceTB/SmolVLM-Instruct", "SmolVLM 2B", "Hugging Face", 2.25,
      16384, "multimodal", "apache-2.0", "2024-11", ["vision"], 3, True, None,
      layers=24, hidden_size=2048, kv_heads=32, head_dim=64),

    # -- IBM Granite --------------------------------------------------------
    m("ibm-granite/granite-4.0-h-small", "Granite 4.0 H Small", "IBM", 32.2,
      131072, "general", "apache-2.0", "2025-10", ["tools"], 4, True,
      "granite4:small-h", active_params_b=9.0, layers=40, hidden_size=4096,
      kv_heads=8, head_dim=128),
    m("ibm-granite/granite-4.0-h-tiny", "Granite 4.0 H Tiny", "IBM", 6.94,
      131072, "general", "apache-2.0", "2025-10", ["tools"], 3, True,
      "granite4:tiny-h", active_params_b=1.0, layers=40, hidden_size=1536,
      kv_heads=4, head_dim=128),
    m("ibm-granite/granite-4.0-micro", "Granite 4.0 Micro", "IBM", 3.4, 131072,
      "general", "apache-2.0", "2025-10", ["tools"], 3, True, "granite4:micro",
      layers=40, hidden_size=2560, kv_heads=8, head_dim=64),
    m("ibm-granite/granite-vision-3.2-2b", "Granite Vision 3.2 2B", "IBM",
      2.98, 131072, "multimodal", "apache-2.0", "2025-02", ["vision"], 3, True,
      None, layers=40, hidden_size=2048, kv_heads=8, head_dim=64),
    m("ibm-granite/granite-embedding-278m-multilingual",
      "Granite Embedding 278M", "IBM", 0.278, 512, "embedding", "apache-2.0",
      "2024-12", ["embedding"], 3, True, None),
    m("ibm-granite/granite-8b-code-instruct-128k", "Granite 8B Code", "IBM",
      8.05, 128000, "coding", "apache-2.0", "2024-07", [], 3, True,
      "granite-code:8b", layers=36, hidden_size=4096, kv_heads=8, head_dim=128),

    # -- Cohere -------------------------------------------------------------
    m("CohereForAI/c4ai-command-a-03-2025", "Command A", "Cohere", 111.0,
      262144, "general", "cc-by-nc-4.0", "2025-03", ["tools"], 5, True,
      "command-a:111b"),
    m("CohereForAI/c4ai-command-r7b-12-2024", "Command R7B", "Cohere", 8.03,
      131072, "general", "cc-by-nc-4.0", "2024-12", ["tools"], 3, True,
      "command-r7b:7b"),
    m("CohereForAI/aya-vision-8b", "Aya Vision 8B", "Cohere", 8.63, 16384,
      "multimodal", "cc-by-nc-4.0", "2025-03", ["vision"], 3, True, None),
    m("CohereForAI/aya-vision-32b", "Aya Vision 32B", "Cohere", 32.3, 16384,
      "multimodal", "cc-by-nc-4.0", "2025-03", ["vision"], 4, True, None),

    # -- Falcon / TII -------------------------------------------------------
    m("tiiuae/Falcon3-1B-Instruct", "Falcon3 1B", "TII", 1.67, 8192, "general",
      "falcon-llm", "2024-12", [], 2, True, "falcon3:1b", layers=18,
      hidden_size=2048, kv_heads=4, head_dim=256),
    m("tiiuae/Falcon3-3B-Instruct", "Falcon3 3B", "TII", 3.23, 32768,
      "general", "falcon-llm", "2024-12", [], 3, True, "falcon3:3b", layers=22,
      hidden_size=3072, kv_heads=4, head_dim=256),
    m("tiiuae/Falcon-H1-7B-Instruct", "Falcon-H1 7B", "TII", 7.59, 262144,
      "general", "falcon-llm", "2025-05", ["tools"], 4, True, None, layers=44,
      hidden_size=3072, kv_heads=2, head_dim=128),
    m("tiiuae/falcon-mamba-7b-instruct", "Falcon Mamba 7B", "TII", 7.27, 8192,
      "general", "falcon-mamba", "2024-08", [], 3, True, None),
    m("tiiuae/falcon-40b-instruct", "Falcon 40B Instruct", "TII", 41.8, 2048,
      "general", "apache-2.0", "2023-05", [], 2, True, "falcon:40b", layers=60,
      hidden_size=8192, kv_heads=8, head_dim=64),

    # -- 01.AI --------------------------------------------------------------
    m("01-ai/Yi-1.5-6B-Chat", "Yi 1.5 6B Chat", "01.AI", 6.06, 4096, "chat",
      "apache-2.0", "2024-05", [], 3, True, "yi:6b", layers=32,
      hidden_size=4096, kv_heads=4, head_dim=128),
    m("01-ai/Yi-Coder-9B-Chat", "Yi Coder 9B", "01.AI", 8.83, 131072, "coding",
      "apache-2.0", "2024-09", [], 3, True, "yi-coder:9b", layers=48,
      hidden_size=4096, kv_heads=4, head_dim=128),

    # -- InternLM / Shanghai AI Lab ----------------------------------------
    m("internlm/internlm3-8b-instruct", "InternLM3 8B", "Shanghai AI Lab", 8.8,
      32768, "general", "apache-2.0", "2025-01", ["tools"], 4, True, None,
      layers=48, hidden_size=4096, kv_heads=2, head_dim=128),
    m("OpenGVLab/InternVL2_5-8B", "InternVL 2.5 8B", "Shanghai AI Lab", 8.08,
      32768, "multimodal", "mit", "2024-12", ["vision"], 4, True, None),
    m("OpenGVLab/InternVL3-14B", "InternVL3 14B", "Shanghai AI Lab", 15.1,
      32768, "multimodal", "apache-2.0", "2025-04", ["vision"], 4, True, None),

    # -- Liquid AI ----------------------------------------------------------
    m("LiquidAI/LFM2-1.2B", "LFM2 1.2B", "Liquid AI", 1.17, 128000, "general",
      "lfm-open", "2025-07", [], 3, True, None, layers=16, hidden_size=2048,
      kv_heads=8, head_dim=64),
    m("LiquidAI/LFM2-VL-1.6B", "LFM2-VL 1.6B", "Liquid AI", 1.6, 128000,
      "multimodal", "lfm-open", "2025-08", ["vision"], 3, True, None,
      layers=16, hidden_size=2048, kv_heads=8, head_dim=64),

    # -- Arcee AI -----------------------------------------------------------
    m("arcee-ai/AFM-4.5B", "Arcee AFM 4.5B", "Arcee AI", 4.5, 65536, "general",
      "apache-2.0", "2025-07", ["tools"], 3, True, None, layers=36,
      hidden_size=2560, kv_heads=4, head_dim=128),
    m("arcee-ai/Arcee-Blitz", "Arcee Blitz 24B", "Arcee AI", 23.6, 32768,
      "general", "apache-2.0", "2025-02", [], 3, True, None, layers=40,
      hidden_size=5120, kv_heads=8, head_dim=128),

    # -- Nous Research ------------------------------------------------------
    m("NousResearch/Hermes-4-14B", "Hermes 4 14B", "Nous Research", 14.8,
      131072, "general", "apache-2.0", "2025-08", ["tools", "reasoning"], 4,
      True, None, layers=40, hidden_size=5120, kv_heads=8, head_dim=128),
    m("NousResearch/DeepHermes-3-Llama-3-8B-Preview", "DeepHermes 3 8B",
      "Nous Research", 8.03, 131072, "reasoning", "llama3", "2025-02",
      ["reasoning"], 3, True, None, layers=32, hidden_size=4096, kv_heads=8,
      head_dim=128),

    # -- Perplexity ---------------------------------------------------------
    m("perplexity-ai/r1-1776", "R1-1776", "Perplexity", 685.0, 131072,
      "reasoning", "mit", "2025-02", ["reasoning"], 5, True, None,
      active_params_b=37.0),
    m("perplexity-ai/r1-1776-distill-llama-70b", "R1-1776 Distill 70B",
      "Perplexity", 70.6, 131072, "reasoning", "llama3.3", "2025-02",
      ["reasoning"], 4, True, None),

    # -- Community fine-tunes ----------------------------------------------
    m("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B Chat",
      "TinyLlama", 1.1, 2048, "chat", "apache-2.0", "2023-12", [], 1, True,
      "tinyllama:1.1b", layers=22, hidden_size=2048, kv_heads=4, head_dim=64),
    m("HuggingFaceH4/zephyr-7b-beta", "Zephyr 7B Beta", "Hugging Face", 7.24,
      32768, "chat", "mit", "2023-10", [], 2, True, "zephyr:7b",
      layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
    m("openchat/openchat-3.5-0106", "OpenChat 3.5", "OpenChat", 7.24, 8192,
      "chat", "apache-2.0", "2024-01", [], 2, True, "openchat:7b", layers=32,
      hidden_size=4096, kv_heads=8, head_dim=128),
    m("Nexusflow/Starling-LM-7B-beta", "Starling LM 7B", "Nexusflow", 7.24,
      8192, "chat", "apache-2.0", "2024-03", [], 2, True, "starling-lm:7b",
      layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
    m("cognitivecomputations/dolphin-2.9.3-mistral-7B-32k", "Dolphin 2.9 7B",
      "Cognitive Computations", 7.24, 32768, "chat", "apache-2.0", "2024-05",
      [], 2, True, "dolphin-mistral:7b", layers=32, hidden_size=4096,
      kv_heads=8, head_dim=128),
    m("upstage/SOLAR-10.7B-Instruct-v1.0", "SOLAR 10.7B", "Upstage", 10.7,
      4096, "general", "cc-by-nc-4.0", "2023-12", [], 3, True, "solar:10.7b",
      layers=48, hidden_size=4096, kv_heads=8, head_dim=128),
    m("microsoft/Orca-2-13b", "Orca 2 13B", "Microsoft", 13.0, 4096,
      "reasoning", "microsoft-research", "2023-11", [], 2, True, None,
      layers=40, hidden_size=5120, kv_heads=40, head_dim=128),
    m("lmsys/vicuna-13b-v1.5", "Vicuna 13B v1.5", "LMSYS", 13.0, 4096, "chat",
      "llama2", "2023-08", [], 1, True, "vicuna:13b", layers=40,
      hidden_size=5120, kv_heads=40, head_dim=128),
    m("WizardLMTeam/WizardLM-2-8x22B", "WizardLM-2 8x22B", "WizardLM", 141.0,
      65536, "general", "apache-2.0", "2024-04", [], 4, True, None,
      active_params_b=39.0),

    # -- Vision -------------------------------------------------------------
    m("llava-hf/llava-v1.6-mistral-7b-hf", "LLaVA 1.6 Mistral 7B", "LLaVA",
      7.57, 32768, "multimodal", "apache-2.0", "2024-01", ["vision"], 3, True,
      "llava:7b"),
    m("llava-hf/llava-v1.6-34b-hf", "LLaVA 1.6 34B", "LLaVA", 34.8, 4096,
      "multimodal", "apache-2.0", "2024-01", ["vision"], 3, True, "llava:34b",
      layers=60, hidden_size=7168, kv_heads=8, head_dim=128),
    m("vikhyatk/moondream2", "Moondream 2", "Moondream", 1.87, 2048,
      "multimodal", "apache-2.0", "2024-03", ["vision"], 3, True,
      "moondream:1.8b"),
    m("microsoft/Florence-2-large", "Florence-2 Large", "Microsoft", 0.77,
      4096, "multimodal", "mit", "2024-06", ["vision"], 3, False, None),
    m("HuggingFaceM4/Idefics3-8B-Llama3", "Idefics3 8B", "Hugging Face", 8.46,
      131072, "multimodal", "apache-2.0", "2024-08", ["vision"], 3, False,
      None, layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
    m("moonshotai/Kimi-VL-A3B-Instruct", "Kimi-VL A3B", "Moonshot AI", 16.4,
      131072, "multimodal", "mit", "2025-04", ["vision"], 4, True, None,
      active_params_b=2.8, layers=27, hidden_size=2048, kv_heads=16,
      head_dim=128),

    # -- Coding -------------------------------------------------------------
    m("infly/OpenCoder-8B-Instruct", "OpenCoder 8B", "INF Tech", 7.77, 8192,
      "coding", "inf-open", "2024-11", [], 3, True, None, layers=32,
      hidden_size=4096, kv_heads=8, head_dim=128),
    m("Qwen/CodeQwen1.5-7B-Chat", "CodeQwen1.5 7B", "Alibaba Qwen", 7.25,
      65536, "coding", "tongyi-qianwen", "2024-04", [], 3, True, "codeqwen:7b",
      layers=32, hidden_size=4096, kv_heads=4, head_dim=128),
    m("WizardLMTeam/WizardCoder-33B-V1.1", "WizardCoder 33B", "WizardLM", 33.3,
      16384, "coding", "deepseek", "2024-01", [], 3, True, None, layers=62,
      hidden_size=7168, kv_heads=8, head_dim=128),
    m("JetBrains/Mellum-4b-base", "Mellum 4B", "JetBrains", 4.02, 8192,
      "coding", "apache-2.0", "2025-04", [], 3, True, None, layers=30,
      hidden_size=3072, kv_heads=24, head_dim=128),

    # -- Embedding / reranking ---------------------------------------------
    m("intfloat/multilingual-e5-large-instruct", "Multilingual E5 Large",
      "Microsoft", 0.56, 512, "embedding", "mit", "2024-02", ["embedding"], 4,
      True, None),
    m("intfloat/e5-large-v2", "E5 Large v2", "Microsoft", 0.335, 512,
      "embedding", "mit", "2023-05", ["embedding"], 3, True, None),
    m("thenlper/gte-large", "GTE Large", "Alibaba NLP", 0.335, 512,
      "embedding", "mit", "2023-07", ["embedding"], 3, True, None),
    m("jinaai/jina-embeddings-v3", "Jina Embeddings v3", "Jina AI", 0.572,
      8192, "embedding", "cc-by-nc-4.0", "2024-09", ["embedding"], 4, True,
      None),
    m("mixedbread-ai/mxbai-embed-large-v1", "mxbai Embed Large", "Mixedbread",
      0.335, 512, "embedding", "apache-2.0", "2024-03", ["embedding"], 4, True,
      "mxbai-embed-large:335m"),
    m("nomic-ai/nomic-embed-text-v2-moe", "Nomic Embed Text v2", "Nomic AI",
      0.475, 512, "embedding", "apache-2.0", "2025-02", ["embedding"], 4, True,
      None, active_params_b=0.305),
    m("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2",
      "Sentence Transformers", 0.109, 384, "embedding", "apache-2.0",
      "2021-08", ["embedding"], 3, True, None),
    m("BAAI/bge-reranker-v2-m3", "BGE Reranker v2 M3", "BAAI", 0.568, 8192,
      "embedding", "apache-2.0", "2024-03", ["reranker"], 4, True, None),
    m("Alibaba-NLP/gte-Qwen2-7B-instruct", "GTE Qwen2 7B", "Alibaba NLP", 7.61,
      131072, "embedding", "apache-2.0", "2024-06", ["embedding"], 4, True,
      None, layers=28, hidden_size=3584, kv_heads=4, head_dim=128),
    m("NovaSearch/stella_en_1.5B_v5", "Stella EN 1.5B v5", "NovaSearch", 1.54,
      131072, "embedding", "mit", "2024-07", ["embedding"], 4, True, None,
      layers=28, hidden_size=1536, kv_heads=2, head_dim=128),

    # -- Alternative architectures -----------------------------------------
    m("Zyphra/Zamba2-7B-Instruct", "Zamba2 7B", "Zyphra", 7.42, 4096,
      "general", "apache-2.0", "2024-10", [], 3, True, None, layers=81,
      hidden_size=3584, kv_heads=32, head_dim=112),
    m("state-spaces/mamba-2.8b-hf", "Mamba 2.8B", "State Spaces", 2.77, 2048,
      "general", "apache-2.0", "2023-12", [], 2, False, None),
    m("RWKV/rwkv-6-world-7b", "RWKV-6 World 7B", "BlinkDL", 7.64, 65536,
      "general", "apache-2.0", "2024-06", [], 3, True, None),
    m("togethercomputer/StripedHyena-Nous-7B", "StripedHyena Nous 7B",
      "Together AI", 7.64, 32768, "general", "apache-2.0", "2023-12", [], 2,
      False, None),

    # == Seed catalog =======================================================
    # Records that were in data/models.json before this script held them.
    # They live here now so that every record has one source.

    # -- 01.AI --------------------------------------------------------------
    m("01-ai/Yi-1.5-34B-Chat", "Yi 1.5 34B Chat", "01.AI", 34.4, 4096,
      "general", "apache-2.0", "2024-05", [], 4, True, "yi:34b", layers=60,
      hidden_size=7168, kv_heads=8, head_dim=128),
    m("01-ai/Yi-1.5-9B-Chat", "Yi 1.5 9B Chat", "01.AI", 8.83, 4096, "chat",
      "apache-2.0", "2024-05", [], 3, True, "yi:9b", layers=48,
      hidden_size=4096, kv_heads=4, head_dim=128),

    # -- Alibaba Qwen -------------------------------------------------------
    m("Qwen/Qwen3-235B-A22B", "Qwen3 235B A22B", "Alibaba Qwen", 235.0, 131072,
      "reasoning", "apache-2.0", "2025-04", ["tools", "thinking", "moe"], 5,
      True, "qwen3:235b", active_params_b=22.0, layers=94, hidden_size=4096,
      kv_heads=4, head_dim=128),
    m("Qwen/Qwen2.5-VL-72B-Instruct", "Qwen2.5-VL 72B Instruct",
      "Alibaba Qwen", 73.4, 128000, "multimodal", "qwen", "2025-01",
      ["vision", "tools"], 5, True, "qwen2.5vl:72b", layers=80,
      hidden_size=8192, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5 72B Instruct", "Alibaba Qwen",
      72.7, 32768, "general", "qwen", "2024-09", ["tools"], 5, True,
      "qwen2.5:72b", layers=80, hidden_size=8192, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL 32B Instruct",
      "Alibaba Qwen", 33.4, 128000, "multimodal", "apache-2.0", "2025-03",
      ["vision", "tools"], 5, True, "qwen2.5vl:32b", layers=64,
      hidden_size=5120, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-32B-Instruct", "Qwen2.5 32B Instruct", "Alibaba Qwen",
      32.8, 32768, "general", "apache-2.0", "2024-09", ["tools"], 5, True,
      "qwen2.5:32b", layers=64, hidden_size=5120, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen2.5 Coder 32B", "Alibaba Qwen",
      32.8, 32768, "coding", "apache-2.0", "2024-11", ["tools", "fim"], 5,
      True, "qwen2.5-coder:32b", layers=64, hidden_size=5120, kv_heads=8,
      head_dim=128),
    m("Qwen/Qwen3-32B", "Qwen3 32B", "Alibaba Qwen", 32.8, 131072, "reasoning",
      "apache-2.0", "2025-04", ["tools", "thinking"], 5, True, "qwen3:32b",
      layers=64, hidden_size=5120, kv_heads=8, head_dim=128),
    m("Qwen/QwQ-32B", "QwQ 32B", "Alibaba Qwen", 32.8, 131072, "reasoning",
      "apache-2.0", "2025-03", ["thinking", "tools"], 5, True, "qwq:32b",
      layers=64, hidden_size=5120, kv_heads=8, head_dim=128),
    m("Qwen/Qwen3-Coder-32B-Instruct", "Qwen3 Coder 32B", "Alibaba Qwen", 32.8,
      262144, "coding", "apache-2.0", "2025-07", ["tools", "fim"], 5, True,
      None, layers=64, hidden_size=5120, kv_heads=8, head_dim=128),
    m("Qwen/Qwen3-30B-A3B", "Qwen3 30B A3B", "Alibaba Qwen", 30.5, 262144,
      "reasoning", "apache-2.0", "2025-04", ["tools", "thinking", "moe"], 5,
      True, "qwen3:30b-a3b", active_params_b=3.3, layers=48, hidden_size=2048,
      kv_heads=4, head_dim=128),
    m("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5 14B Instruct", "Alibaba Qwen",
      14.8, 32768, "general", "apache-2.0", "2024-09", ["tools"], 4, True,
      "qwen2.5:14b", layers=48, hidden_size=5120, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen2.5 Coder 14B", "Alibaba Qwen",
      14.8, 32768, "coding", "apache-2.0", "2024-11", ["tools", "fim"], 4,
      True, "qwen2.5-coder:14b", layers=48, hidden_size=5120, kv_heads=8,
      head_dim=128),
    m("Qwen/Qwen3-14B", "Qwen3 14B", "Alibaba Qwen", 14.8, 131072, "reasoning",
      "apache-2.0", "2025-04", ["tools", "thinking"], 4, True, "qwen3:14b",
      layers=40, hidden_size=5120, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL 7B Instruct", "Alibaba Qwen",
      8.29, 128000, "multimodal", "apache-2.0", "2025-01", ["vision", "tools"],
      4, True, "qwen2.5vl:7b", layers=28, hidden_size=3584, kv_heads=4,
      head_dim=128),
    m("Qwen/Qwen3-8B", "Qwen3 8B", "Alibaba Qwen", 8.19, 131072, "reasoning",
      "apache-2.0", "2025-04", ["tools", "thinking"], 4, True, "qwen3:8b",
      layers=36, hidden_size=4096, kv_heads=8, head_dim=128),
    m("Qwen/Qwen3-Coder-8B-Instruct", "Qwen3 Coder 8B", "Alibaba Qwen", 8.19,
      262144, "coding", "apache-2.0", "2025-07", ["tools", "fim"], 4, True,
      None, layers=36, hidden_size=4096, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct", "Alibaba Qwen", 7.62,
      32768, "general", "apache-2.0", "2024-09", ["tools"], 4, True,
      "qwen2.5:7b", layers=28, hidden_size=3584, kv_heads=4, head_dim=128),
    m("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen2.5 Coder 7B", "Alibaba Qwen",
      7.62, 32768, "coding", "apache-2.0", "2024-09", ["tools", "fim"], 4,
      True, "qwen2.5-coder:7b", layers=28, hidden_size=3584, kv_heads=4,
      head_dim=128),
    m("Qwen/Qwen3-4B", "Qwen3 4B", "Alibaba Qwen", 4.02, 131072, "general",
      "apache-2.0", "2025-04", ["tools", "thinking"], 4, True, "qwen3:4b",
      layers=36, hidden_size=2560, kv_heads=8, head_dim=128),
    m("Qwen/Qwen3-4B-Instruct-2507", "Qwen3 4B Instruct", "Alibaba Qwen", 4.02,
      262144, "general", "apache-2.0", "2025-07", ["tools"], 4, True,
      "qwen3:4b-instruct", layers=36, hidden_size=2560, kv_heads=8,
      head_dim=128),
    m("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL 3B Instruct", "Alibaba Qwen",
      3.75, 128000, "multimodal", "apache-2.0", "2025-01", ["vision"], 3, True,
      "qwen2.5vl:3b", layers=36, hidden_size=2048, kv_heads=2, head_dim=128),
    m("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B Instruct", "Alibaba Qwen", 3.09,
      32768, "general", "qwen-research", "2024-09", ["tools"], 3, True,
      "qwen2.5:3b", layers=36, hidden_size=2048, kv_heads=2, head_dim=128),
    m("Qwen/Qwen3-1.7B", "Qwen3 1.7B", "Alibaba Qwen", 2.03, 32768, "general",
      "apache-2.0", "2025-04", ["thinking"], 3, True, "qwen3:1.7b", layers=28,
      hidden_size=2048, kv_heads=8, head_dim=128),
    m("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", "Alibaba Qwen",
      1.54, 32768, "general", "apache-2.0", "2024-09", ["tools"], 3, True,
      "qwen2.5:1.5b", layers=28, hidden_size=1536, kv_heads=2, head_dim=128),
    m("Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen2.5 Coder 1.5B", "Alibaba Qwen",
      1.54, 32768, "coding", "apache-2.0", "2024-11", ["fim"], 3, True,
      "qwen2.5-coder:1.5b", layers=28, hidden_size=1536, kv_heads=2,
      head_dim=128),
    m("Qwen/Qwen3-0.6B", "Qwen3 0.6B", "Alibaba Qwen", 0.75, 32768, "general",
      "apache-2.0", "2025-04", ["thinking"], 2, True, "qwen3:0.6b", layers=28,
      hidden_size=1024, kv_heads=8, head_dim=128),
    m("Qwen/Qwen3-Embedding-0.6B", "Qwen3 Embedding 0.6B", "Alibaba Qwen", 0.6,
      32768, "embedding", "apache-2.0", "2025-06", ["embedding"], 3, True,
      "qwen3-embedding:0.6b", layers=28, hidden_size=1024, kv_heads=8,
      head_dim=128),
    m("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", "Alibaba Qwen",
      0.49, 32768, "general", "apache-2.0", "2024-09", [], 2, True,
      "qwen2.5:0.5b", layers=24, hidden_size=896, kv_heads=2, head_dim=64),

    # -- Allen AI -----------------------------------------------------------
    m("allenai/OLMo-2-13B-1124-Instruct", "OLMo 2 13B Instruct", "Allen AI",
      13.0, 4096, "general", "apache-2.0", "2024-11", [], 3, True, "olmo2:13b",
      layers=40, hidden_size=5120, kv_heads=40, head_dim=128),
    m("allenai/OLMo-2-7B-1124-Instruct", "OLMo 2 7B Instruct", "Allen AI",
      7.28, 4096, "general", "apache-2.0", "2024-11", [], 3, True, "olmo2:7b",
      layers=32, hidden_size=4096, kv_heads=32, head_dim=128),

    # -- BAAI ---------------------------------------------------------------
    m("BAAI/bge-m3", "BGE-M3", "BAAI", 0.568, 8192, "embedding", "mit",
      "2024-01", ["embedding"], 4, True, "bge-m3"),
    m("BAAI/bge-large-en-v1.5", "BGE Large EN v1.5", "BAAI", 0.335, 512,
      "embedding", "mit", "2023-09", ["embedding"], 3, True, "bge-large"),

    # -- BigCode ------------------------------------------------------------
    m("bigcode/starcoder2-15b", "StarCoder2 15B", "BigCode", 15.7, 16384,
      "coding", "bigcode-openrail-m", "2024-02", ["fim"], 3, True,
      "starcoder2:15b", layers=40, hidden_size=6144, kv_heads=4, head_dim=128),
    m("bigcode/starcoder2-7b", "StarCoder2 7B", "BigCode", 7.17, 16384,
      "coding", "bigcode-openrail-m", "2024-02", ["fim"], 3, True,
      "starcoder2:7b", layers=32, hidden_size=4608, kv_heads=4, head_dim=128),
    m("bigcode/starcoder2-3b", "StarCoder2 3B", "BigCode", 3.03, 16384,
      "coding", "bigcode-openrail-m", "2024-02", ["fim"], 2, True,
      "starcoder2:3b", layers=30, hidden_size=3072, kv_heads=2, head_dim=128),

    # -- Cohere -------------------------------------------------------------
    m("CohereForAI/c4ai-command-r-plus-08-2024", "Command R+ 104B", "Cohere",
      104.0, 131072, "general", "cc-by-nc-4.0", "2024-08", ["tools", "rag"], 5,
      True, "command-r-plus:104b", layers=64, hidden_size=12288, kv_heads=8,
      head_dim=128),
    m("CohereForAI/c4ai-command-r-08-2024", "Command R 35B", "Cohere", 32.3,
      131072, "general", "cc-by-nc-4.0", "2024-08", ["tools", "rag"], 4, True,
      "command-r:35b", layers=40, hidden_size=8192, kv_heads=8, head_dim=128),
    m("CohereForAI/aya-expanse-32b", "Aya Expanse 32B", "Cohere", 32.3, 131072,
      "chat", "cc-by-nc-4.0", "2024-10", ["multilingual"], 4, True,
      "aya-expanse:32b", layers=40, hidden_size=8192, kv_heads=8, head_dim=128),
    m("CohereForAI/aya-expanse-8b", "Aya Expanse 8B", "Cohere", 8.03, 8192,
      "chat", "cc-by-nc-4.0", "2024-10", ["multilingual"], 3, True,
      "aya-expanse:8b", layers=32, hidden_size=4096, kv_heads=8, head_dim=128),

    # -- DeepSeek -----------------------------------------------------------
    m("deepseek-ai/DeepSeek-V3", "DeepSeek V3 671B MoE", "DeepSeek", 671.0,
      163840, "general", "deepseek", "2024-12", ["tools", "moe"], 5, True,
      "deepseek-v3:671b", active_params_b=37.0, layers=61, hidden_size=7168,
      kv_heads=8, head_dim=128),
    m("deepseek-ai/DeepSeek-R1", "DeepSeek R1 671B MoE", "DeepSeek", 671.0,
      163840, "reasoning", "mit", "2025-01", ["thinking", "moe"], 5, True,
      "deepseek-r1:671b", active_params_b=37.0, layers=61, hidden_size=7168,
      kv_heads=8, head_dim=128),
    m("deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
      "DeepSeek R1 Distill Llama 70B", "DeepSeek", 70.6, 131072, "reasoning",
      "mit", "2025-01", ["thinking"], 5, True, "deepseek-r1:70b", layers=80,
      hidden_size=8192, kv_heads=8, head_dim=128),
    m("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
      "DeepSeek R1 Distill Qwen 32B", "DeepSeek", 32.8, 131072, "reasoning",
      "mit", "2025-01", ["thinking"], 5, True, "deepseek-r1:32b", layers=64,
      hidden_size=5120, kv_heads=8, head_dim=128),
    m("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
      "DeepSeek Coder V2 Lite 16B", "DeepSeek", 15.7, 163840, "coding",
      "deepseek", "2024-06", ["fim", "moe"], 4, True, "deepseek-coder-v2:16b",
      active_params_b=2.4, layers=27, hidden_size=2048, kv_heads=4,
      head_dim=128),
    m("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
      "DeepSeek R1 Distill Qwen 14B", "DeepSeek", 14.8, 131072, "reasoning",
      "mit", "2025-01", ["thinking"], 4, True, "deepseek-r1:14b", layers=48,
      hidden_size=5120, kv_heads=8, head_dim=128),
    m("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
      "DeepSeek R1 Distill Llama 8B", "DeepSeek", 8.03, 131072, "reasoning",
      "mit", "2025-01", ["thinking"], 4, True,
      "deepseek-r1:8b-llama-distill-q4_K_M", layers=32, hidden_size=4096,
      kv_heads=8, head_dim=128),
    m("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek R1 Distill Qwen 7B",
      "DeepSeek", 7.62, 131072, "reasoning", "mit", "2025-01", ["thinking"], 4,
      True, "deepseek-r1:7b", layers=28, hidden_size=3584, kv_heads=4,
      head_dim=128),
    m("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek R1 Distill 1.5B",
      "DeepSeek", 1.78, 131072, "reasoning", "mit", "2025-01", ["thinking"], 3,
      True, "deepseek-r1:1.5b", layers=28, hidden_size=1536, kv_heads=2,
      head_dim=128),

    # -- Google -------------------------------------------------------------
    m("google/gemma-3-27b-it", "Gemma 3 27B IT", "Google", 27.4, 131072,
      "multimodal", "gemma", "2025-03", ["vision", "tools"], 5, True,
      "gemma3:27b", layers=62, hidden_size=4608, kv_heads=16, head_dim=128),
    m("google/gemma-2-27b-it", "Gemma 2 27B IT", "Google", 27.2, 8192,
      "general", "gemma", "2024-06", [], 4, True, "gemma2:27b", layers=46,
      hidden_size=4608, kv_heads=16, head_dim=128),
    m("google/gemma-3-12b-it", "Gemma 3 12B IT", "Google", 12.2, 131072,
      "multimodal", "gemma", "2025-03", ["vision", "tools"], 4, True,
      "gemma3:12b", layers=48, hidden_size=3840, kv_heads=8, head_dim=256),
    m("google/gemma-2-9b-it", "Gemma 2 9B IT", "Google", 9.24, 8192, "chat",
      "gemma", "2024-06", [], 3, True, "gemma2:9b", layers=42,
      hidden_size=3584, kv_heads=8, head_dim=256),
    m("google/codegemma-7b-it", "CodeGemma 7B IT", "Google", 8.54, 8192,
      "coding", "gemma", "2024-04", ["fim"], 3, True, "codegemma:7b",
      layers=28, hidden_size=3072, kv_heads=1, head_dim=256),
    m("google/gemma-3-4b-it", "Gemma 3 4B IT", "Google", 4.3, 131072,
      "multimodal", "gemma", "2025-03", ["vision"], 3, True, "gemma3:4b",
      layers=34, hidden_size=2560, kv_heads=4, head_dim=256),
    m("google/gemma-2-2b-it", "Gemma 2 2B IT", "Google", 2.61, 8192, "chat",
      "gemma", "2024-06", [], 3, True, "gemma2:2b", layers=26,
      hidden_size=2304, kv_heads=4, head_dim=256),
    m("google/gemma-3-1b-it", "Gemma 3 1B IT", "Google", 1.0, 32768, "chat",
      "gemma", "2025-03", [], 2, True, "gemma3:1b", layers=26,
      hidden_size=1152, kv_heads=4, head_dim=256),

    # -- Hugging Face -------------------------------------------------------
    m("mradermacher/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2 1.7B Instruct",
      "Hugging Face", 1.71, 8192, "general", "apache-2.0", "2024-11", [], 2,
      True, "smollm2:1.7b", layers=24, hidden_size=2048, kv_heads=32,
      head_dim=64),

    # -- IBM ----------------------------------------------------------------
    m("ibm-granite/granite-3.3-8b-instruct", "Granite 3.3 8B Instruct", "IBM",
      8.17, 131072, "general", "apache-2.0", "2025-04", ["tools", "fim"], 3,
      True, "granite3.3:8b", layers=40, hidden_size=4096, kv_heads=8,
      head_dim=128),
    m("ibm-granite/granite-3.3-2b-instruct", "Granite 3.3 2B Instruct", "IBM",
      2.53, 131072, "general", "apache-2.0", "2025-04", ["tools"], 2, True,
      "granite3.3:2b", layers=40, hidden_size=2048, kv_heads=8, head_dim=64),

    # -- Meta ---------------------------------------------------------------
    m("meta-llama/Llama-4-Maverick-17B-128E-Instruct",
      "Llama 4 Maverick 17B 128E", "Meta", 400.0, 1048576, "general", "llama4",
      "2025-04", ["tools", "vision", "moe"], 5, True, "llama4:maverick",
      active_params_b=17.0, layers=48, hidden_size=5120, kv_heads=8,
      head_dim=256),
    m("meta-llama/Llama-4-Scout-17B-16E-Instruct", "Llama 4 Scout 17B 16E",
      "Meta", 109.0, 524288, "general", "llama4", "2025-04",
      ["tools", "vision", "moe"], 5, True, "llama4:scout",
      active_params_b=17.0, layers=48, hidden_size=5120, kv_heads=8,
      head_dim=256),
    m("meta-llama/Llama-3.2-90B-Vision-Instruct", "Llama 3.2 90B Vision",
      "Meta", 88.8, 131072, "multimodal", "llama3.2", "2024-09", ["vision"], 5,
      True, "llama3.2-vision:90b", layers=80, hidden_size=8192, kv_heads=8,
      head_dim=128),
    m("meta-llama/Llama-3.1-70B-Instruct", "Llama 3.1 70B Instruct", "Meta",
      70.6, 131072, "general", "llama3.1", "2024-07", ["tools"], 5, True,
      "llama3.1:70b", layers=80, hidden_size=8192, kv_heads=8, head_dim=128),
    m("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B Instruct", "Meta",
      70.6, 131072, "general", "llama3.3", "2024-12", ["tools"], 5, True,
      "llama3.3:70b", layers=80, hidden_size=8192, kv_heads=8, head_dim=128),
    m("meta-llama/Llama-3.2-11B-Vision-Instruct", "Llama 3.2 11B Vision",
      "Meta", 10.7, 131072, "multimodal", "llama3.2", "2024-09", ["vision"], 4,
      True, "llama3.2-vision:11b", layers=32, hidden_size=4096, kv_heads=8,
      head_dim=128),
    m("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", "Meta",
      8.03, 131072, "general", "llama3.1", "2024-07", ["tools"], 4, True,
      "llama3.1:8b", layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
    m("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", "Meta",
      3.21, 131072, "general", "llama3.2", "2024-09", ["tools"], 4, True,
      "llama3.2:3b", layers=28, hidden_size=3072, kv_heads=8, head_dim=128),
    m("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", "Meta",
      1.24, 131072, "general", "llama3.2", "2024-09", ["tools"], 3, True,
      "llama3.2:1b", layers=16, hidden_size=2048, kv_heads=8, head_dim=64),

    # -- Microsoft ----------------------------------------------------------
    m("microsoft/Phi-3.5-MoE-instruct", "Phi-3.5 MoE 42B", "Microsoft", 41.9,
      131072, "general", "mit", "2024-08", ["moe"], 4, True, None,
      active_params_b=6.6, layers=32, hidden_size=4096, kv_heads=8,
      head_dim=128),
    m("microsoft/phi-4", "Phi-4 14B", "Microsoft", 14.7, 16384, "reasoning",
      "mit", "2024-12", [], 4, True, "phi4:14b", layers=40, hidden_size=5120,
      kv_heads=10, head_dim=128),
    m("microsoft/Phi-3-medium-4k-instruct", "Phi-3 Medium 14B", "Microsoft",
      14.0, 4096, "general", "mit", "2024-05", [], 4, True, "phi3:14b",
      layers=40, hidden_size=5120, kv_heads=10, head_dim=128),
    m("microsoft/Phi-3-small-8k-instruct", "Phi-3 Small 7B", "Microsoft", 7.39,
      8192, "general", "mit", "2024-05", [], 3, True, None, layers=32,
      hidden_size=4096, kv_heads=8, head_dim=128),
    m("microsoft/Phi-4-mini-instruct", "Phi-4 Mini 3.8B", "Microsoft", 3.84,
      131072, "general", "mit", "2025-02", ["tools"], 3, True,
      "phi4-mini:3.8b", layers=32, hidden_size=3072, kv_heads=8, head_dim=128),
    m("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini 3.8B", "Microsoft", 3.82,
      4096, "general", "mit", "2024-04", [], 3, True, "phi3:3.8b", layers=32,
      hidden_size=3072, kv_heads=32, head_dim=96),
    m("microsoft/Phi-3.5-mini-instruct", "Phi-3.5 Mini 3.8B", "Microsoft",
      3.82, 131072, "general", "mit", "2024-08", [], 3, True, "phi3.5:3.8b",
      layers=32, hidden_size=3072, kv_heads=32, head_dim=96),

    # -- Mistral AI ---------------------------------------------------------
    m("mistralai/Mixtral-8x22B-Instruct-v0.1", "Mixtral 8x22B Instruct",
      "Mistral AI", 141.0, 65536, "general", "apache-2.0", "2024-04",
      ["tools", "moe"], 4, True, "mixtral:8x22b", active_params_b=39.1,
      layers=56, hidden_size=6144, kv_heads=8, head_dim=128),
    m("mistralai/Mistral-Large-Instruct-2411", "Mistral Large 123B",
      "Mistral AI", 123.0, 131072, "general", "mnpl", "2024-11", ["tools"], 5,
      True, "mistral-large:123b", layers=88, hidden_size=12288, kv_heads=8,
      head_dim=128),
    m("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B Instruct",
      "Mistral AI", 46.7, 32768, "general", "apache-2.0", "2023-12", ["moe"],
      4, True, "mixtral:8x7b", active_params_b=12.9, layers=32,
      hidden_size=4096, kv_heads=8, head_dim=128),
    m("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral Small 3.2 24B",
      "Mistral AI", 23.6, 131072, "general", "apache-2.0", "2025-06",
      ["tools", "vision"], 5, True, "mistral-small:24b", layers=40,
      hidden_size=5120, kv_heads=8, head_dim=128),
    m("mistralai/Codestral-2501", "Codestral 22B", "Mistral AI", 22.2, 262144,
      "coding", "mnpl", "2025-01", ["fim"], 5, True, "codestral:22b",
      layers=56, hidden_size=6144, kv_heads=8, head_dim=128),
    m("mistralai/Mistral-Nemo-Instruct-2407", "Mistral NeMo 12B", "Mistral AI",
      12.2, 131072, "general", "apache-2.0", "2024-07", ["tools"], 4, True,
      "mistral-nemo:12b", layers=40, hidden_size=5120, kv_heads=8,
      head_dim=128),
    m("mistralai/Pixtral-12B-2409", "Pixtral 12B", "Mistral AI", 12.2, 131072,
      "multimodal", "apache-2.0", "2024-09", ["vision"], 4, True, None,
      layers=40, hidden_size=5120, kv_heads=8, head_dim=128),
    m("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct v0.3",
      "Mistral AI", 7.25, 32768, "general", "apache-2.0", "2024-05", ["tools"],
      3, True, "mistral:7b", layers=32, hidden_size=4096, kv_heads=8,
      head_dim=128),

    # -- Nexusflow ----------------------------------------------------------
    m("Nexusflow/Athene-V2-Chat-72B", "Athene V2 Chat 72B", "Nexusflow", 72.7,
      131072, "chat", "qwen", "2024-11", [], 5, True, "athene-v2:72b",
      layers=80, hidden_size=8192, kv_heads=8, head_dim=128),

    # -- Nomic AI -----------------------------------------------------------
    m("nomic-ai/nomic-embed-text-v1.5", "Nomic Embed Text v1.5", "Nomic AI",
      0.137, 8192, "embedding", "apache-2.0", "2024-02", ["embedding"], 4,
      True, "nomic-embed-text"),

    # -- Nous Research ------------------------------------------------------
    m("NousResearch/Hermes-3-Llama-3.1-70B", "Hermes 3 Llama 3.1 70B",
      "Nous Research", 70.6, 131072, "chat", "llama3.1", "2024-08", ["tools"],
      5, True, "hermes3:70b", layers=80, hidden_size=8192, kv_heads=8,
      head_dim=128),
    m("NousResearch/Hermes-3-Llama-3.1-8B", "Hermes 3 Llama 3.1 8B",
      "Nous Research", 8.03, 131072, "chat", "llama3.1", "2024-08", ["tools"],
      4, True, "hermes3:8b", layers=32, hidden_size=4096, kv_heads=8,
      head_dim=128),

    # -- NVIDIA -------------------------------------------------------------
    m("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", "Nemotron 70B Instruct",
      "NVIDIA", 70.6, 131072, "general", "llama3.1", "2024-10", ["tools"], 5,
      True, "nemotron:70b", layers=80, hidden_size=8192, kv_heads=8,
      head_dim=128),
    m("nvidia/Mistral-NeMo-Minitron-8B-Instruct", "Minitron 8B Instruct",
      "NVIDIA", 8.41, 8192, "general", "nvidia", "2024-08", [], 3, True, None,
      layers=40, hidden_size=4096, kv_heads=8, head_dim=128),

    # -- Sentence Transformers ----------------------------------------------
    m("sentence-transformers/all-MiniLM-L6-v2", "All-MiniLM-L6-v2",
      "Sentence Transformers", 0.023, 256, "embedding", "apache-2.0",
      "2021-08", ["embedding"], 2, True, "all-minilm:l6-v2"),

    # -- Shanghai AI Lab ----------------------------------------------------
    m("internlm/internlm2_5-20b-chat", "InternLM 2.5 20B Chat",
      "Shanghai AI Lab", 20.0, 32768, "general", "apache-2.0", "2024-07",
      ["tools"], 4, True, "internlm2:20b", layers=48, hidden_size=6144,
      kv_heads=8, head_dim=128),
    m("internlm/internlm2_5-7b-chat", "InternLM 2.5 7B Chat",
      "Shanghai AI Lab", 7.74, 32768, "general", "apache-2.0", "2024-07",
      ["tools"], 3, True, "internlm2:7b", layers=32, hidden_size=4096,
      kv_heads=8, head_dim=128),

    # -- Snowflake ----------------------------------------------------------
    m("Snowflake/snowflake-arctic-embed-l-v2.0", "Arctic Embed L v2",
      "Snowflake", 0.568, 8192, "embedding", "apache-2.0", "2024-11",
      ["embedding"], 4, True, "snowflake-arctic-embed2"),

    # -- Stability AI -------------------------------------------------------
    m("stabilityai/stable-code-instruct-3b", "Stable Code 3B", "Stability AI",
      2.8, 16384, "coding", "stability-ai", "2024-01", ["fim"], 2, True,
      "stable-code:3b", layers=32, hidden_size=2560, kv_heads=32, head_dim=80),

    # -- TII ----------------------------------------------------------------
    m("TII/falcon-11B", "Falcon 11B", "TII", 11.0, 8192, "general",
      "tii-falcon", "2024-05", [], 3, True, "falcon2:11b", layers=60,
      hidden_size=4096, kv_heads=8, head_dim=64),
    m("tiiuae/Falcon3-10B-Instruct", "Falcon 3 10B Instruct", "TII", 10.3,
      32768, "general", "tii-falcon", "2024-12", [], 4, True, "falcon3:10b",
      layers=40, hidden_size=3072, kv_heads=4, head_dim=256),
    m("tiiuae/Falcon3-7B-Instruct", "Falcon 3 7B Instruct", "TII", 7.49, 32768,
      "general", "tii-falcon", "2024-12", [], 3, True, "falcon3:7b", layers=28,
      hidden_size=3072, kv_heads=4, head_dim=256),

    # -- Upstage ------------------------------------------------------------
    m("upstage/solar-pro-preview-instruct", "Solar Pro 22B", "Upstage", 22.1,
      4096, "general", "mit", "2024-09", [], 3, True, "solar-pro:22b",
      layers=64, hidden_size=5120, kv_heads=10, head_dim=128),

    # -- Valiant Labs -------------------------------------------------------
    m("ValiantLabs/Llama3.1-8B-Enigma", "Llama 3.1 8B Enigma", "Valiant Labs",
      8.03, 131072, "chat", "llama3.1", "2024-08", [], 3, True, None,
      layers=32, hidden_size=4096, kv_heads=8, head_dim=128),
]


def build(db):
    """The catalog `db` becomes once MODELS is applied to it.

    Existing records are replaced in place and new ones appended before the
    stable sort, so records that tie on the sort key keep their relative order
    from one run to the next. Records with no entry in MODELS are dropped.
    """
    wanted = {entry["id"]: entry for entry in MODELS}
    if len(wanted) != len(MODELS):
        seen, dupes = set(), set()
        for entry in MODELS:
            (dupes if entry["id"] in seen else seen).add(entry["id"])
        sys.exit(f"duplicate ids in MODELS: {', '.join(sorted(dupes))}")

    kept = [wanted[e["id"]] for e in db["models"] if e["id"] in wanted]
    present = {e["id"] for e in kept}
    added = [entry for entry in MODELS if entry["id"] not in present]
    dropped = [e["id"] for e in db["models"] if e["id"] not in wanted]

    models = kept + added
    models.sort(key=lambda e: (e["provider"].lower(), -e["params_b"]))
    return {**db, "models": models}, added, dropped


def render(db):
    return json.dumps(db, indent=2, ensure_ascii=False) + "\n"


def drift(current, built):
    """One line per record that differs between the catalog and this file."""
    old = {e["id"]: e for e in current["models"]}
    new = {e["id"]: e for e in built["models"]}
    lines = []
    for rid in sorted(old.keys() | new.keys()):
        if rid not in new:
            lines.append(f"  {rid}: in data/models.json but not in this file")
        elif rid not in old:
            lines.append(f"  {rid}: in this file but not in data/models.json")
        elif old[rid] != new[rid]:
            fields = sorted(k for k in old[rid].keys() | new[rid].keys()
                            if old[rid].get(k) != new[rid].get(k))
            lines.append(f"  {rid}: {', '.join(fields)}")
    if not lines:
        lines.append("  record order or formatting")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if data/models.json differs from this file, without writing",
    )
    args = parser.parse_args()

    text = DB.read_text(encoding="utf-8")
    current = json.loads(text)
    built, added, dropped = build(current)
    rendered = render(built)

    if args.check:
        if rendered == text:
            print(f"data/models.json matches scripts/add_models.py ({len(MODELS)} records)")
            return
        print("data/models.json has drifted from scripts/add_models.py:", file=sys.stderr)
        for line in drift(current, built):
            print(line, file=sys.stderr)
        print("Make the change in scripts/add_models.py and run it.", file=sys.stderr)
        sys.exit(1)

    # Bytes, not text: text mode on Windows would write CRLF line endings.
    DB.write_bytes(rendered.encode("utf-8"))
    providers = {e["provider"] for e in built["models"]}
    print(f"added {len(added)}, dropped {len(dropped)}, "
          f"{len(built['models']) - len(added)} kept")
    print(f"{len(built['models'])} models, {len(providers)} providers")


if __name__ == "__main__":
    main()
