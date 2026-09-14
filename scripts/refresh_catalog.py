#!/usr/bin/env python3
"""Check the shipped catalog against the outside world.

    python scripts/refresh_catalog.py                 # human-readable report
    python scripts/refresh_catalog.py --json          # machine-readable
    python scripts/refresh_catalog.py --markdown out.md

Three questions, all of which go stale on their own while the file sits still:

1. Do the Ollama tags we print still resolve? A renamed or withdrawn tag means
   llmspec hands someone a `ollama pull` command that fails.
2. Do the Hugging Face repos we name still exist? Our ids *are* HF ids, so this
   is an exact lookup rather than a guess.
3. Which widely-downloaded models are missing from the catalog?

Only the first two are defects. The third is a reading list: this script never
edits data/models.json, because deciding a model is worth shipping — and
finding its real geometry — is the judgement that makes the catalog worth
having. It tells you what to look at; scripts/add_models.py is where a
decision gets written down.

Standard library only, so CI needs no install step.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "models.json"

OLLAMA_MANIFEST = "https://registry.ollama.ai/v2/library/{name}/manifests/{tag}"
HF_MODEL = "https://huggingface.co/api/models/{id}"
HF_TOP = (
    "https://huggingface.co/api/models"
    "?sort=downloads&direction=-1&limit={limit}&filter=text-generation"
)

USER_AGENT = "llmspec-catalog-check"
TIMEOUT = 20
WORKERS = 6

# Repos that rank high on downloads without being models anyone would run:
# test fixtures, and re-uploads of someone else's weights in one quantization.
NOISE = (
    "testing",
    "tiny-",
    "-tiny",
    "gguf",
    "awq",
    "gptq",
    "-bnb-",
    "int4",
    "int8",
    "-fp8",
    "mlx-community/",
    "unsloth/",
)


def status(url, attempts=3):
    """HTTP status for `url`, or 0 when the request could not be made.

    Retried, because these checks turn into a defect report: a connection
    reset from hammering a registry with a thread pool must not be published
    as "this model is gone". Only a definite answer counts, and 404 is the
    only one this script acts on.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.status
        except urllib.error.HTTPError as e:
            # 404 is an answer; 429 and 5xx are the server asking us to wait.
            if e.code < 500 and e.code != 429:
                return e.code
        except Exception:
            pass
        if attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
    return 0


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def check_ollama_tags(models):
    """Tags in the catalog that the Ollama registry no longer serves."""
    tagged = [m for m in models if m.get("ollama")]

    def check(model):
        reference = model["ollama"]
        name, _, tag = reference.partition(":")
        code = status(OLLAMA_MANIFEST.format(name=name, tag=tag or "latest"))
        return model, reference, code

    stale, unreachable = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for model, reference, code in pool.map(check, tagged):
            if code == 404:
                stale.append({"id": model["id"], "name": model["name"], "tag": reference})
            elif code == 0 or code >= 500:
                unreachable += 1
    stale.sort(key=lambda e: e["id"])
    return {"checked": len(tagged), "stale": stale, "unreachable": unreachable}


def check_hf_repos(models):
    """Catalog entries whose Hugging Face repo has gone."""

    def check(model):
        return model, status(HF_MODEL.format(id=model["id"]))

    missing, unreachable = [], 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for model, code in pool.map(check, models):
            if code == 404:
                missing.append({"id": model["id"], "name": model["name"]})
            elif code == 0 or code >= 500:
                unreachable += 1
    missing.sort(key=lambda e: e["id"])
    return {"checked": len(models), "missing": missing, "unreachable": unreachable}


def find_candidates(models, limit, want):
    """Widely-downloaded models the catalog does not carry."""
    have = {m["id"].lower() for m in models}
    try:
        top = fetch_json(HF_TOP.format(limit=limit))
    except Exception as e:
        return {"error": str(e), "candidates": []}

    candidates = []
    for entry in top:
        model_id = entry.get("id", "")
        lowered = model_id.lower()
        if lowered in have or entry.get("gated") or entry.get("private"):
            continue
        if any(token in lowered for token in NOISE):
            continue
        if entry.get("pipeline_tag") != "text-generation":
            continue
        candidates.append(
            {
                "id": model_id,
                "downloads": entry.get("downloads", 0),
                "likes": entry.get("likes", 0),
            }
        )
        if len(candidates) >= want:
            break
    return {"scanned": len(top), "candidates": candidates}


def render(report):
    """A markdown report, which is also the body of the tracking issue."""
    tags, repos, new = report["ollama"], report["huggingface"], report["candidates"]
    out = []

    out.append(f"The catalog holds **{report['models']} models** from "
               f"**{report['providers']} providers**.\n")

    out.append("## Ollama tags\n")
    if tags["stale"]:
        out.append(
            f"{len(tags['stale'])} of {tags['checked']} tags no longer resolve. "
            "llmspec prints these as the command to run, so each one is a "
            "`pull` that fails for the user:\n"
        )
        out.append("| Model | Tag |")
        out.append("|---|---|")
        for entry in tags["stale"]:
            out.append(f"| {entry['name']} | `{entry['tag']}` |")
        out.append("")
    else:
        out.append(f"All {tags['checked']} tags resolve.\n")
    if tags["unreachable"]:
        out.append(f"_{tags['unreachable']} could not be reached and were not judged._\n")

    out.append("## Hugging Face repos\n")
    if repos["missing"]:
        out.append(f"{len(repos['missing'])} of {repos['checked']} repos are gone:\n")
        for entry in repos["missing"]:
            out.append(f"- `{entry['id']}` ({entry['name']})")
        out.append("")
    else:
        out.append(f"All {repos['checked']} repos exist.\n")
    if repos["unreachable"]:
        out.append(f"_{repos['unreachable']} could not be reached and were not judged._\n")

    out.append("## Candidates\n")
    if new.get("error"):
        out.append(f"The listing could not be fetched: {new['error']}\n")
    elif new["candidates"]:
        out.append(
            "Widely-downloaded models the catalog does not carry. These are for "
            "reading, not merging — a record is only worth shipping once its "
            "geometry is known, which is what `scripts/add_models.py` is for.\n"
        )
        out.append("| Model | Downloads | Likes |")
        out.append("|---|---:|---:|")
        for entry in new["candidates"]:
            out.append(f"| `{entry['id']}` | {entry['downloads']:,} | {entry['likes']:,} |")
        out.append("")
    else:
        out.append("Nothing in the top downloads is missing from the catalog.\n")

    return "\n".join(out)


def main():
    # The report contains characters a cp1252 console would refuse.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the raw findings")
    parser.add_argument("--markdown", type=Path, help="also write the report to this file")
    parser.add_argument("--scan", type=int, default=200, help="HF models to scan (default 200)")
    parser.add_argument("--candidates", type=int, default=15, help="candidates to list")
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="exit non-zero when a tag or repo has gone (candidates never fail)",
    )
    args = parser.parse_args()

    db = json.loads(DB.read_text(encoding="utf-8"))
    models = db["models"]

    report = {
        "models": len(models),
        "providers": len({m["provider"] for m in models}),
        "ollama": check_ollama_tags(models),
        "huggingface": check_hf_repos(models),
        "candidates": find_candidates(models, args.scan, args.candidates),
    }
    report["stale_count"] = len(report["ollama"]["stale"]) + len(
        report["huggingface"]["missing"]
    )

    markdown = render(report)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2) if args.json else markdown)

    if args.fail_on_stale and report["stale_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
