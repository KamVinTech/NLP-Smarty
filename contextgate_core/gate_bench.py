"""
Offline replay: run recorded (or synthetic) OpenAI-format requests through the gate and report MEASURED savings, per
profile and per transform category, with the safety counters. No network. This is how ROI gets proven on YOUR traffic:

  python gateway.py --upstream ... --record traffic.jsonl     # capture originals (contains your prompts: keep private)
  python gate_bench.py traffic.jsonl                          # replay offline

Input: JSONL where each line is a request body or {"ts":..., "body": {...}} (the --record format), or a JSON array.
`--from-docs DIR` builds SYNTHETIC RAG-style requests from real text files in DIR (clearly labelled as synthetic).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
from typing import Any

from . import gate

CATEGORIES = ("user_text", "tool_schemas", "duplicates")


def load_requests(path: str) -> list[dict]:
    raw = open(path, encoding="utf-8").read().strip()
    if not raw:
        return []
    if raw[0] == "[":
        items = json.loads(raw)
    else:
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return [i["body"] if isinstance(i, dict) and isinstance(i.get("body"), dict) else i for i in items if isinstance(i, dict)]


def synthetic_from_docs(root: str, n: int = 30, chunks: int = 6, seed: int = 7, api: str = "openai") -> list[dict]:
    """RAG-style requests: real paragraphs from real files as context, a polite filler-laden question, a tool schema."""
    paras: list[str] = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in {"node_modules", ".git", ".venv", "NLP-start", "__pycache__", ".smarty", "dist", "out"}]
        for fn in fns:
            if fn.lower().endswith((".md", ".txt", ".rst")):
                try:
                    text = open(os.path.join(dp, fn), encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                paras += [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 40]
    if not paras:
        return []
    rng = random.Random(seed)
    qs = ["Could you please explain how this works? I was wondering if you could basically summarise the key points.",
          "I would really appreciate it if you could help me understand the setup, in order to configure it properly.",
          "Can you please tell me what to change, and do not suggest anything outside of these documents?"]
    tools = [{"type": "function", "function": {"name": "search_docs", "description": "Could you please use this tool in order to search the "
              "documentation for the information that the user is asking about.", "parameters": {"type": "object",
              "properties": {"query": {"type": "string", "description": "I would like you to provide the search query to run."}},
              "required": ["query"]}}}]
    out = []
    for _ in range(n):
        ctx = "\n\n".join(f"[doc {i + 1}]\n{rng.choice(paras)}" for i in range(chunks))
        if api == "anthropic":
            a_tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"]} for t in tools]
            out.append({"model": "synthetic", "max_tokens": 512, "system": [{"type": "text", "text": "You answer using only the provided documents."}],
                        "messages": [{"role": "user", "content": [{"type": "text", "text": f"{ctx}\n\n{rng.choice(qs)}"}]}], "tools": a_tools})
            continue
        out.append({"model": "synthetic", "messages": [{"role": "system", "content": "You answer using only the provided documents."},
                    {"role": "user", "content": f"{ctx}\n\n{rng.choice(qs)}"}], "tools": tools})
    return out


def run(requests: list[dict], profile: str) -> dict[str, Any]:
    raw = fwd = fallbacks = errors = applied = 0
    apis: dict[str, int] = {}
    cats = {c: [0, 0] for c in CATEGORIES}
    per, ms = [], []
    for body in requests:
        api = gate.detect_api(body)
        apis[api] = apis.get(api, 0) + 1
        r = gate.gate_request(body, profile, api)
        raw += r.raw_tokens
        fwd += r.fwd_tokens
        fallbacks += r.fallbacks
        errors += 1 if r.error else 0
        applied += 1 if r.applied else 0
        ms.append(r.ms)
        per.append(100 * r.saved / r.raw_tokens if r.raw_tokens else 0.0)
        for c, (a, b) in r.by_category.items():
            cats[c][0] += a
            cats[c][1] += b
    ms.sort()
    return {"profile": profile, "requests": len(requests), "apis": apis, "applied": applied, "raw_tokens": raw, "fwd_tokens": fwd,
            "saved_tokens": raw - fwd, "saved_pct": round(100 * (raw - fwd) / raw, 2) if raw else 0.0,
            "per_request_saved_pct": {"min": round(min(per), 2), "median": round(statistics.median(per), 2), "max": round(max(per), 2)} if per else {},
            "categories": {c: {"before": a, "after": b, "saved": a - b} for c, (a, b) in cats.items()},
            "verification_fallbacks": fallbacks, "errors": errors,
            "gate_ms_p50": round(statistics.median(ms), 2) if ms else 0.0, "gate_ms_p95": round(ms[int(0.95 * (len(ms) - 1))], 2) if ms else 0.0}


def format_report(results: list[dict], label: str) -> str:
    mix = ", ".join(f"{n} {k}" for k, n in sorted(results[0].get("apis", {}).items()))
    L = [f"{label}: {results[0]['requests']} requests ({mix}), {results[0]['raw_tokens']} estimated input tokens before gating", ""]
    L.append(f"{'profile':12}{'saved':>9}{'%':>8}{'median/req':>12}{'max/req':>9}  {'user text':>10}{'schemas':>9}{'dupes':>8}{'fallbacks':>10}{'p95 ms':>8}")
    for r in results:
        c = r["categories"]
        L.append(f"{r['profile']:12}{r['saved_tokens']:>9}{r['saved_pct']:>7.2f}%{r['per_request_saved_pct'].get('median', 0):>11.2f}%"
                 f"{r['per_request_saved_pct'].get('max', 0):>8.2f}%  {c['user_text']['saved']:>10}{c['tool_schemas']['saved']:>9}"
                 f"{c['duplicates']['saved']:>8}{r['verification_fallbacks']:>10}{r['gate_ms_p95']:>8}")
    L.append("\nunits: estimated tokens (pipeline.estimate_tokens). Savings are for whatever workload you replayed: "
             "re-run on your own recorded traffic before quoting a number.")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench", description=__doc__.split("\n")[1])
    ap.add_argument("file", nargs="?", help="JSONL/JSON of recorded requests")
    ap.add_argument("--from-docs", metavar="DIR", help="build SYNTHETIC RAG requests from real text files in DIR")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--api", choices=("openai", "anthropic"), default="openai", help="shape of the --from-docs synthetic requests")
    ap.add_argument("--profiles", nargs="+", default=list(gate.PROFILES), choices=sorted(gate.PROFILES))
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args(argv)
    try:
        if a.from_docs:
            reqs, label = synthetic_from_docs(a.from_docs, a.n, api=a.api), f"SYNTHETIC {a.api} requests built from real documents in {a.from_docs}"
        elif a.file:
            reqs, label = load_requests(a.file), f"recorded requests from {a.file}"
        else:
            print("error: give a recorded-requests file or --from-docs DIR", file=sys.stderr)
            return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not reqs:
        print("error: no requests found", file=sys.stderr)
        return 2
    results = [run(reqs, p) for p in a.profiles]
    print(format_report(results, label))
    if a.json_out:
        json.dump(results, open(a.json_out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
