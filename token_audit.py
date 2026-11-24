"""
`smarty audit`: where do an agent session's tokens actually go, and what is worth optimizing?

Reads real Claude Code transcripts and prices every token with the provider cost model:

    J = sum over API calls t of [ input_t + Wcw * cache_write_t + Wcr * cache_read_t ] + Wout * output_t

(defaults Wcw=1.25, Wcr=0.10, Wout=5.0 relative to the base input price: ASSUMPTIONS, override on the command line).

Key consequences, all computed here from your own data rather than assumed:
  * A token in the cached prefix costs Wcw once and Wcr on every later call, so   unit_cost = Wcw + Wcr * (n - 1)
    (n = API calls in the session). Cutting 1 prefix token saves unit_cost; cutting 1 *call* saves Wcr * prefix.
  * With a prefix of ~70k tokens the fixed prefix is typically >80% of a short session, so tool-output compression and
    prompt compression can only ever touch the small remainder (an independent JetBrains test of RTK found no net saving).
  * The user-controllable slice of the prefix (instruction files, skill listing, connector/MCP names, agent list) is
    attributed per source so the recommendations are concrete.
Stdlib only; no network; no LLM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Any

os.environ.setdefault("SMARTY_LIGHT_TOKENS", "1")

import ab_compare as ab  # noqa: E402
import repo_indexer as ri  # noqa: E402

DEFAULT_WEIGHTS = {"cw": 1.25, "cr": 0.10, "out": 5.0}
HOST_ATTACHMENTS = {"prompt_snapshot"}  # a transcript copy of the host's own system prompt: not user-controllable
CONTROLLABLE = ("instructions", "skill_listing", "deferred_tools_delta", "agent_listing_delta", "mcp_instructions_delta")


def _att_text(a: dict) -> str:
    if a.get("type") == "instructions" and isinstance(a.get("files"), list):
        return "\n".join(str(x.get("content", "")) for x in a["files"] if isinstance(x, dict))
    c = a.get("content")
    if isinstance(c, str):
        return c
    return json.dumps(c if c is not None else {k: v for k, v in a.items() if k != "type"}, ensure_ascii=False)


def analyze_session(path: str, w: dict[str, float] | None = None) -> dict[str, Any] | None:
    """Cost decomposition + first-turn attribution for one transcript. None if it has fewer than 2 API calls."""
    w = w or DEFAULT_WEIGHTS
    usage: dict[str, dict] = {}
    order: list[str] = []
    atts: dict[str, int] = {}
    instr_files: dict[str, int] = {}
    results: dict[str, int] = {}
    call_of_tool: dict[str, int] = {}
    user_tokens = 0
    seen_assistant = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("type")
            if t == "assistant":
                seen_assistant = True
                m = d.get("message", {})
                if m.get("id") and m.get("usage") and m["id"] not in usage:
                    usage[m["id"]] = m["usage"]
                    order.append(m["id"])
                for b in m.get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        call_of_tool[b["id"]] = len(order)
            elif t == "attachment" and not seen_assistant:
                a = d.get("attachment", {})
                typ = a.get("type", "?")
                atts[typ] = atts.get(typ, 0) + ri.est_tokens(_att_text(a))
                if typ == "instructions" and isinstance(a.get("files"), list):
                    for x in a["files"]:
                        if isinstance(x, dict):
                            instr_files[x.get("path", "?")] = ri.est_tokens(str(x.get("content", "")))
            elif t == "user":
                c = d.get("message", {}).get("content")
                if isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            results[b.get("tool_use_id", "")] = ri.est_tokens(ab._result_text(b.get("content")))
                elif not seen_assistant and not user_tokens:
                    user_tokens = ri.est_tokens(ab._prompt_text(d) or "")
                if not seen_assistant and not user_tokens and isinstance(c, list):
                    user_tokens = ri.est_tokens(ab._prompt_text(d) or "")
    U = [usage[i] for i in order]
    n = len(U)
    if n < 2:
        return None

    def total_in(u: dict) -> float:
        return u.get("input_tokens", 0) + w["cw"] * u.get("cache_creation_input_tokens", 0) + w["cr"] * u.get("cache_read_input_tokens", 0)

    P = sum(U[0].get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    cost_in = sum(total_in(u) for u in U)
    out = w["out"] * sum(u.get("output_tokens", 0) for u in U)
    fixed = total_in(U[0]) + (n - 1) * w["cr"] * P
    host = atts.get("prompt_snapshot", 0)
    layer = {k: v for k, v in atts.items() if k not in HOST_ATTACHMENTS}
    unit = w["cw"] + w["cr"] * (n - 1)
    return {
        "path": path, "calls": n, "prefix_tokens": P,
        "prefix_cache_write": U[0].get("cache_creation_input_tokens", 0), "prefix_cache_read": U[0].get("cache_read_input_tokens", 0),
        "cost_total": cost_in + out, "cost_fixed_prefix": fixed, "cost_growth": cost_in - fixed, "cost_output": out,
        "unit_cost_per_prefix_token": unit, "turn_cost": w["cr"] * P,
        "user_prompt_tokens": user_tokens, "host_system_prompt_tokens": host,
        "layer_tokens": layer, "instruction_files": instr_files,
        "tool_definitions_and_other_tokens": max(0, P - sum(layer.values()) - host - user_tokens),
        "tool_result_tokens": sum(results.values()),
        "tool_result_cost": sum(tok * (w["cw"] + w["cr"] * max(0, n - call_of_tool.get(tid, n))) for tid, tok in results.items()),
    }


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in rows)


def aggregate(rows: list[dict], cold: bool = False, w: dict[str, float] | None = None) -> dict[str, Any]:
    w = w or DEFAULT_WEIGHTS
    n = _mean(rows, "calls")
    P = _mean(rows, "prefix_tokens")
    cost = _mean(rows, "cost_total")
    if cold:  # the cache expired between sessions: the whole prefix is a write at call 1, then reads
        delta = (w["cw"] - w["cr"]) * P - (_mean(rows, "prefix_cache_write") * (w["cw"] - w["cr"]))
        cost += delta
    unit = w["cw"] + w["cr"] * (n - 1)
    layer_keys = {k for r in rows for k in r["layer_tokens"]}
    layer = {k: statistics.mean(r["layer_tokens"].get(k, 0) for r in rows) for k in layer_keys}
    files: dict[str, float] = {}
    for r in rows:
        for p, t in r["instruction_files"].items():
            files[p] = files.get(p, 0) + t / len(rows)
    return {
        "sessions": len(rows), "mean_calls": n, "mean_prefix_tokens": P, "cold_cache": cold, "mean_cost_total": cost,
        "share_fixed_prefix": _mean(rows, "cost_fixed_prefix") / cost if not cold else None,
        "share_growth": _mean(rows, "cost_growth") / cost, "share_output": _mean(rows, "cost_output") / cost,
        "share_tool_results": _mean(rows, "tool_result_cost") / cost,
        "share_user_prompt": _mean(rows, "user_prompt_tokens") * unit / cost,
        "unit_cost": unit, "turn_cost": w["cr"] * P, "turn_share": w["cr"] * P / cost,
        "host_system_prompt_tokens": _mean(rows, "host_system_prompt_tokens"),
        "tool_definitions_and_other_tokens": _mean(rows, "tool_definitions_and_other_tokens"),
        "layer_tokens": layer, "instruction_files": files,
        "layer_cost": {k: v * unit for k, v in layer.items()}, "layer_share": {k: v * unit / cost for k, v in layer.items()},
    }


def recommendations(agg: dict[str, Any], reduction: float = 0.5) -> list[dict]:
    """Rule-based, with the projected saving of cutting each controllable source by `reduction`."""
    unit, cost = agg["unit_cost"], agg["mean_cost_total"]
    recs = []

    def add(source: str, tokens: float, action: str) -> None:
        if tokens >= 300:
            recs.append({"source": source, "tokens": round(tokens), "per_session_cost": round(tokens * unit),
                         "projected_saving_pct": round(100 * tokens * reduction * unit / cost, 1), "action": action})

    lt = agg["layer_tokens"]
    for p, t in sorted(agg["instruction_files"].items(), key=lambda kv: -kv[1]):
        add(f"instructions: {p}", t, "loaded into EVERY session: keep only rules the agent cannot infer; deduplicate with project files")
    add("skill listing", lt.get("skill_listing", 0), "every installed skill's name+description is paid per session: remove skills/plugins you do not use")
    add("connector / MCP tool names", lt.get("deferred_tools_delta", 0), "disable MCP servers and connectors this project does not need")
    add("subagent listing", lt.get("agent_listing_delta", 0), "remove unused custom agents")
    add("MCP server instructions", lt.get("mcp_instructions_delta", 0), "servers add instruction text per session: trim or disable unused ones")
    recs.sort(key=lambda r: -r["per_session_cost"])
    return recs


def required_pairs(sd_pct: float, effect_pct: float, z_alpha: float = 1.96, z_power: float = 0.84) -> int:
    """Paired-t sample size (normal approximation) to detect a mean paired change of `effect_pct` given its sd."""
    return max(2, math.ceil(((z_alpha + z_power) * sd_pct / abs(effect_pct)) ** 2))


def report(rows: list[dict], w: dict[str, float]) -> dict[str, Any]:
    warm, cold = aggregate(rows, False, w), aggregate(rows, True, w)
    print(f"sessions: {warm['sessions']}   mean API calls: {warm['mean_calls']:.1f}   prefix at call 1: {warm['mean_prefix_tokens']:.0f} tokens")
    print(f"weights: cache-write x{w['cw']}, cache-read x{w['cr']}, output x{w['out']}   (cost in base-input-token equivalents)\n")
    print(f"{'where the cost goes (warm cache)':44}{'share':>8}")
    for lab, k in (("fixed prefix (system prompt, tools, instructions...)", "share_fixed_prefix"),
                   ("  of which tool results + messages that persist", "share_tool_results"),
                   ("context growth (everything appended)", "share_growth"), ("output tokens", "share_output"),
                   ("your typed prompt", "share_user_prompt")):
        print(f"{lab:44}{100 * warm[k]:>7.1f}%")
    print(f"\none avoided API call saves {warm['turn_cost']:.0f} weighted tokens = {100 * warm['turn_share']:.1f}% of an average session")
    print(f"one prefix token costs {warm['unit_cost']:.2f} weighted tokens per session (write once + {warm['mean_calls'] - 1:.1f} re-reads)")
    print(f"cold cache (prefix expired between sessions): mean session cost {cold['mean_cost_total']:.0f} vs {warm['mean_cost_total']:.0f} warm "
          f"({100 * (cold['mean_cost_total'] / warm['mean_cost_total'] - 1):+.0f}%)")
    print(f"\nprefix composition: host system prompt ~{warm['host_system_prompt_tokens']:.0f}, tool definitions + other host overhead ~"
          f"{warm['tool_definitions_and_other_tokens']:.0f} (not controllable from a project), user layer below (controllable)")
    print(f"{'user-controllable source':34}{'tokens':>8}{'cost/session':>14}{'share':>8}")
    for k in CONTROLLABLE:
        if warm["layer_tokens"].get(k):
            print(f"{k:34}{warm['layer_tokens'][k]:>8.0f}{warm['layer_cost'][k]:>14.0f}{100 * warm['layer_share'][k]:>7.1f}%")
    recs = recommendations(warm)
    print("\nrecommendations (projected saving if the source is cut by 50%):")
    for r in recs:
        print(f"  - {r['source']}: {r['tokens']} tokens, ~{r['per_session_cost']} weighted/session -> up to -{r['projected_saving_pct']}%\n      {r['action']}")
    total_pct = sum(r["projected_saving_pct"] for r in recs)
    print(f"\nupper bound from trimming all of the above by 50%: about -{total_pct:.1f}% of session cost")
    return {"warm": warm, "cold": cold, "recommendations": recs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="token_audit", description=__doc__.split("\n")[1])
    ap.add_argument("paths", nargs="*", help=".jsonl transcripts and/or project directories")
    ap.add_argument("--match", help="select Claude Code project directories by name substring")
    ap.add_argument("--last", type=int, help="newest N sessions")
    ap.add_argument("--w-cw", type=float, default=DEFAULT_WEIGHTS["cw"])
    ap.add_argument("--w-cr", type=float, default=DEFAULT_WEIGHTS["cr"])
    ap.add_argument("--w-out", type=float, default=DEFAULT_WEIGHTS["out"])
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--power", nargs=2, type=float, metavar=("SD_PCT", "EFFECT_PCT"),
                    help="print the paired sample size needed to detect EFFECT_PCT given SD_PCT, then exit")
    a = ap.parse_args(argv)
    if a.power:
        print(f"paired runs needed (alpha=0.05, power=0.8): {required_pairs(*a.power)}")
        return 0
    try:
        paths = ab.expand_inputs(a.paths, a.last, a.match)
    except ab.InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    w = {"cw": a.w_cw, "cr": a.w_cr, "out": a.w_out}
    rows = [r for r in (analyze_session(p, w) for p in paths) if r]
    if not rows:
        print("error: no session with at least 2 API calls found", file=sys.stderr)
        return 2
    out = report(rows, w)
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
