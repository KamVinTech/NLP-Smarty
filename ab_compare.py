"""
A/B comparison for NLP-Smarty: does the retrieval capsule actually reduce what an agent reads/spends?

Two complementary layers (neither alone is proof):

  simulate     Offline, deterministic, zero-LLM MODEL of an agent locating code with vs. without smarty.
               Baseline agent = grep distinctive keywords, then read matching files (best-ranked first) until the target
               file is read. Savings are EXPLORATION avoided minus capsule overhead; a miss or a silent gate saves
               nothing (a wrong capsule is pure cost). Assumptions are explicit and conservative (see below).

  transcripts  Real measurement from Claude Code session transcripts (~/.claude/projects/<proj>/<session>.jsonl):
               tokens by component, exploration tool calls/result size before the first edit, smarty usage.
               Compare groups of runs (A = baseline, B = smarty).

Protocol for a real A/B (the only evidence that settles the question):
  1. Same commit, same model, same prompt text, fresh session per run. >=5 distinct tasks x >=3 runs per arm
     (LLM runs vary a lot; one pair of runs proves nothing).
  2. Arm A (baseline): a `git worktree add ../repo-baseline` checkout. It lacks the untracked .claude/.mcp.json, so no
     hook and no MCP tools. Arm B: the main checkout where `smarty init` was run.
  3. Finish each run (or stop at the same point, e.g. first successful edit), then:
       python ab_compare.py transcripts --a A1.jsonl A2.jsonl --b B1.jsonl B2.jsonl
Stdlib only; no network; no LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from typing import Any

os.environ.setdefault("SMARTY_LIGHT_TOKENS", "1")  # same estimator for both arms; --bpe switches to the BPE-calibrated one

import repo_indexer as ri  # noqa: E402

MAX_READS = 12          # a baseline agent that has not found the target after this many reads keeps going: cap the model
GREP_LINES = 20         # matching lines shown per grep
MAX_KEYWORDS = 4        # distinctive keywords a baseline agent greps for


# ── label loading ─────────────────────────────────────────────────────────────
def load_labels(path: str) -> tuple[list[dict], list[str]]:
    """Accepts tests/golden_queries.json style ([{q, expect: ["path:Symbol"]}]) or {"code": [...], "silent": [...]}."""
    data = json.load(open(path))
    code, silent = (data.get("code", []), data.get("silent", [])) if isinstance(data, dict) else (data, [])
    norm = [{"q": c["q"], "expect": sorted({e.split(":")[0] for e in c["expect"]})} for c in code]
    return norm, list(silent)


# ── offline simulation ────────────────────────────────────────────────────────
class RepoText:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.text: dict[str, str] = {}
        for rel in ri.iter_repo_files(self.root):
            try:
                with open(os.path.join(self.root, rel), encoding="utf-8", errors="replace") as f:
                    self.text[rel] = f.read()
            except OSError:
                pass
        self.lower = {p: t.lower() for p, t in self.text.items()}
        self.tokens = {p: ri.est_tokens(t) for p, t in self.text.items()}


def path_candidates(repo: RepoText, words: set[str]) -> tuple[list[str], int]:
    """Files an agent would find by globbing file names for query words (4-char stems: 'healing' -> *heal*).
    Words that match many paths (src, test, utils) are ignored. Returns (ordered paths, tokens of the glob listing)."""
    stems = {w[:4] if len(w) >= 5 else w for w in words}
    flat = {p: p.lower() for p in repo.lower}
    n = max(1, len(flat))
    usable = {s for s in stems if 0 < sum(s in f for f in flat.values()) <= n * 0.2}
    scored = [(-sum(s in f for s in usable), p) for p, f in flat.items() if any(s in f for s in usable)]
    order = [p for _, p in sorted(scored)]
    return order, ri.est_tokens("\n".join(order[:25]))


def baseline_explore(repo: RepoText, query: str, expect: list[str], path_discovery: bool = True) -> dict[str, Any] | None:
    """Model of a grep-then-read agent (optionally also globbing file names first). None = nothing to grep for."""
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", query.lower()) if w not in ri.STOPWORDS}
    n = max(1, len(repo.lower))
    df = {w: sum(1 for t in repo.lower.values() if w in t) for w in words}
    kws = sorted((w for w in words if 0 < df[w] <= n * 0.5), key=lambda w: (df[w], w))[:MAX_KEYWORDS]
    if not kws:
        return None
    grep_cost = 0
    for k in kws:  # each grep returns up to GREP_LINES "path:line:text" rows
        shown = 0
        for p, t in repo.lower.items():
            for i, ln in enumerate(t.split("\n"), 1):
                if k in ln:
                    grep_cost += ri.est_tokens(f"{p}:{i}:{repo.text[p].split(chr(10))[i - 1][:160]}")
                    shown += 1
                    if shown >= GREP_LINES:
                        break
            if shown >= GREP_LINES:
                break
    scored = []
    for p, t in repo.lower.items():
        hit = [k for k in kws if k in t]
        if hit:
            scored.append((-len(hit), -sum(t.count(k) for k in hit), p))
    order = [p for _, _, p in sorted(scored)]
    if path_discovery:
        by_name, glob_cost = path_candidates(repo, words)
        grep_cost += glob_cost
        order = by_name + [p for p in order if p not in by_name]
    explore, read, found = 0, 0, None
    for p in order[:MAX_READS]:
        read += 1
        if p in expect:
            found = p
            break
        explore += repo.tokens[p]
    target = found or next((e for e in expect if e in repo.tokens), None)
    if target is None:
        return None
    return {"explore": explore + grep_cost, "reads": read, "found": found is not None, "target": target,
            "target_read": repo.tokens[target], "keywords": kws}


def span_tokens(repo: RepoText, path: str, start: int, end: int) -> int:
    return ri.est_tokens("\n".join(repo.text[path].split("\n")[start - 1:end]))


def simulate(root: str, code: list[dict], silent: list[str], min_confidence: float, budget: int,
             path_discovery: bool = True) -> dict[str, Any]:
    ri.refresh(root)
    repo = RepoText(root)
    rows = []
    for c in code:
        base = baseline_explore(repo, c["q"], c["expect"], path_discovery)
        if base is None:
            rows.append({"q": c["q"], "skipped": True})
            continue
        cap = ri.build_capsule(root, c["q"], token_budget=budget, min_confidence=min_confidence)
        base_total = base["explore"] + base["target_read"]
        emitted = cap["reason"] == "ok"
        hit_hit = next((h for h in cap["hits"] if h["path"] in c["expect"]), None) if emitted else None
        if not emitted:
            outcome, cons, opt = "silent", base_total, base_total
        elif hit_hit is None:
            outcome, cons, opt = "miss", cap["tokens"] + base_total, cap["tokens"] + base_total
        else:
            outcome = "hit"
            cons = cap["tokens"] + base["target_read"]  # agent still reads the whole target file
            opt = cap["tokens"] + min(base["target_read"], span_tokens(repo, hit_hit["path"], hit_hit["start"], hit_hit["end"]))
        rows.append({"q": c["q"], "outcome": outcome, "baseline": base_total, "explore": base["explore"],
                     "target_read": base["target_read"], "capsule": cap["tokens"] if emitted else 0,
                     "conservative": cons, "optimistic": opt, "baseline_found": base["found"], "reads": base["reads"]})
    used = [r for r in rows if not r.get("skipped")]
    wasted = [ri.build_capsule(root, q, token_budget=budget, min_confidence=min_confidence) for q in silent]
    false_inj = [w for w in wasted if w["reason"] == "ok"]
    n = len(used)

    def mean(key: str) -> float:
        return round(sum(r[key] for r in used) / n, 1) if n else 0.0

    def saving(key: str) -> float:
        b = sum(r["baseline"] for r in used)
        return round(100 * (1 - sum(r[key] for r in used) / b), 1) if b else 0.0

    per_task = sorted(1 - r["conservative"] / r["baseline"] for r in used) if used else []
    return {
        "min_confidence": min_confidence, "baseline_model": "grep+filename-glob" if path_discovery else "grep-only", "tasks": n, "skipped_no_keywords": len(rows) - n,
        "hit": sum(r["outcome"] == "hit" for r in used), "miss": sum(r["outcome"] == "miss" for r in used),
        "silent": sum(r["outcome"] == "silent" for r in used),
        "mean_baseline_tokens": mean("baseline"), "mean_explore_tokens": mean("explore"),
        "mean_target_read_tokens": mean("target_read"), "mean_capsule_tokens_when_emitted": round(
            sum(r["capsule"] for r in used if r["capsule"]) / max(1, sum(1 for r in used if r["capsule"])), 1),
        "mean_with_smarty_conservative": mean("conservative"), "mean_with_smarty_optimistic": mean("optimistic"),
        "saving_pct_conservative": saving("conservative"), "saving_pct_optimistic": saving("optimistic"),
        "median_task_saving_pct_conservative": round(100 * statistics.median(per_task), 1) if per_task else 0.0,
        "tasks_where_smarty_costs_more": sum(r["conservative"] > r["baseline"] for r in used),
        "baseline_never_found_target": sum(not r["baseline_found"] for r in used),
        "chatty_prompts": len(silent), "chatty_false_injections": len(false_inj),
        "chatty_wasted_tokens": sum(w["tokens"] for w in false_inj), "rows": rows,
    }


# ── real transcripts ──────────────────────────────────────────────────────────
INSTRUCTIONS_MARKER = "nlp-smarty index:"  # start of mcp_server.INSTRUCTIONS; present when the agent received them
EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
EXPLORE_TOOLS = {"Read", "Grep", "Glob", "LS"}
_BASH_EXPLORE = re.compile(r"(^|[\s;&|])(grep|rg|find|ls|cat|head|tail|tree|wc|git\s+(grep|ls-files)|sed\s+-n)\b")


def _is_explore(u: dict) -> bool:
    if u["name"] in EXPLORE_TOOLS:
        return True
    return u["name"] == "Bash" and bool(_BASH_EXPLORE.search(str(u["input"].get("command", ""))))


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def analyze_transcript(path: str) -> dict[str, Any]:
    usage: dict[str, dict] = {}          # message.id -> usage (lines repeat per content block; count each message once)
    uses: list[dict] = []                # tool_use in order
    seen_tool: set[str] = set()
    results: dict[str, int] = {}         # tool_use_id -> estimated result tokens
    hooks = hook_tokens = 0
    instr_seen = 0
    ts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("type")
            if d.get("timestamp") and t in ("user", "assistant"):
                ts.append(d["timestamp"])
            if t == "assistant":
                m = d.get("message", {})
                if m.get("id") and m.get("usage"):
                    usage[m["id"]] = m["usage"]
                for b in m.get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") not in seen_tool:
                        seen_tool.add(b["id"])
                        uses.append({"id": b["id"], "name": b.get("name", ""), "input": b.get("input", {})})
            elif t == "user":
                c = d.get("message", {}).get("content")
                for b in c if isinstance(c, list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        results[b.get("tool_use_id", "")] = ri.est_tokens(_result_text(b.get("content")))
            elif t == "attachment":
                a = d.get("attachment", {})
                if a.get("type") == "hook_additional_context" and "retrieved_context" in str(a.get("content", "")):
                    hooks += 1
                    hook_tokens += ri.est_tokens(str(a.get("content", "")))
                elif INSTRUCTIONS_MARKER in json.dumps(a):
                    instr_seen = 1
    inp = sum(u.get("input_tokens", 0) for u in usage.values())
    cw = sum(u.get("cache_creation_input_tokens", 0) for u in usage.values())
    cr = sum(u.get("cache_read_input_tokens", 0) for u in usage.values())
    out = sum(u.get("output_tokens", 0) for u in usage.values())
    first_edit = next((i for i, u in enumerate(uses) if u["name"] in EDIT_TOOLS), len(uses))

    pre = [u for u in uses[:first_edit] if _is_explore(u)]
    return {
        "api_calls": len(usage), "input_tokens": inp, "cache_write_tokens": cw, "cache_read_tokens": cr,
        "output_tokens": out,
        # ASSUMPTION: Anthropic-style relative pricing (cache write 1.25x, cache read 0.1x of base input). Adjust to taste.
        "weighted_input_equiv": round(inp + 1.25 * cw + 0.1 * cr),
        "tool_calls": len(uses), "tool_calls_before_first_edit": first_edit,
        "explore_calls_before_first_edit": len(pre),
        "explore_result_tokens_before_first_edit": sum(results.get(u["id"], 0) for u in pre),
        "had_edit": first_edit < len(uses),
        "smarty_tool_calls": sum("smarty" in u["name"].lower() for u in uses),
        "smarty_tools_used": sorted({u["name"].split("__")[-1] for u in uses if "smarty" in u["name"].lower()}),
        "hook_injections": hooks, "hook_injection_tokens": hook_tokens,
        "smarty_instructions_seen": instr_seen,
        "wall_seconds": _span(ts),
    }


def _prompt_text(d: dict) -> str | None:
    """Text of a real user prompt entry (None for tool results / empty). Harness <system-reminder> blocks are dropped."""
    c = d.get("message", {}).get("content")
    if isinstance(c, list):
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            return None
        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    t = re.sub(r"<system-reminder>.*?</system-reminder>", "", c or "", flags=re.S).strip()
    return t or None


SEGMENT_METRICS = ["weighted_input_equiv", "output_tokens", "api_calls", "tool_calls",
                   "explore_calls_before_first_edit", "explore_result_tokens_before_first_edit",
                   "smarty_tool_calls", "hook_injections", "hook_injection_tokens"]


def analyze_segments(path: str) -> list[dict[str, Any]]:
    """Split a session at each real user prompt and attribute usage, tool calls and hook injections to the prompt
    (task) they belong to. A sequence of tasks in one session is then comparable task by task."""
    segs: list[dict[str, Any]] = []
    seen_msg: set[str] = set()
    seen_tool: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("type")
            if t == "user":
                txt = _prompt_text(d)
                if txt is not None:
                    segs.append({"prompt": txt, "usage": [], "uses": [], "results": {}, "hooks": 0, "hook_tokens": 0})
                    continue
            if not segs:
                continue
            cur = segs[-1]
            if t == "user":
                c = d.get("message", {}).get("content")
                for b in c if isinstance(c, list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        cur["results"][b.get("tool_use_id", "")] = ri.est_tokens(_result_text(b.get("content")))
            elif t == "assistant":
                m = d.get("message", {})
                if m.get("id") and m.get("usage") and m["id"] not in seen_msg:
                    seen_msg.add(m["id"])
                    cur["usage"].append(m["usage"])
                for b in m.get("content", []) or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") not in seen_tool:
                        seen_tool.add(b["id"])
                        cur["uses"].append({"id": b["id"], "name": b.get("name", ""), "input": b.get("input", {})})
            elif t == "attachment":
                a = d.get("attachment", {})
                if a.get("type") == "hook_additional_context" and "retrieved_context" in str(a.get("content", "")):
                    cur["hooks"] += 1
                    cur["hook_tokens"] += ri.est_tokens(str(a.get("content", "")))
    out = []
    for sg in segs:
        u = sg["usage"]
        inp, cw = sum(x.get("input_tokens", 0) for x in u), sum(x.get("cache_creation_input_tokens", 0) for x in u)
        cr, outp = sum(x.get("cache_read_input_tokens", 0) for x in u), sum(x.get("output_tokens", 0) for x in u)
        uses = sg["uses"]
        first_edit = next((i for i, x in enumerate(uses) if x["name"] in EDIT_TOOLS), len(uses))
        pre = [x for x in uses[:first_edit] if _is_explore(x)]
        out.append({"prompt": sg["prompt"], "api_calls": len(u), "weighted_input_equiv": round(inp + 1.25 * cw + 0.1 * cr),
                    "output_tokens": outp, "tool_calls": len(uses), "explore_calls_before_first_edit": len(pre),
                    "explore_result_tokens_before_first_edit": sum(sg["results"].get(x["id"], 0) for x in pre),
                    "smarty_tool_calls": sum("smarty" in x["name"].lower() for x in uses),
                    "hook_injections": sg["hooks"], "hook_injection_tokens": sg["hook_tokens"]})
    return out


def compare_segments(a: list[list[dict]], b: list[list[dict]]) -> list[dict]:
    """Pair prompt k of every run in A with prompt k of every run in B; report mean per metric per task."""
    rows = []
    for k in range(min(max(map(len, a), default=0), max(map(len, b), default=0))):
        ga = [r[k] for r in a if len(r) > k]
        gb = [r[k] for r in b if len(r) > k]
        if not ga or not gb:
            continue
        task = {"task": k + 1, "prompt": (ga[0]["prompt"] or "")[:70], "n_a": len(ga), "n_b": len(gb), "metrics": []}
        for m in SEGMENT_METRICS:
            ma, mb = statistics.mean(x[m] for x in ga), statistics.mean(x[m] for x in gb)
            task["metrics"].append({"metric": m, "A": round(ma, 1), "B": round(mb, 1),
                                    "delta_pct": round(100 * (mb - ma) / ma, 1) if ma else None})
        rows.append(task)
    return rows


class InputError(Exception):
    pass


PROJECTS_DIR = "~/.claude/projects"


def match_project_dirs(match: str, root: str | None = None) -> list[str]:
    """Claude Code project directories whose name contains `match` (avoids shell-glob 'no matches found' errors)."""
    root = os.path.expanduser(root or PROJECTS_DIR)
    if not os.path.isdir(root):
        raise InputError(f"Claude Code projects directory not found: {root}")
    found = sorted(os.path.join(root, d) for d in os.listdir(root) if match in d and os.path.isdir(os.path.join(root, d)))
    if not found:
        raise InputError(f"no Claude Code project directory matching '{match}' under {root} yet "
                         f"(a directory only appears once a session has run in that folder)")
    return found


def expand_inputs(paths: list[str], last: int | None = None, match: str | None = None) -> list[str]:
    """Files are used as given. Directories expand to their *.jsonl sessions; with `last`, only the newest N sessions
    across ALL given directories are kept (so several worktree project dirs can be passed with one --last)."""
    files: list[str] = []
    from_dirs: list[str] = []
    seen_dirs: list[str] = []
    if match:
        paths = list(paths) + match_project_dirs(match)
    if not paths:
        raise InputError("give transcript paths/directories or --match NAME")
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            from_dirs.extend(os.path.join(p, f) for f in os.listdir(p) if f.endswith(".jsonl"))
            seen_dirs.append(p)
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise InputError(f"not found: {p}  (substitute your real transcript paths or a project directory under "
                             f"~/.claude/projects/; 'A1.jsonl'-style names in the docs are placeholders)")
    if seen_dirs and not from_dirs and not files:  # empty dirs are fine inside a glob; nothing anywhere is an error
        raise InputError(f"no .jsonl transcripts in {', '.join(seen_dirs)} (has a session been run there yet?)")
    from_dirs.sort(key=os.path.getmtime, reverse=True)
    return files + (from_dirs[:last] if last else from_dirs)


def _span(ts: list[str]) -> float:
    from datetime import datetime

    try:
        parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
        return round((parse(max(ts)) - parse(min(ts))).total_seconds(), 1) if ts else 0.0
    except ValueError:
        return 0.0


METRICS = ["weighted_input_equiv", "input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens",
           "api_calls", "tool_calls_before_first_edit", "explore_calls_before_first_edit",
           "explore_result_tokens_before_first_edit", "smarty_tool_calls", "hook_injections", "hook_injection_tokens",
           "smarty_instructions_seen", "wall_seconds"]


def guess_arm(a: dict) -> str:
    """A = no smarty activity; B = hook (and/or tools) but no instructions; C = the agent received the MCP instructions."""
    if a["smarty_instructions_seen"]:
        return "C"
    return "B" if (a["hook_injections"] or a["smarty_tool_calls"]) else "A"


def compare(a: list[dict], b: list[dict]) -> list[dict]:
    rows = []
    for m in METRICS:
        ma, mb = statistics.mean(x[m] for x in a), statistics.mean(x[m] for x in b)
        rows.append({"metric": m, "A": round(ma, 1), "B": round(mb, 1),
                     "delta_pct": round(100 * (mb - ma) / ma, 1) if ma else None})
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print_sim(s: dict) -> None:
    print(f"\n== baseline={s['baseline_model']}  min_confidence={s['min_confidence']}  tasks={s['tasks']} (skipped {s['skipped_no_keywords']}: nothing to grep)")
    print(f"   outcomes: hit={s['hit']} miss(wasted capsule)={s['miss']} silent={s['silent']}")
    print(f"   baseline tokens/task: {s['mean_baseline_tokens']}  (exploration {s['mean_explore_tokens']} + target read {s['mean_target_read_tokens']})")
    print(f"   with smarty         : {s['mean_with_smarty_conservative']} conservative | {s['mean_with_smarty_optimistic']} optimistic"
          f"   (capsule ~{s['mean_capsule_tokens_when_emitted']} tokens when emitted)")
    print(f"   SAVING              : {s['saving_pct_conservative']}% conservative | {s['saving_pct_optimistic']}% optimistic"
          f"   (median task {s['median_task_saving_pct_conservative']}%; costs MORE on {s['tasks_where_smarty_costs_more']} tasks)")
    print(f"   chatty prompts      : {s['chatty_false_injections']}/{s['chatty_prompts']} wrongly injected ({s['chatty_wasted_tokens']} wasted tokens)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ab_compare", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="offline model of exploration with vs. without smarty")
    s.add_argument("--root", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--min-confidence", type=float, nargs="+", default=[ri.MIN_CONFIDENCE])
    s.add_argument("--budget", type=int, default=450)
    s.add_argument("--baseline", choices=("grep+path", "grep"), nargs="+", default=["grep+path"],
                   help="baseline agent model(s): grep+path also globs file names (fairer, default); grep is the strawman")
    s.add_argument("--bpe", action="store_true", help="use the BPE-calibrated estimator instead of the light heuristic")
    s.add_argument("--json", dest="json_out")
    i = sub.add_parser("inspect", help="one line per session: inferred arm, hook/instructions/tool use, cost (validity check)")
    i.add_argument("paths", nargs="*", help=".jsonl files and/or project directories")
    i.add_argument("--match", help="select project directories by name substring, e.g. NexusAI-smarty-c (no shell glob needed)")
    i.add_argument("--last", type=int, help="newest N sessions across the given directories")
    t = sub.add_parser("transcripts", help="compare real Claude Code sessions: A (baseline) vs B (smarty)")
    t.add_argument("--smarty-vs-smarty", action="store_true",
                   help="reference arm (--a) is hook-only smarty and --b is hook + MCP instructions; adjusts the validity checks")
    t.add_argument("--a", nargs="*", default=[], help="baseline transcripts: .jsonl files and/or project directories")
    t.add_argument("--b", nargs="*", default=[], help="smarty transcripts: .jsonl files and/or project directories")
    t.add_argument("--by-arm", nargs=2, metavar=("REF", "CAND"), choices=("A", "B", "C"),
                   help="gather sessions via --match/--a/--b paths, group them by the arm each one ACTUALLY was (A none / B hook "
                        "only / C hook+instructions, inferred from content) and compare REF vs CAND. Needed because Claude "
                        "Code names project dirs after the repo root, so folder names cannot identify an arm.")
    t.add_argument("--match", help="with --by-arm: select project directories by name substring")
    t.add_argument("--a-match", help="select arm-A project directories by name substring")
    t.add_argument("--b-match", help="select arm-B project directories by name substring")
    t.add_argument("--per-prompt", action="store_true",
                   help="split each session at its user prompts and compare task by task (use when a session ran several tasks)")
    t.add_argument("--last", type=int, help="when a directory is given, use only its newest N sessions")
    t.add_argument("--json", dest="json_out")
    a = ap.parse_args(argv)
    if a.cmd == "inspect":
        try:
            paths = expand_inputs(a.paths, a.last, a.match)
        except InputError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"{'modified':12}{'arm':4}{'hook':>5}{'instr':>6}{'mcp':>5}{'explore':>8}{'api':>5}{'wIn':>9} {'edit':5} prompt / smarty tools used")
        for p in sorted(paths, key=os.path.getmtime):
            x = analyze_transcript(p)
            seg = analyze_segments(p)
            when = __import__("time").strftime("%m-%d %H:%M", __import__("time").localtime(os.path.getmtime(p)))
            prompt = (seg[0]["prompt"] if seg else "")[:46].replace("\n", " ")
            tools = ",".join(x["smarty_tools_used"]) or "-"
            print(f"{when:12}{guess_arm(x):4}{x['hook_injections']:>5}{x['smarty_instructions_seen']:>6}{x['smarty_tool_calls']:>5}"
                  f"{x['explore_calls_before_first_edit']:>8}{x['api_calls']:>5}{x['weighted_input_equiv']:>9} "
                  f"{'yes' if x['had_edit'] else 'no':5} {prompt!r} [{tools}]" + (f"  (+{len(seg)-1} more prompts)" if len(seg) > 1 else ""))
        return 0
    if a.cmd == "simulate":
        if a.bpe:
            ri._ESTIMATOR = None
            os.environ.pop("SMARTY_LIGHT_TOKENS", None)
        code, silent = load_labels(a.labels)
        out = [simulate(a.root, code, silent, mc, a.budget, bm == "grep+path")
               for bm in a.baseline for mc in a.min_confidence]
        for r in out:
            _print_sim(r)
        print("\nMODEL ASSUMPTIONS: baseline agent globs file names and greps <=4 distinctive keywords, reads matching files best-first (<=12) until it "
              "reads the target; 'conservative' still reads the whole target file after a hit; a miss or a silent gate saves "
              "nothing. This is a MODEL, not a measurement: use `transcripts` on real runs to confirm.")
        if a.json_out:
            json.dump(out, open(a.json_out, "w"), indent=2)
        return 0
    try:
        if a.by_arm:
            allp = expand_inputs(a.a + a.b, a.last, a.match)
            arms = {p: guess_arm(analyze_transcript(p)) for p in allp}
            pa = [p for p in allp if arms[p] == a.by_arm[0]]
            pb = [p for p in allp if arms[p] == a.by_arm[1]]
            counts = {k: sum(1 for v in arms.values() if v == k) for k in "ABC"}
            print(f"sessions by inferred arm: A={counts['A']} B={counts['B']} C={counts['C']}  (comparing {a.by_arm[0]} vs {a.by_arm[1]})")
            if not pa or not pb:
                raise InputError(f"need sessions in both arms; found {a.by_arm[0]}={len(pa)}, {a.by_arm[1]}={len(pb)}")
            if set(a.by_arm) == {"B", "C"}:
                a.smarty_vs_smarty = True
        else:
            pa, pb = expand_inputs(a.a, a.last, a.a_match), expand_inputs(a.b, a.last, a.b_match)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ga, gb = [analyze_transcript(p) for p in pa], [analyze_transcript(p) for p in pb]
    if a.per_prompt:
        sa, sb = [analyze_segments(p) for p in pa], [analyze_segments(p) for p in pb]
        for task in compare_segments(sa, sb):
            print(f"\n== task {task['task']}: {task['prompt']!r}   (A n={task['n_a']}, B n={task['n_b']})")
            print(f"{'metric':44}{'A':>12}{'B':>12}{'B vs A':>10}")
            for r in task["metrics"]:
                print(f"{r['metric']:44}{r['A']:>12}{r['B']:>12}{(str(r['delta_pct']) + '%') if r['delta_pct'] is not None else '-':>10}")
        print("\n--- whole-session totals (all tasks together) ---")
    rows = compare(ga, gb)
    print(f"A (baseline) n={len(ga)}   B (smarty) n={len(gb)}")
    if a.smarty_vs_smarty:
        if any(not x["hook_injections"] for x in ga + gb):
            print("WARNING: some runs had no hook injection; both arms are supposed to use the hook.")
        if any(x["smarty_instructions_seen"] for x in ga):
            print("WARNING: reference arm (--a) received the MCP instructions - it is not hook-only; comparison invalid.")
        if not all(x["smarty_instructions_seen"] for x in gb):
            print("WARNING: not every --b run received the MCP instructions (smarty_instructions_seen); check the server is "
                  "current and SMARTY_MCP_INSTRUCTIONS is not 'off' in that folder.")
        used = sum(1 for x in gb if x["smarty_tool_calls"])
        print(f"MCP tools were called in {used}/{len(gb)} instruction-arm runs "
              f"(and {sum(1 for x in ga if x['smarty_tool_calls'])}/{len(ga)} in the reference arm).")
    else:
        if any(x["hook_injections"] or x["smarty_tool_calls"] for x in ga):
            print("WARNING: arm A shows smarty activity - the baseline is contaminated; this comparison is invalid.")
        if not any(x["hook_injections"] or x["smarty_tool_calls"] for x in gb):
            print("WARNING: arm B shows NO smarty activity (0 hook injections, 0 smarty tool calls) - smarty may not have loaded.")
    print(f"{'metric':44}{'A':>12}{'B':>12}{'B vs A':>10}")
    for r in rows:
        print(f"{r['metric']:44}{r['A']:>12}{r['B']:>12}{(str(r['delta_pct']) + '%') if r['delta_pct'] is not None else '-':>10}")
    if min(len(ga), len(gb)) < 3:
        print("\nWARNING: fewer than 3 runs per arm. LLM runs vary widely; treat these deltas as anecdotes, not evidence.")
    if not all(x["had_edit"] for x in ga + gb):
        print("NOTE: some runs never edited a file; 'before first edit' metrics then cover the whole run.")
    if a.json_out:
        json.dump({"A": ga, "B": gb, "compare": rows,
                   "per_prompt": compare_segments(sa, sb) if a.per_prompt else None}, open(a.json_out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
