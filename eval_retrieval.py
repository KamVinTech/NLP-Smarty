"""
Deterministic retrieval evaluation from git history (replaces unmeasured token-savings claims).

For each recent non-merge commit C touching indexable files:
  * query        = C's commit message (subject + body)
  * ground truth = indexable files changed by C
  * repo state   = C's PARENT tree (exported to a temp dir), so the index never sees the answer
Metrics: file recall@k (k=1,3,5), hit-rate@5, capsule tokens, and token cost vs. two baselines:
  - "read changed files" : raw tokens of the ground-truth files (optimistic baseline: agent knows where to look)
  - "read all candidates": raw tokens of every file in the repo index (pessimistic blind-exploration baseline)
Commit messages are a noisy proxy for a real task; treat results as relative signal (regression gate), not truth.

Usage: python eval_retrieval.py [--root .] [-n 20] [--json out.json] [--min-recall5 0.0]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

import repo_indexer as ri


def _git(root: str, *args: str) -> str:
    p = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"git {' '.join(args)} failed")
    return p.stdout


def list_commits(root: str, n: int) -> list[str]:
    return _git(root, "rev-list", "--no-merges", "--max-parents=1", f"--max-count={n * 3}", "HEAD").split()


def _export_tree(root: str, rev: str, dest: str) -> None:
    p = subprocess.run(["git", "-C", root, "archive", "--format=tar", rev], capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    import io

    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
        for m in tf.getmembers():  # path-traversal guard
            if m.name.startswith("/") or ".." in m.name.split("/"):
                continue
            tf.extract(m, dest)


def _indexable(path: str) -> bool:
    parts = path.split("/")
    return (os.path.splitext(path)[1].lower() in ri.LANG_BY_EXT and not any(p in ri.IGNORE_DIRS for p in parts[:-1])
            and parts[-1] not in ri.IGNORE_FILES)


def evaluate(root: str, n: int = 20, budget: int = 450, k: int = 5) -> dict[str, Any]:
    root = os.path.abspath(root)
    cases: list[dict[str, Any]] = []
    for rev in list_commits(root, n):
        if len(cases) >= n:
            break
        msg = _git(root, "log", "-1", "--format=%B", rev).strip()
        changed = [f for f in _git(root, "diff-tree", "--no-commit-id", "--name-only", "-r", "--diff-filter=AM", rev).split("\n")
                   if f and _indexable(f)]
        if not msg or not changed:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                _export_tree(root, f"{rev}^", tmp)
            except RuntimeError:
                continue  # root commit
            con = ri._connect(tmp)
            ri.refresh(tmp, con)
            existing = {r["path"]: r["tokens_raw"] for r in con.execute("SELECT path, tokens_raw FROM files")}
            truth = [f for f in changed if f in existing]  # files added by C cannot be retrieved from the parent
            if not truth:
                con.close()
                continue
            res = ri.search(tmp, msg, k=k, recency=False, con=con, auto_refresh=False)
            con.close()
            ranked_files: list[str] = []
            for h in res["hits"]:
                if h["path"] not in ranked_files:
                    ranked_files.append(h["path"])
            cap = ri.build_capsule(tmp, msg, token_budget=budget, k=k)
            cases.append({
                "commit": rev[:8], "subject": msg.split("\n")[0][:70], "truth": truth, "retrieved": ranked_files,
                "recall@1": len(set(ranked_files[:1]) & set(truth)) / len(truth),
                "recall@3": len(set(ranked_files[:3]) & set(truth)) / len(truth),
                "recall@5": len(set(ranked_files[:5]) & set(truth)) / len(truth),
                "hit@5": bool(set(ranked_files[:5]) & set(truth)),
                "capsule_tokens": cap["tokens"], "capsule_reason": cap["reason"],
                "baseline_changed_tokens": sum(existing[f] for f in truth),
                "baseline_all_tokens": sum(existing.values()),
            })
    return {"cases": cases, "summary": _summarize(cases)}


def evaluate_golden(root: str, path: str | None = None, k: int = 5) -> dict[str, Any]:
    """Hand-labelled task -> symbol queries (tests/golden_queries.json). Metrics: recall@k, top-1, gate emit rate."""
    root = os.path.abspath(root)
    path = path or os.path.join(root, "tests", "golden_queries.json")
    cases = json.load(open(path))
    ri.refresh(root)
    rows = []
    for c in cases:
        got = [f"{h['path']}:{h['name']}" for h in ri.search(root, c["q"], k=k, recency=False)["hits"]]
        exp = set(c["expect"])
        rows.append({"q": c["q"], "set": c.get("set", "dev"), "recall@k": bool(exp & set(got)),
                     "top1": bool(got and got[0] in exp), "emit": ri.build_capsule(root, c["q"])["reason"] == "ok",
                     "got": got[:3]})
    def agg(rs):
        n = len(rs) or 1
        return {"n": len(rs), f"recall@{k}": round(sum(r["recall@k"] for r in rs) / n, 3),
                "top1": round(sum(r["top1"] for r in rs) / n, 3), "emit_rate": round(sum(r["emit"] for r in rs) / n, 3)}
    return {"all": agg(rows), "dev": agg([r for r in rows if r["set"] == "dev"]),
            "heldout": agg([r for r in rows if r["set"] == "heldout"]), "rows": rows}


def _summarize(cases: list[dict]) -> dict[str, Any]:
    if not cases:
        return {"n": 0}
    m = lambda key: sum(c[key] for c in cases) / len(cases)  # noqa: E731
    emitted = [c for c in cases if c["capsule_reason"] == "ok"]
    e_cap = sum(c["capsule_tokens"] for c in emitted)
    e_chg = sum(c["baseline_changed_tokens"] for c in emitted)
    e_all = sum(c["baseline_all_tokens"] for c in emitted)
    return {
        "n": len(cases),
        "recall@1": round(m("recall@1"), 3), "recall@3": round(m("recall@3"), 3), "recall@5": round(m("recall@5"), 3),
        "hit_rate@5": round(sum(c["hit@5"] for c in cases) / len(cases), 3),
        "capsule_emitted_rate": round(len(emitted) / len(cases), 3),
        "n_emitted": len(emitted),
        "avg_capsule_tokens_emitted": round(e_cap / len(emitted), 1) if emitted else 0,
        # savings are computed ONLY over cases where a capsule was actually emitted (empty capsules save nothing useful)
        "emitted_vs_changed_files_pct": round(100 * (1 - e_cap / e_chg), 1) if e_chg else None,
        "emitted_vs_all_files_pct": round(100 * (1 - e_cap / e_all), 1) if e_all else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=".")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=450)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--golden", action="store_true", help="evaluate tests/golden_queries.json instead of git history")
    ap.add_argument("--min-recall5", type=float, default=0.0, help="exit 1 if mean recall@5 is below this (CI gate)")
    a = ap.parse_args(argv)
    if a.golden:
        g = evaluate_golden(a.root)
        for r in g["rows"]:
            print(f"{'OK  ' if r['recall@k'] else 'MISS'} {'top1' if r['top1'] else '    '} {'emit' if r['emit'] else '    '} "
                  f"[{r['set']}] {r['q'][:60]} -> {r['got'][:2]}")
        print(json.dumps({k: v for k, v in g.items() if k != "rows"}, indent=2))
        return 1 if g["all"]["recall@5"] < a.min_recall5 else 0
    out = evaluate(a.root, a.n, a.budget)
    for c in out["cases"]:
        print(f"{c['commit']} r@1={c['recall@1']:.2f} r@5={c['recall@5']:.2f} cap={c['capsule_tokens']:>4}t "
              f"[{c['capsule_reason']}] {c['subject']}")
    print(json.dumps(out["summary"], indent=2))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=2)
    return 1 if out["summary"].get("n", 0) and out["summary"]["recall@5"] < a.min_recall5 else 0


if __name__ == "__main__":
    sys.exit(main())
