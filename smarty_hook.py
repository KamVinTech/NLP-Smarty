"""
Claude Code `UserPromptSubmit` hook: injects a budgeted <retrieved_context> capsule before Turn 1.

stdin : hook JSON {"session_id", "prompt", "cwd", ...}
stdout: {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "..."}} or nothing
Design rules: fail-open (always exit 0, never block a prompt), stdlib-only fast path (no spaCy/pipeline import),
session ledger so a symbol is never injected twice, silence below the confidence gate.

Register in .claude/settings.json:
  {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python3 /abs/path/smarty_hook.py"}]}]}}
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

os.environ.setdefault("SMARTY_LIGHT_TOKENS", "1")  # must precede repo_indexer: skip the heavy BPE estimator import

import repo_indexer as ri  # noqa: E402

FIRST_TURN_BUDGET = 450
FOLLOWUP_BUDGET = 250
MIN_QUERY_TERMS = 2
MAX_LEDGER = 200
LEDGER_TTL_S = 7 * 24 * 3600


def _ledger_path(root: str, session_id: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "default")[:64]
    d = os.path.join(root, ri.INDEX_DIR, "sessions")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{sid}.json")


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("delivered"), list):
            return data
    except (OSError, ValueError):
        pass
    return {"turns": 0, "delivered": []}


def _gc(root: str) -> None:
    d = os.path.join(root, ri.INDEX_DIR, "sessions")
    now = time.time()
    try:
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if now - os.path.getmtime(p) > LEDGER_TTL_S:
                os.remove(p)
    except OSError:
        pass


_WRAPPERS = re.compile(r"<(system-reminder|ide_[a-z_]+|command-[a-z-]+)>.*?</\1>", re.S)


def clean_prompt(prompt: str) -> str:
    """Drop harness-injected wrappers (worktree notices, IDE context, ...) so only what the user typed is searched."""
    return _WRAPPERS.sub(" ", prompt).strip()


def process(event: dict) -> dict | None:
    """Return the hook output dict, or None to stay silent."""
    prompt = event.get("prompt")
    if isinstance(prompt, str):
        prompt = clean_prompt(prompt)
    cwd = event.get("cwd") or os.getcwd()
    if not isinstance(prompt, str) or not prompt.strip() or prompt.lstrip().startswith("/"):
        return None
    # project root: explicit override > Claude Code's project dir (cwd may be a subdirectory) > event cwd
    root = os.path.abspath(os.environ.get("SMARTY_ROOT") or os.environ.get("CLAUDE_PROJECT_DIR") or cwd)
    if len(ri.tokenize(prompt, drop_stop=True)) < MIN_QUERY_TERMS:
        return None

    lp = _ledger_path(root, str(event.get("session_id") or "default"))
    ledger = _load(lp)
    first = ledger["turns"] == 0
    delivered = [tuple(x) for x in ledger["delivered"] if isinstance(x, list) and len(x) == 2]
    out = ri.build_capsule(root, prompt, token_budget=FIRST_TURN_BUDGET if first else FOLLOWUP_BUDGET,
                           k=5 if first else 3, exclude=delivered)
    ledger["turns"] += 1
    if out["reason"] == "ok":
        ledger["delivered"] = (ledger["delivered"] + [list(d) for d in out.get("delivered", [])])[-MAX_LEDGER:]
    try:
        with open(lp, "w") as f:
            json.dump(ledger, f)
        if first:
            _gc(root)
    except OSError:
        pass
    if out["reason"] != "ok" or not out["capsule"]:
        return None
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": out["capsule"]}}


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
        result = process(event if isinstance(event, dict) else {})
        if result:
            sys.stdout.write(json.dumps(result))
    except Exception as exc:  # fail-open: a broken hook must never block the user's prompt
        sys.stderr.write(f"[nlp-smarty hook] ignored error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
