"""
`smarty` CLI: bootstrap, verify and remove NLP-Smarty repo intelligence in any repository.

    smarty init      [--root DIR] [--dry-run] [--scope local|project] [--no-hooks] [--no-mcp] [--no-index]
                     [--agents-md [FILE]] [--python PATH]
    smarty doctor    [--root DIR]
    smarty uninstall [--root DIR] [--dry-run] [--purge]

Guarantees: idempotent; merges (never overwrites) existing JSON config and aborts on files it cannot parse; every
changed pre-existing file gets a one-time `.smarty-bak` backup; `uninstall` removes only what `init` added.
Stdlib only; no network; no LLM.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any

HOME = os.path.dirname(os.path.abspath(__file__))
SERVER_NAME = "nlp-smarty"
HOOK_MARK = "smarty_hook.py"
INDEX_MARK = "repo_indexer.py"
MCP_MARK = "mcp_server.py"
BEGIN, END = "<!-- smarty:begin -->", "<!-- smarty:end -->"
CARTRIDGE = f"""{BEGIN}
## NLP-Smarty repo intelligence (auto-managed; edit outside these markers only)
- MCP tools `smarty_get_context`, `smarty_find_symbol`, `smarty_repo_map`, `smarty_get_slice` query a local symbol index: use them before grepping or reading whole files.
- A `<retrieved_context>` block may be injected before your turn. It is lexical retrieval and can be incomplete; verify with grep if unsure.
- The index lives in `.smarty/` (gitignored). Do not edit it.
{END}
"""


class InitError(Exception):
    pass


# ── file helpers ──────────────────────────────────────────────────────────────
def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _load_json(path: str) -> dict[str, Any] | None:
    raw = _read(path)
    if raw is None:
        return None
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError as exc:
        raise InitError(f"{path} is not valid JSON ({exc}); refusing to modify it. Fix or remove it and retry.")
    if not isinstance(data, dict):
        raise InitError(f"{path} must contain a JSON object; refusing to modify it.")
    return data


def _dump(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2) + "\n"


def _canon(path: str) -> str:
    """Follow symlinks (e.g. CLAUDE.md -> AGENTS.md) so we edit the target instead of replacing the link."""
    return os.path.realpath(path) if os.path.islink(path) else path


class Plan:
    """Collects file writes so --dry-run and real runs share one code path."""

    def __init__(self) -> None:
        self.writes: dict[str, tuple[str | None, str | None]] = {}  # path -> (old, new); new None = delete
        self.notes: list[str] = []

    def write(self, path: str, new: str | None) -> None:
        path = _canon(path)
        old = self.writes[path][0] if path in self.writes else _read(path)
        self.writes[path] = (old, new)

    def current(self, path: str) -> str | None:
        path = _canon(path)
        return self.writes[path][1] if path in self.writes else _read(path)

    def changed(self) -> list[str]:
        return [p for p, (o, n) in self.writes.items() if o != n]

    def show(self) -> None:
        for p in self.changed():
            old, new = self.writes[p]
            print(f"\n--- {p}  ({'delete' if new is None else 'create' if old is None else 'modify'})")
            diff = difflib.unified_diff((old or "").splitlines(), (new or "").splitlines(), "before", "after", lineterm="", n=1)
            for line in diff:
                print("   " + line)

    def apply(self, root: str, backup: bool = True) -> None:
        git_dir = os.path.join(root, ".git") + os.sep
        for p in self.changed():
            old, new = self.writes[p]
            if new is None:
                os.remove(p)
                continue
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            if backup and old is not None and not p.startswith(git_dir):
                _backup(root, p, old)
            tmp = p + ".smarty-tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(new)
            os.replace(tmp, p)


def _backup(root: str, path: str, content: str) -> None:
    """One-time pre-change copy under .smarty/backups/ (gitignored; never clutters the working tree)."""
    d = os.path.join(root, ".smarty")
    bp = os.path.join(d, "backups", os.path.relpath(path, root).replace(os.sep, "__"))
    if os.path.exists(bp):
        return
    os.makedirs(os.path.dirname(bp), exist_ok=True)
    gi = os.path.join(d, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w") as f:
            f.write("*\n")
    with open(bp, "w", encoding="utf-8") as f:
        f.write(content)


# ── settings (hooks) ──────────────────────────────────────────────────────────
def _hooks_of(settings: dict, event: str) -> list:
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        raise InitError("'hooks' in settings must be an object")
    groups = hooks.get(event, [])
    if not isinstance(groups, list):
        raise InitError(f"'hooks.{event}' in settings must be a list")
    return groups


def _has_marker(settings: dict, event: str, marker: str) -> bool:
    for g in _hooks_of(settings, event):
        for h in (g.get("hooks", []) if isinstance(g, dict) else []):
            if isinstance(h, dict) and marker in str(h.get("command", "")):
                return True
    return False


def _add_hook(settings: dict, event: str, command: str, timeout: int) -> None:
    _hooks_of(settings, event)  # structure check
    settings.setdefault("hooks", {}).setdefault(event, []).append(
        {"hooks": [{"type": "command", "command": command, "timeout": timeout}]})


def _strip_marker(settings: dict, event: str, marker: str) -> bool:
    groups = _hooks_of(settings, event)
    changed = False
    keep = []
    for g in groups:
        if isinstance(g, dict) and isinstance(g.get("hooks"), list):
            inner = [h for h in g["hooks"] if not (isinstance(h, dict) and marker in str(h.get("command", "")))]
            if len(inner) != len(g["hooks"]):
                changed = True
            if inner:
                keep.append({**g, "hooks": inner})
        else:
            keep.append(g)
    if changed:
        if keep:
            settings["hooks"][event] = keep
        else:
            settings["hooks"].pop(event, None)
            if not settings["hooks"]:
                settings.pop("hooks")
    return changed


def _settings_paths(root: str) -> dict[str, str]:
    d = os.path.join(root, ".claude")
    return {"project": os.path.join(d, "settings.json"), "local": os.path.join(d, "settings.local.json")}


def _plan_hooks(plan: Plan, root: str, scope: str, python: str, home: str) -> None:
    paths = _settings_paths(root)
    loaded = {k: _load_json(p) for k, p in paths.items()}
    hook_cmd = f"{shlex.quote(python)} {shlex.quote(os.path.join(home, HOOK_MARK))}"
    index_cmd = (f"{shlex.quote(python)} {shlex.quote(os.path.join(home, INDEX_MARK))} "
                 f"--root \"$CLAUDE_PROJECT_DIR\" index >/dev/null 2>&1 || true")
    target_path = paths[scope]
    target = loaded[scope] if loaded[scope] is not None else {}
    before = json.dumps(target, sort_keys=True)
    for event, marker, cmd, timeout in (("UserPromptSubmit", HOOK_MARK, hook_cmd, 10),
                                        ("SessionStart", INDEX_MARK, index_cmd, 30)):
        where = [k for k, s in loaded.items() if s and _has_marker(s, event, marker)]
        if where:
            plan.notes.append(f"{event} hook already configured in {paths[where[0]]} - left as is")
        else:
            _add_hook(target, event, cmd, timeout)
    if json.dumps(target, sort_keys=True) != before:
        plan.write(target_path, _dump(target))
        if scope == "local":
            _plan_git_exclude(plan, root, os.path.join(".claude", "settings.local.json"))


def _plan_git_exclude(plan: Plan, root: str, rel: str) -> None:
    """Keep machine-specific local settings out of commits via .git/info/exclude (never edits .gitignore)."""
    if not os.path.isdir(os.path.join(root, ".git")):
        return
    ignored = subprocess.run(["git", "-C", root, "check-ignore", "-q", rel], capture_output=True).returncode == 0
    if ignored:
        return
    ex = os.path.join(root, ".git", "info", "exclude")
    cur = plan.current(ex) or ""
    if rel not in cur.split("\n"):
        plan.write(ex, cur + ("" if cur.endswith("\n") or not cur else "\n") + rel + "\n")


# ── MCP ───────────────────────────────────────────────────────────────────────
def _plan_mcp(plan: Plan, root: str, python: str, home: str, scope: str) -> None:
    path = os.path.join(root, ".mcp.json")
    data = _load_json(path)
    cfg = data if data is not None else {}
    servers = cfg.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise InitError(".mcp.json 'mcpServers' must be an object")
    existing = servers.get(SERVER_NAME)
    if existing is not None:
        if MCP_MARK in json.dumps(existing):
            plan.notes.append(f"MCP server '{SERVER_NAME}' already in .mcp.json - left as is")
        else:
            plan.notes.append(f"WARNING: .mcp.json already has an unrelated '{SERVER_NAME}' server - not touching it")
    else:
        servers[SERVER_NAME] = {"command": python, "args": [os.path.join(home, MCP_MARK)],
                                "env": {"SMARTY_ROOT": "."}}
        cfg["mcpServers"] = servers
        plan.write(path, _dump(cfg))
        plan.notes.append(".mcp.json contains machine-specific absolute paths; avoid committing it (or share a relative one)")
    # approval so the server starts without a prompt
    sp = _settings_paths(root)[scope]
    cur = plan.current(sp)
    try:
        s = json.loads(cur) if cur and cur.strip() else {}
    except ValueError as exc:
        raise InitError(f"{sp} is not valid JSON ({exc}); refusing to modify it.")
    if not isinstance(s, dict) or not isinstance(s.get("enabledMcpjsonServers", []), list):
        raise InitError(f"{sp}: unexpected structure for 'enabledMcpjsonServers'")
    approved = s.setdefault("enabledMcpjsonServers", [])
    if SERVER_NAME not in approved and existing is None:
        approved.append(SERVER_NAME)
        plan.write(sp, _dump(s))


# ── cartridge (static AGENTS.md/CLAUDE.md block) ──────────────────────────────
def _cartridge_target(root: str, name: str) -> str:
    if name != "AUTO":
        return os.path.join(root, name)
    for n in ("CLAUDE.md", "AGENTS.md"):
        if os.path.exists(os.path.join(root, n)):
            return os.path.join(root, n)
    return os.path.join(root, "AGENTS.md")


def _with_block(text: str | None) -> str:
    text = text or ""
    if BEGIN in text and END in text:
        a, b = text.index(BEGIN), text.index(END) + len(END)
        return text[:a] + CARTRIDGE.rstrip("\n") + text[b:]
    return text + ("" if not text or text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n") + CARTRIDGE


def _without_block(text: str) -> str:
    if BEGIN not in text or END not in text:
        return text
    a, b = text.index(BEGIN), text.index(END) + len(END)
    out = (text[:a].rstrip("\n") + "\n" + text[b:].lstrip("\n")) if text[:a].strip() else text[b:].lstrip("\n")
    return out


# ── commands ──────────────────────────────────────────────────────────────────
def _resolve_root(arg: str) -> str:
    root = os.path.abspath(arg)
    if not os.path.isdir(root):
        raise InitError(f"root is not a directory: {root}")
    return root


def _preflight(home: str) -> None:
    missing = [f for f in (HOOK_MARK, INDEX_MARK, MCP_MARK, "pipeline.py") if not os.path.exists(os.path.join(home, f))]
    if missing:
        raise InitError(f"NLP-Smarty install at {home} is missing: {', '.join(missing)} (use --home)")


def cmd_init(a: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    root, home = _resolve_root(a.root), os.path.abspath(a.home)
    _preflight(home)
    python = os.path.abspath(a.python) if a.python else sys.executable
    if not os.path.isfile(python):
        raise InitError(f"python interpreter not found: {python}")
    plan = Plan()
    if not os.path.isdir(os.path.join(root, ".git")):
        plan.notes.append("not a git repository: only a plain directory walk will be indexed (no .gitignore awareness)")
    if not a.no_hooks:
        _plan_hooks(plan, root, a.scope, python, home)
    if not a.no_mcp:
        _plan_mcp(plan, root, python, home, a.scope)
    if a.agents_md:
        tgt = _cartridge_target(root, a.agents_md)
        new = _with_block(plan.current(tgt))
        if new != plan.current(tgt):
            plan.write(tgt, new)
    changed = plan.changed()
    if a.dry_run:
        print(f"[dry-run] would change {len(changed)} file(s) under {root}")
        plan.show()
    else:
        plan.apply(root)
        for p in changed:
            print(f"  wrote {os.path.relpath(p, root)}")
        if not changed:
            print("  nothing to change (already configured)")
    for n in plan.notes:
        print(f"  note: {n}")
    if not a.no_index and not a.dry_run:
        os.environ.setdefault("SMARTY_LIGHT_TOKENS", "1")
        sys.path.insert(0, home)
        import repo_indexer as ri  # noqa: WPS433

        st = ri.refresh(root)
        info = ri.stats(root)
        print(f"  indexed {info['files']} files / {info['symbols']} symbols "
              f"({st['reparsed']} parsed, {st['unchanged']} unchanged)")
    if not a.dry_run:
        print(f"done in {time.perf_counter() - t0:.2f}s - run `smarty doctor` to verify; restart Claude Code to load hooks/MCP")
    return 0


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    return ok


def cmd_doctor(a: argparse.Namespace) -> int:
    root, home = _resolve_root(a.root), os.path.abspath(a.home)
    print(f"smarty doctor: {root}")
    ok = True
    try:
        _preflight(home)
        ok &= _check("install files present", True, home)
    except InitError as exc:
        return 1 if not _check("install files present", False, str(exc)) else 0
    s = {k: _load_json(p) or {} for k, p in _settings_paths(root).items()}
    for event, mark in (("UserPromptSubmit", HOOK_MARK), ("SessionStart", INDEX_MARK)):
        ok &= _check(f"{event} hook registered", any(_has_marker(v, event, mark) for v in s.values()))
    mcp = _load_json(os.path.join(root, ".mcp.json")) or {}
    ok &= _check("MCP server registered", MCP_MARK in json.dumps(mcp.get("mcpServers", {}).get(SERVER_NAME, {})))
    env = {**os.environ, "CLAUDE_PROJECT_DIR": root}
    ev = json.dumps({"session_id": "doctor", "prompt": "how does the prompt cache lookup work", "cwd": root})
    p = subprocess.run([sys.executable, os.path.join(home, HOOK_MARK)], input=ev, capture_output=True, text=True,
                       env=env, timeout=60)
    ok &= _check("hook smoke test (fail-open, valid output)", p.returncode == 0 and
                 (p.stdout == "" or "hookSpecificOutput" in json.loads(p.stdout)), f"exit={p.returncode}")
    sess = os.path.join(root, ".smarty", "sessions", "doctor.json")
    if os.path.exists(sess):
        os.remove(sess)
    msgs = "\n".join(json.dumps(m) for m in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})) + "\n"
    p = subprocess.run([sys.executable, os.path.join(home, MCP_MARK)], input=msgs, capture_output=True, text=True,
                       env={**env, "SMARTY_ROOT": root}, timeout=60, cwd=home)
    try:
        tools = [t["name"] for t in json.loads(p.stdout.splitlines()[-1])["result"]["tools"]]
    except (ValueError, KeyError, IndexError):
        tools = []
    ok &= _check("MCP handshake + tools/list", "smarty_get_context" in tools, f"{len(tools)} tools")
    sys.path.insert(0, home)
    os.environ.setdefault("SMARTY_LIGHT_TOKENS", "1")
    import repo_indexer as ri  # noqa: WPS433

    info = ri.stats(root) if os.path.isdir(os.path.join(root, ".smarty")) else {"files": 0, "symbols": 0}
    ok &= _check("index built", info["files"] > 0, f"{info['files']} files, {info['symbols']} symbols")
    print("all checks passed" if ok else "some checks FAILED")
    return 0 if ok else 1


def cmd_audit(a: argparse.Namespace) -> int:
    sys.path.insert(0, HOME)
    import token_audit  # noqa: WPS433

    return token_audit.main(a.audit_args)


def cmd_uninstall(a: argparse.Namespace) -> int:
    root = _resolve_root(a.root)
    plan = Plan()
    for path in _settings_paths(root).values():
        s = _load_json(path)
        if s is None:
            continue
        for event, mark in (("UserPromptSubmit", HOOK_MARK), ("SessionStart", INDEX_MARK)):
            _strip_marker(s, event, mark)
        ap = s.get("enabledMcpjsonServers")
        if isinstance(ap, list) and SERVER_NAME in ap:
            ap.remove(SERVER_NAME)
            if not ap:
                s.pop("enabledMcpjsonServers")
        new = _dump(s) if s else None
        if new != _read(path):
            plan.write(path, new)
    mp = os.path.join(root, ".mcp.json")
    m = _load_json(mp)
    if m and isinstance(m.get("mcpServers"), dict) and MCP_MARK in json.dumps(m["mcpServers"].get(SERVER_NAME, {})):
        del m["mcpServers"][SERVER_NAME]
        if not m["mcpServers"]:
            m.pop("mcpServers")
        plan.write(mp, _dump(m) if m else None)
    seen: list[str] = []
    for name in ("CLAUDE.md", "AGENTS.md", "agents.md"):
        p = os.path.join(root, name)
        if not os.path.isfile(p) or any(os.path.samefile(p, q) for q in seen):
            continue  # also dedupes case-insensitive filesystems and symlinks
        seen.append(p)
        t = _read(p)
        if t and BEGIN in t and END in t:
            rest = _without_block(t)
            plan.write(p, rest if rest.strip() else None)  # file held only our block -> it was ours
    changed = plan.changed()
    if a.dry_run:
        print(f"[dry-run] would change {len(changed)} file(s)")
        plan.show()
        return 0
    plan.apply(root, backup=False)
    claude_dir = os.path.join(root, ".claude")
    if os.path.isdir(claude_dir) and not os.listdir(claude_dir):
        os.rmdir(claude_dir)
    for p in changed:
        print(f"  {'removed' if plan.writes[p][1] is None else 'updated'} {os.path.relpath(p, root)}")
    if a.purge:
        import shutil

        d = os.path.join(root, ".smarty")
        if os.path.isdir(d):
            shutil.rmtree(d)
            print("  removed .smarty/")
    print("uninstalled" if changed or a.purge else "nothing to remove")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="smarty", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default=".", help="target repository (default: cwd)")
        p.add_argument("--home", default=HOME, help=argparse.SUPPRESS)

    i = sub.add_parser("init", help="register hooks + MCP server, build the index")
    common(i)
    i.add_argument("--dry-run", action="store_true", help="show diffs, write nothing")
    i.add_argument("--scope", choices=("local", "project"), default="local",
                   help="where new hooks go: .claude/settings.local.json (default, machine-specific) or settings.json")
    i.add_argument("--python", help="interpreter for hooks/MCP (default: the one running smarty)")
    i.add_argument("--no-hooks", action="store_true")
    i.add_argument("--no-mcp", action="store_true")
    i.add_argument("--no-index", action="store_true")
    i.add_argument("--agents-md", nargs="?", const="AUTO", default=None, metavar="FILE",
                   help="append a static marker-fenced cartridge to CLAUDE.md/AGENTS.md (or FILE)")
    i.set_defaults(fn=cmd_init)
    d = sub.add_parser("doctor", help="verify an install end to end")
    common(d)
    d.set_defaults(fn=cmd_doctor)
    au = sub.add_parser("audit", help="where do your agent sessions' tokens go? (reads Claude Code transcripts)")
    au.add_argument("audit_args", nargs=argparse.REMAINDER, help="passed to token_audit (see: python token_audit.py -h)")
    au.set_defaults(fn=cmd_audit)
    u = sub.add_parser("uninstall", help="remove what init added")
    common(u)
    u.add_argument("--dry-run", action="store_true")
    u.add_argument("--purge", action="store_true", help="also delete the .smarty/ index")
    u.set_defaults(fn=cmd_uninstall)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except InitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
