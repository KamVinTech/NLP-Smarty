"""The code lives in ONE importable package; root scripts are thin launchers (see docs/RESTRUCTURE.md)."""
import ast
import importlib
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHERS = {"smarty_cli": "cli", "smarty_hook": "hook", "bridge_server": "bridge", "gateway": "gateway", "ledger": "ledger",
             "gate_bench": "gate_bench", "mcp_server": "mcp_server", "repo_indexer": "repo_indexer", "token_audit": "token_audit",
             "ab_compare": "ab_compare", "eval_retrieval": "eval_retrieval"}


def test_no_flat_engine_modules_remain_at_the_root():
    root_py = {f[:-3] for f in os.listdir(ROOT) if f.endswith(".py")}
    assert root_py <= set(LAUNCHERS) | {"test_harness"}, root_py - set(LAUNCHERS)


def test_every_launcher_is_thin_and_points_at_an_existing_package_main():
    for name, mod in LAUNCHERS.items():
        src = open(os.path.join(ROOT, f"{name}.py"), encoding="utf-8").read()
        assert len(src.splitlines()) < 20 and f"from contextgate_core.{mod} import main" in src, name
        assert callable(importlib.import_module(f"contextgate_core.{mod}").main), mod


def test_pyproject_ships_exactly_one_package_and_console_scripts_resolve():
    s = open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    assert 'packages = ["contextgate_core"]' in s and "py-modules" not in s
    for target in re.findall(r'^[\w-]+ = "([\w.]+):(\w+)"$', s, re.M):
        assert callable(getattr(importlib.import_module(target[0]), target[1])), target


def test_importing_the_package_is_light_and_the_public_api_is_lazy():
    code = ("import sys, contextgate_core; heavy = [m for m in ('contextgate_core.pipeline','contextgate_core.gate','spacy') if m in sys.modules];"
            "print(heavy); from contextgate_core import gate_text, optimize, estimate_tokens, gate_request; print(callable(gate_text))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.split("\n")
    assert out[0] == "[]" and out[1] == "True"


def test_no_module_uses_flat_imports_of_its_siblings():
    siblings = {f[:-3] for f in os.listdir(os.path.join(ROOT, "contextgate_core")) if f.endswith(".py")}
    for f in os.listdir(os.path.join(ROOT, "contextgate_core")):
        if not f.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(open(os.path.join(ROOT, "contextgate_core", f), encoding="utf-8").read())):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
            assert not (set(names) & siblings), (f, names)
