import json
import os
import subprocess
import sys

import pytest

from mcp_server import handle_message

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rpc(method, params=None, mid=1):
    return handle_message({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def test_initialize_negotiates():
    r = rpc("initialize", {"protocolVersion": "2024-11-05"})
    assert r["result"]["protocolVersion"] == "2024-11-05"
    assert rpc("initialize", {"protocolVersion": "1999"})["result"]["protocolVersion"] == "2025-06-18"


def test_notification_gets_no_response():
    assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list():
    tools = rpc("tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} >= {
        "smarty_optimize", "smarty_audit_response", "smarty_compress_context", "smarty_compress_tool_schema"}
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


def _call(name, args):
    r = rpc("tools/call", {"name": name, "arguments": args})["result"]
    return r["isError"], json.loads(r["content"][0]["text"])


def test_optimize_tool():
    err, out = _call("smarty_optimize", {"text": "Please write a function to reverse a string without slicing"})
    assert not err and out["payload"] and "trace" in out


def test_invalid_args_reported_in_band():
    err, out = _call("smarty_optimize", {"text": 3})
    assert err and out["error_code"] == "INVALID_REQUEST"


def test_compress_context_drops_irrelevant():
    err, out = _call("smarty_compress_context", {
        "query": "redis cache", "chunks": ["redis is a cache", "bananas are yellow", "redis cache eviction"]})
    assert not err and 1 not in out["kept_indices"]


def test_compress_schema():
    err, out = _call("smarty_compress_tool_schema", {"schema": {"name": "x", "description": "Please kindly use this in order to x"}})
    assert not err and out["tokens_saved"] >= 0


def test_audit_tool():
    err, out = _call("smarty_audit_response", {"text": "write code without loops", "response": "def f(): pass"})
    assert not err and out


def test_unknown_tool_and_method():
    assert rpc("tools/call", {"name": "nope"})["error"]["code"] == -32602
    assert rpc("bogus")["error"]["code"] == -32601


def test_stdio_end_to_end():
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    p = subprocess.run([sys.executable, os.path.join(ROOT, "mcp_server.py")], cwd=ROOT,
                       input="\n".join(json.dumps(m) for m in msgs) + "\nnot json\n",
                       capture_output=True, text=True, timeout=60)
    lines = [json.loads(l) for l in p.stdout.splitlines()]
    assert [l.get("id") for l in lines] == [1, 2, None]
    assert lines[2]["error"]["code"] == -32700


def test_repo_tools(monkeypatch, tmp_path):
    (tmp_path / "svc.py").write_text("def charge_card(amount):\n    '''Charge a credit card.'''\n    return amount\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    monkeypatch.setenv("SMARTY_ROOT", str(tmp_path))
    err, out = _call("smarty_get_context", {"query": "charge credit card amount"})
    assert not err and out["reason"] == "ok" and "charge_card" in out["capsule"]
    err, out = _call("smarty_find_symbol", {"name": "charge_card"})
    assert not err and out["matches"][0]["path"] == "svc.py"
    err, out = _call("smarty_get_slice", {"path": "svc.py", "start": 1, "end": 2})
    assert not err and "charge_card" in out["text"]
    err, out = _call("smarty_repo_map", {})
    assert not err and "svc.py" in out["repo_map"]
    for bad in ({"path": "../x", "start": 1, "end": 2}, {"path": ".env", "start": 1, "end": 1},
                {"path": "svc.py", "start": 5, "end": 2}, {"path": "svc.py", "start": 1, "end": 9999}):
        err, out = _call("smarty_get_slice", bad)
        assert err and out["error_code"] == "INVALID_REQUEST"


def test_resources(monkeypatch, tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    monkeypatch.setenv("SMARTY_ROOT", str(tmp_path))
    assert rpc("resources/list")["result"]["resources"][0]["uri"] == "smarty://repo_map"
    assert "a.py" in rpc("resources/read", {"uri": "smarty://repo_map"})["result"]["contents"][0]["text"]
    assert rpc("resources/read", {"uri": "smarty://nope"})["error"]


# ── server instructions (opt-in) ──────────────────────────────────────────────
def test_instructions_are_off_by_default(monkeypatch):
    monkeypatch.delenv("SMARTY_MCP_INSTRUCTIONS", raising=False)
    result = rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]
    assert "instructions" not in result and result["serverInfo"]["name"] == "nlp-smarty"


@pytest.mark.parametrize("value", ["on", "1", "true", "YES", " On "])
def test_instructions_opt_in_are_short_and_accurate(monkeypatch, value):
    monkeypatch.setenv("SMARTY_MCP_INSTRUCTIONS", value)
    import re

    import repo_indexer as ri
    from mcp_server import INSTRUCTIONS, TOOLS

    text = rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]["instructions"]
    assert text == INSTRUCTIONS and text.startswith("nlp-smarty index:")
    assert ri._heuristic_tokens(text) <= 125  # paid on every session when enabled: keep it small
    named = set(re.findall(r"smarty_[a-z_]+", text))
    assert named and named <= set(TOOLS), f"instructions mention tools that do not exist: {named - set(TOOLS)}"
    assert "smarty_find_references" in named and "grep" in text and "retrieved_context" in text


@pytest.mark.parametrize("value", ["off", "0", "false", "NO", "", "maybe"])
def test_anything_else_keeps_instructions_off(monkeypatch, value):
    monkeypatch.setenv("SMARTY_MCP_INSTRUCTIONS", value)
    assert "instructions" not in rpc("initialize", {})["result"]


def test_instructions_over_stdio():
    env = {**os.environ, "SMARTY_MCP_INSTRUCTIONS": "on"}
    p = subprocess.run([sys.executable, os.path.join(ROOT, "mcp_server.py")], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=60, input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n")
    assert json.loads(p.stdout.splitlines()[0])["result"]["instructions"].startswith("nlp-smarty index:")


# ── smarty_find_references ────────────────────────────────────────────────────
def test_find_references_tool(monkeypatch, tmp_path):
    (tmp_path / "core.py").write_text("def charge(amount):\n    return amount\n")
    (tmp_path / "api.py").write_text("from core import charge\n\ndef checkout(a):\n    return charge(a)\n")
    monkeypatch.setenv("SMARTY_ROOT", str(tmp_path))
    assert "smarty_find_references" in {t["name"] for t in rpc("tools/list")["result"]["tools"]}
    err, out = _call("smarty_find_references", {"name": "charge"})
    assert not err and out["counts"]["definitions"] == 1 and out["counts"]["callers"] == 1
    assert "api.py" in out["text"] and "checkout" in out["text"] and out["truncated"] is False
    for bad in ({"name": ""}, {"name": "a b"}, {"name": "x", "include_tests": "yes"}, {"name": "x", "token_budget": 5}):
        err, out = _call("smarty_find_references", bad)
        assert err and out["error_code"] == "INVALID_REQUEST"
