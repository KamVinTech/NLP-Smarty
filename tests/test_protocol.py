import json

import pytest

from bridge_server import process_line
from protocol import PROTOCOL_VERSION, ProtocolError, validate_request


def call(obj):
    return process_line(obj if isinstance(obj, str) else json.dumps(obj))


def test_ping_has_version():
    r = call({"id": "1", "command": "ping"})
    assert r["ok"] and r["protocol_version"] == PROTOCOL_VERSION


def test_capabilities():
    r = call({"id": "1", "command": "capabilities"})
    assert r["ok"] and "optimize" in r["commands"]


def test_invalid_json():
    r = call("{nope")
    assert not r["ok"] and r["error_code"] == "INVALID_JSON" and r["id"] == "unknown"


def test_unknown_command_not_silently_optimized():
    r = call({"id": "x", "command": "bogus", "text": "hi"})
    assert r["error_code"] == "UNKNOWN_COMMAND" and r["id"] == "x"


@pytest.mark.parametrize("bad", [
    {"text": 5},
    {"text": ""},
    {"text": "a", "max_total_tokens": -1},
    {"text": "a", "max_total_tokens": True},
    {"text": "a", "context_chunks": "str"},
    {"text": "a", "conversation_history": [{"role": "user"}]},
    {"text": "a", "compression_intensity": "EXTREME"},
    {"text": "a", "enable_cache": "yes"},
])
def test_invalid_requests(bad):
    r = call({"id": "1", "command": "optimize", **bad})
    assert not r["ok"] and r["error_code"] == "INVALID_REQUEST" and isinstance(r["error"], str)


def test_non_object_request():
    with pytest.raises(ProtocolError):
        validate_request([1, 2])
    assert call("[1]")["error_code"] == "INVALID_REQUEST"


def test_optimize_roundtrip():
    r = call({"id": "1", "command": "optimize", "text": "Please write a function to add two numbers"})
    assert r["ok"] and r["payload"] and r["protocol_version"] == PROTOCOL_VERSION


def test_audit_pair():
    r = call({"id": "1", "command": "audit_pair", "text": "write a function without using loops",
              "raw_response": "def f(): pass", "optimized_response": "def f(): pass"})
    assert r["ok"] and "optimized_audit" in r
