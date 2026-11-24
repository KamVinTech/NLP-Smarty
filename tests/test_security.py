"""Adversarial inputs against the gate and the proxy: each must be rejected or fail safe, never hang or crash."""
import http.client
import json
import os
import socket
import stat
import time

import pytest

from contextgate import gate
from tests.test_gateway import PROMPT, SECRET, chat, make_gw, post, up  # noqa: F401  (fixtures)


@pytest.mark.parametrize("text", ["{" * 100_000, "[" * 100_000, '{"a": ' * 30_000, "[1," * 60_000, '["a"] ' * 30_000])
def test_gate_json_scan_is_bounded_and_fails_safe(text):
    t0 = time.time()
    out, _ok = gate.gate_text(text, 2)
    assert time.time() - t0 < 5
    assert out == text or isinstance(out, str)


def test_gate_still_protects_real_json():
    doc = 'Please basically fix this: {"keep": "  spaced   value ", "n": [1, 2, 3]} in order to be safe.'
    out, ok = gate.gate_text(doc, 2)
    assert ok and '{"keep": "  spaced   value ", "n": [1, 2, 3]}' in out


def _raw(port, payload: bytes, wait=2.0) -> bytes:
    s = socket.create_connection(("127.0.0.1", port), timeout=wait)
    s.sendall(payload)
    s.settimeout(wait)
    try:
        return s.recv(65536)
    except socket.timeout:
        return b""
    finally:
        s.close()


@pytest.mark.parametrize("target", ["/v1/../admin", "/v1/%2e%2e/admin", "/v1/%2E%2E%2fadmin", "http://evil.example/v1/x", "/v1/a\\b"])
def test_bad_request_targets_are_rejected_not_forwarded(make_gw, up, target):
    port, _ = make_gw()
    out = _raw(port, f"GET {target} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    assert out.startswith(b"HTTP/1.") and b" 400 " in out.split(b"\r\n")[0]
    assert not up.seen


@pytest.mark.parametrize("cl", ["-5", "abc", "1e3", "0x10"])
def test_bad_content_length_is_400(make_gw, up, cl):
    port, _ = make_gw()
    out = _raw(port, f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {cl}\r\n\r\n".encode())
    assert b" 400 " in out.split(b"\r\n")[0]
    assert not up.seen


def test_te_plus_content_length_is_rejected(make_gw, up):
    port, _ = make_gw()
    out = _raw(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n0\r\n\r\n")
    assert b" 400 " in out.split(b"\r\n")[0] and not up.seen


@pytest.mark.parametrize("size", ["zz", "-1", "FFFFFFFFFFFFFFFF"])
def test_bad_or_huge_chunk_sizes_never_hang_or_allocate(make_gw, up, size):
    port, _ = make_gw(max_body=1000)
    t0 = time.time()
    out = _raw(port, f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n{size}\r\nabc".encode())
    assert time.time() - t0 < 3
    status = out.split(b"\r\n")[0]
    assert b" 400 " in status or b" 413 " in status
    assert not up.seen


def test_deeply_nested_json_does_not_crash_and_is_forwarded_verbatim(make_gw, up):
    port, _ = make_gw()
    body = b"[" * 200_000 + b"]" * 200_000
    c, r = post(port, "/v1/chat/completions", body, raw=True)
    r.read()
    assert r.status == 200
    assert up.seen[-1]["body"] == body
    c.close()


def test_server_survives_the_abuse(make_gw, up):
    port, _ = make_gw()
    _raw(port, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n")
    c, r = post(port, "/v1/chat/completions", chat())
    assert r.status == 200
    r.read(); c.close()


def test_record_file_permissions_are_tightened_when_preexisting(make_gw, up, tmp_path):
    rec = tmp_path / "rec.jsonl"
    rec.write_text("")
    os.chmod(rec, 0o666)
    port, _ = make_gw(record_path=str(rec))
    post(port, "/v1/chat/completions", chat())[1].read()
    assert stat.S_IMODE(os.stat(rec).st_mode) == 0o600


def test_verbose_log_cannot_be_injected_via_newlines(make_gw, up, capsys):
    port, _ = make_gw(verbose=True)
    _raw(port, b"GET /v1/models?x=%0d%0aFAKE HTTP/1.1\r\nHost: x\r\n\r\n")
    assert "\r" not in capsys.readouterr().err


@pytest.mark.parametrize("text", ["[1," * 60_000, "[" * 100_000, '{"a": ' * 30_000])
def test_gate_json_decoder_work_is_bounded_by_attempts_not_by_how_fast_this_python_is(text, monkeypatch):
    """A wall-clock assertion is machine-dependent (3.13 recurses much deeper in the C decoder than 3.9): count the work instead."""
    calls = []
    real = gate._JSON_DECODER.raw_decode

    class Counting:
        def raw_decode(self, s):
            calls.append(1)
            return real(s)

    monkeypatch.setattr(gate, "_JSON_DECODER", Counting())
    gate.gate_text(text, 2)
    assert len(calls) <= 100, len(calls)
