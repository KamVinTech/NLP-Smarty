import http.client
import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from contextgate_core import gate
from contextgate_core import gateway
from contextgate_core import ledger as lg

FILLER = "Hey! Could you please help me and I was wondering if you could basically fix this, in order to be safe? "
PROMPT = FILLER + "Do not use recursion and keep `get_user_by_id` and https://example.com/a?b=1 verbatim, without using slicing."
TOOLS = [{"type": "function", "function": {"name": "f", "description": "Could you please use this tool in order to fetch the user.",
                                           "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}]
SECRET = "Bearer sk-test-SECRET-TOKEN-123"


class FakeUpstream:
    def __init__(self):
        self.seen, self.mode, self.reply = [], "json", "def f(x):\n    return x\n"
        self.release = threading.Event()
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _go(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                owner.seen.append({"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                if owner.mode == "error":
                    data = json.dumps({"error": {"message": "slow down"}}).encode()
                    self.send_response(429); self.send_header("Content-Type", "application/json")
                    self.send_header("Retry-After", "7"); self.send_header("Content-Length", str(len(data))); self.end_headers()
                    return self.wfile.write(data)
                if owner.mode == "sse":
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    chunk = lambda d, extra=None: ("data: " + json.dumps({"choices": [{"delta": {"content": d}}], **(extra or {})}) + "\n\n").encode()
                    self.wfile.write(chunk("Hel")); self.wfile.flush()
                    owner.release.wait(5)
                    self.wfile.write(chunk("lo [::-1]")); self.wfile.write(chunk("", {"usage": {"prompt_tokens": 55, "completion_tokens": 9}}))
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                    return
                payload = {"id": "x", "model": "m1", "choices": [{"message": {"role": "assistant", "content": owner.reply}}],
                           "usage": {"prompt_tokens": 123, "completion_tokens": 7}} if self.path.endswith("/chat/completions") else {"data": [{"id": "m1"}]}
                data = json.dumps(payload).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            do_GET = do_POST = _go

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.release.set()
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def up():
    u = FakeUpstream()
    yield u
    u.stop()


@pytest.fixture
def make_gw(tmp_path, up):
    started = []

    def make(**kw):
        led = lg.Ledger(str(tmp_path / "l.db"))
        cfg = gateway.Config(kw.pop("upstream", f"http://127.0.0.1:{up.port}"), ledger=led, **kw)
        srv = gateway.make_server("127.0.0.1", 0, cfg)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append((srv, led))
        return srv.server_address[1], led
    yield make
    for srv, led in started:
        srv.shutdown(); srv.server_close(); led.close()


def wait_rows(led, n=1, timeout=5.0):
    """The ledger row is written AFTER the response is sent (no added latency), so wait for it instead of racing."""
    import time
    end = time.time() + timeout
    while len(led.rows()) < n and time.time() < end:
        time.sleep(0.02)
    return led.rows()


def post(port, path, body, headers=None, raw=False):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = body if raw else json.dumps(body).encode()
    c.request("POST", path, body=data, headers={"Content-Type": "application/json", "Authorization": SECRET, **(headers or {})})
    return c, c.getresponse()


def chat(**kw):
    return {"model": "m1", "messages": [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": PROMPT}], "tools": TOOLS, **kw}


def test_request_is_compressed_credentials_forwarded_response_relayed_and_recorded(make_gw, up):
    port, led = make_gw()
    c, r = post(port, "/v1/chat/completions", chat())
    body = r.read()
    sent = up.seen[0]
    got = json.loads(sent["body"])
    assert r.status == 200 and json.loads(body)["usage"]["prompt_tokens"] == 123  # upstream response relayed
    assert gate.estimate_tokens(sent["body"].decode()) < gate.estimate_tokens(json.dumps(chat()))
    user = got["messages"][1]["content"]
    for span in ("`get_user_by_id`", "https://example.com/a?b=1", "Do not use recursion", "without using slicing"):
        assert span in user
    assert "Could you please" not in user and got["messages"][0] == chat()["messages"][0]  # filler gone, system untouched
    assert sent["headers"]["authorization"] == SECRET and sent["headers"]["accept-encoding"] == "identity"
    assert sent["headers"]["host"] == f"127.0.0.1:{up.port}" and sent["path"] == "/v1/chat/completions"
    assert r.getheader("X-ContextGate-Applied") == "1" and int(r.getheader("X-ContextGate-Tokens-Saved-Est")) > 0
    row = wait_rows(led)[0]
    assert row["model"] == "m1" and row["applied"] == 1 and row["up_prompt_tokens"] == 123 and row["up_completion_tokens"] == 7
    assert row["raw_tokens"] > row["fwd_tokens"] and row["status"] == 200 and row["audit_score"] == 100
    c.close()


def test_ledger_and_logs_never_contain_prompt_text_or_credentials(make_gw, up, tmp_path):
    port, led = make_gw()
    post(port, "/v1/chat/completions", chat())[1].read()
    wait_rows(led)
    led.close()
    raw = open(tmp_path / "l.db", "rb").read() + (open(str(tmp_path / "l.db") + "-wal", "rb").read() if os.path.exists(str(tmp_path / "l.db") + "-wal") else b"")
    for needle in (b"get_user_by_id", b"example.com", b"sk-test-SECRET", b"Bearer", b"recursion"):
        assert needle not in raw


def test_streaming_is_relayed_incrementally_and_audited_at_the_end(make_gw, up):
    up.mode = "sse"
    port, led = make_gw()
    c, r = post(port, "/v1/chat/completions", chat(stream=True))
    first = r.fp.readline()  # arrives while the upstream is still waiting: proves the stream is not buffered
    assert first.startswith(b"data:") and b"Hel" in first and not up.release.is_set()
    up.release.set()
    rest = r.read()
    assert b"lo [::-1]" in rest and rest.rstrip().endswith(b"[DONE]")
    c.close()
    import time
    for _ in range(50):
        if led.rows():
            break
        time.sleep(0.05)
    row = led.rows()[0]
    assert row["stream"] == 1 and row["up_prompt_tokens"] == 55 and row["up_completion_tokens"] == 9
    assert row["audit_score"] == 60 and row["neg_ok"] == 0  # "without using slicing" violated by [::-1]


def test_upstream_errors_are_relayed_unchanged_and_counted(make_gw, up):
    up.mode = "error"
    port, led = make_gw()
    c, r = post(port, "/v1/chat/completions", chat())
    assert r.status == 429 and r.getheader("Retry-After") == "7" and json.loads(r.read())["error"]["message"] == "slow down"
    assert lg.summarize(wait_rows(led))["upstream_errors"] == 1
    c.close()


def test_gate_failure_fails_open_with_the_exact_original_bytes(make_gw, up, monkeypatch):
    port, led = make_gw()
    monkeypatch.setattr(gateway.gate, "gate_openai_request", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    original = json.dumps(chat()).encode()
    c, r = post(port, "/v1/chat/completions", original, raw=True)
    assert r.status == 200 and up.seen[0]["body"] == original
    row = wait_rows(led)[0]
    assert row["error"] == "RuntimeError" and row["applied"] == 0
    c.close()


def test_invalid_json_and_other_routes_pass_through_untouched(make_gw, up):
    port, led = make_gw()
    c, r = post(port, "/v1/chat/completions", b"not json at all", raw=True)
    assert r.status == 200 and up.seen[-1]["body"] == b"not json at all"
    c.close()
    emb = json.dumps({"input": FILLER + "text", "model": "e"}).encode()
    c, r = post(port, "/v1/embeddings", emb, raw=True)
    assert up.seen[-1]["body"] == emb and up.seen[-1]["path"] == "/v1/embeddings"  # only chat is gated
    c.close()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/v1/models", headers={"Authorization": SECRET})
    r = c.getresponse()
    assert r.status == 200 and json.loads(r.read())["data"][0]["id"] == "m1" and up.seen[-1]["method"] == "GET"
    assert all(row["route"].endswith("/chat/completions") for row in wait_rows(led))  # non-chat traffic is not ledgered
    c.close()


def test_unreachable_upstream_returns_502_and_is_recorded(make_gw):
    port, led = make_gw(upstream="http://127.0.0.1:1")
    c, r = post(port, "/v1/chat/completions", chat())
    assert r.status == 502 and json.loads(r.read())["error"]["type"] == "contextgate_upstream_error"
    assert wait_rows(led)[0]["error"].startswith("upstream")
    c.close()


def test_record_writes_original_bodies_with_private_permissions(make_gw, up, tmp_path):
    rec = str(tmp_path / "rec.jsonl")
    port, led = make_gw(record_path=rec)
    post(port, "/v1/chat/completions", chat())[1].read()
    line = json.loads(open(rec).readline())
    assert line["body"]["messages"][1]["content"] == PROMPT  # the ORIGINAL, not the compressed text
    assert stat.S_IMODE(os.stat(rec).st_mode) == 0o600


def test_default_run_records_nothing(make_gw, up, tmp_path):
    port, led = make_gw()
    post(port, "/v1/chat/completions", chat())[1].read()
    assert not (tmp_path / "rec.jsonl").exists()


def test_oversized_bodies_are_rejected_and_chunked_bodies_work(make_gw, up):
    port, led = make_gw(max_body=1000)
    c, r = post(port, "/v1/chat/completions", b"x" * 2000, raw=True)
    assert r.status == 413
    c.close()
    port2, _ = make_gw()
    c = http.client.HTTPConnection("127.0.0.1", port2, timeout=10)
    data = json.dumps(chat()).encode()
    c.request("POST", "/v1/chat/completions", body=iter([data[:50], data[50:]]), headers={"Authorization": SECRET, "Transfer-Encoding": "chunked", "Content-Type": "application/json"}, encode_chunked=True)
    assert c.getresponse().status == 200 and json.loads(up.seen[-1]["body"])["model"] == "m1"
    c.close()


def test_force_usage_only_when_asked(make_gw, up):
    up.mode = "sse"; up.release.set()
    port, _ = make_gw(force_usage=True)
    post(port, "/v1/chat/completions", chat(stream=True))[1].read()
    assert json.loads(up.seen[-1]["body"])["stream_options"] == {"include_usage": True}
    port2, _ = make_gw()
    post(port2, "/v1/chat/completions", chat(stream=True))[1].read()
    assert "stream_options" not in json.loads(up.seen[-1]["body"])


def test_profile_and_config_validation_and_loopback_guard(capsys):
    with pytest.raises(ValueError):
        gateway.Config("ftp://x")
    with pytest.raises(ValueError):
        gateway.Config("http://x", profile="reckless")
    assert gateway.main(["--upstream", "http://example.com", "--host", "0.0.0.0", "--no-ledger"]) == 2
    assert "non-loopback" in capsys.readouterr().err
    assert gateway.main(["--upstream", "nope", "--no-ledger"]) == 2


def test_sigterm_right_after_a_response_does_not_lose_the_ledger_row(up, tmp_path):
    """Regression: a request answered just before SIGTERM used to lose its ledger row (daemon handler threads)."""
    import socket
    import subprocess
    import sys
    import time

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    db = str(tmp_path / "t.db")
    proc = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gateway.py"),
                             "--upstream", f"http://127.0.0.1:{up.port}", "--port", str(port), "--ledger", db],
                            stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        for _ in range(6):
            c, r = post(port, "/v1/chat/completions", chat())
            r.read()
            c.close()
        proc.terminate()  # immediately after the last response
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    led = lg.Ledger(db)
    assert len(led.rows()) == 6
    led.close()
