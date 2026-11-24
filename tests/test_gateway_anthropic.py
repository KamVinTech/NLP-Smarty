import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from contextgate_core import gate
from contextgate_core import gateway
from contextgate_core import ledger as lg

KEY = "sk-ant-test-SECRET-KEY"
HDRS = {"x-api-key": KEY, "anthropic-version": "2023-06-01", "anthropic-beta": "prompt-caching-2024-07-31"}
REMINDER = "<system-reminder>\nInstructions: please always just be careful, basically.\n</system-reminder>"
SIG = "CAQSmgUKEAgUGAI4AUIIdGhpbmtpbmc" * 4
FILLER = "Hey! Could you please help me and I was wondering if you could basically fix this, in order to be safe? "
USAGE = {"input_tokens": 20, "cache_read_input_tokens": 700, "cache_creation_input_tokens": 100, "output_tokens": 9}


def body(**kw):
    return {"model": "claude-test", "max_tokens": 64,
            "system": [{"type": "text", "text": "Could you please be basically helpful.", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": REMINDER},
                                             {"type": "text", "text": FILLER + "Keep `get_user_by_id` verbatim and do not use slicing."}]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": SIG},
                                                  {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls  -la"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file_a  file_b"}]}],
            "tools": [{"name": "Bash", "description": "Could you please use this tool in order to run commands.",
                       "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}, "cache_control": {"type": "ephemeral"}}],
            **kw}


class FakeAnthropic:
    def __init__(self):
        self.seen, self.mode, self.reply = [], "json", "Done."
        self.release = threading.Event()
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _go(self):
                n = int(self.headers.get("Content-Length") or 0)
                owner.seen.append({"method": self.command, "path": self.path, "body": self.rfile.read(n) if n else b"",
                                   "headers": {k.lower(): v for k, v in self.headers.items()}})
                if owner.mode == "error":
                    data = json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}).encode()
                    self.send_response(529); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                    self.end_headers(); return self.wfile.write(data)
                if owner.mode == "sse":
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    ev = lambda name, d: f"event: {name}\ndata: {json.dumps(d)}\n\n".encode()
                    self.wfile.write(ev("message_start", {"type": "message_start", "message": {"id": "m", "usage": {**USAGE, "output_tokens": 1}}}))
                    self.wfile.write(ev("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
                    self.wfile.write(ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}))
                    self.wfile.flush()
                    owner.release.wait(5)
                    self.wfile.write(ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo [::-1]"}}))
                    self.wfile.write(ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}}))
                    self.wfile.write(ev("message_stop", {"type": "message_stop"})); self.wfile.flush()
                    return
                data = json.dumps({"id": "m", "type": "message", "role": "assistant", "model": "claude-test",
                                   "content": [{"type": "text", "text": owner.reply}, {"type": "tool_use", "id": "t", "name": "Bash", "input": {}}],
                                   "usage": USAGE}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)

            do_GET = do_POST = _go

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.release.set(); self.srv.shutdown(); self.srv.server_close()


@pytest.fixture
def up():
    u = FakeAnthropic()
    yield u
    u.stop()


@pytest.fixture
def gw(tmp_path, up):
    started = []

    def make(**kw):
        led = lg.Ledger(str(tmp_path / "l.db"))
        srv = gateway.make_server("127.0.0.1", 0, gateway.Config(f"http://127.0.0.1:{up.port}", ledger=led, **kw))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append((srv, led))
        return srv.server_address[1], led
    yield make
    for srv, led in started:
        srv.shutdown(); srv.server_close(); led.close()


def wait_rows(led, n=1):
    end = time.time() + 5
    while len(led.rows()) < n and time.time() < end:
        time.sleep(0.02)
    return led.rows()


def post(port, path, payload, raw=False, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, body=payload if raw else json.dumps(payload).encode(), headers={"Content-Type": "application/json", **HDRS, **(headers or {})})
    return c, c.getresponse()


def test_messages_request_is_gated_safely_and_usage_is_normalised(gw, up):
    port, led = gw()
    original = body()
    c, r = post(port, "/v1/messages", original)
    data = r.read()
    sent = json.loads(up.seen[0]["body"])
    assert r.status == 200 and json.loads(data)["usage"] == USAGE  # response relayed as the upstream sent it
    typed = sent["messages"][0]["content"][1]["text"]
    assert "`get_user_by_id`" in typed and "do not use slicing" in typed and "Could you please" not in typed
    assert sent["messages"][0]["content"][0]["text"] == REMINDER  # system-reminder wrapper untouched
    assert sent["messages"][1] == original["messages"][1]  # thinking signature + tool_use byte-identical
    assert sent["messages"][2] == original["messages"][2] and sent["system"] == original["system"]
    assert sent["tools"][0]["cache_control"] == {"type": "ephemeral"} and "in order to" not in sent["tools"][0]["description"]
    h = up.seen[0]["headers"]
    assert h["x-api-key"] == KEY and h["anthropic-version"] == "2023-06-01" and h["anthropic-beta"] == "prompt-caching-2024-07-31"
    assert r.getheader("X-ContextGate-Applied") == "1"
    row = wait_rows(led)[0]
    assert row["api"] == "anthropic" and row["model"] == "claude-test" and row["applied"] == 1
    assert row["up_prompt_tokens"] == 20 + 700 + 100  # fresh + cache reads + cache writes = the whole prompt
    assert row["up_cache_read_tokens"] == 700 and row["up_cache_write_tokens"] == 100 and row["up_completion_tokens"] == 9
    assert row["raw_tokens"] > row["fwd_tokens"] and row["audit_score"] == 100
    c.close()


def test_stream_events_are_relayed_incrementally_and_usage_audited_at_the_end(gw, up):
    up.mode = "sse"
    port, led = gw()
    c, r = post(port, "/v1/messages", body(stream=True))
    first = r.fp.readline() + r.fp.readline()
    assert b"message_start" in first and not up.release.is_set()  # arrives while the upstream is still mid-stream
    up.release.set()
    rest = r.read()
    assert b"lo [::-1]" in rest and rest.rstrip().endswith(b'"type": "message_stop"}')
    c.close()
    row = wait_rows(led)[0]
    assert row["stream"] == 1 and row["api"] == "anthropic" and row["up_prompt_tokens"] == 820
    assert row["up_completion_tokens"] == 9  # the final message_delta wins over message_start's placeholder
    assert row["audit_score"] == 60 and row["neg_ok"] == 0  # "do not use slicing" violated by [::-1]


def test_count_tokens_and_other_routes_pass_through_ungated(gw, up):
    port, led = gw()
    raw = json.dumps(body()).encode()
    c, r = post(port, "/v1/messages/count_tokens", raw, raw=True)
    assert up.seen[-1]["body"] == raw and up.seen[-1]["path"] == "/v1/messages/count_tokens"  # byte-identical
    c.close()
    time.sleep(0.2)
    assert led.rows() == []  # not a gated route, so not ledgered


def test_anthropic_errors_are_relayed_unchanged(gw, up):
    up.mode = "error"
    port, led = gw()
    c, r = post(port, "/v1/messages", body())
    assert r.status == 529 and json.loads(r.read())["error"]["type"] == "overloaded_error"
    assert lg.summarize(wait_rows(led))["upstream_errors"] == 1
    c.close()


def test_credentials_never_reach_the_ledger_file(gw, up, tmp_path):
    port, led = gw()
    post(port, "/v1/messages", body())[1].read()
    wait_rows(led)
    led.close()
    import os
    raw = open(tmp_path / "l.db", "rb").read()
    wal = str(tmp_path / "l.db") + "-wal"
    raw += open(wal, "rb").read() if os.path.exists(wal) else b""
    for needle in (b"sk-ant-test", b"get_user_by_id", b"prompt-caching", b"SECRET"):
        assert needle not in raw


# ── units ─────────────────────────────────────────────────────────────────────
def test_route_kind():
    assert gateway.route_kind("POST", "/v1/chat/completions") == "openai"
    assert gateway.route_kind("POST", "/v1/messages?beta=true") == "anthropic"
    assert gateway.route_kind("POST", "/v1/messages/count_tokens") is None
    assert gateway.route_kind("POST", "/v1/messages/batches") is None
    assert gateway.route_kind("GET", "/v1/messages") is None and gateway.route_kind("POST", "/v1/embeddings") is None


def test_user_text_for_audit_skips_tool_results_and_reminders():
    b = {"messages": [{"role": "user", "content": [{"type": "text", "text": REMINDER + "\nFix it without using recursion."}]},
                      {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]}]}
    assert gateway._user_text(b, "anthropic") == "Fix it without using recursion."  # the instruction, found behind a tool turn
    assert gateway._user_text({"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}, "openai") == "b"


def test_usage_normalisation_and_stream_state():
    assert gateway._anthropic_usage(USAGE) == {"prompt": 820, "completion": 9, "cache_read": 700, "cache_write": 100}
    assert gateway._anthropic_usage({}) == {"prompt": None, "completion": None, "cache_read": None, "cache_write": None}
    assert gateway._openai_usage({"prompt_tokens": 50, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 40}}) == \
        {"prompt": 50, "completion": 3, "cache_read": 40, "cache_write": None}
    st = gateway.StreamState("anthropic")
    for ev in ({"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 1}}},
               {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
               {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{}"}},
               {"type": "message_delta", "usage": {"output_tokens": 7}}):
        st.feed(b"data: " + json.dumps(ev).encode())
    st.feed(b"event: ping"); st.feed(b"data: [DONE]"); st.feed(b"data: not json")
    assert st.result() == ("hi", {"prompt": 5, "completion": 7, "cache_read": 0, "cache_write": 0})
