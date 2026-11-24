"""
OpenAI-compatible gateway proxy: point any SDK's base_url at http://127.0.0.1:8787/v1 and requests are gated, forwarded,
relayed (streaming included), post-audited and recorded in the ledger. Stdlib only.

  python gateway.py --upstream https://api.openai.com [--profile strict] [--port 8787]

Guarantees: fail-open (any gate error forwards the original request); credentials are forwarded untouched and never
logged or stored; the ledger stores counts only; binds to loopback unless --allow-remote; the response relayed to the
client is exactly what the upstream sent (the gateway only ADDS X-ContextGate-* headers).
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import os
import signal
import ssl
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import gate
import ledger as lg
import pipeline as _p
from data_contracts import OptimizationTrace

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade"}
MAX_BODY = 64 * 1024 * 1024


class Config:
    def __init__(self, upstream: str, profile: str = "strict", ledger: lg.Ledger | None = None, record_path: str | None = None,
                 timeout: float = 600.0, force_usage: bool = False, audit: bool = True, verbose: bool = False,
                 max_body: int = MAX_BODY) -> None:
        u = urllib.parse.urlsplit(upstream)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"--upstream must be an http(s) URL, got {upstream!r}")
        if profile not in gate.PROFILES:
            raise ValueError(f"unknown profile {profile!r}; choose from {', '.join(gate.PROFILES)}")
        self.upstream, self.scheme, self.host, self.port = upstream, u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.base_path = u.path.rstrip("/")
        self.profile, self.ledger, self.record_path = profile, ledger, record_path
        self.timeout, self.force_usage, self.audit, self.verbose, self.max_body = timeout, force_usage, audit, verbose, max_body


# ── helpers ───────────────────────────────────────────────────────────────────
def _last_user_text(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""


def audit_response(original_body: dict, response_text: str) -> dict[str, Any] | None:
    """Deterministic post-audit against constraints found in the ORIGINAL last user message.
    Transparent mode has no intent routing, so format checks do not apply; the negation check is the heuristic one in
    pipeline.audit_agent_response (it detects slicing and recursion violations only). entity_ok is therefore None."""
    if not response_text:
        return None
    text = _last_user_text(original_body)
    trace = OptimizationTrace()
    trace.negations = _p._extract_negations(text)
    trace.extracted_entities = _p._extract_entities(text, text)
    res = _p.audit_agent_response(response_text, trace)
    neg = next((c["passed"] for c in res["checks"] if c["check"].startswith("Negation")), None)
    return {"score": res["compliance_score"], "neg_ok": neg, "entity_ok": None}


def _read_request_body(handler: BaseHTTPRequestHandler, limit: int) -> bytes | None:
    te = (handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in te:
        data = b""
        while True:
            size = int(handler.rfile.readline().split(b";")[0].strip() or b"0", 16)
            if size == 0:
                handler.rfile.readline()
                return data
            data += handler.rfile.read(size)
            handler.rfile.readline()
            if len(data) > limit:
                return None
    n = int(handler.headers.get("Content-Length") or 0)
    if n > limit:
        return None
    return handler.rfile.read(n) if n else b""


def _extract_text_and_usage(payload: dict) -> tuple[str, dict]:
    text = ""
    try:
        text = payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    return text if isinstance(text, str) else "", (payload.get("usage") or {})


def make_server(host: str, port: int, cfg: Config) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        server_version = "contextgate/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # request lines only, never bodies or headers
            if cfg.verbose:
                sys.stderr.write("[gateway] " + (fmt % args) + "\n")

        # every verb goes through the same relay
        def do_GET(self) -> None: self._handle()
        def do_POST(self) -> None: self._handle()
        def do_PUT(self) -> None: self._handle()
        def do_PATCH(self) -> None: self._handle()
        def do_DELETE(self) -> None: self._handle()
        def do_OPTIONS(self) -> None: self._handle()

        def _send_error_json(self, status: int, message: str, kind: str) -> None:
            body = json.dumps({"error": {"message": message, "type": kind}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self) -> None:
            t_req = time.perf_counter()
            raw = _read_request_body(self, cfg.max_body) if self.command in ("POST", "PUT", "PATCH") else b""
            if raw is None:
                return self._send_error_json(413, "request body too large", "contextgate_request_too_large")
            path = self.path
            is_chat = self.command == "POST" and urllib.parse.urlsplit(path).path.endswith("/chat/completions")
            original: dict | None = None
            forward_body, res, error = raw, None, ""
            if is_chat:
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                    original = parsed if isinstance(parsed, dict) else None
                except (ValueError, UnicodeDecodeError):
                    original = None
                if original is not None:
                    try:
                        res = gate.gate_openai_request(original, cfg.profile)
                        send = res.body
                        if cfg.force_usage and original.get("stream") and "stream_options" not in send:
                            send = {**send, "stream_options": {"include_usage": True}}
                        if res.applied or send is not res.body:
                            forward_body = json.dumps(send, ensure_ascii=False).encode("utf-8")
                        error = res.error
                        if cfg.record_path:
                            self._record(original)
                    except Exception as exc:  # fail open: forward exactly what we received
                        res, forward_body, error = None, raw, f"{type(exc).__name__}"
            gate_ms = (time.perf_counter() - t_req) * 1000
            self._relay(path, forward_body, original, res, gate_ms, error, is_chat)

        def _record(self, body: dict) -> None:
            fd = os.open(cfg.record_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "body": body}, ensure_ascii=False) + "\n")

        def _relay(self, path: str, body: bytes, original: dict | None, res: Any, gate_ms: float, error: str, is_chat: bool) -> None:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP | {"host", "content-length", "accept-encoding"}}
            headers["Host"] = cfg.host if cfg.port in (80, 443) else f"{cfg.host}:{cfg.port}"
            headers["Accept-Encoding"] = "identity"
            if body or self.command in ("POST", "PUT", "PATCH"):
                headers["Content-Length"] = str(len(body))
            row = dict(route=urllib.parse.urlsplit(path).path, host=cfg.host, model=(original or {}).get("model"), profile=cfg.profile,
                       stream=bool((original or {}).get("stream")), applied=bool(res and res.applied),
                       raw_tokens=res.raw_tokens if res else None, fwd_tokens=res.fwd_tokens if res else None,
                       gate_ms=gate_ms, fallbacks=res.fallbacks if res else 0, protected_ok=res.protected_ok if res else True,
                       error=error or None)
            t_up = time.perf_counter()
            conn = None
            try:
                if cfg.scheme == "https":
                    conn = http.client.HTTPSConnection(cfg.host, cfg.port, timeout=cfg.timeout, context=ssl.create_default_context())
                else:
                    conn = http.client.HTTPConnection(cfg.host, cfg.port, timeout=cfg.timeout)
                conn.request(self.command, cfg.base_path + path, body=body if (body or self.command in ("POST", "PUT", "PATCH")) else None,
                             headers=headers)
                resp = conn.getresponse()
            except Exception as exc:
                self._send_error_json(502, f"upstream unreachable: {type(exc).__name__}", "contextgate_upstream_error")
                row.update(status=502, error=f"upstream {type(exc).__name__}", upstream_ms=(time.perf_counter() - t_up) * 1000)
                self._ledger(is_chat, row)
                if conn:
                    conn.close()
                return
            ctype = (resp.getheader("Content-Type") or "").lower()
            streaming = "text/event-stream" in ctype
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() in HOP_BY_HOP or k.lower() == "content-length":
                    continue
                self.send_header(k, v)
            if res is not None:
                self.send_header("X-ContextGate-Profile", cfg.profile)
                self.send_header("X-ContextGate-Applied", "1" if res.applied else "0")
                self.send_header("X-ContextGate-Tokens-Saved-Est", str(max(0, res.saved)))
            row["status"] = resp.status
            try:
                if streaming:
                    self.end_headers()
                    acc, usage = [], {}
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
                        if line.startswith(b"data:") and b"[DONE]" not in line:
                            try:
                                chunk = json.loads(line[5:].decode("utf-8", "replace"))
                                usage = chunk.get("usage") or usage
                                acc.append((chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or "")
                            except (ValueError, AttributeError, IndexError, TypeError):
                                pass
                    text = "".join(x for x in acc if isinstance(x, str))
                else:
                    data = resp.read()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    text, usage = "", {}
                    enc = (resp.getheader("Content-Encoding") or "").lower()
                    if is_chat and resp.status == 200 and enc in ("", "identity", "gzip"):
                        try:
                            payload = json.loads((gzip.decompress(data) if enc == "gzip" else data).decode("utf-8"))
                            text, usage = _extract_text_and_usage(payload)
                        except (ValueError, OSError, UnicodeDecodeError):
                            pass
                row["upstream_ms"] = (time.perf_counter() - t_up) * 1000
                row["up_prompt_tokens"] = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                row["up_completion_tokens"] = usage.get("completion_tokens") if isinstance(usage, dict) else None
                if cfg.audit and is_chat and original is not None and resp.status == 200:
                    try:
                        a = audit_response(original, text)
                        if a:
                            row.update(audit_score=a["score"], neg_ok=a["neg_ok"], entity_ok=a["entity_ok"])
                    except Exception:
                        row["error"] = row.get("error") or "AuditError"
            except (BrokenPipeError, ConnectionResetError):
                row["error"] = row.get("error") or "client_disconnected"
                row["upstream_ms"] = (time.perf_counter() - t_up) * 1000
            finally:
                conn.close()
                self._ledger(is_chat, row)

        def _ledger(self, is_chat: bool, row: dict) -> None:
            if cfg.ledger is not None and is_chat:
                try:
                    cfg.ledger.record(**row)
                except Exception:
                    pass

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = False  # server_close() then WAITS for in-flight requests, so their ledger rows are never lost
    srv.block_on_close = True
    return srv


def serve(host: str, port: int, cfg: Config) -> None:
    srv = make_server(host, port, cfg)
    print(f"gateway listening on http://{host}:{srv.server_address[1]}  ->  {cfg.upstream}  (profile: {cfg.profile}, "
          f"ledger: {cfg.ledger.path if cfg.ledger else 'off'})", file=sys.stderr)
    print(f"point your SDK at base_url=http://{host}:{srv.server_address[1]}/v1", file=sys.stderr)
    def _stop(*_: Any) -> None:  # SIGTERM (docker stop, systemd, kill) shuts down gracefully like Ctrl-C
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _stop)
    except ValueError:  # not the main thread (embedded use): the caller owns shutdown
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()  # waits for in-flight handler threads
        if cfg.ledger is not None:
            cfg.ledger.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="gateway", description="OpenAI-compatible compression gateway")
    ap.add_argument("--upstream", default=os.environ.get("SMARTY_UPSTREAM"), required=not os.environ.get("SMARTY_UPSTREAM"),
                    help="upstream base URL, e.g. https://api.openai.com or https://router.huggingface.co")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--allow-remote", action="store_true", help="allow binding to a non-loopback address (credentials pass through!)")
    ap.add_argument("--profile", default="strict", choices=sorted(gate.PROFILES))
    ap.add_argument("--ledger", default=lg.DEFAULT_PATH)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--record", help="append ORIGINAL request bodies to this JSONL file for offline `bench` (contains your prompts!)")
    ap.add_argument("--force-usage", action="store_true", help="add stream_options.include_usage to streaming requests")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.allow_remote:
        print("error: refusing to bind to a non-loopback address without --allow-remote "
              "(the gateway forwards your API credentials)", file=sys.stderr)
        return 2
    try:
        cfg = Config(a.upstream, a.profile, None if a.no_ledger else lg.Ledger(a.ledger), a.record, a.timeout,
                     a.force_usage, not a.no_audit, a.verbose)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if a.record:
        print(f"WARNING: recording original request bodies (your prompts) to {a.record}", file=sys.stderr)
    serve(a.host, a.port, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
