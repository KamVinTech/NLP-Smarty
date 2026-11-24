"""
NLP-Smarty MCP server (stdio, newline-delimited JSON-RPC 2.0). Zero third-party deps.

Exposes the deterministic, zero-LLM optimizer to any MCP client (Claude Code, Cursor, Copilot...).
Register:  claude mcp add nlp-smarty -- ./NLP-start/bin/python /abs/path/mcp_server.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import traceback
from typing import Any, Callable

from bridge_server import _build_raw_request
from pipeline import _compress_tool_schema, _prune_external_context_mmr, audit_agent_response, optimize
import repo_indexer as ri
from protocol import ProtocolError, __version__, validate_request

SERVER_INFO = {"name": "nlp-smarty", "version": __version__}
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

# Sent once per session in the initialize result; the client adds it to the agent's context. Keep it short: it is paid
# on every session. The "nlp-smarty index:" prefix is a marker that ab_compare.py uses to confirm delivery.
INSTRUCTIONS = (
    "nlp-smarty index: this repo's symbols are indexed with exact line spans. To locate code, call smarty_get_context "
    "(describe the task) or smarty_find_symbol (a name); for usages call smarty_find_references; then smarty_get_slice "
    "to read only the lines you need. A <retrieved_context> block may already be in your prompt; use these tools for "
    "what it does not cover. Results are lexical and can be incomplete: fall back to grep if they come back empty."
)


def server_instructions() -> str | None:
    """OFF by default: in a 5-run live test the agent never used the tools even with these instructions (it never loaded
    the deferred tools), so they only cost ~100 tokens per session. Opt in with SMARTY_MCP_INSTRUCTIONS=on|1|true."""
    if os.environ.get("SMARTY_MCP_INSTRUCTIONS", "off").strip().lower() in ("on", "1", "true", "yes"):
        return INSTRUCTIONS
    return None

_PIPELINE_PROPS: dict[str, Any] = {
    "text": {"type": "string", "description": "Raw prompt / task text."},
    "context_chunks": {"type": "array", "items": {"type": "string"}, "description": "RAG chunks to prune via MMR."},
    "conversation_history": {
        "type": "array",
        "items": {"type": "object", "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                  "required": ["role", "content"]},
    },
    "compression_intensity": {"type": "string", "enum": ["AUTO", "LIGHT", "MEDIUM", "AGGRESSIVE"]},
    "max_total_tokens": {"type": "integer", "minimum": 0},
}


def _text_result(obj: Any, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(obj, indent=2)}], "isError": is_error}


def _tool_optimize(args: dict[str, Any]) -> Any:
    validate_request({"command": "optimize", **args})
    compiled = optimize(_build_raw_request(args), few_shot_k=int(args.get("few_shot_k", 2)))
    return {
        "status": compiled.status,
        "clarification_request": compiled.clarification_request,
        "payload": compiled.payload,
        "trace": dataclasses.asdict(compiled.trace),
    }


def _tool_audit(args: dict[str, Any]) -> Any:
    validate_request({"command": "audit_pair", **args})
    resp = args.get("response")
    if not isinstance(resp, str):
        raise ProtocolError("INVALID_REQUEST", "'response' must be a string")
    compiled = optimize(_build_raw_request(args))
    return audit_agent_response(resp, compiled.trace)


def _tool_compress_context(args: dict[str, Any]) -> Any:
    query, chunks = args.get("query"), args.get("chunks")
    if not isinstance(query, str) or not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        raise ProtocolError("INVALID_REQUEST", "'query' (string) and 'chunks' (list of strings) are required")
    text, kept = _prune_external_context_mmr(
        query, chunks, int(args.get("context_budget", 160)), int(args.get("top_k", 3)))
    return {"context": text, "kept_indices": kept}


def _tool_compress_schema(args: dict[str, Any]) -> Any:
    schema = args.get("schema")
    if not isinstance(schema, (dict, str)):
        raise ProtocolError("INVALID_REQUEST", "'schema' must be a JSON object or string")
    compressed, saved = _compress_tool_schema(schema)
    return {"compressed_schema": compressed, "tokens_saved": saved}


TOOLS: dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], Any]]] = {
    "smarty_optimize": ({
        "name": "smarty_optimize",
        "description": "Deterministically compress/normalize a prompt (+context, history, tools) into a token-budgeted compiled payload with an optimization trace. Zero-LLM.",
        "inputSchema": {"type": "object", "properties": _PIPELINE_PROPS, "required": ["text"]},
    }, _tool_optimize),
    "smarty_audit_response": ({
        "name": "smarty_audit_response",
        "description": "Score (0-100) an LLM/agent response against the original prompt for format compliance, entity fidelity and negation adherence.",
        "inputSchema": {"type": "object", "properties": {**_PIPELINE_PROPS, "response": {"type": "string"}},
                        "required": ["text", "response"]},
    }, _tool_audit),
    "smarty_compress_context": ({
        "name": "smarty_compress_context",
        "description": "Select the most query-relevant, non-redundant chunks (BM25+TF-IDF+MMR) within a token budget.",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string"}, "chunks": {"type": "array", "items": {"type": "string"}},
            "context_budget": {"type": "integer", "minimum": 0}, "top_k": {"type": "integer", "minimum": 1}},
            "required": ["query", "chunks"]},
    }, _tool_compress_context),
    "smarty_compress_tool_schema": ({
        "name": "smarty_compress_tool_schema",
        "description": "Compress an OpenAI/MCP JSON tool schema, stripping verbose prose while preserving structure; returns tokens saved.",
        "inputSchema": {"type": "object", "properties": {"schema": {"type": ["object", "string"]}},
                        "required": ["schema"]},
    }, _tool_compress_schema),
}


# ── repo-intelligence tools (root is fixed server-side: SMARTY_ROOT or cwd; never agent-supplied) ──
def _root() -> str:
    return os.path.abspath(os.environ.get("SMARTY_ROOT") or os.getcwd())


def _int(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    v = args.get(key, default)
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ProtocolError("INVALID_REQUEST", f"'{key}' must be an integer in [{lo}, {hi}]")
    return v


def _tool_get_context(args: dict[str, Any]) -> Any:
    q = args.get("query")
    if not isinstance(q, str) or not q.strip():
        raise ProtocolError("INVALID_REQUEST", "'query' must be a non-empty string")
    out = ri.build_capsule(_root(), q, token_budget=_int(args, "token_budget", 450, 100, 4000),
                           k=_int(args, "k", 5, 1, 15))
    return {"capsule": out["capsule"], "tokens": out["tokens"], "confidence": out["confidence"],
            "reason": out["reason"],
            "symbols": [f"{h['path']}:L{h['start']}-{h['end']} {h['name']}" for h in out["hits"]]}


def _tool_repo_map(args: dict[str, Any]) -> Any:
    return {"repo_map": ri.repo_map(_root(), _int(args, "token_budget", 800, 100, 8000))}


def _tool_find_symbol(args: dict[str, Any]) -> Any:
    n = args.get("name")
    if not isinstance(n, str) or not n.strip():
        raise ProtocolError("INVALID_REQUEST", "'name' must be a non-empty string")
    return {"matches": ri.find_symbol(_root(), n, _int(args, "limit", 10, 1, 50))}


def _tool_find_references(args: dict[str, Any]) -> Any:
    n = args.get("name")
    if not isinstance(n, str) or not n.strip():
        raise ProtocolError("INVALID_REQUEST", "'name' must be a non-empty string")
    inc = args.get("include_tests", True)
    if not isinstance(inc, bool):
        raise ProtocolError("INVALID_REQUEST", "'include_tests' must be a boolean")
    try:
        out = ri.find_references(_root(), n, _int(args, "limit", 40, 1, 200), _int(args, "token_budget", 700, 100, 4000), inc)
    except ValueError as exc:
        raise ProtocolError("INVALID_REQUEST", str(exc))
    return {"text": out["text"], "counts": out["counts"], "truncated": out["truncated"]}


def _tool_get_slice(args: dict[str, Any]) -> Any:
    path = args.get("path")
    if not isinstance(path, str) or not path:
        raise ProtocolError("INVALID_REQUEST", "'path' must be a non-empty string")
    start, end = _int(args, "start", 1, 1, 10**7), _int(args, "end", 1, 1, 10**7)
    if end < start or end - start > 400:
        raise ProtocolError("INVALID_REQUEST", "need start <= end and at most 400 lines")
    try:
        return ri.get_slice(_root(), path, start, end, context=_int(args, "context", 0, 0, 50))
    except (ValueError, OSError) as exc:
        raise ProtocolError("INVALID_REQUEST", str(exc))


_obj = lambda props, req: {"type": "object", "properties": props, "required": req}  # noqa: E731
TOOLS.update({
    "smarty_get_context": ({
        "name": "smarty_get_context",
        "description": "Budgeted code slices (file:Lstart-Lend) most relevant to a task, from the local index. Use before grepping/reading files.",
        "inputSchema": _obj({"query": {"type": "string"}, "token_budget": {"type": "integer"}, "k": {"type": "integer"}},
                            ["query"]),
    }, _tool_get_context),
    "smarty_repo_map": ({
        "name": "smarty_repo_map",
        "description": "Compact map of files and top-level symbols with line spans; modified files first.",
        "inputSchema": _obj({"token_budget": {"type": "integer"}}, []),
    }, _tool_repo_map),
    "smarty_find_symbol": ({
        "name": "smarty_find_symbol",
        "description": "Locate a symbol by name (exact, prefix, or Class.method suffix); returns path and line span.",
        "inputSchema": _obj({"name": {"type": "string"}, "limit": {"type": "integer"}}, ["name"]),
    }, _tool_find_symbol),
    "smarty_find_references": ({
        "name": "smarty_find_references",
        "description": "Where a symbol is used: definitions, then callers, then other mentions as file:line, ranked and "
                       "token-budgeted. Use instead of grep to find usages.",
        "inputSchema": _obj({"name": {"type": "string"}, "limit": {"type": "integer"}, "token_budget": {"type": "integer"},
                             "include_tests": {"type": "boolean"}}, ["name"]),
    }, _tool_find_references),
    "smarty_get_slice": ({
        "name": "smarty_get_slice",
        "description": "Numbered lines start..end (max 400) of a workspace file; secrets redacted.",
        "inputSchema": _obj({"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"},
                             "context": {"type": "integer"}}, ["path", "start", "end"]),
    }, _tool_get_slice),
})

RESOURCES = [{"uri": "smarty://repo_map", "name": "repo_map", "mimeType": "text/markdown",
              "description": "Compact repo symbol map with line spans."}]


def handle_message(msg: Any) -> dict[str, Any] | None:
    """Return a JSON-RPC response, or None for notifications."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    is_notification = "id" not in msg

    def ok(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    if is_notification:
        return None
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
        result: dict[str, Any] = {"protocolVersion": version,
                                  "capabilities": {"tools": {"listChanged": False}, "resources": {}},
                                  "serverInfo": SERVER_INFO}
        instructions = server_instructions()
        if instructions:
            result["instructions"] = instructions
        return ok(result)
    if method == "ping":
        return ok({})
    if method == "resources/list":
        return ok({"resources": RESOURCES})
    if method == "resources/read":
        if params.get("uri") != "smarty://repo_map":
            return err(-32002, f"Unknown resource: {params.get('uri')}")
        return ok({"contents": [{"uri": "smarty://repo_map", "mimeType": "text/markdown",
                                 "text": ri.repo_map(_root(), 1200)}]})
    if method == "tools/list":
        return ok({"tools": [t[0] for t in TOOLS.values()]})
    if method == "tools/call":
        name = params.get("name")
        if name not in TOOLS:
            return err(-32602, f"Unknown tool: {name}")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            return err(-32602, "arguments must be an object")
        try:
            return ok(_text_result(TOOLS[name][1](args)))
        except ProtocolError as exc:
            return ok(_text_result({"error_code": exc.code, "error": exc.message}, is_error=True))
        except Exception as exc:  # tool failures are reported in-band per MCP spec
            sys.stderr.write(f"[NLP-Smarty MCP] {exc}\n{traceback.format_exc()}\n")
            return ok(_text_result({"error_code": "INTERNAL_ERROR", "error": str(exc)}, is_error=True))
    return err(-32601, f"Method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle_message(json.loads(line))
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
