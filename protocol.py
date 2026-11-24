"""
Bridge protocol contract: version, error codes, deterministic request validation.
Zero third-party dependencies. No LLM / network usage.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.6.0"  # the single source of truth: tools/check_release.py keeps pyproject/package.json/lock in sync with it
PROTOCOL_VERSION = "1.0"

# Error codes
E_INVALID_JSON = "INVALID_JSON"
E_INVALID_REQUEST = "INVALID_REQUEST"
E_UNKNOWN_COMMAND = "UNKNOWN_COMMAND"
E_INTERNAL = "INTERNAL_ERROR"

INTENSITIES = ("AUTO", "LIGHT", "MEDIUM", "AGGRESSIVE")
BUDGET_FIELDS = (
    "max_total_tokens", "core_query_budget", "context_budget",
    "history_budget", "tools_budget", "few_shot_budget",
)

_PIPELINE_CMDS = ("optimize", "compare_hf", "audit_pair")
COMMANDS = ("optimize", "compare_hf", "audit_pair", "clear_cache", "ping", "capabilities")


class ProtocolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def error_response(req_id: Any, code: str, message: str, **extra: Any) -> dict[str, Any]:
    # `error` stays a plain string for backward compatibility with extension.ts.
    return {
        "id": req_id, "ok": False, "error": message, "error_code": code,
        "protocol_version": PROTOCOL_VERSION, **extra,
    }


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_request(req: Any) -> None:
    """Raise ProtocolError on malformed requests."""
    if not isinstance(req, dict):
        raise ProtocolError(E_INVALID_REQUEST, "request must be a JSON object")

    cmd = req.get("command", "optimize")
    if not isinstance(cmd, str) or cmd not in COMMANDS:
        raise ProtocolError(E_UNKNOWN_COMMAND, f"unknown command: {cmd!r}; supported: {', '.join(COMMANDS)}")

    if cmd not in _PIPELINE_CMDS:
        return

    def bad(msg: str) -> ProtocolError:
        return ProtocolError(E_INVALID_REQUEST, msg)

    text = req.get("text", "")
    if not isinstance(text, str):
        raise bad("'text' must be a string")
    if cmd in _PIPELINE_CMDS and not text.strip():
        raise bad("'text' is required and must be non-empty")

    for key in ("task_hint", "hf_model_id", "raw_response", "optimized_response"):
        if req.get(key) is not None and not isinstance(req[key], str):
            raise bad(f"'{key}' must be a string")

    chunks = req.get("context_chunks")
    if chunks is not None and (not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks)):
        raise bad("'context_chunks' must be a list of strings")

    hist = req.get("conversation_history")
    if hist is not None:
        if not isinstance(hist, list) or not all(
            isinstance(h, dict) and isinstance(h.get("role"), str) and isinstance(h.get("content"), str)
            for h in hist
        ):
            raise bad("'conversation_history' must be a list of {role: str, content: str}")

    files = req.get("workspace_files")
    if files is not None:
        if not isinstance(files, list) or not all(
            isinstance(f, dict) and isinstance(f.get("content", ""), str) and isinstance(f.get("path", ""), str)
            for f in files
        ):
            raise bad("'workspace_files' must be a list of {path: str, content: str}")

    for key in BUDGET_FIELDS + ("few_shot_k",):
        if key in req and (not _is_int(req[key]) or req[key] < 0):
            raise bad(f"'{key}' must be a non-negative integer")

    inten = req.get("compression_intensity")
    if inten is not None and inten not in INTENSITIES:
        raise bad(f"'compression_intensity' must be one of {list(INTENSITIES)}")

    if "enable_cache" in req and not isinstance(req["enable_cache"], bool):
        raise bad("'enable_cache' must be a boolean")


def capabilities() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "commands": list(COMMANDS),
        "compression_intensities": list(INTENSITIES),
        "budget_fields": list(BUDGET_FIELDS),
        "error_codes": [E_INVALID_JSON, E_INVALID_REQUEST, E_UNKNOWN_COMMAND, E_INTERNAL],
    }
