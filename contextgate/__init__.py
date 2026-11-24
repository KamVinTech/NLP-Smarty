"""ContextGate: a deterministic, zero-LLM gate for prompts and LLM requests.

Public API (imported lazily so `import contextgate` stays instant; the Claude Code hook depends on that):

    from contextgate import gate_text, gate_request, optimize, estimate_tokens
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from .protocol import __version__

_LAZY = {
    "gate_text": "gate",
    "gate_request": "gate",
    "optimize": "pipeline",
    "estimate_tokens": "pipeline",
    "audit_agent_response": "pipeline",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module 'contextgate' has no attribute {name!r}")
