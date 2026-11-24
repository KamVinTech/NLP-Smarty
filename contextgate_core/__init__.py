"""ContextGate: a deterministic, zero-LLM gate for prompts and LLM requests.

Public API (every name below is covered by tests/test_public_api.py; imports are lazy so ``import contextgate_core`` stays instant,
which the Claude Code hook depends on)::

    from contextgate_core import gate_text, gate_request, optimize, estimate_tokens

    text, ok = gate_text("Could you please fix this in order to be safe", "balanced")   # str -> (str, bool)
    result = gate_request(chat_body)                                                   # dict -> GateResult
    compiled = optimize(RawRequest(text="..."))                                        # RawRequest -> CompiledPrompt
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .protocol import __version__

if TYPE_CHECKING:  # static analysis / IDE completion only; never executed at runtime
    from .data_contracts import CompiledPrompt, OptimizationTrace, RawRequest, TokenBudget
    from .gate import PROFILES, GateResult, gate_request, gate_text
    from .pipeline import audit_agent_response, estimate_tokens, optimize

_LAZY = {
    "gate_text": "gate",
    "gate_request": "gate",
    "GateResult": "gate",
    "PROFILES": "gate",
    "optimize": "pipeline",
    "estimate_tokens": "pipeline",
    "audit_agent_response": "pipeline",
    "RawRequest": "data_contracts",
    "TokenBudget": "data_contracts",
    "CompiledPrompt": "data_contracts",
    "OptimizationTrace": "data_contracts",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module 'contextgate_core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
