from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawRequest:
    text: str
    task_hint: str = ""          # optional caller-supplied hint


@dataclass
class OptimizationTrace:
    raw_token_count: int = 0
    optimized_token_count: int = 0
    detected_intent: str = ""
    intent_confidence: float = 0.0
    stripped_tokens: list[str] = field(default_factory=list)
    injected_example_ids: list[str] = field(default_factory=list)
    injected_tool_ids: list[str] = field(default_factory=list)
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    negations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class CompiledPrompt:
    payload: str                 # final XML/Markdown string for downstream agent
    trace: OptimizationTrace
    status: str = "OK"           # "OK" | "NEEDS_CLARIFICATION" | "INSUFFICIENT_CONTEXT"
    clarification_request: str = ""
