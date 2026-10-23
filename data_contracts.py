from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenBudget:
    """
    Step 1: Token-Aware Budgeting contract.
    Controls the maximum token ceiling for each compiled prompt section.
    """
    max_total_tokens: int = 512
    core_query_budget: int = 120
    context_budget: int = 160
    history_budget: int = 80
    tools_budget: int = 60
    few_shot_budget: int = 90


@dataclass
class RawRequest:
    text: str
    task_hint: str = ""                                      # optional caller-supplied intent hint
    context_chunks: list[str] = field(default_factory=list)  # Step 3: external RAG chunks to prune via MMR
    conversation_history: list[dict[str, str]] = field(default_factory=list)  # Step 7: [{"role": "user"|"assistant", "content": "..."}]
    token_budget: TokenBudget = field(default_factory=TokenBudget)            # Step 1: token-aware budget
    compression_intensity: str = "AUTO"                      # Step 8: "AUTO" | "LIGHT" | "MEDIUM" | "AGGRESSIVE"
    enable_cache: bool = True                                # Step 6: deterministic prompt cache toggle


@dataclass
class OptimizationTrace:
    # Step 9: Comprehensive Before & After Token Measurements
    raw_token_count: int = 0
    optimized_token_count: int = 0
    query_token_savings_pct: float = 0.0
    unoptimized_payload_token_count: int = 0
    compiled_payload_token_count: int = 0
    payload_token_savings_pct: float = 0.0
    tool_schema_tokens_saved: int = 0

    # Routing & Extraction Audit
    detected_intent: str = ""
    secondary_intents: list[str] = field(default_factory=list)
    intent_confidence: float = 0.0
    compression_intensity_used: str = "LIGHT"
    pruning_strategy_used: str = "MMR+BM25+TFIDF"

    # Transformation Details
    stripped_tokens: list[str] = field(default_factory=list)
    normalized_phrases: list[str] = field(default_factory=list)
    injected_example_ids: list[str] = field(default_factory=list)
    injected_tool_ids: list[str] = field(default_factory=list)
    kept_context_indices: list[int] = field(default_factory=list)
    kept_history_turns: int = 0
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    negations: list[str] = field(default_factory=list)

    # Step 6: Caching & Telemetry
    cache_hit: bool = False
    cache_type: str = "NONE"          # "NONE" | "CANONICAL_EXACT" | "SIMHASH_NEAR"
    cache_key: str = ""
    simhash_signature: str = ""
    latency_ms: float = 0.0


@dataclass
class CompiledPrompt:
    payload: str                      # final XML/Markdown string for downstream agent
    trace: OptimizationTrace
    status: str = "OK"                # "OK" | "NEEDS_CLARIFICATION" | "INSUFFICIENT_CONTEXT"
    clarification_request: str = ""