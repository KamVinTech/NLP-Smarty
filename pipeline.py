"""
Deterministic Prompt Optimization Pipeline
==========================================
Stage 1 — Lexical & Syntactic Intent Router   (FlashText trie + spaCy POS)
Stage 2 — Rule-Based Prompt Compressor         (regex filler-strip + BM25 sentence pruning)
Stage 3 — BM25 Context & Few-Shot Pruner       (rank_bm25)
Stage 4 — Structural Compiler & Guardrail Injector (string templating)

Zero LLM calls. All transformations are deterministic and fully logged.
"""

import re
import time
from typing import Any

import spacy
from flashtext import KeywordProcessor
from rank_bm25 import BM25Okapi

from data_contracts import CompiledPrompt, OptimizationTrace, RawRequest
from registry import (
    FEW_SHOT_EXAMPLES,
    GUARDRAIL_RULES,
    REQUIRED_SLOTS,
    TOOL_REGISTRY,
)

# ── Lazy-load spaCy model once ─────────────────────────────────────────────────
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _nlp


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Intent Router
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "CODE_GENERATION":  ["write", "generate", "create", "implement", "code", "function",
                         "class", "script", "sql", "query"],
    "DATA_EXTRACTION":  ["extract", "parse", "find all", "pull out", "scrape", "regex",
                         "identify", "list all"],
    "TOOL_EXECUTION":   ["run", "execute", "search", "send", "open", "read file",
                         "call", "invoke", "trigger"],
    "FACTUAL_QA":       ["what is", "who is", "when did", "where is", "how many",
                         "define", "explain", "tell me"],
}

_NEGATION_PATTERN = re.compile(
    r"\b(not|never|without|except|no |don't|doesn't|isn't|aren't|won't|cannot|can't)\b",
    re.IGNORECASE,
)

def _build_intent_processor() -> dict[str, KeywordProcessor]:
    processors = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        kp = KeywordProcessor(case_sensitive=False)
        kp.add_keywords_from_list(keywords)
        processors[intent] = kp
    return processors

_INTENT_PROCESSORS = _build_intent_processor()


def _route_intent(text: str) -> tuple[str, float]:
    """Returns (intent_label, confidence_score 0-1)."""
    scores: dict[str, int] = {}
    for intent, kp in _INTENT_PROCESSORS.items():
        scores[intent] = len(kp.extract_keywords(text))

    total = sum(scores.values())
    if total == 0:
        return "AMBIGUOUS_QUERY", 0.0

    best = max(scores, key=lambda k: scores[k])
    confidence = round(scores[best] / total, 3)
    return best, confidence


def _extract_negations(text: str) -> list[str]:
    return _NEGATION_PATTERN.findall(text)


def _check_required_slots(intent: str, entities: dict) -> list[str]:
    missing = []
    for slot in REQUIRED_SLOTS.get(intent, []):
        if slot not in entities or not entities[slot]:
            missing.append(slot)
    return missing


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Prompt Compressor
# ══════════════════════════════════════════════════════════════════════════════

# Filler phrases to strip (order matters — longer phrases first)
_FILLER_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"could you (please |kindly )?",
        r"i was wondering if (you could )?",
        r"i('d| would) (like|love|appreciate it) (if you could |to )?",
        r"would you (be able to |mind )?",
        r"can you (please |kindly )?",
        r"please (help me |assist me )?(to )?",
        r"i need (you to |help )?",
        r"i want (you to )?",
        r"just (quickly |briefly )?",
        r"as (an AI|a language model|an assistant),?\s*",
    ]
]

# Tokens that must NEVER be stripped
_PROTECTED_PATTERN = re.compile(
    r'(".*?"|\'.*?\'|`[^`]*`|https?://\S+|not|never|without|except)',
    re.IGNORECASE | re.DOTALL,
)


def _protect_and_strip_filler(text: str) -> tuple[str, list[str]]:
    """
    1. Temporarily replace protected spans with placeholders.
    2. Strip filler phrases.
    3. Restore protected spans.
    Returns (cleaned_text, list_of_stripped_tokens).
    """
    placeholders: dict[str, str] = {}
    counter = [0]

    def _protect(m: re.Match) -> str:
        key = f"__PROT{counter[0]}__"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    protected = _PROTECTED_PATTERN.sub(_protect, text)

    stripped: list[str] = []
    for pattern in _FILLER_PATTERNS:
        match = pattern.search(protected)
        if match:
            stripped.append(match.group(0).strip())
        protected = pattern.sub("", protected)

    # Restore placeholders
    for key, val in placeholders.items():
        protected = protected.replace(key, val)

    cleaned = re.sub(r"\s{2,}", " ", protected).strip()
    # Capitalise first letter if it was lowered by stripping
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned, stripped


def _bm25_sentence_prune(text: str, token_budget: int = 120) -> str:
    """
    If text exceeds token_budget words, score sentences with BM25 against
    the full query terms and keep only the top-scoring ones.
    """
    words = text.split()
    if len(words) <= token_budget:
        return text

    nlp = _get_nlp()
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    if len(sentences) <= 1:
        return text

    tokenized = [s.lower().split() for s in sentences]
    bm25 = BM25Okapi(tokenized)
    query_tokens = words[:50]  # use first 50 words as the "query"
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(zip(scores, sentences), reverse=True)
    kept, count = [], 0
    for score, sent in ranked:
        kept.append(sent)
        count += len(sent.split())
        if count >= token_budget:
            break

    # Re-order kept sentences in original document order
    order = {s: i for i, s in enumerate(sentences)}
    kept.sort(key=lambda s: order.get(s, 999))
    return " ".join(kept)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — BM25 Context & Few-Shot Pruner
# ══════════════════════════════════════════════════════════════════════════════

def _bm25_top_k(query: str, corpus: list[dict], text_key: str, k: int) -> list[dict]:
    """Generic BM25 top-K selector over a list of dicts."""
    if not corpus:
        return []
    tokenized = [item[text_key].lower().split() for item in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, corpus), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked[:k]]


def _select_tools(query: str) -> list[dict]:
    """Return tools whose trigger_keywords overlap with the query."""
    query_lower = query.lower()
    matched = [
        t for t in TOOL_REGISTRY
        if any(kw in query_lower for kw in t["trigger_keywords"])
    ]
    return matched or []


def _select_few_shots(query: str, k: int = 2) -> list[dict]:
    return _bm25_top_k(query, FEW_SHOT_EXAMPLES, "input", k)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Structural Compiler
# ══════════════════════════════════════════════════════════════════════════════

_COMPILED_TEMPLATE = """\
<agent_instructions>
You are executing a pre-processed, deterministically structured task.
Follow every rule in <anti_hallucination_protocol> and <output_constraints> strictly.
</agent_instructions>

<anti_hallucination_protocol>
1. SOURCE-ONLY GROUNDING: Rely ONLY on facts and schemas in <verified_context> and <core_task>.
2. ABSTENTION RULE: If required information is missing, output STATUS: INSUFFICIENT_CONTEXT followed by missing fields.
3. ENTITY FIDELITY: Preserve all identifiers, dates, URLs, and numbers verbatim.
4. TOOL STRICTNESS: Only invoke tools listed in <allowed_tools>.
{guardrails}
</anti_hallucination_protocol>

<task_metadata>
- Detected Intent: {intent}
- Complexity Tier: {complexity}
- Negation / Exclusion Constraints: {negations}
</task_metadata>

<extracted_entities>
{entities}
</extracted_entities>

<verified_context>
{context}
</verified_context>

<allowed_tools>
{tools}
</allowed_tools>

<few_shot_examples>
{examples}
</few_shot_examples>

<core_task>
{query}
</core_task>

<output_constraints>
- Format: {output_format}
- Pre-Output Verification: Verify every claim traces to <verified_context> and no <task_metadata> constraints are violated.
</output_constraints>
"""


def _extract_entities(text: str) -> dict[str, Any]:
    """Deterministic regex-based NER for common entity types."""
    entities: dict[str, Any] = {}

    emails = re.findall(r"[\w.\-]+@[\w.\-]+\.\w+", text)
    if emails:
        entities["emails"] = emails

    urls = re.findall(r"https?://\S+", text)
    if urls:
        entities["urls"] = urls

    dates = re.findall(
        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
        text, re.IGNORECASE,
    )
    if dates:
        entities["dates"] = dates

    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if numbers:
        entities["numbers"] = numbers

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    flat_quoted = [q[0] or q[1] for q in quoted]
    if flat_quoted:
        entities["quoted_strings"] = flat_quoted

    # Root verb via spaCy dependency parse
    nlp = _get_nlp()
    doc = nlp(text)
    root_verbs = [t.lemma_ for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"]
    if root_verbs:
        entities["action"] = root_verbs[0]

    return entities


def _complexity_tier(token_count: int) -> str:
    if token_count < 20:
        return "LOW"
    if token_count < 60:
        return "MEDIUM"
    return "HIGH"


def _format_entities(entities: dict) -> str:
    if not entities:
        return "NONE"
    return "\n".join(f"- {k}: {v}" for k, v in entities.items())


def _format_tools(tools: list[dict]) -> str:
    if not tools:
        return "NONE"
    return "\n".join(f"- {t['id']}: {t['schema']}" for t in tools)


def _format_examples(examples: list[dict]) -> str:
    if not examples:
        return "NONE"
    parts = []
    for ex in examples:
        parts.append(f"[{ex['id']}]\nInput:  {ex['input']}\nOutput: {ex['output']}")
    return "\n\n".join(parts)


def _infer_output_format(intent: str) -> str:
    return {
        "CODE_GENERATION":  "Fenced code block with language tag",
        "DATA_EXTRACTION":  "JSON array of extracted values",
        "TOOL_EXECUTION":   "Tool call JSON: {name, params}",
        "FACTUAL_QA":       "Plain text, ≤3 sentences",
        "AMBIGUOUS_QUERY":  "Clarification question",
    }.get(intent, "Plain text")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def optimize(request: RawRequest, few_shot_k: int = 2) -> CompiledPrompt:
    t0 = time.perf_counter()
    trace = OptimizationTrace()

    raw_tokens = request.text.split()
    trace.raw_token_count = len(raw_tokens)

    # ── Stage 1: Intent routing ────────────────────────────────────────────
    intent, confidence = _route_intent(request.text)
    negations = _extract_negations(request.text)
    entities = _extract_entities(request.text)

    trace.detected_intent = intent
    trace.intent_confidence = confidence
    trace.negations = negations
    trace.extracted_entities = entities

    # Early short-circuit if required slots are missing
    missing = _check_required_slots(intent, entities)
    if missing:
        trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return CompiledPrompt(
            payload="",
            trace=trace,
            status="NEEDS_CLARIFICATION",
            clarification_request=(
                f"To proceed with {intent}, please provide: {', '.join(missing)}"
            ),
        )

    # ── Stage 2: Compression ───────────────────────────────────────────────
    compressed, stripped = _protect_and_strip_filler(request.text)
    compressed = _bm25_sentence_prune(compressed)
    trace.stripped_tokens = stripped

    # ── Stage 3: BM25 pruning ──────────────────────────────────────────────
    tools = _select_tools(compressed)
    examples = _select_few_shots(compressed, k=few_shot_k)
    trace.injected_tool_ids = [t["id"] for t in tools]
    trace.injected_example_ids = [e["id"] for e in examples]

    # ── Stage 4: Compile ───────────────────────────────────────────────────
    guardrail_block = "\n".join(f"{i+5}. {r}" for i, r in enumerate(GUARDRAIL_RULES))
    opt_tokens = compressed.split()
    trace.optimized_token_count = len(opt_tokens)

    payload = _COMPILED_TEMPLATE.format(
        guardrails=guardrail_block,
        intent=intent,
        complexity=_complexity_tier(trace.raw_token_count),
        negations=", ".join(negations) if negations else "NONE",
        entities=_format_entities(entities),
        context="[No external context provided — agent must rely on task only]",
        tools=_format_tools(tools),
        examples=_format_examples(examples),
        query=compressed,
        output_format=_infer_output_format(intent),
    )

    trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return CompiledPrompt(payload=payload, trace=trace, status="OK")
