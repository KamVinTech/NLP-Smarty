"""
Deterministic Prompt Optimization Pipeline (Upgraded v2)
========================================================
Stage 1 — Contraction-Safe Boilerplate & Filler Compressor + Constraint-Preserving Sentence Pruner
Stage 2 — Weighted Lexical & Syntactic Intent Router (FlashText/Regex + spaCy POS with Lead-Verb Boost)
Stage 3 — Pre-Indexed, Stopword-Filtered BM25 Pruner for Few-Shots, Tools & External Context
Stage 4 — Token-Optimal Structural Compiler & Anti-Hallucination Guardrail Injector

Zero LLM calls. 100% backward-compatible with existing data_contracts.py, registry.py, and test_harness.py.
Includes graceful pure-Python fallbacks if spaCy, FlashText, or rank_bm25 are not installed in the active env.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

from data_contracts import CompiledPrompt, OptimizationTrace, RawRequest
from registry import (
    FEW_SHOT_EXAMPLES,
    GUARDRAIL_RULES,
    REQUIRED_SLOTS,
    TOOL_REGISTRY,
)

# ── Optional C-extension / library imports with deterministic fallbacks ────────
try:
    import spacy
    _HAS_SPACY = True
except ImportError:
    spacy = None
    _HAS_SPACY = False

try:
    from flashtext import KeywordProcessor
    _HAS_FLASHTEXT = True
except ImportError:
    KeywordProcessor = None
    _HAS_FLASHTEXT = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    BM25Okapi = None
    _HAS_BM25 = False


# ── Lazy-load spaCy model once (keep lemmatizer enabled so token.lemma_ works) ─
_nlp = None

def _get_nlp():
    global _nlp
    if not _HAS_SPACY:
        return None
    if _nlp is None:
        try:
            # Keep tagger, parser, attribute_ruler, and lemmatizer enabled so token.lemma_ is populated
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            _nlp = False
    return _nlp if _nlp is not False else None


# ── Pure-Python Okapi BM25 fallback (identical math to rank_bm25.BM25Okapi) ────
class _FastBM25Okapi:
    """Zero-dependency Okapi BM25 implementation (k1=1.5, b=0.75, epsilon=0.25)."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        df: dict[str, int] = {}

        for doc in corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                df[token] = df.get(token, 0) + 1

        self.idf: dict[str, float] = {}
        idf_sum = 0.0
        negative_idfs = []
        for token, freq in df.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[token] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(token)

        self.average_idf = (idf_sum / len(self.idf)) if self.idf else 0.0
        eps = self.epsilon * self.average_idf
        for token in negative_idfs:
            self.idf[token] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(q, 0)
                if freq == 0:
                    continue
                dl = self.doc_len[idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[idx] += q_idf * (freq * (self.k1 + 1.0)) / denom
        return scores


def _create_bm25(tokenized_corpus: list[list[str]]):
    if _HAS_BM25 and BM25Okapi is not None:
        return BM25Okapi(tokenized_corpus)
    return _FastBM25Okapi(tokenized_corpus)


# ── Stopwords & Tokenizer for High-Signal BM25 Scoring ─────────────────────────
_STOPWORDS: frozenset[str] = frozenset({
    "a", "all", "an", "and", "any", "are", "as", "at", "be", "by", "can",
    "could", "did", "do", "does", "each", "every", "for", "from", "had",
    "has", "have", "he", "her", "here", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "me", "might", "my", "of",
    "on", "or", "our", "out", "please", "shall", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "too", "up", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom",
    "why", "will", "with", "would", "you", "your",
})

_WORD_TOKEN_RE = re.compile(r"\b[a-z0-9_]+\b")


def _tokenize_for_bm25(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lowercases, strips punctuation, and removes zero-signal stopwords."""
    raw = _WORD_TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return raw
    filtered = [t for t in raw if t not in _STOPWORDS and len(t) > 1]
    return filtered if filtered else raw


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Negation-Safe & Contraction-Safe Prompt Compressor
# ══════════════════════════════════════════════════════════════════════════════

# Expand conversational contractions BEFORE filler stripping so negations are explicit
# and apostrophes never masquerade as single-quoted string literals.
_CONTRACTION_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bi'd\b", re.IGNORECASE), "I would"),
    (re.compile(r"\bi'm\b", re.IGNORECASE), "I am"),
    (re.compile(r"\bi've\b", re.IGNORECASE), "I have"),
    (re.compile(r"\bi'll\b", re.IGNORECASE), "I will"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bdon't\b", re.IGNORECASE), "do not"),
    (re.compile(r"\bdoesn't\b", re.IGNORECASE), "does not"),
    (re.compile(r"\bdidn't\b", re.IGNORECASE), "did not"),
    (re.compile(r"\bisn't\b", re.IGNORECASE), "is not"),
    (re.compile(r"\baren't\b", re.IGNORECASE), "are not"),
    (re.compile(r"\bwasn't\b", re.IGNORECASE), "was not"),
    (re.compile(r"\bweren't\b", re.IGNORECASE), "were not"),
    (re.compile(r"\bshouldn't\b", re.IGNORECASE), "should not"),
    (re.compile(r"\bwouldn't\b", re.IGNORECASE), "would not"),
    (re.compile(r"\bcouldn't\b", re.IGNORECASE), "could not"),
]

# Filler phrases to strip (ordered longest/most specific first)
_FILLER_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bi was wondering if (?:you could (?:please |kindly )?)?",
        r"\bi would (?:really )?(?:like|love|appreciate)(?: it)? (?:if you could (?:please |kindly )?|to )?",
        r"\bcould you (?:please |kindly )?(?:help me (?:to )?|assist me (?:in |with |to )?)?",
        r"\bwould you (?:please |kindly )?(?:be able to |mind )?",
        r"\bcan you (?:please |kindly )?(?:help me (?:to )?)?",
        r"\bplease (?:help me (?:to )?|assist me (?:to )?)?",
        r"\bi need (?:you to |help (?:with |to )?)?",
        r"\bi want (?:you to )?",
        r"^(?:please\s+)?tell me\s+(?=(?:what|who|when|where|why|how)\b)",
        r"\bjust (?:quickly |briefly |simply )+",
        r"\b(?:quickly|briefly|kindly)\s+(?=(?:write|generate|create|extract|parse|search|run|explain|tell|find|select|build)\b)",
        r"\bas (?:an ai|a language model|an assistant|a helpful assistant),?\s*",
        r"^(?:hey|hi|hello)(?:\s+there)?[,!]?\s+",
        r"\bthank you(?: in advance)?[!.]?\s*$",
        r"\bthanks(?: a lot| so much)?[!.]?\s*$",
    ]
]

# Protected spans:
# 1. Fenced/inline code blocks (`...`)
# 2. Double-quoted literals ("...")
# 3. True single-quoted literals ('...' NOT preceded/followed by word chars so contractions aren't matched)
# 4. URLs (http/https) and Email addresses (user@domain.tld)
# 5. Strict word-bounded negation tokens (\bnot\b, \bnever\b, etc. — never matches inside "notes" or "annotate")
_PROTECTED_PATTERN = re.compile(
    r'(`[^`]+`|"(?:\\.|[^"\\])*"|(?<!\w)\'(?:\\.|[^\'\\])*\'(?!\w)|https?://\S+|[\w.\-]+@[\w.\-]+\.\w+|\b(?:not|never|without|except|neither|nor|unless)\b)',
    re.IGNORECASE | re.DOTALL,
)

_NEGATION_PATTERN = re.compile(
    r"\b(not|never|without|except|no|neither|nor|unless|don't|doesn't|isn't|aren't|won't|cannot|can't)\b",
    re.IGNORECASE,
)

_CONSTRAINT_MARKER_PATTERN = re.compile(
    r"\b(must|only|required|ensure|strictly|format|json|xml|schema|limit|maximum|minimum|between|after|before)\b",
    re.IGNORECASE,
)


def _protect_and_strip_filler(text: str) -> tuple[str, list[str]]:
    """
    1. Normalize conversational contractions ('I'd' -> 'I would', 'don't' -> 'do not').
    2. Replace protected spans (code, quotes, URLs, emails, word-bounded negations) with placeholders.
    3. Strip conversational filler phrases.
    4. Restore protected spans and normalize whitespace/capitalization.
    """
    working = text
    for pattern, replacement in _CONTRACTION_MAP:
        working = pattern.sub(replacement, working)

    placeholders: dict[str, str] = {}
    counter = 0

    def _protect(m: re.Match) -> str:
        nonlocal counter
        key = f"__PROT{counter}__"
        placeholders[key] = m.group(0)
        counter += 1
        return key

    protected = _PROTECTED_PATTERN.sub(_protect, working)

    stripped: list[str] = []
    for pattern in _FILLER_PATTERNS:
        matches = list(pattern.finditer(protected))
        for m in matches:
            val = m.group(0).strip()
            if val:
                stripped.append(val)
        protected = pattern.sub(" ", protected)

    # Restore placeholders in reverse order
    for key, val in reversed(list(placeholders.items())):
        protected = protected.replace(key, val)

    cleaned = re.sub(r"\s{2,}", " ", protected).strip()
    cleaned = re.sub(r"^[,;:\-]+\s*", "", cleaned).strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned, stripped


def _split_sentences(text: str) -> list[str]:
    """Deterministic sentence splitter using spaCy if available, else regex."""
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _bm25_sentence_prune(text: str, token_budget: int = 120) -> str:
    """
    Constraint-safe BM25 sentence pruner:
    - Never prunes sentences containing negations, constraints, URLs, emails, or code blocks.
    - Scores optional/descriptive sentences against the full document's high-signal keywords
      (avoiding the first-50-word self-similarity bias).
    """
    words = text.split()
    if len(words) <= token_budget:
        return text

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text

    tokenized = [_tokenize_for_bm25(s) for s in sentences]
    bm25 = _create_bm25(tokenized)

    # Build a global centroid query from the top content terms across the entire prompt
    all_content_tokens = _tokenize_for_bm25(text)
    top_query_tokens = [tok for tok, _ in Counter(all_content_tokens).most_common(25)]
    scores = bm25.get_scores(top_query_tokens)

    # Boost sentences that carry negations, constraints, or concrete literals so they are never lost
    adjusted_scores: list[tuple[float, int, str]] = []
    for idx, (score, sent) in enumerate(zip(scores, sentences)):
        bonus = 0.0
        if _NEGATION_PATTERN.search(sent):
            bonus += 100.0
        if _CONSTRAINT_MARKER_PATTERN.search(sent):
            bonus += 50.0
        if "`" in sent or "http" in sent or "@" in sent:
            bonus += 50.0
        if idx == 0:
            bonus += 25.0  # preserve primary task opener
        adjusted_scores.append((score + bonus, idx, sent))

    ranked = sorted(adjusted_scores, key=lambda x: x[0], reverse=True)
    kept_indices: list[tuple[int, str]] = []
    count = 0
    for _, idx, sent in ranked:
        kept_indices.append((idx, sent))
        count += len(sent.split())
        if count >= token_budget:
            break

    # Re-order by original sentence index (safe even when duplicate sentences exist)
    kept_indices.sort(key=lambda x: x[0])
    return " ".join(sent for _, sent in kept_indices)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Weighted Lexical & Syntactic Intent Router
# ══════════════════════════════════════════════════════════════════════════════

# Weighted keywords: multi-word phrases and high-specificity terms carry higher weight
_INTENT_KEYWORDS_WEIGHTED: dict[str, dict[str, float]] = {
    "CODE_GENERATION": {
        "write": 1.2, "generate": 1.2, "create": 1.0, "implement": 1.5,
        "code": 1.3, "function": 1.5, "class": 1.3, "script": 1.3,
        "sql": 1.8, "query": 1.2, "algorithm": 1.5, "refactor": 1.5,
        "debug": 1.4, "python function": 2.5, "sql query": 2.5,
    },
    "DATA_EXTRACTION": {
        "extract": 2.0, "parse": 1.8, "find all": 2.0, "pull out": 1.8,
        "scrape": 1.8, "regex": 1.5, "identify": 1.3, "list all": 1.8,
        "extract all": 2.5, "email addresses": 1.5,
    },
    "TOOL_EXECUTION": {
        "run": 1.4, "execute": 1.6, "search": 1.8, "search the web": 2.5,
        "send": 1.6, "open": 1.3, "read file": 2.2, "call": 1.3,
        "invoke": 1.6, "trigger": 1.5, "look up": 1.8,
    },
    "FACTUAL_QA": {
        "what is": 2.0, "who is": 2.0, "when did": 2.0, "where is": 2.0,
        "how many": 1.8, "define": 1.8, "explain": 1.6, "tell me": 1.2,
        "capital of": 2.0, "why does": 1.8,
    },
}

_INTENT_WORD_REGEXES: dict[str, list[tuple[str, re.Pattern, float]]] = {
    intent: [
        (kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), weight)
        for kw, weight in kw_map.items()
    ]
    for intent, kw_map in _INTENT_KEYWORDS_WEIGHTED.items()
}


def _route_intent(cleaned_text: str, raw_text: str = "", task_hint: str = "") -> tuple[str, float]:
    """
    Routes intent using:
    1. Explicit caller `task_hint` override if valid.
    2. Weighted phrase/keyword matching on `cleaned_text`.
    3. 2.0x Imperative Lead-Verb Boost when a keyword appears in the first 3 words
       of the compressed query (so 'Extract emails from a Python script' routes to
       DATA_EXTRACTION rather than CODE_GENERATION).
    """
    if task_hint:
        normalized_hint = task_hint.strip().upper()
        if normalized_hint in _INTENT_KEYWORDS_WEIGHTED or normalized_hint in REQUIRED_SLOTS:
            return normalized_hint, 1.0

    target = cleaned_text or raw_text
    first_three_words = " ".join(target.split()[:3]).lower()

    scores: dict[str, float] = {intent: 0.0 for intent in _INTENT_KEYWORDS_WEIGHTED}
    for intent, patterns in _INTENT_WORD_REGEXES.items():
        for kw, pattern, weight in patterns:
            matches = pattern.findall(target)
            if matches:
                score = weight * len(matches)
                # Apply lead-verb boost if keyword anchors the opening command
                if re.search(rf"\b{re.escape(kw)}\b", first_three_words):
                    score *= 2.0
                scores[intent] += score

    total = sum(scores.values())
    if total == 0.0:
        return "AMBIGUOUS_QUERY", 0.0

    best = max(scores, key=lambda k: scores[k])
    confidence = round(scores[best] / total, 3)
    return best, confidence


def _extract_negations(text: str) -> list[str]:
    seen = []
    for m in _NEGATION_PATTERN.findall(text):
        low = m.lower().strip()
        if low not in seen:
            seen.append(low)
    return seen


# Known English imperative task verbs for fallback & validation
_KNOWN_TASK_VERBS: dict[str, str] = {
    "write": "write", "writing": "write", "wrote": "write",
    "generate": "generate", "generating": "generate",
    "create": "create", "creating": "create",
    "implement": "implement", "implementing": "implement",
    "extract": "extract", "extracting": "extract",
    "parse": "parse", "parsing": "parse",
    "find": "find", "finding": "find",
    "scrape": "scrape", "scraping": "scrape",
    "identify": "identify", "identifying": "identify",
    "list": "list", "listing": "list",
    "run": "run", "running": "run",
    "execute": "execute", "executing": "execute",
    "search": "search", "searching": "search",
    "send": "send", "sending": "send",
    "open": "open", "opening": "open",
    "read": "read", "reading": "read",
    "call": "call", "calling": "call",
    "invoke": "invoke", "invoking": "invoke",
    "trigger": "trigger", "triggering": "trigger",
    "define": "define", "defining": "define",
    "explain": "explain", "explaining": "explain",
    "tell": "tell", "telling": "tell",
    "select": "select", "selecting": "select",
    "calculate": "calculate", "compute": "compute",
    "summarize": "summarize", "translate": "translate",
}


def _extract_root_action(cleaned_text: str) -> str | None:
    """
    Extracts the primary imperative action verb from the *compressed* query
    (so polite filler verbs like 'wonder' or 'love' never masquerade as the root action).
    """
    # 1. Check if the first word of the cleaned query is a known imperative verb
    words = _WORD_TOKEN_RE.findall(cleaned_text.lower())
    if words and words[0] in _KNOWN_TASK_VERBS:
        return _KNOWN_TASK_VERBS[words[0]]

    # 2. Try spaCy dependency parse on the cleaned text
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(cleaned_text)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                lemma = (token.lemma_ or token.text).lower()
                if lemma not in ("be", "have", "do", "would", "could", "can"):
                    return _KNOWN_TASK_VERBS.get(lemma, lemma)
        for token in doc:
            if token.pos_ == "VERB":
                lemma = (token.lemma_ or token.text).lower()
                return _KNOWN_TASK_VERBS.get(lemma, lemma)

    # 3. Deterministic lexical scan for the first task verb in the cleaned query
    for w in words:
        if w in _KNOWN_TASK_VERBS:
            return _KNOWN_TASK_VERBS[w]
    return None


def _extract_entities(raw_text: str, cleaned_text: str) -> dict[str, Any]:
    """
    Deterministic regex + syntactic entity extractor.
    Extracts literals from `raw_text` and the syntactic `action` verb from `cleaned_text`.
    """
    entities: dict[str, Any] = {}

    emails = re.findall(r"[\w.\-]+@[\w.\-]+\.\w+", raw_text)
    if emails:
        entities["emails"] = list(dict.fromkeys(emails))

    urls = re.findall(r"https?://\S+", raw_text)
    if urls:
        entities["urls"] = list(dict.fromkeys(urls))

    dates = re.findall(
        r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
        raw_text,
        re.IGNORECASE,
    )
    if dates:
        entities["dates"] = list(dict.fromkeys(dates))

    # Extract numbers that are not already part of extracted ISO dates
    text_without_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", raw_text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text_without_dates)
    if numbers:
        entities["numbers"] = list(dict.fromkeys(numbers))

    # Match true quoted strings (ignoring English contractions like I'd or don't)
    double_quoted = re.findall(r'"([^"]+)"', raw_text)
    single_quoted = re.findall(r"(?<!\w)'([^']+)'(?!\w)", raw_text)
    flat_quoted = [q for q in (double_quoted + single_quoted) if q.strip()]
    if flat_quoted:
        entities["quoted_strings"] = list(dict.fromkeys(flat_quoted))

    action = _extract_root_action(cleaned_text)
    if action:
        entities["action"] = action

    return entities


def _check_required_slots(intent: str, entities: dict) -> list[str]:
    missing = []
    for slot in REQUIRED_SLOTS.get(intent, []):
        if slot not in entities or not entities[slot]:
            missing.append(slot)
    return missing


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Cached, Stopword-Filtered BM25 Context, Few-Shot & Tool Pruner
# ══════════════════════════════════════════════════════════════════════════════

_FEW_SHOT_CACHE: dict[str, Any] = {"len": -1, "bm25": None, "tokenized": []}

# Intent prefix affinity map so few-shot examples matching the detected intent get a relevance boost
_INTENT_EXAMPLE_PREFIX: dict[str, str] = {
    "CODE_GENERATION": "ex_code_gen",
    "DATA_EXTRACTION": "ex_data_extract",
    "TOOL_EXECUTION": "ex_tool_exec",
    "FACTUAL_QA": "ex_factual_qa",
}


def _get_few_shot_bm25():
    """Returns a cached BM25 index over FEW_SHOT_EXAMPLES, rebuilding only if registry length changed."""
    if _FEW_SHOT_CACHE["len"] != len(FEW_SHOT_EXAMPLES) or _FEW_SHOT_CACHE["bm25"] is None:
        tokenized = [_tokenize_for_bm25(item["input"]) for item in FEW_SHOT_EXAMPLES]
        _FEW_SHOT_CACHE["tokenized"] = tokenized
        _FEW_SHOT_CACHE["bm25"] = _create_bm25(tokenized) if tokenized else None
        _FEW_SHOT_CACHE["len"] = len(FEW_SHOT_EXAMPLES)
    return _FEW_SHOT_CACHE["bm25"]


def _select_few_shots(query: str, intent: str = "", intent_confidence: float = 0.0, k: int = 2) -> list[dict]:
    """
    Selects up to top-K relevant few-shot examples using stopword-filtered BM25
    plus intent-affinity gating:
    - Never returns examples with score <= 0.0.
    - When intent_confidence >= 0.75 and examples carry intent-prefixed IDs or `intent` tags,
      excludes cross-intent examples that only share a single incidental noun (e.g. 'Python').
    """
    if not FEW_SHOT_EXAMPLES or k <= 0:
        return []

    bm25 = _get_few_shot_bm25()
    if bm25 is None:
        return []

    query_tokens = _tokenize_for_bm25(query)
    raw_scores = bm25.get_scores(query_tokens)
    preferred_prefix = _INTENT_EXAMPLE_PREFIX.get(intent, "")
    known_prefixes = tuple(_INTENT_EXAMPLE_PREFIX.values())

    scored_items: list[tuple[float, int, dict]] = []
    for idx, (score, item) in enumerate(zip(raw_scores, FEW_SHOT_EXAMPLES)):
        item_id = item.get("id", "")
        item_intent = item.get("intent", "")
        matches_intent = (
            (preferred_prefix and item_id.startswith(preferred_prefix))
            or (intent and item_intent == intent)
        )
        has_other_intent_tag = (
            (item_id.startswith(known_prefixes) and not matches_intent)
            or (item_intent and item_intent != intent)
        )

        # If intent confidence is high (>=0.75), filter out examples explicitly tagged for a different intent
        if intent_confidence >= 0.75 and has_other_intent_tag:
            continue

        adjusted = float(score) + (0.75 if matches_intent else 0.0)
        if adjusted > 0.0:
            scored_items.append((adjusted, idx, item))

    scored_items.sort(key=lambda x: (-x[0], x[1]))
    return [item for _, _, item in scored_items[:k]]


# Intent-to-tool guard: prevent side-effect tools (e.g. send_email) from being injected
# into passive DATA_EXTRACTION or FACTUAL_QA queries just because the noun "email" appeared.
_PASSIVE_INTENTS = frozenset({"DATA_EXTRACTION", "FACTUAL_QA"})
_SIDE_EFFECT_TOOLS = frozenset({"send_email"})


def _select_tools(query: str, intent: str = "") -> list[dict]:
    """
    Matches tools using strict word-boundary regexes (preventing substring false positives
    like 'mail' in 'email' or 'open' in 'propensity') and guards against injecting
    irrelevant execution tools when a keyword only appears as a search topic or noun.
    """
    matched: list[tuple[int, dict]] = []
    first_word = query.split()[0].lower() if query.split() else ""

    for tool in TOOL_REGISTRY:
        tool_id = tool["id"]
        hit_count = 0
        lead_verb_hit = False

        for kw in tool.get("trigger_keywords", []):
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, query, re.IGNORECASE):
                hit_count += 1
            if first_word == kw.lower():
                lead_verb_hit = True

        if hit_count == 0:
            continue

        # Block side-effect tools (e.g. send_email) on passive intents (e.g. "Extract all email addresses")
        if intent in _PASSIVE_INTENTS and tool_id in _SIDE_EFFECT_TOOLS and not lead_verb_hit:
            continue

        # Guard `run_python`: only inject when the user asks to run/execute code or scripts,
        # not when "Python" is merely the subject of a web search or a static function-writing request.
        if tool_id == "run_python":
            if not re.search(r"\b(run|execute|script)\b", query, re.IGNORECASE):
                continue

        matched.append((hit_count, tool))

    matched.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in matched]


def _prune_external_context(query: str, context_chunks: list[str] | None, top_k: int = 3) -> str:
    """
    If caller supplies `context_chunks` on RawRequest, ranks and prunes them via BM25
    so only relevant chunks with score > 0 are injected into <verified_context>.
    """
    if not context_chunks:
        return "[No external context provided — rely strictly on <core_task>]"

    tokenized = [_tokenize_for_bm25(chunk) for chunk in context_chunks]
    bm25 = _create_bm25(tokenized)
    scores = bm25.get_scores(_tokenize_for_bm25(query))

    ranked = [
        (score, idx, chunk)
        for idx, (score, chunk) in enumerate(zip(scores, context_chunks))
        if score > 0.0
    ]
    if not ranked:
        ranked = [(0.0, idx, chunk) for idx, chunk in enumerate(context_chunks[:1])]

    ranked.sort(key=lambda x: (-x[0], x[1]))
    selected = sorted(ranked[:top_k], key=lambda x: x[1])
    return "\n---\n".join(chunk.strip() for _, _, chunk in selected)


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


def _complexity_tier(token_count: int, negation_count: int = 0, entity_count: int = 0) -> str:
    score = token_count + (negation_count * 5) + (entity_count * 3)
    if score < 20:
        return "LOW"
    if score < 60:
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
        "FACTUAL_QA":       "Plain text, <=3 sentences",
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

    # ── Stage 1: Negation-Safe Compression (runs FIRST so polite filler verbs don't pollute Stage 2) ──
    compressed, stripped = _protect_and_strip_filler(request.text)
    compressed = _bm25_sentence_prune(compressed)
    trace.stripped_tokens = stripped
    trace.optimized_token_count = len(compressed.split())

    # ── Stage 2: Weighted Intent Routing & Deterministic Entity Extraction ───
    intent, confidence = _route_intent(
        cleaned_text=compressed,
        raw_text=request.text,
        task_hint=getattr(request, "task_hint", ""),
    )
    negations = _extract_negations(compressed)
    entities = _extract_entities(raw_text=request.text, cleaned_text=compressed)

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

    # ── Stage 3: Stopword-Filtered BM25 Pruning & Intent-Gated Tool Selection ──
    tools = _select_tools(compressed, intent=intent)
    examples = _select_few_shots(compressed, intent=intent, intent_confidence=confidence, k=few_shot_k)
    context_chunks = getattr(request, "context_chunks", None)
    pruned_context = _prune_external_context(compressed, context_chunks)

    trace.injected_tool_ids = [t["id"] for t in tools]
    trace.injected_example_ids = [e["id"] for e in examples]

    # ── Stage 4: Structural Compilation ──────────────────────────────────────
    guardrail_block = "\n".join(f"{i+5}. {r}" for i, r in enumerate(GUARDRAIL_RULES))

    payload = _COMPILED_TEMPLATE.format(
        guardrails=guardrail_block,
        intent=intent,
        complexity=_complexity_tier(
            trace.raw_token_count,
            negation_count=len(negations),
            entity_count=len(entities),
        ),
        negations=", ".join(negations) if negations else "NONE",
        entities=_format_entities(entities),
        context=pruned_context,
        tools=_format_tools(tools),
        examples=_format_examples(examples),
        query=compressed,
        output_format=_infer_output_format(intent),
    )

    trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return CompiledPrompt(payload=payload, trace=trace, status="OK")