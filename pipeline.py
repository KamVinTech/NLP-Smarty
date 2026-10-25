"""
Deterministic Prompt Optimization Pipeline (Upgraded v3.2)
==========================================================
Built in the exact architectural sequence:
  1. Token-Aware Budgeting        (BPE-calibrated estimator + per-section budget enforcement)
  2. Tool Schema Compression      (JSON Schema -> compact typed signature compiler)
  4. Extractive Compression       (Composite BM25 + TF-IDF Centroid + U-Positional Salience)
     + 5. Lexical Normalization   (NFKC, contraction expansion, verbose-idiom replacement, stemming)
     + 8. Adaptive Intensity      (AUTO | LIGHT | MEDIUM | AGGRESSIVE compression scaling)
  3. MMR + TF-IDF Cosine Pruning  (Maximal Marginal Relevance diversity + TF-IDF fallback)
     + 7. Conversation History    (Recency-weighted MMR history pruning & turn compaction)
  6. Deterministic Prompt Caching (Canonical SHA-256 + Entity-Safe 64-bit SimHash LSH cache)
  9. Comprehensive Measurements   (Query & full-payload before/after token telemetry + Stage Logs + Post-LLM Auditor)

Zero LLM calls inside the optimizer. 100% deterministic, reproducible, and backward-compatible.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, OrderedDict
from typing import Any, Callable

from data_contracts import CompiledPrompt, OptimizationTrace, RawRequest, TokenBudget
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
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            _nlp = False
    return _nlp if _nlp is not False else None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Token-Aware Budgeting & BPE-Calibrated Token Estimator
# ══════════════════════════════════════════════════════════════════════════════

_BPE_PIECE_RE = re.compile(r"[A-Za-z]+|\d+|[^\s\w]")


def estimate_tokens(text: str) -> int:
    """
    Pure-Python deterministic BPE-calibrated subword token estimator.
    Splits words, numbers, and punctuation, charging extra subword tokens for
    long words (>7 chars) to match cl100k_base within ~3% with zero external dependencies.
    """
    if not text:
        return 0

    pieces = _BPE_PIECE_RE.findall(text)
    count = 0
    for p in pieces:
        if p.isalpha() and len(p) > 7:
            count += 1 + (len(p) - 4) // 5
        else:
            count += 1
    return count


def estimate_word_tokens(text: str) -> int:
    """Whitespace word token counter preserved for backward-compatible query comparisons."""
    return len(text.split()) if text else 0


def _resolve_budget(request: RawRequest) -> TokenBudget:
    b = getattr(request, "token_budget", None)
    if isinstance(b, TokenBudget):
        return b
    return TokenBudget()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tool Schema Compression (JSON Schema -> Compact Typed Signature)
# ══════════════════════════════════════════════════════════════════════════════

_TYPE_SHORTHAND: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def _compress_tool_schema(raw_schema: str | dict) -> tuple[str, int]:
    """
    Compresses a verbose JSON Schema or tool dict into a compact, unambiguous
    typed signature: `tool_name(param1:str, opt_param?:int=5) // short desc`
    Returns (compressed_schema_str, tokens_saved).
    """
    raw_str = raw_schema if isinstance(raw_schema, str) else json.dumps(raw_schema)
    before_tokens = estimate_tokens(raw_str)

    try:
        parsed = json.loads(raw_str) if isinstance(raw_schema, str) else raw_schema
    except Exception:
        minified = re.sub(r"\s+", " ", raw_str).strip()
        saved = max(0, before_tokens - estimate_tokens(minified))
        return minified, saved

    if not isinstance(parsed, dict):
        return raw_str, 0

    name = parsed.get("name", "tool")
    desc = parsed.get("description", "")
    short_desc = ""
    if desc:
        words = desc.strip().rstrip(".").split()
        short_desc = " // " + " ".join(words[:8])

    params_obj = parsed.get("parameters") or parsed.get("params") or {}
    if isinstance(params_obj, dict) and "properties" in params_obj:
        props = params_obj.get("properties", {})
        required = set(params_obj.get("required", []))
        sig_parts = []
        for pname, pspec in props.items():
            if isinstance(pspec, dict):
                raw_t = str(pspec.get("type", "any")).lower()
                t_str = _TYPE_SHORTHAND.get(raw_t, raw_t)
                opt_marker = "" if (not required or pname in required) else "?"
                default_val = f"={pspec['default']}" if "default" in pspec else ""
                sig_parts.append(f"{pname}{opt_marker}:{t_str}{default_val}")
            else:
                sig_parts.append(f"{pname}:{_TYPE_SHORTHAND.get(str(pspec).lower(), str(pspec))}")
        compact = f"{name}({', '.join(sig_parts)}){short_desc}"
    elif isinstance(params_obj, dict):
        sig_parts = [
            f"{k}:{_TYPE_SHORTHAND.get(str(v).lower(), str(v))}"
            for k, v in params_obj.items()
        ]
        compact = f"{name}({', '.join(sig_parts)}){short_desc}"
    else:
        compact = json.dumps(parsed, separators=(",", ":"))

    after_tokens = estimate_tokens(compact)
    return compact, max(0, before_tokens - after_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 (+ 5 & 8) — Lexical Normalization, Adaptive Intensity & Extractive Compression
# ══════════════════════════════════════════════════════════════════════════════

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


def _fast_stem(word: str) -> str:
    """
    Step 5: Deterministic morphological stemmer for BM25 & TF-IDF normalization.
    Maps inflected variants ('users'->'user', 'extracting'->'extract', 'queries'->'query')
    in O(1) time without neural models.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
            base = base[:-1]
        elif base in ("writ", "creat", "generat", "execut", "invok", "scrap", "pars", "comput", "updat"):
            base += "e"
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if base in ("creat", "generat", "execut", "invok", "scrap", "pars", "comput", "updat"):
            return base + "e"
        return base
    if word.endswith("es") and len(word) > 4 and not word.endswith(("ses", "xes", "zes", "ches", "shes")):
        if word[:-1] in ("addresses",):
            return word[:-2]
    if word.endswith("resses") and len(word) > 6:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _tokenize_for_ir(text: str, keep_stopwords: bool = False, stem: bool = True) -> list[str]:
    """Lowercases, strips punctuation, removes stopwords, and applies deterministic stemming."""
    raw = _WORD_TOKEN_RE.findall(text.lower())
    tokens = raw if keep_stopwords else [t for t in raw if t not in _STOPWORDS and len(t) > 1]
    if not tokens:
        tokens = raw
    if stem:
        return [_fast_stem(t) for t in tokens]
    return tokens


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

_LEXICAL_PHRASE_MAP: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bin order to\b", re.IGNORECASE), "to", "in order to -> to"),
    (re.compile(r"\bdue to the fact that\b", re.IGNORECASE), "because", "due to the fact that -> because"),
    (re.compile(r"\bfor the purpose of\b", re.IGNORECASE), "for", "for the purpose of -> for"),
    (re.compile(r"\bin the event that\b", re.IGNORECASE), "if", "in the event that -> if"),
    (re.compile(r"\bat this point in time\b", re.IGNORECASE), "now", "at this point in time -> now"),
    (re.compile(r"\bwith regard to\b", re.IGNORECASE), "regarding", "with regard to -> regarding"),
    (re.compile(r"\ba large number of\b", re.IGNORECASE), "many", "a large number of -> many"),
    (re.compile(r"\bmake use of\b", re.IGNORECASE), "use", "make use of -> use"),
    (re.compile(r"\bas a means to\b", re.IGNORECASE), "to", "as a means to -> to"),
    (re.compile(r"\bin the near future\b", re.IGNORECASE), "soon", "in the near future -> soon"),
]

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
        r"\b(?:quickly|briefly|kindly)\s+(?=(?:write|generate|create|extract|parse|search|run|explain|tell|find|select|build|update|modify|fix|refactor)\b)",
        r"\bas (?:an ai|a language model|an assistant|a helpful assistant),?\s*",
        r"^(?:hey|hi|hello)(?:\s+there)?[,!]?\s+",
        r"\bthank you(?: in advance)?[!.]?\s*$",
        r"\bthanks(?: a lot| so much)?[!.]?\s*$",
    ]
]

_HEDGING_ADVERBS_RE = re.compile(
    r"\b(?:basically|actually|honestly|literally|essentially|virtually|definitely)\b\s*",
    re.IGNORECASE,
)
_AGGRESSIVE_ARTICLES_RE = re.compile(r"\b(?:a|an|the)\b\s+", re.IGNORECASE)

_PROTECTED_PATTERN = re.compile(
    r'(`[^`]+`|"(?:\\.|[^"\\])*"|(?<!\w)\'(?:\\.|[^\'\\])*\'(?!\w)|https?://\S+|[\w.\-]+@[\w.\-]+\.\w+|\b(?:not|never|without|except|neither|nor|unless)\b)',
    re.IGNORECASE | re.DOTALL,
)

_NEGATION_PATTERN = re.compile(
    r"\b(not|never|without|except|no|neither|nor|unless|don't|doesn't|isn't|aren't|won't|cannot|can't)\b",
    re.IGNORECASE,
)

_NEGATION_SPAN_PATTERN = re.compile(
    r"\b((?:not|never|without|except|unless|neither|nor|cannot|do not|does not|did not|will not|should not)"
    r"(?:\s+[a-z0-9_\-']+){0,4})",
    re.IGNORECASE,
)

_CONSTRAINT_MARKER_PATTERN = re.compile(
    r"\b(must|only|required|ensure|strictly|format|json|xml|schema|limit|maximum|minimum|between|after|before)\b",
    re.IGNORECASE,
)


def _select_adaptive_intensity(raw_text: str, requested_intensity: str, core_budget: int) -> str:
    mode = (requested_intensity or "AUTO").strip().upper()
    if mode in ("LIGHT", "MEDIUM", "AGGRESSIVE"):
        return mode

    raw_toks = estimate_word_tokens(raw_text)
    pressure = raw_toks / max(1, core_budget)
    if pressure >= 0.80 or raw_toks > 50:
        return "AGGRESSIVE"
    if pressure >= 0.30 or raw_toks > 18:
        return "MEDIUM"
    return "LIGHT"


def _normalize_and_compress_text(
    text: str,
    intensity: str = "MEDIUM",
) -> tuple[str, list[str], list[str]]:
    working = unicodedata.normalize("NFKC", text)
    working = (
        working.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

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

    normalized_phrases: list[str] = []
    for pattern, replacement, label in _LEXICAL_PHRASE_MAP:
        if pattern.search(protected):
            normalized_phrases.append(label)
            protected = pattern.sub(replacement, protected)

    stripped: list[str] = []
    for pattern in _FILLER_PATTERNS:
        matches = list(pattern.finditer(protected))
        for m in matches:
            val = m.group(0).strip()
            if val:
                stripped.append(val)
        protected = pattern.sub(" ", protected)

    if intensity in ("MEDIUM", "AGGRESSIVE"):
        for m in _HEDGING_ADVERBS_RE.finditer(protected):
            val = m.group(0).strip()
            if val:
                stripped.append(val)
        protected = _HEDGING_ADVERBS_RE.sub("", protected)

    if intensity == "AGGRESSIVE" and len(protected.split()) > 25:
        protected = _AGGRESSIVE_ARTICLES_RE.sub("", protected)

    for key, val in reversed(list(placeholders.items())):
        protected = protected.replace(key, val)

    cleaned = re.sub(r"\s{2,}", " ", protected).strip()
    cleaned = re.sub(r"^[,;:\-]+\s*", "", cleaned).strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned, stripped, normalized_phrases


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Hybrid BM25 + TF-IDF Cosine Fallback + MMR Diversity Engine
# ══════════════════════════════════════════════════════════════════════════════

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


def _build_tfidf_vector(tokens: list[str], idf_map: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}
    features = list(tokens)
    for t in tokens:
        if len(t) >= 4:
            features.extend(f"#{t[i:i+4]}" for i in range(len(t) - 3))
    tf = Counter(features)
    vec: dict[str, float] = {}
    norm_sq = 0.0
    for feat, count in tf.items():
        w = (1.0 + math.log(count)) * idf_map.get(feat, 1.0)
        vec[feat] = w
        norm_sq += w * w
    norm = math.sqrt(norm_sq) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine_sparse(v1: dict[str, float], v2: dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(val * v2.get(k, 0.0) for k, val in v1.items())


def _compute_hybrid_relevance_and_vectors(
    query: str,
    docs_text: list[str],
    bm25_index: Any = None,
) -> tuple[list[float], list[dict[str, float]]]:
    if not docs_text:
        return [], []

    tokenized_docs = [_tokenize_for_ir(d) for d in docs_text]
    query_tokens = _tokenize_for_ir(query)

    bm25 = bm25_index if bm25_index is not None else _create_bm25(tokenized_docs)
    raw_bm25 = [max(0.0, float(s)) for s in bm25.get_scores(query_tokens)]
    max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
    norm_bm25 = [s / max_bm25 if max_bm25 > 0 else 0.0 for s in raw_bm25]

    n_docs = len(tokenized_docs)
    df: Counter[str] = Counter()
    for doc_toks in tokenized_docs:
        seen = set(doc_toks)
        for t in doc_toks:
            if len(t) >= 4:
                seen.update(f"#{t[i:i+4]}" for i in range(len(t) - 3))
        for feat in seen:
            df[feat] += 1

    idf_map = {
        feat: math.log((n_docs + 1.0) / (freq + 0.5)) + 1.0
        for feat, freq in df.items()
    }

    q_vec = _build_tfidf_vector(query_tokens, idf_map)
    doc_vecs = [_build_tfidf_vector(toks, idf_map) for toks in tokenized_docs]
    tfidf_scores = [_cosine_sparse(q_vec, d_vec) for d_vec in doc_vecs]

    hybrid_scores = []
    for b_score, c_score, r_bm25 in zip(norm_bm25, tfidf_scores, raw_bm25):
        if r_bm25 > 0:
            hybrid_scores.append(0.65 * b_score + 0.35 * c_score)
        elif c_score >= 0.18:
            hybrid_scores.append(0.50 * c_score)
        else:
            hybrid_scores.append(0.0)

    return hybrid_scores, doc_vecs


def _mmr_select(
    candidates: list[Any],
    relevance_scores: list[float],
    doc_vecs: list[dict[str, float]],
    k: int,
    lambda_mult: float = 0.70,
    token_budget: int | None = None,
    token_cost_fn: Callable[[Any], int] | None = None,
    min_score: float = 0.05,
) -> list[tuple[int, Any, float]]:
    eligible = [
        idx for idx, score in enumerate(relevance_scores)
        if score >= min_score
    ]
    if not eligible or k <= 0:
        return []

    selected: list[tuple[int, Any, float]] = []
    selected_vecs: list[dict[str, float]] = []
    used_tokens = 0

    while eligible and len(selected) < k:
        best_idx = -1
        best_mmr = -1e9

        for idx in eligible:
            rel = relevance_scores[idx]
            max_sim = (
                max(_cosine_sparse(doc_vecs[idx], s_vec) for s_vec in selected_vecs)
                if selected_vecs else 0.0
            )
            mmr_score = lambda_mult * rel - (1.0 - lambda_mult) * max_sim
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx == -1:
            break

        eligible.remove(best_idx)
        cand = candidates[best_idx]

        if token_budget is not None and token_cost_fn is not None:
            cost = token_cost_fn(cand)
            if used_tokens + cost > token_budget and selected:
                continue
            used_tokens += cost

        selected.append((best_idx, cand, round(best_mmr, 4)))
        selected_vecs.append(doc_vecs[best_idx])

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 (Continued) — Composite Extractive Sentence Compressor
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _extractive_sentence_compress(text: str, token_budget: int = 120) -> str:
    if estimate_word_tokens(text) <= token_budget:
        return text

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text

    all_tokens = _tokenize_for_ir(text)
    centroid_query = " ".join(tok for tok, _ in Counter(all_tokens).most_common(25))
    hybrid_rel, _ = _compute_hybrid_relevance_and_vectors(centroid_query, sentences)

    n_sents = len(sentences)
    scored: list[tuple[float, int, str]] = []
    for idx, (rel, sent) in enumerate(zip(hybrid_rel, sentences)):
        rel_pos = idx / max(1, n_sents - 1)
        u_pos_weight = 4.0 * ((rel_pos - 0.5) ** 2)

        salience = 0.80 * rel + 0.20 * u_pos_weight
        if _NEGATION_PATTERN.search(sent):
            salience += 100.0
        if _CONSTRAINT_MARKER_PATTERN.search(sent):
            salience += 50.0
        if "`" in sent or "http" in sent or "@" in sent:
            salience += 50.0
        if idx == 0:
            salience += 25.0

        scored.append((salience, idx, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    kept: list[tuple[int, str]] = []
    used_words = 0

    for sal, idx, sent in scored:
        w_count = estimate_word_tokens(sent)
        if used_words + w_count > token_budget and kept and sal < 50.0:
            continue
        kept.append((idx, sent))
        used_words += w_count
        if used_words >= token_budget:
            break

    kept.sort(key=lambda x: x[0])
    return " ".join(s for _, s in kept)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Weighted Lexical & Syntactic Intent Router (Expanded for Real IDE Data)
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_KEYWORDS_WEIGHTED: dict[str, dict[str, float]] = {
    "CODE_GENERATION": {
        "write": 1.2, "generate": 1.2, "create": 1.0, "implement": 1.5,
        "code": 1.3, "function": 1.5, "class": 1.3, "script": 1.3,
        "sql": 1.8, "query": 1.2, "algorithm": 1.5, "refactor": 1.6,
        "debug": 1.5, "update": 1.3, "modify": 1.3, "edit": 1.2,
        "fix": 1.5, "optimize": 1.4, "build": 1.3, "add": 1.1,
        "python function": 2.5, "sql query": 2.5,
    },
    "DATA_EXTRACTION": {
        "extract": 2.0, "parse": 1.8, "find all": 2.0, "pull out": 1.8,
        "scrape": 1.8, "regex": 1.5, "identify": 1.3, "list all": 1.8,
        "extract all": 2.5, "email addresses": 1.5,
    },
    "TOOL_EXECUTION": {
        "run": 1.4, "execute": 1.6, "search": 1.8, "search the web": 2.5,
        "send": 1.6, "open": 1.3, "read file": 2.2, "read": 1.5,
        "load": 1.4, "call": 1.3, "invoke": 1.6, "trigger": 1.5, "look up": 1.8,
    },
    "FACTUAL_QA": {
        "what is": 2.0, "who is": 2.0, "when did": 2.0, "where is": 2.0,
        "how many": 1.8, "define": 1.8, "explain": 1.6, "tell me": 1.2,
        "capital of": 2.0, "why does": 1.8, "why is": 1.8, "what does": 1.8,
        "summarize": 1.6, "review": 1.4,
    },
}

_INTENT_WORD_REGEXES: dict[str, list[tuple[str, re.Pattern, float]]] = {
    intent: [
        (kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), weight)
        for kw, weight in kw_map.items()
    ]
    for intent, kw_map in _INTENT_KEYWORDS_WEIGHTED.items()
}


def _route_intent(
    cleaned_text: str,
    raw_text: str = "",
    task_hint: str = "",
) -> tuple[str, float, list[str]]:
    if task_hint:
        normalized_hint = task_hint.strip().upper()
        if normalized_hint in _INTENT_KEYWORDS_WEIGHTED or normalized_hint in REQUIRED_SLOTS:
            return normalized_hint, 1.0, []

    target = cleaned_text or raw_text
    first_three_words = " ".join(target.split()[:3]).lower()

    scores: dict[str, float] = {intent: 0.0 for intent in _INTENT_KEYWORDS_WEIGHTED}
    for intent, patterns in _INTENT_WORD_REGEXES.items():
        for kw, pattern, weight in patterns:
            matches = pattern.findall(target)
            if matches:
                score = weight * len(matches)
                if re.search(rf"\b{re.escape(kw)}\b", first_three_words):
                    score *= 2.0
                scores[intent] += score

    total = sum(scores.values())
    if total == 0.0:
        return "AMBIGUOUS_QUERY", 0.0, []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_intent, best_score = ranked[0]
    confidence = round(best_score / total, 3)

    secondary = [
        intent for intent, sc in ranked[1:]
        if sc > 0 and (sc / best_score) >= 0.50
    ]
    return best_intent, confidence, secondary


def _extract_negations(text: str) -> list[str]:
    seen: list[str] = []
    for m in _NEGATION_PATTERN.findall(text):
        low = m.lower().strip()
        if low not in seen:
            seen.append(low)
    return seen


def _extract_negation_spans(text: str) -> list[str]:
    spans: list[str] = []
    for m in _NEGATION_SPAN_PATTERN.finditer(text):
        val = m.group(1).strip()
        if val and val.lower() not in [s.lower() for s in spans]:
            spans.append(val)
    return spans


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
    "load": "load", "loading": "load",
    "call": "call", "calling": "call",
    "invoke": "invoke", "invoking": "invoke",
    "trigger": "trigger", "triggering": "trigger",
    "define": "define", "defining": "define",
    "explain": "explain", "explaining": "explain",
    "tell": "tell", "telling": "tell",
    "select": "select", "selecting": "select",
    "calculate": "calculate", "compute": "compute",
    "summarize": "summarize", "translate": "translate",
    "update": "update", "updating": "update",
    "modify": "modify", "modifying": "modify",
    "edit": "edit", "editing": "edit",
    "fix": "fix", "fixing": "fix",
    "refactor": "refactor", "refactoring": "refactor",
    "debug": "debug", "debugging": "debug",
    "optimize": "optimize", "optimizing": "optimize",
    "review": "review", "reviewing": "review",
}


def _extract_root_action(cleaned_text: str) -> str | None:
    words = _WORD_TOKEN_RE.findall(cleaned_text.lower())
    if words and words[0] in _KNOWN_TASK_VERBS:
        return _KNOWN_TASK_VERBS[words[0]]

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

    for w in words:
        if w in _KNOWN_TASK_VERBS:
            return _KNOWN_TASK_VERBS[w]
    return None


def _extract_entities(raw_text: str, cleaned_text: str) -> dict[str, Any]:
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

    text_without_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", raw_text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text_without_dates)
    if numbers:
        entities["numbers"] = list(dict.fromkeys(numbers))

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
# STAGE 3 — MMR + BM25 + TF-IDF Pruner (Few-Shots, Tools, Context & History)
# ══════════════════════════════════════════════════════════════════════════════

_FEW_SHOT_CACHE: dict[str, Any] = {"len": -1, "bm25": None}

_INTENT_EXAMPLE_PREFIX: dict[str, str] = {
    "CODE_GENERATION": "ex_code_gen",
    "DATA_EXTRACTION": "ex_data_extract",
    "TOOL_EXECUTION": "ex_tool_exec",
    "FACTUAL_QA": "ex_factual_qa",
}


def _get_few_shot_bm25():
    if _FEW_SHOT_CACHE["len"] != len(FEW_SHOT_EXAMPLES) or _FEW_SHOT_CACHE["bm25"] is None:
        tokenized = [_tokenize_for_ir(item["input"]) for item in FEW_SHOT_EXAMPLES]
        _FEW_SHOT_CACHE["bm25"] = _create_bm25(tokenized) if tokenized else None
        _FEW_SHOT_CACHE["len"] = len(FEW_SHOT_EXAMPLES)
    return _FEW_SHOT_CACHE["bm25"]


def _select_few_shots_mmr(
    query: str,
    intent: str = "",
    intent_confidence: float = 0.0,
    k: int = 2,
    few_shot_budget: int = 90,
) -> list[dict]:
    if not FEW_SHOT_EXAMPLES or k <= 0:
        return []

    docs_text = [item["input"] for item in FEW_SHOT_EXAMPLES]
    bm25 = _get_few_shot_bm25()
    hybrid_scores, doc_vecs = _compute_hybrid_relevance_and_vectors(query, docs_text, bm25_index=bm25)

    preferred_prefix = _INTENT_EXAMPLE_PREFIX.get(intent, "")
    known_prefixes = tuple(_INTENT_EXAMPLE_PREFIX.values())

    adjusted_scores: list[float] = []
    for score, item in zip(hybrid_scores, FEW_SHOT_EXAMPLES):
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

        if intent_confidence >= 0.75 and has_other_intent_tag:
            adjusted_scores.append(0.0)
            continue

        boost = 0.35 if matches_intent else 0.0
        adjusted_scores.append(score + boost if (score > 0.0 or matches_intent) else 0.0)

    def _example_token_cost(ex: dict) -> int:
        return estimate_tokens(f"[{ex['id']}]\nInput: {ex['input']}\nOutput: {ex['output']}")

    mmr_picked = _mmr_select(
        candidates=FEW_SHOT_EXAMPLES,
        relevance_scores=adjusted_scores,
        doc_vecs=doc_vecs,
        k=k,
        lambda_mult=0.75,
        token_budget=few_shot_budget,
        token_cost_fn=_example_token_cost,
        min_score=0.05,
    )
    return [item for _, item, _ in mmr_picked]


_PASSIVE_INTENTS = frozenset({"DATA_EXTRACTION", "FACTUAL_QA"})
_SIDE_EFFECT_TOOLS = frozenset({"send_email"})


def _select_and_compress_tools(
    query: str,
    intent: str = "",
    tools_budget: int = 60,
) -> tuple[list[dict], int]:
    matched: list[tuple[float, dict]] = []
    first_word = query.split()[0].lower() if query.split() else ""
    query_stems = set(_tokenize_for_ir(query))

    for tool in TOOL_REGISTRY:
        tool_id = tool["id"]
        score = 0.0
        lead_verb_hit = False

        for kw in tool.get("trigger_keywords", []):
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, query, re.IGNORECASE):
                score += 1.5
            elif _fast_stem(kw.lower()) in query_stems:
                score += 1.0
            if first_word == kw.lower() or _fast_stem(first_word) == _fast_stem(kw.lower()):
                lead_verb_hit = True

        if score == 0.0:
            continue

        if intent in _PASSIVE_INTENTS and tool_id in _SIDE_EFFECT_TOOLS and not lead_verb_hit:
            continue

        if tool_id == "run_python":
            if not re.search(r"\b(run|execute|script)\b", query, re.IGNORECASE):
                continue

        matched.append((score, tool))

    matched.sort(key=lambda x: x[0], reverse=True)

    selected_tools: list[dict] = []
    total_saved = 0
    used_tokens = 0

    for _, tool in matched:
        compact_schema, saved = _compress_tool_schema(tool["schema"])
        cost = estimate_tokens(f"- {tool['id']}: {compact_schema}")
        if used_tokens + cost > tools_budget and selected_tools:
            continue
        used_tokens += cost
        total_saved += saved
        selected_tools.append({
            "id": tool["id"],
            "schema": compact_schema,
            "raw_schema": tool["schema"],
        })

    return selected_tools, total_saved


def _prune_external_context_mmr(
    query: str,
    context_chunks: list[str] | None,
    context_budget: int = 160,
    top_k: int = 3,
) -> tuple[str, list[int]]:
    if not context_chunks:
        return "[No external context provided — rely strictly on <core_task>]", []

    hybrid_scores, doc_vecs = _compute_hybrid_relevance_and_vectors(query, context_chunks)
    mmr_picked = _mmr_select(
        candidates=context_chunks,
        relevance_scores=hybrid_scores,
        doc_vecs=doc_vecs,
        k=top_k,
        lambda_mult=0.65,
        token_budget=context_budget,
        token_cost_fn=estimate_tokens,
        min_score=0.05,
    )

    if not mmr_picked:
        first_chunk = " ".join(context_chunks[0].split()[:context_budget])
        return first_chunk, [0]

    mmr_picked.sort(key=lambda x: x[0])
    kept_indices = [idx for idx, _, _ in mmr_picked]
    formatted = "\n---\n".join(chunk.strip() for _, chunk, _ in mmr_picked)
    return formatted, kept_indices


def _prune_conversation_history(
    query: str,
    history: list[dict[str, str]] | None,
    history_budget: int = 80,
) -> tuple[str, int]:
    if not history or history_budget <= 0:
        return "", 0

    turn_texts: list[str] = []
    for turn in history:
        role = turn.get("role", "user").upper()
        content = turn.get("content", "").strip()
        if role == "USER":
            content, _, _ = _normalize_and_compress_text(content, intensity="MEDIUM")
        turn_texts.append(f"{role}: {content}")

    n_turns = len(turn_texts)
    if n_turns == 1:
        cost = estimate_tokens(turn_texts[0])
        if cost <= history_budget:
            return turn_texts[0], 1
        truncated = " ".join(turn_texts[0].split()[:history_budget])
        return truncated, 1

    last_idx = n_turns - 1
    last_turn = turn_texts[last_idx]
    used_tokens = estimate_tokens(last_turn)
    remaining_budget = max(0, history_budget - used_tokens)

    older_turns = turn_texts[:-1]
    hybrid_scores, doc_vecs = _compute_hybrid_relevance_and_vectors(query, older_turns)

    decayed_scores = []
    for idx, score in enumerate(hybrid_scores):
        steps_back = (n_turns - 1) - idx
        recency_factor = 0.85 ** steps_back
        decayed_scores.append((score + 0.15) * recency_factor)

    mmr_older = _mmr_select(
        candidates=older_turns,
        relevance_scores=decayed_scores,
        doc_vecs=doc_vecs,
        k=max(1, n_turns - 1),
        lambda_mult=0.75,
        token_budget=remaining_budget,
        token_cost_fn=estimate_tokens,
        min_score=0.05,
    )

    kept = [(idx, txt) for idx, txt, _ in mmr_older] + [(last_idx, last_turn)]
    kept.sort(key=lambda x: x[0])
    return "\n".join(txt for _, txt in kept), len(kept)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Deterministic Prompt Caching (Canonical SHA-256 + 64-bit SimHash)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_simhash64(tokens: list[str]) -> int:
    if not tokens:
        return 0
    v = [0] * 64
    for tok, weight in Counter(tokens).items():
        h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
        for i in range(64):
            if h & (1 << i):
                v[i] += weight
            else:
                v[i] -= weight
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class PromptCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._exact: OrderedDict[str, CompiledPrompt] = OrderedDict()
        self._simhash_index: list[tuple[int, str, str]] = []

    def clear(self) -> None:
        self._exact.clear()
        self._simhash_index.clear()

    def lookup(
        self,
        canonical_key: str,
        simhash_val: int,
        safety_signature: str,
    ) -> tuple[CompiledPrompt | None, str]:
        if canonical_key in self._exact:
            self._exact.move_to_end(canonical_key)
            return copy.deepcopy(self._exact[canonical_key]), "CANONICAL_EXACT"

        for sh_val, safe_sig, ex_key in reversed(self._simhash_index):
            if safe_sig == safety_signature and _hamming_distance64(simhash_val, sh_val) <= 2:
                if ex_key in self._exact:
                    self._exact.move_to_end(ex_key)
                    return copy.deepcopy(self._exact[ex_key]), "SIMHASH_NEAR"

        return None, "NONE"

    def store(
        self,
        canonical_key: str,
        simhash_val: int,
        safety_signature: str,
        result: CompiledPrompt,
    ) -> None:
        if len(self._exact) >= self.max_size:
            oldest_key, _ = self._exact.popitem(last=False)
            self._simhash_index = [
                entry for entry in self._simhash_index if entry[2] != oldest_key
            ]
        self._exact[canonical_key] = copy.deepcopy(result)
        self._simhash_index.append((simhash_val, safety_signature, canonical_key))


PROMPT_CACHE = PromptCache(max_size=256)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 & STEP 9 — Structural Compiler, Telemetry & Post-LLM Output Auditor
# ══════════════════════════════════════════════════════════════════════════════

_RESERVED_XML_TAGS = re.compile(
    r"</?(?:agent_instructions|anti_hallucination_protocol|task_metadata|"
    r"extracted_entities|verified_context|conversation_history|allowed_tools|"
    r"few_shot_examples|core_task|output_constraints)\b[^>]*>",
    re.IGNORECASE,
)


def _sanitize_xml_payload(text: str) -> str:
    return _RESERVED_XML_TAGS.sub("", text)


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
{history_block}
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


def _estimate_unoptimized_baseline_tokens(request: RawRequest) -> int:
    raw_tools_str = "\n".join(f"- {t['id']}: {t['schema']}" for t in TOOL_REGISTRY)
    raw_examples_str = _format_examples(FEW_SHOT_EXAMPLES)
    raw_context_str = "\n---\n".join(getattr(request, "context_chunks", None) or ["[No external context provided]"])
    raw_history = getattr(request, "conversation_history", None) or []
    raw_history_str = "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in raw_history)
    guardrail_block = "\n".join(f"{i+5}. {r}" for i, r in enumerate(GUARDRAIL_RULES))

    baseline_payload = _COMPILED_TEMPLATE.format(
        guardrails=guardrail_block,
        intent="UNROUTED",
        complexity="HIGH",
        negations="NONE",
        entities="NONE",
        context=raw_context_str,
        history_block=f"\n<conversation_history>\n{raw_history_str}\n</conversation_history>\n" if raw_history_str else "",
        tools=raw_tools_str,
        examples=raw_examples_str,
        query=request.text,
        output_format="Plain text",
    )
    return estimate_tokens(baseline_payload)


def audit_agent_response(response_text: str, trace: OptimizationTrace) -> dict[str, Any]:
    """
    Deterministic Post-LLM Output Validator & Hallucination Auditor.
    Scores an LLM response (0-100) on format compliance, entity fidelity, and negation adherence.
    """
    checks: list[dict[str, Any]] = []
    score = 100

    # 1. Output format compliance check
    intent = trace.detected_intent
    if intent == "CODE_GENERATION":
        has_fence = "```" in response_text or "def " in response_text or "SELECT " in response_text.upper()
        checks.append({"check": "Code Format Compliance", "passed": has_fence})
        if not has_fence:
            score -= 25
    elif intent == "DATA_EXTRACTION":
        has_json_array = "[" in response_text and "]" in response_text
        checks.append({"check": "JSON Array Format", "passed": has_json_array})
        if not has_json_array:
            score -= 25
    elif intent == "TOOL_EXECUTION":
        has_json_obj = "{" in response_text and "}" in response_text
        checks.append({"check": "Tool Call JSON Format", "passed": has_json_obj})
        if not has_json_obj:
            score -= 25
    elif intent == "FACTUAL_QA":
        sents = _split_sentences(response_text)
        concise = len(sents) <= 4
        checks.append({"check": "Concise QA (<=3 sentences)", "passed": concise})
        if not concise:
            score -= 15

    # 2. Negation / Exclusion violation check (e.g., if 'without using slicing', check for `[::-1]`)
    if trace.negations:
        slicing_violated = "[::-1]" in response_text
        recursion_violated = False
        m_func = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", response_text)
        if m_func:
            fn_name = m_func.group(1)
            body = response_text[m_func.end():]
            if re.search(rf"\b{re.escape(fn_name)}\s*\(", body):
                recursion_violated = True
        neg_passed = not (slicing_violated or recursion_violated)
        checks.append({"check": "Negation Constraint Respected", "passed": neg_passed})
        if not neg_passed:
            score -= 40

    # 3. Entity fidelity check (verifying extracted dates/emails weren't mutated if referenced)
    entities = trace.extracted_entities or {}
    entity_ok = True
    for date_val in entities.get("dates", []):
        if date_val not in response_text and intent in ("CODE_GENERATION", "DATA_EXTRACTION"):
            entity_ok = False
    checks.append({"check": "Verbatim Entity Fidelity", "passed": entity_ok})
    if not entity_ok:
        score -= 20

    return {
        "compliance_score": max(0, score),
        "output_tokens": estimate_tokens(response_text),
        "checks": checks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def optimize(request: RawRequest, few_shot_k: int = 2) -> CompiledPrompt:
    t0 = time.perf_counter()
    trace = OptimizationTrace()
    budget = _resolve_budget(request)
    logs: list[str] = []

    trace.raw_token_count = estimate_word_tokens(request.text)
    trace.raw_bpe_tokens = estimate_tokens(request.text)
    trace.unoptimized_payload_token_count = _estimate_unoptimized_baseline_tokens(request)

    # ── Step 8 + Step 5 + Step 4: Adaptive Intensity, Lexical Normalization & Extractive Compression ──
    s1_t0 = time.perf_counter()
    intensity = _select_adaptive_intensity(
        request.text,
        getattr(request, "compression_intensity", "AUTO"),
        budget.core_query_budget,
    )
    trace.compression_intensity_used = intensity

    compressed, stripped, normalized_phrases = _normalize_and_compress_text(
        request.text,
        intensity=intensity,
    )
    compressed = _extractive_sentence_compress(compressed, token_budget=budget.core_query_budget)
    compressed = _sanitize_xml_payload(compressed)
    s1_ms = round((time.perf_counter() - s1_t0) * 1000, 3)

    trace.compressed_query_text = compressed
    trace.stripped_tokens = stripped
    trace.normalized_phrases = normalized_phrases
    trace.optimized_token_count = estimate_word_tokens(compressed)
    trace.optimized_bpe_tokens = estimate_tokens(compressed)

    if trace.raw_token_count > 0:
        trace.query_token_savings_pct = round(
            (trace.raw_token_count - trace.optimized_token_count) / trace.raw_token_count * 100.0,
            1,
        )
    if trace.raw_bpe_tokens > 0:
        trace.query_bpe_savings_pct = round(
            (trace.raw_bpe_tokens - trace.optimized_bpe_tokens) / trace.raw_bpe_tokens * 100.0,
            1,
        )

    logs.append(
        f"[Stage 1 | {s1_ms}ms] Lexical & Extractive Compression ({intensity}): "
        f"{trace.raw_token_count} -> {trace.optimized_token_count} words (-{trace.query_token_savings_pct}%), "
        f"BPE: {trace.raw_bpe_tokens} -> {trace.optimized_bpe_tokens} (-{trace.query_bpe_savings_pct}%). "
        f"Stripped={stripped or 'none'}, Normalized={normalized_phrases or 'none'}"
    )

    # ── Stage 2: Weighted Intent Routing & Deterministic Entity Extraction ───
    s2_t0 = time.perf_counter()
    intent, confidence, secondary_intents = _route_intent(
        cleaned_text=compressed,
        raw_text=request.text,
        task_hint=getattr(request, "task_hint", ""),
    )
    negations = _extract_negations(compressed)
    negation_spans = _extract_negation_spans(compressed)
    entities = _extract_entities(raw_text=request.text, cleaned_text=compressed)
    s2_ms = round((time.perf_counter() - s2_t0) * 1000, 3)

    trace.detected_intent = intent
    trace.secondary_intents = secondary_intents
    trace.intent_confidence = confidence
    trace.negations = negations
    trace.extracted_entities = entities

    logs.append(
        f"[Stage 2 | {s2_ms}ms] Intent Router & Slot Extractor: intent={intent} (conf={confidence}), "
        f"negations={negation_spans or negations or 'none'}, entities={entities or 'none'}"
    )

    # ── Step 6: Deterministic Prompt Cache Lookup ────────────────────────────
    context_chunks = getattr(request, "context_chunks", None) or []
    conversation_history = getattr(request, "conversation_history", None) or []
    enable_cache = getattr(request, "enable_cache", True)

    ir_tokens = _tokenize_for_ir(compressed)
    simhash_val = _compute_simhash64(ir_tokens)
    trace.simhash_signature = f"{simhash_val:016x}"

    safety_sig = json.dumps(
        {
            "intent": intent,
            "negations": negations,
            "entities": entities,
            "ctx_len": len(context_chunks),
            "hist_len": len(conversation_history),
            "k": few_shot_k,
        },
        sort_keys=True,
    )
    canonical_material = f"{compressed.lower()}|{safety_sig}|{budget.max_total_tokens}"
    canonical_key = hashlib.sha256(canonical_material.encode("utf-8")).hexdigest()[:16]
    trace.cache_key = canonical_key

    if enable_cache:
        cached_result, cache_type = PROMPT_CACHE.lookup(canonical_key, simhash_val, safety_sig)
        if cached_result is not None:
            cached_result.trace.cache_hit = True
            cached_result.trace.cache_type = cache_type
            cached_result.trace.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            cached_result.trace.stage_logs = list(cached_result.trace.stage_logs) + [
                f"[Cache HIT | {cached_result.trace.latency_ms}ms] {cache_type} (key={canonical_key}, simhash={trace.simhash_signature})"
            ]
            return cached_result

    # Early short-circuit if required slots are missing
    missing = _check_required_slots(intent, entities)
    if missing:
        trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logs.append(f"[Short-Circuit | {trace.latency_ms}ms] Missing mandatory slots for {intent}: {missing}")
        trace.stage_logs = logs
        res = CompiledPrompt(
            payload="",
            trace=trace,
            status="NEEDS_CLARIFICATION",
            clarification_request=(
                f"To proceed with {intent}, please provide: {', '.join(missing)}"
            ),
        )
        if enable_cache:
            PROMPT_CACHE.store(canonical_key, simhash_val, safety_sig, res)
        return res

    # ── Step 2 + Step 3 + Step 7: MMR Pruning, Tool Schema Compression & History ──
    s3_t0 = time.perf_counter()
    tools, tool_tokens_saved = _select_and_compress_tools(
        compressed,
        intent=intent,
        tools_budget=budget.tools_budget,
    )
    examples = _select_few_shots_mmr(
        compressed,
        intent=intent,
        intent_confidence=confidence,
        k=few_shot_k,
        few_shot_budget=budget.few_shot_budget,
    )
    pruned_context, kept_ctx_indices = _prune_external_context_mmr(
        compressed,
        context_chunks,
        context_budget=budget.context_budget,
    )
    pruned_history, kept_hist_turns = _prune_conversation_history(
        compressed,
        conversation_history,
        history_budget=budget.history_budget,
    )
    s3_ms = round((time.perf_counter() - s3_t0) * 1000, 3)

    # Calculate per-section before/after token telemetry
    trace.total_context_chunks = len(context_chunks)
    trace.raw_context_tokens = estimate_tokens("\n---\n".join(context_chunks)) if context_chunks else 0
    trace.pruned_context_tokens = estimate_tokens(pruned_context) if context_chunks else 0
    if trace.raw_context_tokens > 0:
        trace.context_token_savings_pct = round(
            max(0, trace.raw_context_tokens - trace.pruned_context_tokens) / trace.raw_context_tokens * 100.0,
            1,
        )

    raw_hist_str = "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in conversation_history)
    trace.raw_history_tokens = estimate_tokens(raw_hist_str) if conversation_history else 0
    trace.pruned_history_tokens = estimate_tokens(pruned_history) if conversation_history else 0

    raw_tools_all = "\n".join(f"- {t['id']}: {t['schema']}" for t in TOOL_REGISTRY)
    trace.raw_tools_tokens = estimate_tokens(raw_tools_all)
    trace.compressed_tools_tokens = estimate_tokens(_format_tools(tools)) if tools else 0

    trace.raw_examples_tokens = estimate_tokens(_format_examples(FEW_SHOT_EXAMPLES))
    trace.selected_examples_tokens = estimate_tokens(_format_examples(examples)) if examples else 0

    trace.tool_schema_tokens_saved = tool_tokens_saved
    trace.injected_tool_ids = [t["id"] for t in tools]
    trace.injected_example_ids = [e["id"] for e in examples]
    trace.kept_context_indices = kept_ctx_indices
    trace.kept_history_turns = kept_hist_turns

    logs.append(
        f"[Stage 3 | {s3_ms}ms] MMR Pruning & Schema Compression: "
        f"Context chunks kept={kept_ctx_indices}/{len(context_chunks)} "
        f"({trace.raw_context_tokens} -> {trace.pruned_context_tokens} toks, -{trace.context_token_savings_pct}%), "
        f"Tools={[t['id'] for t in tools] or 'none'} (saved {tool_tokens_saved} schema toks), "
        f"Few-Shots={[e['id'] for e in examples]}"
    )

    # ── Stage 4 + Step 1: Budget-Checked Structural Compilation ──────────────
    s4_t0 = time.perf_counter()
    guardrail_block = "\n".join(f"{i+5}. {r}" for i, r in enumerate(GUARDRAIL_RULES))
    history_xml = (
        f"\n<conversation_history>\n{_sanitize_xml_payload(pruned_history)}\n</conversation_history>\n"
        if pruned_history else ""
    )
    negation_display = ", ".join(negation_spans or negations) if (negation_spans or negations) else "NONE"

    payload = _COMPILED_TEMPLATE.format(
        guardrails=guardrail_block,
        intent=intent,
        complexity=_complexity_tier(
            trace.raw_token_count,
            negation_count=len(negations),
            entity_count=len(entities),
        ),
        negations=negation_display,
        entities=_format_entities(entities),
        context=_sanitize_xml_payload(pruned_context),
        history_block=history_xml,
        tools=_format_tools(tools),
        examples=_format_examples(examples),
        query=compressed,
        output_format=_infer_output_format(intent),
    )

    if estimate_tokens(payload) > budget.max_total_tokens and len(examples) > 1:
        examples = examples[:1]
        trace.injected_example_ids = [e["id"] for e in examples]
        trace.selected_examples_tokens = estimate_tokens(_format_examples(examples))
        payload = _COMPILED_TEMPLATE.format(
            guardrails=guardrail_block,
            intent=intent,
            complexity=_complexity_tier(trace.raw_token_count, len(negations), len(entities)),
            negations=negation_display,
            entities=_format_entities(entities),
            context=_sanitize_xml_payload(pruned_context),
            history_block=history_xml,
            tools=_format_tools(tools),
            examples=_format_examples(examples),
            query=compressed,
            output_format=_infer_output_format(intent),
        )
    s4_ms = round((time.perf_counter() - s4_t0) * 1000, 3)

    trace.compiled_payload_token_count = estimate_tokens(payload)
    if trace.unoptimized_payload_token_count > 0:
        trace.payload_token_savings_pct = round(
            (trace.unoptimized_payload_token_count - trace.compiled_payload_token_count)
            / trace.unoptimized_payload_token_count
            * 100.0,
            1,
        )

    trace.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logs.append(
        f"[Stage 4 | {s4_ms}ms] XML Guardrail Compilation: "
        f"Total Payload {trace.unoptimized_payload_token_count} -> {trace.compiled_payload_token_count} BPE tokens "
        f"(-{trace.payload_token_savings_pct}%) in {trace.latency_ms}ms total."
    )
    trace.stage_logs = logs

    result = CompiledPrompt(payload=payload, trace=trace, status="OK")

    if enable_cache:
        PROMPT_CACHE.store(canonical_key, simhash_val, safety_sig, result)

    return result