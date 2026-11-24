"""
Deterministic request gate (the core engine behind the proxy, `smarty bench` and the ledger).

Design rules (each is enforced by tests, not by convention):
  1. PROTECTED SPANS ARE NEVER TOUCHED. Text is split into protected spans (fenced/inline code, URLs, emails, paths,
     identifiers, quoted strings, JSON, negation clauses) and prose; only prose is compressed. A verification pass then
     checks every protected span (and every digit-bearing token) still appears verbatim, and falls back to the original
     text if anything is missing. The guarantee does not depend on compression intensity.
  2. PREFIX STABILITY. Providers cache by exact prefix, so every transform is deterministic and CAUSAL: the transformed
     message i depends only on messages 0..i, never on later ones. transform(msgs[:k]) == transform(msgs)[:k].
  3. VALID STRUCTURE. Tool schemas keep their exact JSON structure (types, enum, required, ...): only `description`
     strings are shortened. System/assistant/tool messages are untouched by the strict profile.
  4. FAIL OPEN. Any internal error returns the request unchanged.
No network, no LLM; only the deterministic rule tables of pipeline.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pipeline as _p  # deterministic rule tables + estimate_tokens (pure Python, no heavy dependencies)

estimate_tokens = _p.estimate_tokens


@dataclass(frozen=True)
class Profile:
    name: str
    prose_level: int        # 1 filler + lexical phrases | 2 + hedging adverbs | 3 + articles in long prose
    dedupe: bool            # causal exact-duplicate collapse of long messages
    schemas: bool = True    # shorten tool-schema descriptions


PROFILES = {
    "strict": Profile("strict", 1, False),
    "balanced": Profile("balanced", 2, True),
    "aggressive": Profile("aggressive", 3, True),
}
DEDUPE_MIN_TOKENS = 200


# ── protected spans ───────────────────────────────────────────────────────────
_PROT_RES = [re.compile(p) for p in (
    r"(?:```|~~~)[\s\S]*?(?:```|~~~|\Z)",                                   # fenced code (also unterminated)
    r"`[^`\n]+`",                                                           # inline code
    r"https?://[^\s<>\"')\]]+",                                             # URLs
    r"[\w.+\-]+@[\w\-]+\.[\w.\-]+",                                         # emails
    r"(?:\.{0,2}/)?(?:[\w.\-]+/)+[\w.\-]*",                                 # paths with a slash
    r"\b[\w\-]+\.(?:py|ts|tsx|js|jsx|json|md|ya?ml|toml|txt|sh|go|rs|java|rb|cc?|cpp|hpp?|css|html|sql)\b",  # file names
    r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b",                                   # snake_case
    r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b",                                   # CONSTANT_CASE
    r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b",                               # CamelCase
    r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b",                                       # lowerCamel
    r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b",                                 # dotted names (os.path.join)
    r"\"(?:\\.|[^\"\\\n]){2,}\"",                                           # double-quoted literals
)]
_NEG_SPAN = re.compile(_p._NEGATION_SPAN_PATTERN.pattern, re.IGNORECASE)    # "never modify the input" (word + up to 4)
_DIGIT_TOKEN = re.compile(r"\d[\w.\-:/,%]*")


def _json_ranges(text: str) -> list[tuple[int, int]]:
    """Brace-balanced {...} / [...] regions that parse as JSON are protected verbatim."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "{[":
            close, depth, j, in_str, esc = "}" if c == "{" else "]", 0, i, False, False
            while j < n and j - i < 200_000:
                ch = text[j]
                if in_str:
                    esc = (ch == "\\") and not esc
                    if ch == '"' and not esc:
                        in_str = False
                    if ch != "\\":
                        esc = False
                elif ch == '"':
                    in_str = True
                elif ch == c:
                    depth += 1
                elif ch == close:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j < n and depth == 0:
                try:
                    json.loads(text[i:j + 1])
                    out.append((i, j + 1))
                    i = j + 1
                    continue
                except ValueError:
                    pass
        i += 1
    return out


def protected_ranges(text: str) -> list[tuple[int, int]]:
    spans = [m.span() for rx in _PROT_RES for m in rx.finditer(text)]
    spans += [m.span() for m in _NEG_SPAN.finditer(text)]
    spans += _json_ranges(text)
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ── prose compression ─────────────────────────────────────────────────────────
def _compress_prose(seg: str, level: int) -> str:
    out = seg
    for pat, rep, _label in _p._LEXICAL_PHRASE_MAP:
        out = pat.sub(rep, out)
    for pat in _p._FILLER_PATTERNS:
        out = pat.sub(" ", out)
    if level >= 2:
        out = _p._HEDGING_ADVERBS_RE.sub("", out)
    if level >= 3 and len(out.split()) > 25:
        out = _p._AGGRESSIVE_ARTICLES_RE.sub("", out)
    out = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", out)       # collapse inner runs only: never indentation
    out = re.sub(r"[ \t]+([,;.!?])", r"\1", out)             # " ," left behind by a removed filler
    lead, trail = seg[:len(seg) - len(seg.lstrip(" \t"))], seg[len(seg.rstrip(" \t")):]
    core = out.strip(" \t")
    if not core:
        return lead + trail if (lead or trail) else ""
    return lead + core + trail


def _compress_segments(text: str, level: int) -> str:
    parts: list[str] = []
    pos = 0
    for s, e in protected_ranges(text):
        if s > pos:
            parts.append(_compress_prose(text[pos:s], level))
        parts.append(text[s:e])
        pos = e
    if pos < len(text):
        parts.append(_compress_prose(text[pos:], level))
    out = "".join(parts)
    if text[:1].isupper() and out[:1].islower():              # a removed opener must not lowercase the sentence
        out = out[:1].upper() + out[1:]
    return out


def verify(original: str, compressed: str) -> list[str]:
    """Protected spans and digit-bearing tokens that did not survive verbatim (empty = safe)."""
    missing: list[str] = []
    for span, n in Counter(original[s:e] for s, e in protected_ranges(original)).items():
        if compressed.count(span) < n:
            missing.append(span)
    for tok, n in Counter(_DIGIT_TOKEN.findall(original)).items():
        if compressed.count(tok) < n:
            missing.append(tok)
    return missing


def gate_text(text: str, level: int) -> tuple[str, bool]:
    """(text, ok): compressed text, or the ORIGINAL with ok=False if verification failed."""
    if not text or not text.strip():
        return text, True
    out = _compress_segments(text, level)
    if out != text and verify(text, out):
        return text, False
    return out, True


# ── structured requests ───────────────────────────────────────────────────────
@dataclass
class GateResult:
    body: dict
    raw_tokens: int
    fwd_tokens: int
    by_category: dict[str, tuple[int, int]] = field(default_factory=dict)
    fallbacks: int = 0
    protected_ok: bool = True
    ms: float = 0.0
    applied: bool = False
    error: str = ""

    @property
    def saved(self) -> int:
        return self.raw_tokens - self.fwd_tokens


def _tok(obj: Any) -> int:
    return estimate_tokens(obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def _walk_descriptions(node: Any, level: int, state: dict) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "description" and isinstance(v, str):
                new, ok = gate_text(v, level)
                state["fallbacks"] += 0 if ok else 1
                out[k] = new
            else:
                out[k] = _walk_descriptions(v, level, state)
        return out
    if isinstance(node, list):
        return [_walk_descriptions(x, level, state) for x in node]
    return node


def _text_parts(content: Any):
    """Yield (container, key) pairs for every text string of an OpenAI-style message content."""
    if isinstance(content, str):
        return
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                yield part, "text"


def gate_openai_request(body: dict, profile: Profile | str = "strict") -> GateResult:
    """Transform an OpenAI-compatible chat request. Never raises: on any error the original body is returned."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    t0 = time.perf_counter()
    try:
        out = copy.deepcopy(body)
        state = {"fallbacks": 0}
        cats = {"user_text": [0, 0], "tool_schemas": [0, 0], "duplicates": [0, 0]}
        seen: dict[str, int] = {}
        for i, msg in enumerate(out.get("messages") or []):
            if not isinstance(msg, dict):
                continue
            role, content = msg.get("role"), msg.get("content")
            if role == "user":
                if isinstance(content, str):
                    new, ok = gate_text(content, prof.prose_level)
                    cats["user_text"][0] += _tok(content)
                    cats["user_text"][1] += _tok(new)
                    state["fallbacks"] += 0 if ok else 1
                    msg["content"] = new
                else:
                    for part, key in _text_parts(content):
                        new, ok = gate_text(part[key], prof.prose_level)
                        cats["user_text"][0] += _tok(part[key])
                        cats["user_text"][1] += _tok(new)
                        state["fallbacks"] += 0 if ok else 1
                        part[key] = new
            if prof.dedupe and isinstance(msg.get("content"), str):  # causal: depends only on messages 0..i
                text = msg["content"]
                tk = _tok(text)
                if tk >= DEDUPE_MIN_TOKENS:
                    h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
                    if h in seen:
                        ref = f"[identical to message #{seen[h]}]"
                        cats["duplicates"][0] += tk
                        cats["duplicates"][1] += _tok(ref)
                        msg["content"] = ref
                    else:
                        seen[h] = i
        if prof.schemas and isinstance(out.get("tools"), list):
            before = _tok(out["tools"])
            out["tools"] = _walk_descriptions(out["tools"], prof.prose_level, state)
            cats["tool_schemas"] = [before, _tok(out["tools"])]
        raw_t, fwd_t = _tok({"messages": body.get("messages"), "tools": body.get("tools")}), \
            _tok({"messages": out.get("messages"), "tools": out.get("tools")})
        if fwd_t >= raw_t:  # never forward something larger than what we received
            return GateResult(body, raw_t, raw_t, {k: (a, a) for k, (a, _) in cats.items()}, state["fallbacks"],
                              state["fallbacks"] == 0, (time.perf_counter() - t0) * 1000, False)
        return GateResult(out, raw_t, fwd_t, {k: (a, b) for k, (a, b) in cats.items()}, state["fallbacks"],
                          state["fallbacks"] == 0, (time.perf_counter() - t0) * 1000, True)
    except Exception as exc:  # fail open
        t = _tok({"messages": body.get("messages"), "tools": body.get("tools")}) if isinstance(body, dict) else 0
        return GateResult(body, t, t, {}, 0, True, (time.perf_counter() - t0) * 1000, False, error=f"{type(exc).__name__}: {exc}")


# ── Anthropic /v1/messages ────────────────────────────────────────────────────
# Real Claude Code requests (checked against recorded request blobs): user text blocks are mostly <system-reminder>
# wrappers (instruction text: never rewritten), assistant `thinking` blocks carry signatures (changing one character
# invalidates them), `tool_use` blocks hold exact commands. So only the user's own plain-text blocks and tool-schema
# descriptions are ever touched; system, thinking, tool_use and (in strict) tool_result are forwarded verbatim.
_NEVER_TOUCH = ("thinking", "redacted_thinking", "tool_use", "server_tool_use", "image", "document")


def _is_prose(text: str) -> bool:
    s = text.lstrip()
    return bool(s) and not s.startswith("<") and "<system-reminder>" not in text


def _maybe_backref(text: str, idx: int, seen: dict[str, int], cat: list[int]) -> str | None:
    tk = _tok(text)
    if tk < DEDUPE_MIN_TOKENS:
        return None
    h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    if h in seen:
        ref = f"[identical to message #{seen[h]}]"
        cat[0] += tk
        cat[1] += _tok(ref)
        return ref
    seen[h] = idx
    return None


def _anthropic_payload(b: dict) -> dict:
    return {"system": b.get("system"), "messages": b.get("messages"), "tools": b.get("tools")}


def gate_anthropic_request(body: dict, profile: Profile | str = "strict") -> GateResult:
    """Transform an Anthropic Messages request. Never raises: on any error the original body is returned."""
    prof = PROFILES[profile] if isinstance(profile, str) else profile
    t0 = time.perf_counter()
    try:
        out = copy.deepcopy(body)
        state = {"fallbacks": 0}
        cats = {"user_text": [0, 0], "tool_schemas": [0, 0], "duplicates": [0, 0]}
        seen: dict[str, int] = {}
        for i, msg in enumerate(out.get("messages") or []):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if msg.get("role") == "user":
                if isinstance(content, str):
                    if _is_prose(content):
                        new, ok = gate_text(content, prof.prose_level)
                        cats["user_text"][0] += _tok(content)
                        cats["user_text"][1] += _tok(new)
                        state["fallbacks"] += 0 if ok else 1
                        msg["content"] = new
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str) \
                                and _is_prose(block["text"]):
                            new, ok = gate_text(block["text"], prof.prose_level)
                            cats["user_text"][0] += _tok(block["text"])
                            cats["user_text"][1] += _tok(new)
                            state["fallbacks"] += 0 if ok else 1
                            block["text"] = new
            if prof.dedupe:  # causal exact-duplicate collapse; same rules as the OpenAI path
                content = msg.get("content")
                if isinstance(content, str):
                    ref = _maybe_backref(content, i, seen, cats["duplicates"])
                    if ref:
                        msg["content"] = ref
                elif isinstance(content, list):
                    for block in content:
                        if not (isinstance(block, dict) and block.get("type") == "tool_result") or block.get("type") in _NEVER_TOUCH:
                            continue
                        inner = block.get("content")
                        text = inner if isinstance(inner, str) else (
                            "\n".join(x["text"] for x in inner) if isinstance(inner, list) and inner and all(
                                isinstance(x, dict) and x.get("type") == "text" and isinstance(x.get("text"), str) for x in inner) else None)
                        if text:
                            ref = _maybe_backref(text, i, seen, cats["duplicates"])
                            if ref:
                                block["content"] = ref
        if prof.schemas and isinstance(out.get("tools"), list):
            before = _tok(out["tools"])
            out["tools"] = _walk_descriptions(out["tools"], prof.prose_level, state)  # also reaches input_schema descriptions
            cats["tool_schemas"] = [before, _tok(out["tools"])]
        raw_t, fwd_t = _tok(_anthropic_payload(body)), _tok(_anthropic_payload(out))
        if fwd_t >= raw_t:
            return GateResult(body, raw_t, raw_t, {k: (a, a) for k, (a, _) in cats.items()}, state["fallbacks"],
                              state["fallbacks"] == 0, (time.perf_counter() - t0) * 1000, False)
        return GateResult(out, raw_t, fwd_t, {k: (a, b) for k, (a, b) in cats.items()}, state["fallbacks"],
                          state["fallbacks"] == 0, (time.perf_counter() - t0) * 1000, True)
    except Exception as exc:  # fail open
        t = _tok(_anthropic_payload(body)) if isinstance(body, dict) else 0
        return GateResult(body, t, t, {}, 0, True, (time.perf_counter() - t0) * 1000, False, error=f"{type(exc).__name__}: {exc}")


def detect_api(body: Any) -> str:
    """'anthropic' or 'openai', from the request shape (used by offline replay; the proxy knows the route)."""
    if not isinstance(body, dict):
        return "openai"
    msgs = body.get("messages") if isinstance(body.get("messages"), list) else []
    if "system" in body and not any(isinstance(m, dict) and m.get("role") == "system" for m in msgs):
        return "anthropic"
    tools = body.get("tools")
    if isinstance(tools, list) and tools and isinstance(tools[0], dict) and "input_schema" in tools[0]:
        return "anthropic"
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result", "thinking") for b in c):
            return "anthropic"
    return "openai"


def gate_request(body: dict, profile: Profile | str = "strict", api: str | None = None) -> GateResult:
    return (gate_anthropic_request if (api or detect_api(body)) == "anthropic" else gate_openai_request)(body, profile)
