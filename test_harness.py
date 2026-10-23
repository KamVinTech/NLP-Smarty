"""
Verification, Regression & Benchmarking Harness (Step 9)
Run: python test_harness.py
"""

from data_contracts import RawRequest, TokenBudget
from pipeline import PROMPT_CACHE, optimize

# ── Original 5 Baseline Prompts (100% Backward-Compatible) ─────────────────────
NOISY_PROMPTS = [
    "Could you please help me write a Python function to reverse a string without using slicing?",
    "I was wondering if you could search the web for the latest Python 3.12 release notes.",
    "I'd love it if you could extract all email addresses from this text: contact us at hello@example.com or support@test.org",
    "Can you please just quickly generate a SQL query to select all users created after 2023-01-01 from the users table?",
    "Tell me what is the capital of France?",
]

# ── Extended V3 Regression Suite (Validating Enhancements 1 through 9) ─────────
REGRESSION_SUITE = [
    {
        "name": "6. Lexical Normalization & Adaptive MEDIUM Intensity",
        "request": RawRequest(
            text=(
                "Hey there! Could you please help me in order to write a Python function "
                "due to the fact that we need to make use of binary search at this point in time "
                "without using recursion?"
            ),
        ),
        "assert_fn": lambda r: (
            r.trace.detected_intent == "CODE_GENERATION"
            and len(r.trace.normalized_phrases) >= 3
            and "without" in r.trace.negations
        ),
    },
    {
        "name": "7. Extractive Sentence Compression & AGGRESSIVE Intensity (Token-Budgeted)",
        "request": RawRequest(
            text=(
                "I was wondering if you could help me write a Python script to parse server logs. "
                "The weather today in Hyderabad is quite sunny and we had coffee this morning. "
                "Our team usually meets on Tuesdays to discuss general office supplies and parking. "
                "Yesterday we talked about painting the hallway blue or green. "
                "You must strictly format the output as JSON and never include debug stack traces."
            ),
            token_budget=TokenBudget(max_total_tokens=450, core_query_budget=28),
            compression_intensity="AGGRESSIVE",
        ),
        "assert_fn": lambda r: (
            r.trace.compression_intensity_used == "AGGRESSIVE"
            and "never" in r.trace.negations
            and "JSON" in r.payload
            and r.trace.optimized_token_count <= 32
        ),
    },
    {
        "name": "8. MMR External Context Pruning (Deduplicating Redundant RAG Chunks)",
        "request": RawRequest(
            text="What is the rate limit and authentication header for the Payments API?",
            context_chunks=[
                "Payments API v2 uses Bearer token authentication via the X-Pay-Token header and enforces a strict rate limit of 100 requests per minute.",
                "Payments API v2 requires the X-Pay-Token Bearer header for authentication and limits callers to 100 requests per minute.",  # near-duplicate of chunk 0
                "The Legacy Billing export job runs every Sunday at 02:00 UTC and writes CSV archives to cold storage.",  # irrelevant chunk
                "If a client exceeds the 100 req/min rate limit on Payments API, the gateway returns HTTP 429 with a Retry-After header.",  # complementary relevant chunk
            ],
            token_budget=TokenBudget(context_budget=65),
        ),
        "assert_fn": lambda r: (
            # MMR must pick exactly ONE of the near-duplicate chunks {0, 1}, keep complementary chunk 3, and drop irrelevant chunk 2
            ((0 in r.trace.kept_context_indices) ^ (1 in r.trace.kept_context_indices))
            and (3 in r.trace.kept_context_indices)
            and (2 not in r.trace.kept_context_indices)
        ),
    },
    {
        "name": "9. Conversation History Pruning (Recency-Weighted MMR)",
        "request": RawRequest(
            text="Can you please update that SQL query to also filter out inactive accounts?",
            conversation_history=[
                {"role": "user", "content": "Could you please write a SQL query to select users created after 2023-01-01?"},
                {"role": "assistant", "content": "SELECT * FROM users WHERE created_at > '2023-01-01';"},
                {"role": "user", "content": "By the way, what is the capital of France?"},
                {"role": "assistant", "content": "Paris is the capital of France."},
                {"role": "user", "content": "Now we are working on the users table SQL query again."},
            ],
            token_budget=TokenBudget(history_budget=45),
        ),
        "assert_fn": lambda r: (
            r.trace.kept_history_turns >= 2
            and "<conversation_history>" in r.payload
        ),
    },
    {
        "name": "10. Tool Schema Compression & Morphological Stemming Match",
        "request": RawRequest(
            text="Please open and load the configuration settings from config.json",
        ),
        "assert_fn": lambda r: (
            "read_file" in r.trace.injected_tool_ids
            and r.trace.tool_schema_tokens_saved > 20
            and "read_file(path:str, encoding?:str=utf-8)" in r.payload
        ),
    },
    {
        "name": "11. Deterministic Canonical Cache Hit (Paraphrased Polite Prefix)",
        "request": RawRequest(
            # Paraphrased polite wrapper around Test 1 — normalizes to identical canonical query!
            text="Hey! I would really appreciate it if you could please write a Python function to reverse a string without using slicing?",
        ),
        "assert_fn": lambda r: (
            r.trace.cache_hit is True
            and r.trace.cache_type in ("CANONICAL_EXACT", "SIMHASH_NEAR")
            and r.trace.latency_ms < 1.5
        ),
    },
    {
        "name": "12. Missing Mandatory Slot Early Short-Circuit (Zero Downstream Tokens)",
        "request": RawRequest(
            text="Just quickly the latest release notes.",
            task_hint="TOOL_EXECUTION",
        ),
        "assert_fn": lambda r: (
            r.status == "NEEDS_CLARIFICATION"
            and r.payload == ""
        ),
    },
]

SEP = "─" * 78


def _token_delta(raw: int, opt: int) -> str:
    saved = raw - opt
    pct = (saved / raw * 100) if raw else 0.0
    return f"{raw} -> {opt} tokens  (saved {saved}, {pct:.1f}%)"


def main():
    PROMPT_CACHE.clear()
    print("=" * 78)
    print(" PART A: ORIGINAL 5 BENCHMARK PROMPTS (BEFORE vs. AFTER TELEMETRY)")
    print("=" * 78)

    for i, prompt in enumerate(NOISY_PROMPTS, 1):
        result = optimize(RawRequest(text=prompt))
        t = result.trace

        print(f"\n{SEP}")
        print(f"TEST {i}")
        print(f"{SEP}")
        print(f"RAW INPUT        :\n  {prompt}")
        print(f"\nSTATUS           : {result.status}")
        print(f"INTENT           : {t.detected_intent}  (confidence={t.intent_confidence}, intensity={t.compression_intensity_used})")
        print(f"QUERY TOKENS     : {_token_delta(t.raw_token_count, t.optimized_token_count)}")
        print(f"PAYLOAD TOKENS   : {_token_delta(t.unoptimized_payload_token_count, t.compiled_payload_token_count)}")
        print(f"SCHEMA SAVED     : {t.tool_schema_tokens_saved} tokens")
        print(f"STRIPPED         : {t.stripped_tokens or 'none'}")
        print(f"NEGATIONS        : {t.negations or 'none'}")
        print(f"ENTITIES         : {t.extracted_entities}")
        print(f"TOOLS            : {t.injected_tool_ids or 'none'}")
        print(f"EXAMPLES         : {t.injected_example_ids}")
        print(f"CACHE            : hit={t.cache_hit} ({t.cache_type}) key={t.cache_key}")
        print(f"LATENCY          : {t.latency_ms} ms")

    print("\n\n" + "=" * 78)
    print(" PART B: EXTENDED V3 REGRESSION SUITE (FEATURES 1 -> 2 -> 4 -> 3 -> 6 -> 9)")
    print("=" * 78)

    passed = 0
    for case in REGRESSION_SUITE:
        res = optimize(case["request"])
        tr = res.trace
        ok = bool(case["assert_fn"](res))
        passed += int(ok)
        verdict = "PASS" if ok else "FAIL"

        print(f"\n[{verdict}] {case['name']}")
        print(f"  Status / Intent : {res.status} | {tr.detected_intent} (intensity={tr.compression_intensity_used})")
        print(f"  Query Tokens    : {_token_delta(tr.raw_token_count, tr.optimized_token_count)}")
        print(f"  Payload Tokens  : {_token_delta(tr.unoptimized_payload_token_count, tr.compiled_payload_token_count)}")
        if tr.normalized_phrases:
            print(f"  Normalized      : {tr.normalized_phrases}")
        if tr.kept_context_indices:
            print(f"  Kept Context IDs: {tr.kept_context_indices}")
        if tr.kept_history_turns:
            print(f"  Kept History    : {tr.kept_history_turns} turns")
        if tr.tool_schema_tokens_saved:
            print(f"  Tool Compression: saved {tr.tool_schema_tokens_saved} schema tokens")
        if tr.cache_hit:
            print(f"  Cache Hit       : {tr.cache_type} (key={tr.cache_key}, simhash={tr.simhash_signature})")
        print(f"  Latency         : {tr.latency_ms} ms")

    print(f"\n{SEP}")
    print(f"REGRESSION SUMMARY: {passed}/{len(REGRESSION_SUITE)} extended checks passed.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()