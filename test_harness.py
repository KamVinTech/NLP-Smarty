"""
Verification & Benchmarking Harness
Run: python test_harness.py
"""

from data_contracts import RawRequest
from pipeline import optimize

NOISY_PROMPTS = [
    "Could you please help me write a Python function to reverse a string without using slicing?",
    "I was wondering if you could search the web for the latest Python 3.12 release notes.",
    "I'd love it if you could extract all email addresses from this text: contact us at hello@example.com or support@test.org",
    "Can you please just quickly generate a SQL query to select all users created after 2023-01-01 from the users table?",
    "Tell me what is the capital of France?",
]

SEP = "─" * 70

def _token_delta(raw: int, opt: int) -> str:
    saved = raw - opt
    pct = (saved / raw * 100) if raw else 0
    return f"{raw} → {opt} tokens  (saved {saved}, {pct:.1f}%)"

def main():
    for i, prompt in enumerate(NOISY_PROMPTS, 1):
        result = optimize(RawRequest(text=prompt))
        t = result.trace

        print(f"\n{SEP}")
        print(f"TEST {i}")
        print(f"{SEP}")
        print(f"RAW INPUT :\n  {prompt}")
        print(f"\nSTATUS    : {result.status}")

        if result.status == "NEEDS_CLARIFICATION":
            print(f"CLARIFY   : {result.clarification_request}")
            continue

        print(f"INTENT    : {t.detected_intent}  (confidence={t.intent_confidence})")
        print(f"TOKENS    : {_token_delta(t.raw_token_count, t.optimized_token_count)}")
        print(f"STRIPPED  : {t.stripped_tokens or 'none'}")
        print(f"NEGATIONS : {t.negations or 'none'}")
        print(f"ENTITIES  : {t.extracted_entities}")
        print(f"TOOLS     : {t.injected_tool_ids or 'none'}")
        print(f"EXAMPLES  : {t.injected_example_ids}")
        print(f"LATENCY   : {t.latency_ms} ms")
        print(f"\nCOMPILED PROMPT (first 600 chars):\n{result.payload[:600]}...")

    print(f"\n{SEP}\nAll tests complete.\n")

if __name__ == "__main__":
    main()
