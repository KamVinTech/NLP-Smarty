import pytest

from contextgate_core.data_contracts import RawRequest
from contextgate_core.pipeline import PROMPT_CACHE, optimize
from test_harness import NOISY_PROMPTS, REGRESSION_SUITE


@pytest.fixture(autouse=True, scope="module")  # regression cases 10→11 share cache state
def _clear_cache():
    PROMPT_CACHE.clear()
    for p in NOISY_PROMPTS:  # mirrors harness order: case 11 expects a cache hit on prompt 1
        optimize(RawRequest(text=p))


@pytest.mark.parametrize("case", REGRESSION_SUITE, ids=[c["name"] for c in REGRESSION_SUITE])
def test_regression_suite(case):
    assert case["assert_fn"](optimize(case["request"]))


@pytest.mark.parametrize("prompt", NOISY_PROMPTS)
def test_noisy_prompts_never_grow(prompt):
    t = optimize(RawRequest(text=prompt)).trace
    assert t.optimized_token_count <= t.raw_token_count


@pytest.mark.parametrize("neg", ["without using slicing", "do not use recursion", "never modify the input"])
def test_negation_survives_in_payload(neg):
    res = optimize(RawRequest(text=f"Could you please write a Python function to reverse a string, {neg}?"))
    assert neg.split()[0] in res.payload.lower()
    assert res.trace.negations


def test_url_preserved_verbatim():
    url = "https://example.com/api/v2/items?id=42"
    res = optimize(RawRequest(text=f"Please fetch {url} and summarize the response"))
    assert url in res.payload


def test_deterministic():
    r = RawRequest(text="Please help me refactor this function to be faster", enable_cache=False)
    assert optimize(r).payload == optimize(r).payload
