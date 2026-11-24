"""The documented library surface. If one of these tests fails you are changing the public API: do it on purpose."""
import doctest
import inspect
import os
import re

import pytest

import contextgate_core as cg
from contextgate_core import gate as gate_mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PUBLIC = {"__version__", "gate_text", "gate_request", "GateResult", "PROFILES", "optimize", "estimate_tokens",
          "audit_agent_response", "RawRequest", "TokenBudget", "CompiledPrompt", "OptimizationTrace"}
SIGNATURES = {  # name -> [(parameter, default)]
    "gate_text": [("text", inspect.Parameter.empty), ("level", 1)],
    "gate_request": [("body", inspect.Parameter.empty), ("profile", "strict"), ("api", None)],
    "optimize": [("request", inspect.Parameter.empty), ("few_shot_k", 2)],
    "estimate_tokens": [("text", inspect.Parameter.empty)],
    "audit_agent_response": [("response_text", inspect.Parameter.empty), ("trace", inspect.Parameter.empty)],
}
GATE_RESULT_FIELDS = ["body", "raw_tokens", "fwd_tokens", "by_category", "fallbacks", "protected_ok", "ms", "applied", "error"]


def test_the_public_names_are_exactly_these():
    assert set(cg.__all__) == PUBLIC
    for name in PUBLIC:
        assert getattr(cg, name) is not None


def test_public_signatures_are_stable():
    for name, params in SIGNATURES.items():
        got = [(p.name, p.default) for p in inspect.signature(getattr(cg, name)).parameters.values()]
        assert got == params, name


def test_gate_result_fields_and_derived_saved():
    assert [f for f in cg.GateResult.__dataclass_fields__] == GATE_RESULT_FIELDS
    r = cg.GateResult(body={}, raw_tokens=10, fwd_tokens=7)
    assert r.saved == 3


def test_every_public_callable_is_documented():
    for name in ("gate_text", "gate_request", "optimize", "estimate_tokens", "audit_agent_response"):
        assert len((getattr(cg, name).__doc__ or "").strip()) > 40, name


def test_package_is_typed_and_ships_py_typed():
    assert os.path.exists(os.path.join(ROOT, "contextgate_core", "py.typed"))
    assert re.search(r'package-data\]\s*\ncontextgate_core = \["py.typed"\]', open(os.path.join(ROOT, "pyproject.toml")).read())


def test_gate_text_docstring_example_runs():
    assert doctest.run_docstring_examples(gate_mod.gate_text, vars(gate_mod), name="gate_text") is None
    result = doctest.DocTestRunner(verbose=False)
    for t in doctest.DocTestFinder().find(gate_mod.gate_text, "gate_text", globs=dict(vars(gate_mod))):
        result.run(t)
    assert result.failures == 0 and result.tries >= 1


def test_gate_text_levels_and_errors():
    s = "Could you please basically fix this in order to be safe."
    assert cg.gate_text(s, 1) == cg.gate_text(s, "strict") and cg.gate_text(s, 2) == cg.gate_text(s, "balanced")
    assert cg.gate_text(s, 3) == cg.gate_text(s, "aggressive") and cg.gate_text(s) == cg.gate_text(s, 1)
    for bad in (0, 4, -1, True, "nope", 2.0):
        with pytest.raises(ValueError):
            cg.gate_text(s, bad)
    assert cg.gate_text("", 1) == ("", True) and cg.gate_text("   ", 2) == ("   ", True)


@pytest.mark.parametrize("text", ["", "ok", "Please do not use `rm -rf /` ever.", "{" * 50_000, "a b c " * 2000,
                                  "Could you please basically fix this in order to be safe. " * 50])
def test_gate_text_never_returns_longer_text_and_failures_return_the_original(text):
    out, ok = cg.gate_text(text, "aggressive")
    assert len(out) <= len(text)
    if not ok:
        assert out == text


@pytest.mark.parametrize("body", [{}, {"messages": None}, {"messages": [1, 2]}, {"messages": [{"role": "user"}]},
                                  {"model": 1, "messages": [{"role": "user", "content": None}]},
                                  {"messages": [{"role": "user", "content": [{"type": "text", "text": 5}]}]},
                                  {"tools": "x", "messages": []}, {"system": 3, "messages": [{"role": "user", "content": "hi"}]}])
def test_gate_request_never_raises_and_never_forwards_a_larger_request(body):
    before = repr(body)
    r = cg.gate_request(body)
    assert isinstance(r, cg.GateResult) and r.saved >= 0
    assert repr(body) == before  # the caller's dict is never mutated


def test_the_readme_example_runs_exactly_as_documented(capsys):
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    block = re.search(r"## Use as a library\n\n```python\n(.*?)```", readme, re.S).group(1)
    exec(compile(block, "README.md", "exec"), {})  # asserts inside the example must hold
    assert capsys.readouterr().out.strip()


def test_import_stays_light():
    import subprocess
    import sys

    code = "import sys, contextgate_core; print([m for m in ('contextgate_core.pipeline','contextgate_core.gate','spacy') if m in sys.modules])"
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True).stdout.strip() == "[]"
