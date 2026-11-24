import json

import pytest

from contextgate_core import gate
from contextgate_core import gate_bench as gb
from contextgate_core import cli as smarty_cli

FILLER = "Could you please help me, and I was wondering if you could basically fix this in order to be safe? "


def req(n=1):
    return {"model": "m", "messages": [{"role": "user", "content": FILLER + "Keep get_user_by_id verbatim. " * n}],
            "tools": [{"type": "function", "function": {"name": "f", "description": "Could you please use this tool in order to fetch it."}}]}


def test_load_formats(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text("\n".join([json.dumps(req()), json.dumps({"ts": 1, "body": req(2)}), ""]))
    assert [len(r["messages"]) for r in gb.load_requests(str(a))] == [1, 1]
    b = tmp_path / "b.json"
    b.write_text(json.dumps([req(), req()]))
    assert len(gb.load_requests(str(b))) == 2
    (tmp_path / "e.jsonl").write_text("")
    assert gb.load_requests(str(tmp_path / "e.jsonl")) == []


def test_run_measures_savings_by_category_and_safety():
    r = gb.run([req(1), req(3)], "strict")
    assert r["requests"] == 2 and r["applied"] == 2 and r["saved_tokens"] > 0 and r["verification_fallbacks"] == 0
    assert r["categories"]["user_text"]["saved"] > 0 and r["categories"]["tool_schemas"]["saved"] > 0
    assert r["categories"]["duplicates"]["saved"] == 0  # strict never collapses duplicates
    assert r["saved_pct"] == round(100 * r["saved_tokens"] / r["raw_tokens"], 2)
    assert r["gate_ms_p95"] < 100 and r["errors"] == 0


def test_profiles_are_ordered_by_aggressiveness_on_duplicated_content():
    dup = "tool output line " * 100
    body = {"messages": [{"role": "tool", "tool_call_id": "a", "content": dup}, {"role": "tool", "tool_call_id": "b", "content": dup}]}
    s, b = gb.run([body], "strict"), gb.run([body], "balanced")
    assert s["saved_tokens"] == 0 and b["saved_tokens"] > 0 and b["categories"]["duplicates"]["saved"] > 0


def test_synthetic_from_docs_is_deterministic_and_labelled(tmp_path, capsys):
    (tmp_path / "doc.md").write_text(("The service retries failed calls in order to stay available. " * 8 + "\n\n") * 6)
    a, b = gb.synthetic_from_docs(str(tmp_path), n=3), gb.synthetic_from_docs(str(tmp_path), n=3)
    assert a == b and len(a) == 3 and a[0]["messages"][1]["content"].count("[doc ") == 6
    assert gb.synthetic_from_docs(str(tmp_path / "nothing")) == []
    assert gb.main(["--from-docs", str(tmp_path), "--n", "3"]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out


def test_cli_and_error_paths(tmp_path, capsys):
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps(req()) + "\n")
    out = tmp_path / "o.json"
    assert gb.main([str(f), "--profiles", "strict", "balanced", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "strict" in text and "balanced" in text and "re-run on your own recorded traffic" in text
    assert [r["profile"] for r in json.load(open(out))] == ["strict", "balanced"]
    assert gb.main([]) == 2 and gb.main([str(tmp_path / "missing.jsonl")]) == 2
    (tmp_path / "e.jsonl").write_text("")
    assert gb.main([str(tmp_path / "e.jsonl")]) == 2


def test_smarty_cli_wires_bench_stats_and_serve(tmp_path, capsys):
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps(req()) + "\n")
    assert smarty_cli.main(["bench", str(f), "--profiles", "strict"]) == 0 and "strict" in capsys.readouterr().out
    assert smarty_cli.main(["stats", "--ledger", str(tmp_path / "none.db")]) == 2
    with pytest.raises(SystemExit) as e:
        smarty_cli.main(["serve", "--help"])
    assert e.value.code == 0 and "upstream" in capsys.readouterr().out


def test_bench_replays_both_api_shapes_and_reports_the_mix(tmp_path, capsys):
    anthropic = {"model": "c", "max_tokens": 9, "system": "s", "messages": [
        {"role": "user", "content": [{"type": "text", "text": FILLER + "Keep get_user_by_id verbatim."}]}],
        "tools": [{"name": "f", "description": "Could you please use this tool in order to fetch it.", "input_schema": {"type": "object"}}]}
    f = tmp_path / "mixed.jsonl"
    f.write_text(json.dumps(req()) + "\n" + json.dumps({"ts": 1, "body": anthropic}) + "\n")
    r = gb.run(gb.load_requests(str(f)), "strict")
    assert r["apis"] == {"openai": 1, "anthropic": 1} and r["applied"] == 2 and r["saved_tokens"] > 0
    assert gb.main([str(f), "--profiles", "strict"]) == 0
    assert "1 anthropic, 1 openai" in capsys.readouterr().out


def test_synthetic_anthropic_requests_have_the_real_shape(tmp_path):
    (tmp_path / "doc.md").write_text(("The service retries failed calls in order to stay available. " * 8 + "\n\n") * 6)
    reqs = gb.synthetic_from_docs(str(tmp_path), n=2, api="anthropic")
    assert gate.detect_api(reqs[0]) == "anthropic" and reqs[0]["tools"][0]["input_schema"]["type"] == "object"
    assert isinstance(reqs[0]["system"], list) and reqs[0]["messages"][0]["content"][0]["type"] == "text"
    assert gb.main(["--from-docs", str(tmp_path), "--n", "2", "--api", "anthropic", "--profiles", "strict"]) == 0
