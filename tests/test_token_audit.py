import json

import pytest

from contextgate_core import token_audit as ta

W = ta.DEFAULT_WEIGHTS
WORDS = "word " * 600  # ~600+ estimated tokens


def _line(**kw):
    return json.dumps(kw)


def _usage(inp, cw, cr, out):
    return {"input_tokens": inp, "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr, "output_tokens": out}


def _asst(mid, usage, tool=None, tid=None):
    block = [{"type": "tool_use", "id": tid, "name": tool, "input": {}}] if tool else [{"type": "text", "text": "x"}]
    return _line(type="assistant", message={"id": mid, "usage": usage, "content": block})


def _result(tid, text):
    return _line(type="user", message={"content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]})


@pytest.fixture
def session(tmp_path):
    p = tmp_path / "s.jsonl"
    lines = [
        _line(type="user", message={"content": "fix the bug"}),
        _line(type="attachment", attachment={"type": "instructions", "files": [{"path": "/home/AGENTS.md", "content": WORDS}]}),
        _line(type="attachment", attachment={"type": "skill_listing", "content": WORDS * 2}),
        _line(type="attachment", attachment={"type": "prompt_snapshot", "systemPrompt": ["sys " * 500]}),
        _asst("a1", _usage(10, 1000, 4000, 100), "Grep", "t1"),
        _result("t1", "hit " * 200),
        _asst("a2", _usage(10, 200, 5000, 100), "Edit", "t2"),
        _result("t2", "ok"),
        _asst("a3", _usage(10, 50, 5200, 300)),
    ]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_decomposition_adds_up_and_prices_every_term(session):
    r = ta.analyze_session(session)
    assert r["calls"] == 3 and r["prefix_tokens"] == 10 + 1000 + 4000
    cost_in = sum(i + 1.25 * cw + 0.1 * cr for i, cw, cr in ((10, 1000, 4000), (10, 200, 5000), (10, 50, 5200)))
    assert r["cost_total"] == pytest.approx(cost_in + 5.0 * 500)
    assert r["cost_fixed_prefix"] + r["cost_growth"] + r["cost_output"] == pytest.approx(r["cost_total"])
    assert r["cost_output"] == pytest.approx(2500)
    assert r["turn_cost"] == pytest.approx(0.1 * 5010)
    assert r["unit_cost_per_prefix_token"] == pytest.approx(1.25 + 0.1 * 2)


def test_attribution_separates_host_user_layer_and_prompt(session):
    r = ta.analyze_session(session)
    assert "prompt_snapshot" not in r["layer_tokens"] and r["host_system_prompt_tokens"] > 400  # host copy is not controllable
    assert set(r["layer_tokens"]) == {"instructions", "skill_listing"}
    assert r["instruction_files"] == {"/home/AGENTS.md": r["layer_tokens"]["instructions"]}
    assert r["layer_tokens"]["skill_listing"] > r["layer_tokens"]["instructions"]
    assert 0 < r["user_prompt_tokens"] < 10
    assert r["tool_result_tokens"] > 150 and r["tool_result_cost"] > 0


def test_single_call_sessions_are_skipped(tmp_path):
    p = tmp_path / "one.jsonl"
    p.write_text(_asst("a1", _usage(1, 1, 1, 1)) + "\n")
    assert ta.analyze_session(str(p)) is None


def test_cold_cache_costs_more_and_recommendations_follow_the_model(session):
    rows = [ta.analyze_session(session)]
    warm, cold = ta.aggregate(rows), ta.aggregate(rows, cold=True)
    assert cold["mean_cost_total"] > warm["mean_cost_total"]
    recs = ta.recommendations(warm, reduction=0.5)
    by = {r["source"]: r for r in recs}
    assert "skill listing" in by and any(k.startswith("instructions: /home/AGENTS.md") for k in by)
    s = by["skill listing"]
    expected = 100 * s["tokens"] * 0.5 * warm["unit_cost"] / warm["mean_cost_total"]
    assert s["projected_saving_pct"] == pytest.approx(round(expected, 1), abs=0.11)
    assert recs == sorted(recs, key=lambda r: -r["per_session_cost"])  # biggest first
    assert all(r["tokens"] >= 300 for r in recs)  # tiny sources are not worth a recommendation


def test_required_pairs_matches_the_paired_t_formula_and_is_monotonic():
    assert ta.required_pairs(15, 20) == 5  # ceil((2.8 * 15 / 20) ** 2) = ceil(4.41)
    assert ta.required_pairs(15, 10) > ta.required_pairs(15, 20) > ta.required_pairs(15, 40)
    assert ta.required_pairs(30, 20) > ta.required_pairs(15, 20)


def test_cli(session, tmp_path, capsys):
    out = tmp_path / "o.json"
    assert ta.main([session, "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "where the cost goes" in text and "one avoided API call saves" in text and "recommendations" in text
    assert "warm" in json.load(open(out))
    assert ta.main(["--power", "15", "20"]) == 0 and "needed" in capsys.readouterr().out
    assert ta.main([str(tmp_path / "missing.jsonl")]) == 2
    assert ta.main([]) == 2


def test_smarty_cli_audit_subcommand(session, capsys):
    from contextgate_core import cli as smarty_cli

    assert smarty_cli.main(["audit", session]) == 0
    assert "where the cost goes" in capsys.readouterr().out


def test_smarty_audit_accepts_leading_options(session, capsys):
    from contextgate_core import cli as smarty_cli

    assert smarty_cli.main(["audit", "--w-out", "3", session]) == 0  # an option first used to be rejected by argparse
    assert "output x3.0" in capsys.readouterr().out  # and the option actually took effect
