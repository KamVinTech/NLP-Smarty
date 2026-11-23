import json
import os

import pytest

import ab_compare as ab
import repo_indexer as ri


# ── labels ────────────────────────────────────────────────────────────────────
def test_load_labels_both_formats(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps([{"q": "a b", "expect": ["x.py:f", "x.py:g", "y.py"]}]))
    code, silent = ab.load_labels(str(g))
    assert code == [{"q": "a b", "expect": ["x.py", "y.py"]}] and silent == []
    n = tmp_path / "n.json"
    n.write_text(json.dumps({"code": [{"q": "c", "expect": ["z.ts"]}], "silent": ["thanks"]}))
    assert ab.load_labels(str(n)) == ([{"q": "c", "expect": ["z.ts"]}], ["thanks"])


# ── offline simulation ────────────────────────────────────────────────────────
PAD = "\n".join(f"# amount card charge credit ledger line {i}" for i in range(300))


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pay.py").write_text("def charge_credit_card(amount):\n    return amount\n")
    (tmp_path / "ledger.py").write_text("def summarize():\n    return 1\n" + PAD)  # distractor: more keyword hits, big
    for i in range(6):
        (tmp_path / f"misc{i}.py").write_text(f"def helper_{i}():\n    return {i}\n")
    return str(tmp_path)


def test_hit_accounting(repo):
    s = ab.simulate(repo, [{"q": "charge credit card amount", "expect": ["pay.py"]}], [], 0.25, 450)
    r = s["rows"][0]
    assert r["outcome"] == "hit"
    assert r["baseline"] == r["explore"] + r["target_read"]
    assert r["explore"] > 0  # the model reads the (bigger, better-ranked) distractor first
    assert r["conservative"] == r["capsule"] + r["target_read"]  # still reads the whole target file
    assert r["optimistic"] <= r["conservative"] < r["baseline"]
    assert s["saving_pct_conservative"] > 0 and s["hit"] == 1


def test_miss_is_pure_overhead(repo):
    s = ab.simulate(repo, [{"q": "charge credit card amount", "expect": ["ledger.py"]}], [], 0.25, 450)
    r = s["rows"][0]
    assert r["outcome"] == "miss" and r["conservative"] == r["capsule"] + r["baseline"]
    assert s["saving_pct_conservative"] < 0 and s["tasks_where_smarty_costs_more"] == 1


def test_silent_gate_saves_nothing_and_costs_nothing(repo):
    s = ab.simulate(repo, [{"q": "refund amount", "expect": ["ledger.py"]}], [], 0.25, 450)
    r = s["rows"][0]
    assert r["outcome"] == "silent" and r["capsule"] == 0 and r["conservative"] == r["baseline"] == r["optimistic"]


def test_chatty_prompts_counted_as_waste_only_when_injected(repo):
    s = ab.simulate(repo, [{"q": "charge credit card amount", "expect": ["pay.py"]}],
                    ["thanks that makes sense", "charge credit card amount"], 0.25, 450)
    assert s["chatty_prompts"] == 2 and s["chatty_false_injections"] == 1 and s["chatty_wasted_tokens"] > 0


def test_tasks_with_nothing_to_grep_are_skipped_not_counted(repo):
    s = ab.simulate(repo, [{"q": "zzqq xxww", "expect": ["pay.py"]}], [], 0.25, 450)
    assert s["tasks"] == 0 and s["skipped_no_keywords"] == 1


def test_filename_discovery_makes_baseline_cheaper(tmp_path):
    for i in range(9):  # filler without the query words, so the words stay distinctive (< 50% of files)
        (tmp_path / f"filler{i}.py").write_text(f"def f{i}():\n    return {i}\n")
    for i in range(4):  # distractors that mention the words a lot, so a grep-only agent reads them first
        (tmp_path / f"noise{i}.py").write_text("# invoice generator numbers\n" * 200)
    (tmp_path / "invoice_generator.py").write_text("# invoice generator numbers\ndef make():\n    return 1\n")
    r = ab.RepoText(str(tmp_path))
    q = "invoice generator numbers"
    grep_only = ab.baseline_explore(r, q, ["invoice_generator.py"], path_discovery=False)
    with_glob = ab.baseline_explore(r, q, ["invoice_generator.py"], path_discovery=True)
    assert grep_only["reads"] > 1 and with_glob["reads"] == 1
    assert with_glob["explore"] < grep_only["explore"]


# ── transcripts ───────────────────────────────────────────────────────────────
def _line(**kw):
    return json.dumps(kw)


def _assistant(mid, tool=None, tid=None, inp=None, usage=None, ts="2026-01-01T00:00:00Z"):
    block = [{"type": "tool_use", "id": tid, "name": tool, "input": inp or {}}] if tool else [{"type": "text", "text": "x"}]
    return _line(type="assistant", timestamp=ts, message={
        "id": mid, "content": block,
        "usage": usage or {"input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
                           "output_tokens": 50}})


def _result(tid, text):
    return _line(type="user", timestamp="2026-01-01T00:01:00Z",
                 message={"content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]})


@pytest.fixture
def transcript(tmp_path):
    p = tmp_path / "s.jsonl"
    lines = [
        _line(type="session-owner", timestamp="1999-01-01T00:00:00Z"),  # must not stretch wall time
        _line(type="attachment", attachment={"type": "hook_additional_context",
                                             "content": "<retrieved_context>symbols: x</retrieved_context>"}),
        _assistant("m1", "Grep", "t1"), _assistant("m1", "Grep", "t1"),  # same message repeated per content block
        _result("t1", "word " * 400),
        _assistant("m2", "Bash", "t2", {"command": "git grep -n foo"}), _result("t2", "hit " * 100),
        _assistant("m3", "Bash", "t3", {"command": "npm test"}), _result("t3", "ok"),  # not exploration
        _assistant("m4", "mcp__nlp-smarty__smarty_find_symbol", "t4"), _result("t4", "{}"),
        _assistant("m5", "Edit", "t5"), _result("t5", "done"),
        _assistant("m6", "Read", "t6"), _result("t6", "after edit"),  # after first edit: not counted as pre-edit
        _assistant("m7", ts="2026-01-01T00:10:00Z"),
    ]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_transcript_dedupes_messages_and_sums_usage(transcript):
    a = ab.analyze_transcript(transcript)
    assert a["api_calls"] == 7  # m1 appears twice but counts once
    assert a["input_tokens"] == 70 and a["cache_write_tokens"] == 700 and a["cache_read_tokens"] == 7000
    assert a["output_tokens"] == 350
    assert a["weighted_input_equiv"] == round(70 + 1.25 * 700 + 0.1 * 7000)
    assert a["tool_calls"] == 6  # duplicated tool_use id counted once


def test_transcript_exploration_before_first_edit(transcript):
    a = ab.analyze_transcript(transcript)
    assert a["had_edit"] and a["tool_calls_before_first_edit"] == 4
    assert a["explore_calls_before_first_edit"] == 2  # Grep + `git grep`; `npm test` and the smarty call are not
    assert a["explore_result_tokens_before_first_edit"] > 0
    assert a["smarty_tool_calls"] == 1


def test_transcript_hook_injection_and_wall_time(transcript):
    a = ab.analyze_transcript(transcript)
    assert a["hook_injections"] == 1 and a["hook_injection_tokens"] > 0
    assert a["wall_seconds"] == 600.0  # 00:00 -> 00:10, ignoring the 1999 bookkeeping entry


def test_compare_and_cli(transcript, tmp_path, capsys):
    a = ab.analyze_transcript(transcript)
    b = dict(a, weighted_input_equiv=a["weighted_input_equiv"] // 2)
    row = next(r for r in ab.compare([a], [b]) if r["metric"] == "weighted_input_equiv")
    assert row["delta_pct"] == pytest.approx(-50.0, abs=0.1)
    assert ab.main(["transcripts", "--a", transcript, "--b", transcript]) == 0
    out = capsys.readouterr().out
    assert "weighted_input_equiv" in out and "fewer than 3 runs" in out


def test_simulate_cli_json(repo, tmp_path, capsys):
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps([{"q": "charge credit card amount", "expect": ["pay.py"]}]))
    out = tmp_path / "o.json"
    assert ab.main(["simulate", "--root", repo, "--labels", str(labels), "--baseline", "grep", "grep+path",
                    "--min-confidence", "0.25", "0.34", "--json", str(out)]) == 0
    data = json.load(open(out))
    assert len(data) == 4 and {d["baseline_model"] for d in data} == {"grep-only", "grep+filename-glob"}
    assert "MODEL ASSUMPTIONS" in capsys.readouterr().out


def test_inputs_accept_directories_and_report_missing_paths(tmp_path, transcript, capsys):
    d = tmp_path / "proj"
    d.mkdir()
    for i, name in enumerate(("old.jsonl", "new.jsonl")):
        (d / name).write_text(open(transcript).read())
        os.utime(d / name, (1000 + i * 100, 1000 + i * 100))
    assert [os.path.basename(p) for p in ab.expand_inputs([str(d)])] == ["new.jsonl", "old.jsonl"]  # newest first
    assert [os.path.basename(p) for p in ab.expand_inputs([str(d)], last=1)] == ["new.jsonl"]
    with pytest.raises(ab.InputError, match="placeholders"):
        ab.expand_inputs(["A1.jsonl"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ab.InputError, match="no .jsonl"):
        ab.expand_inputs([str(empty)])
    assert ab.main(["transcripts", "--a", "A1.jsonl", "--b", "B1.jsonl"]) == 2
    assert "error:" in capsys.readouterr().err


def test_contamination_and_missing_smarty_warnings(transcript, tmp_path, capsys):
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps({"type": "assistant", "timestamp": "2026-01-01T00:00:00Z",
                                 "message": {"id": "z", "usage": {"input_tokens": 1}, "content": []}}) + "\n")
    ab.main(["transcripts", "--a", transcript, "--b", str(clean)])  # A has smarty activity, B has none: both warnings
    out = capsys.readouterr().out
    assert "baseline is contaminated" in out and "may not have loaded" in out


def test_last_applies_across_all_directories(tmp_path, transcript):
    d1, d2 = tmp_path / "p1", tmp_path / "p1--claude-worktrees-x"
    for d, stamp in ((d1, 1000), (d2, 2000)):
        d.mkdir()
        for i in range(2):
            f = d / f"s{i}.jsonl"
            f.write_text(open(transcript).read())
            os.utime(f, (stamp + i, stamp + i))
    got = ab.expand_inputs([str(d1), str(d2)], last=3)
    assert len(got) == 3 and all(os.path.getmtime(g) >= 1001 for g in got)  # newest 3 of 4, regardless of directory


# ── per-prompt segmentation ───────────────────────────────────────────────────
def _prompt(text, ts="2026-01-01T00:00:00Z"):
    return _line(type="user", timestamp=ts, message={"role": "user", "content": text})


@pytest.fixture
def two_task_transcript(tmp_path):
    p = tmp_path / "two.jsonl"
    reminder = "<system-reminder>\nWorktree path: /x\n</system-reminder>\n\n "
    lines = [
        _prompt(reminder + "task one"),
        _line(type="attachment", attachment={"type": "hook_additional_context", "content": "<retrieved_context>a</retrieved_context>"}),
        _assistant("a1", "Grep", "g1"), _result("g1", "word " * 100),
        _assistant("a2", "Edit", "e1"), _result("e1", "ok"),
        _prompt("task two"),
        _line(type="attachment", attachment={"type": "hook_additional_context", "content": "<retrieved_context>b</retrieved_context>"}),
        _assistant("b1", "mcp__nlp-smarty__smarty_get_slice", "s1"), _result("s1", "{}"),
        _assistant("b2", "Edit", "e2"), _result("e2", "ok"),
    ]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_segments_attribute_work_to_the_right_prompt(two_task_transcript):
    segs = ab.analyze_segments(two_task_transcript)
    assert [s["prompt"] for s in segs] == ["task one", "task two"]  # system-reminder stripped, tool results are not prompts
    assert segs[0]["hook_injections"] == 1 and segs[1]["hook_injections"] == 1
    assert segs[0]["explore_calls_before_first_edit"] == 1 and segs[0]["explore_result_tokens_before_first_edit"] > 0
    assert segs[1]["smarty_tool_calls"] == 1 and segs[1]["explore_calls_before_first_edit"] == 0
    assert segs[0]["api_calls"] == 2 and segs[1]["api_calls"] == 2
    total = ab.analyze_transcript(two_task_transcript)
    assert sum(s["output_tokens"] for s in segs) == total["output_tokens"]  # nothing lost or double-counted


def test_compare_segments_pairs_by_task_index(two_task_transcript):
    a = [ab.analyze_segments(two_task_transcript)]
    b = [[dict(s, weighted_input_equiv=s["weighted_input_equiv"] // 2) for s in a[0]]]
    rows = ab.compare_segments(a, b)
    assert [r["task"] for r in rows] == [1, 2]
    w = next(m for m in rows[0]["metrics"] if m["metric"] == "weighted_input_equiv")
    assert w["delta_pct"] == pytest.approx(-50.0, abs=0.1)


def test_cli_per_prompt(two_task_transcript, capsys):
    assert ab.main(["transcripts", "--per-prompt", "--a", two_task_transcript, "--b", two_task_transcript]) == 0
    out = capsys.readouterr().out
    assert "== task 1" in out and "== task 2" in out and "whole-session totals" in out
