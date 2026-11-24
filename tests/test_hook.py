import json
import os
import subprocess
import sys

import pytest

from contextgate_core import hook as smarty_hook

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "pay.py").write_text("def charge_card(amount):\n    '''Charge a credit card.'''\n    return amount\n\n"
                                     "def refund_order(order_id):\n    '''Refund an order.'''\n    return order_id\n")
    monkeypatch.setenv("SMARTY_ROOT", str(tmp_path))
    return tmp_path


def ev(prompt, sid="s1", cwd=None):
    return {"session_id": sid, "prompt": prompt, "cwd": cwd or "/nonexistent"}


def ctx(res):
    return res["hookSpecificOutput"]["additionalContext"]


def test_injects_then_dedupes(repo):
    r1 = smarty_hook.process(ev("fix the charge credit card amount bug"))
    assert r1["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit" and "charge_card" in ctx(r1)
    r2 = smarty_hook.process(ev("still wrong: charge credit card amount"))  # same symbol already delivered
    assert r2 is None or "charge_card" not in ctx(r2).split("symbols:")[-1]
    r3 = smarty_hook.process(ev("now refund order id handling", sid="s1"))
    assert r3 and "refund_order" in ctx(r3)


def test_sessions_are_isolated(repo):
    assert smarty_hook.process(ev("charge credit card amount", sid="a"))
    assert smarty_hook.process(ev("charge credit card amount", sid="b"))


@pytest.mark.parametrize("prompt", ["", "   ", "/help", "ok", "thanks", "zzzz qqqq gibberish unrelated"])
def test_silent_cases(repo, prompt):
    assert smarty_hook.process(ev(prompt)) is None


def test_ledger_session_id_cannot_escape(repo):
    smarty_hook.process(ev("charge credit card amount", sid="../../evil"))
    assert not (repo.parent / "evil.json").exists()
    assert os.listdir(repo / ".smarty" / "sessions")


def _run(stdin, env_root):
    env = {**os.environ, "SMARTY_ROOT": str(env_root)}
    return subprocess.run([sys.executable, os.path.join(ROOT, "smarty_hook.py")], input=stdin, capture_output=True,
                          text=True, env=env, timeout=60)


def test_cli_fail_open_and_output(repo):
    bad = _run("not json", repo)
    assert bad.returncode == 0 and bad.stdout == ""
    good = _run(json.dumps(ev("charge credit card amount")), repo)
    assert good.returncode == 0 and "charge_card" in json.loads(good.stdout)["hookSpecificOutput"]["additionalContext"]


def test_project_dir_beats_event_cwd(repo, monkeypatch, tmp_path_factory):
    monkeypatch.delenv("SMARTY_ROOT")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo))
    other = tmp_path_factory.mktemp("elsewhere")
    res = smarty_hook.process(ev("charge credit card amount", cwd=str(other)))
    assert res and "charge_card" in ctx(res)
    assert not (other / ".smarty").exists()  # never indexes/pollutes the event cwd


def test_system_reminder_wrapper_is_not_searched(tmp_path, monkeypatch):
    """Real-session regression: the app prefixes prompts with a <system-reminder> naming the worktree path; those words
    matched README headings and injected irrelevant docs."""
    (tmp_path / "README.md").write_text("# Acme Test Framework\nintro\n")
    (tmp_path / "svc.py").write_text("def charge_card(amount):\n    return amount\n")
    monkeypatch.setenv("SMARTY_ROOT", str(tmp_path))
    wrapped = ("<system-reminder>\nYou are operating in a git worktree.\nWorktree path: /x/Acme-Test-Framework/.claude/"
               "worktrees/w1\nWorktree name: w1\n</system-reminder>\n\n list the tools you have")
    assert smarty_hook.process(ev(wrapped)) is None
    assert smarty_hook.clean_prompt(wrapped) == "list the tools you have"
    real = smarty_hook.process(ev("<system-reminder>\nnoise Acme Framework\n</system-reminder>\ncharge the card amount"))
    assert real and "charge_card" in ctx(real)
