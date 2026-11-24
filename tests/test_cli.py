import json
import os
import subprocess

import pytest

from contextgate_core import cli as cli


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pay.py").write_text("def charge_card(amount):\n    return amount\n")
    return tmp_path


def run(repo, *args):
    return cli.main([args[0], "--root", str(repo), *args[1:]])


def snap(repo):
    out = {}
    for dp, dns, fns in os.walk(repo):
        dns[:] = [d for d in dns if d not in (".git", ".smarty")]
        for f in fns:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, repo)] = open(p).read()
    return out


def jl(repo, rel):
    return json.load(open(os.path.join(repo, rel)))


def cmds(settings, event):
    return [h["command"] for g in settings["hooks"][event] for h in g["hooks"]]


def test_dry_run_writes_nothing(repo):
    before = snap(repo)
    assert run(repo, "init", "--dry-run", "--agents-md") == 0
    assert snap(repo) == before and not (repo / ".smarty").exists()


def test_init_creates_expected_files_and_index(repo):
    assert run(repo, "init", "--agents-md") == 0
    s = jl(repo, ".claude/settings.local.json")
    assert any("smarty_hook.py" in c for c in cmds(s, "UserPromptSubmit"))
    assert any("repo_indexer.py" in c for c in cmds(s, "SessionStart"))
    assert s["enabledMcpjsonServers"] == ["nlp-smarty"]
    assert "mcp_server.py" in json.dumps(jl(repo, ".mcp.json")["mcpServers"]["nlp-smarty"])
    assert cli.BEGIN in (repo / "AGENTS.md").read_text()
    assert (repo / ".smarty" / "index.db").exists()
    # machine-specific local settings are excluded from git without touching .gitignore
    assert not (repo / ".gitignore").exists()
    assert subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", ".claude/settings.local.json"]).returncode == 0


def test_idempotent(repo):
    run(repo, "init", "--agents-md")
    first = snap(repo)
    run(repo, "init", "--agents-md")
    assert snap(repo) == first
    s = jl(repo, ".claude/settings.local.json")
    assert len(cmds(s, "UserPromptSubmit")) == 1 and len(cmds(s, "SessionStart")) == 1


def test_merge_preserves_existing_config_and_backs_up(repo):
    (repo / ".claude").mkdir()
    orig = {"permissions": {"allow": ["Bash(ls)"]},
            "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo other"}]}],
                      "Stop": [{"hooks": [{"type": "command", "command": "echo stop"}]}]}}
    p = repo / ".claude" / "settings.local.json"
    p.write_text(json.dumps(orig))
    (repo / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "extra": 1}))
    run(repo, "init")
    s = jl(repo, ".claude/settings.local.json")
    assert s["permissions"] == orig["permissions"] and s["hooks"]["Stop"] == orig["hooks"]["Stop"]
    assert "echo other" in cmds(s, "UserPromptSubmit") and len(cmds(s, "UserPromptSubmit")) == 2
    m = jl(repo, ".mcp.json")
    assert "other" in m["mcpServers"] and "nlp-smarty" in m["mcpServers"] and m["extra"] == 1
    assert json.loads((repo / ".smarty" / "backups" / ".claude__settings.local.json").read_text()) == orig
    assert not list(repo.rglob("*.smarty-bak"))  # no clutter in the working tree


def test_invalid_json_aborts_and_touches_nothing(repo, capsys):
    (repo / ".claude").mkdir()
    bad = repo / ".claude" / "settings.local.json"
    bad.write_text("{ not json")
    before = snap(repo)
    assert run(repo, "init", "--agents-md") == 2
    assert snap(repo) == before
    assert "not valid JSON" in capsys.readouterr().err


def test_unexpected_structure_aborts(repo):
    (repo / ".claude").mkdir()
    (repo / ".claude" / "settings.local.json").write_text(json.dumps({"hooks": []}))
    assert run(repo, "init") == 2
    assert not (repo / ".mcp.json").exists()


def test_existing_project_hooks_are_not_duplicated(repo):
    (repo / ".claude").mkdir()
    proj = {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "py /x/smarty_hook.py"}]}],
                      "SessionStart": [{"hooks": [{"type": "command", "command": "py /x/repo_indexer.py index"}]}]}}
    (repo / ".claude" / "settings.json").write_text(json.dumps(proj))
    run(repo, "init")
    assert jl(repo, ".claude/settings.json") == proj
    local = jl(repo, ".claude/settings.local.json")  # only the MCP approval, no duplicate hooks
    assert "hooks" not in local and local["enabledMcpjsonServers"] == ["nlp-smarty"]


def test_unrelated_server_with_same_name_not_overwritten(repo, capsys):
    (repo / ".mcp.json").write_text(json.dumps({"mcpServers": {"nlp-smarty": {"command": "something-else"}}}))
    run(repo, "init", "--no-hooks")
    assert jl(repo, ".mcp.json")["mcpServers"]["nlp-smarty"] == {"command": "something-else"}
    assert "unrelated" in capsys.readouterr().out


def test_cartridge_preserves_user_text_and_updates_in_place(repo):
    md = repo / "CLAUDE.md"
    md.write_text("# My rules\n\n- keep it short\n")
    run(repo, "init", "--agents-md", "--no-hooks", "--no-mcp")
    t = md.read_text()
    assert t.startswith("# My rules\n\n- keep it short\n") and t.count(cli.BEGIN) == 1
    md.write_text(t.replace("Do not edit it.", "STALE") + "\n## After\ntext\n")
    run(repo, "init", "--agents-md", "--no-hooks", "--no-mcp")
    t2 = md.read_text()
    assert "STALE" not in t2 and t2.count(cli.BEGIN) == 1 and t2.endswith("## After\ntext\n")
    assert not (repo / "AGENTS.md").exists()  # AUTO picks the existing CLAUDE.md


def test_flags_no_hooks_no_mcp(repo):
    run(repo, "init", "--no-hooks", "--no-mcp")
    assert not (repo / ".claude").exists() and not (repo / ".mcp.json").exists()
    assert (repo / ".smarty" / "index.db").exists()


def test_non_git_directory_works(tmp_path, capsys):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    assert cli.main(["init", "--root", str(tmp_path)]) == 0
    assert "not a git repository" in capsys.readouterr().out
    assert (tmp_path / ".claude" / "settings.local.json").exists()


def test_uninstall_removes_only_ours(repo):
    (repo / ".claude").mkdir()
    (repo / ".claude" / "settings.local.json").write_text(json.dumps(
        {"permissions": {"allow": ["x"]},
         "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo other"}]}]}}))
    (repo / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    (repo / "CLAUDE.md").write_text("# Mine\n")
    run(repo, "init", "--agents-md")
    assert run(repo, "uninstall", "--purge") == 0
    s = jl(repo, ".claude/settings.local.json")
    assert s == {"permissions": {"allow": ["x"]},
                 "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo other"}]}]}}
    assert jl(repo, ".mcp.json") == {"mcpServers": {"other": {"command": "x"}}}
    assert (repo / "CLAUDE.md").read_text().strip() == "# Mine"
    assert not (repo / ".smarty").exists()


def test_uninstall_deletes_files_that_become_empty(repo):
    run(repo, "init")
    run(repo, "uninstall")
    assert not (repo / ".claude" / "settings.local.json").exists() and not (repo / ".mcp.json").exists()
    assert run(repo, "uninstall") == 0  # second run is a no-op


def test_uninstall_dry_run(repo):
    run(repo, "init")
    before = snap(repo)
    run(repo, "uninstall", "--dry-run")
    assert snap(repo) == before


def test_doctor_fails_before_and_passes_after(repo):
    assert run(repo, "doctor") == 1
    run(repo, "init")
    assert run(repo, "doctor") == 0
    assert not (repo / ".smarty" / "sessions" / "doctor.json").exists()


def test_bad_inputs(tmp_path, repo):
    assert cli.main(["init", "--root", str(tmp_path / "nope")]) == 2
    assert run(repo, "init", "--python", "/no/such/python") == 2
    assert cli.main(["init", "--root", str(repo), "--home", str(tmp_path)]) == 2  # incomplete install dir


def test_uninstall_deletes_cartridge_only_file_and_empty_claude_dir(repo):
    run(repo, "init", "--agents-md")
    assert (repo / "AGENTS.md").exists()
    run(repo, "uninstall")
    assert not (repo / "AGENTS.md").exists() and not (repo / ".claude").exists()
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                          text=True).stdout.split() == ["??", "pay.py"]  # nothing of ours left behind


def test_symlinked_claude_md_is_edited_through_not_replaced(repo):
    (repo / "AGENTS.md").write_text("# Shared rules\n")
    os.symlink("AGENTS.md", repo / "CLAUDE.md")
    run(repo, "init", "--agents-md", "--no-hooks", "--no-mcp")
    assert (repo / "CLAUDE.md").is_symlink() and cli.BEGIN in (repo / "AGENTS.md").read_text()
    run(repo, "uninstall")
    assert (repo / "CLAUDE.md").is_symlink() and (repo / "AGENTS.md").read_text().strip() == "# Shared rules"


# ── installed mode (pip install: no root launchers next to the package) ──────────
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Pretend `smarty` runs from a pip install: HOME has no launchers, and the package is importable from anywhere."""
    empty = tmp_path / "site-packages"
    empty.mkdir()
    monkeypatch.setattr(cli, "HOME", str(empty))
    monkeypatch.setenv("PYTHONPATH", ENGINE_ROOT)
    return empty


def test_installed_mode_registers_python_dash_m_not_script_paths(repo, installed):
    assert run(repo, "init") == 0
    hook = cmds(jl(repo, ".claude/settings.local.json"), "UserPromptSubmit")[0]
    index = cmds(jl(repo, ".claude/settings.local.json"), "SessionStart")[0]
    assert "-m contextgate_core.hook" in hook and "-m contextgate_core.repo_indexer" in index and ".py" not in hook
    mcp = jl(repo, ".mcp.json")["mcpServers"]["nlp-smarty"]
    assert mcp["args"] == ["-m", "contextgate_core.mcp_server"] and mcp["env"] == {"SMARTY_ROOT": "."}


def test_installed_mode_doctor_passes_and_uninstall_is_clean(repo, installed, capsys):
    before = snap(repo)
    assert run(repo, "init", "--agents-md") == 0
    assert run(repo, "doctor") == 0, capsys.readouterr().out
    assert run(repo, "init", "--agents-md") == 0  # idempotent: recognises its own `-m` entries
    assert len(cmds(jl(repo, ".claude/settings.local.json"), "UserPromptSubmit")) == 1
    assert run(repo, "uninstall", "--purge") == 0
    assert snap(repo) == before


def test_entries_written_in_clone_mode_are_recognised_after_switching_to_installed_mode(repo, installed, monkeypatch):
    monkeypatch.setattr(cli, "HOME", ENGINE_ROOT)  # clone mode first
    assert run(repo, "init") == 0
    assert "smarty_hook.py" in cmds(jl(repo, ".claude/settings.local.json"), "UserPromptSubmit")[0]
    monkeypatch.setattr(cli, "HOME", str(installed))  # then a pip-installed smarty looks at the same repo
    assert run(repo, "init") == 0
    assert len(cmds(jl(repo, ".claude/settings.local.json"), "UserPromptSubmit")) == 1
    assert run(repo, "uninstall") == 0
    assert not (repo / ".mcp.json").exists()
