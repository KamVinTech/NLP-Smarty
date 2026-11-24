import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bump_version as bv  # noqa: E402
import check_release as cr  # noqa: E402

HAS_GIT = shutil.which("git") is not None and os.path.exists(os.path.join(ROOT, ".git"))  # a file in git worktrees


def fixture_repo(tmp_path, version="1.2.3", **over):
    v = {"pyproject": version, "protocol": version, "pkg": version, "lock": version, "lockroot": version, **over}
    (tmp_path / "pyproject.toml").write_text(f'[build-system]\nrequires = ["x"]\n\n[project]\nname = "n"\nversion = "{v["pyproject"]}"\n\n[tool.x]\nversion = "9"\n')
    (tmp_path / "contextgate_core").mkdir(exist_ok=True)
    (tmp_path / "contextgate_core" / "protocol.py").write_text(f'__version__ = "{v["protocol"]}"\nPROTOCOL_VERSION = "1.0"\n')
    (tmp_path / "package.json").write_text(json.dumps({"name": "n", "version": v["pkg"]}))
    (tmp_path / "package-lock.json").write_text(json.dumps({"name": "n", "version": v["lock"], "packages": {"": {"version": v["lockroot"]}}}))
    return str(tmp_path)


# ── versions ──────────────────────────────────────────────────────────────────
def test_repo_versions_are_aligned():
    assert cr.check_versions(ROOT) == []
    assert set(cr.read_versions(ROOT).values()) == {"0.6.0"}


def test_version_mismatch_and_tag_checks(tmp_path):
    root = fixture_repo(tmp_path)
    assert cr.check_versions(root) == [] and cr.check_versions(root, "v1.2.3") == []
    assert "tag v1.2.4 does not match" in cr.check_versions(root, "v1.2.4")[0]
    bad = fixture_repo(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, pkg="1.2.0")
    assert "version mismatch" in cr.check_versions(bad)[0] and "package.json=1.2.0" in cr.check_versions(bad)[0]
    root2 = fixture_repo(tmp_path / "c" if (tmp_path / "c").mkdir() is None else tmp_path, lockroot="0.0.1")
    assert cr.check_versions(root2)  # the lock file's root-package entry is checked separately from its top-level version


def test_bump_updates_every_location_and_ignores_unrelated_version_keys(tmp_path):
    root = fixture_repo(tmp_path)
    changed = bv.bump(root, "2.0.0")
    assert set(changed) == {"pyproject.toml", "contextgate_core/protocol.py", "package.json", "package-lock.json"}
    assert cr.check_versions(root, "v2.0.0") == []
    assert 'version = "9"' in open(os.path.join(root, "pyproject.toml")).read()  # [tool.x] version untouched
    for bad in ("2.0", "v2.0.0", "2.0.0rc1", ""):
        with pytest.raises(ValueError):
            bv.bump(root, bad)


# ── tracked files ─────────────────────────────────────────────────────────────
@pytest.mark.skipif(not shutil.which("git"), reason="git required")
def test_tracked_junk_detector(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ("ok.py", "__pycache__/x.cpython-310.pyc", "run.log", "out/extension.js", ".mcp.json", "a.vsix", ".smarty-tmp"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A", "-f"], check=True)
    found = set(cr.tracked_junk(str(tmp_path)))
    assert found == {"__pycache__/x.cpython-310.pyc", "run.log", "out/extension.js", ".mcp.json", "a.vsix", ".smarty-tmp"}
    assert cr.tracked_junk(str(tmp_path / "nowhere")) == []


@pytest.mark.skipif(not HAS_GIT, reason="needs the real repository")
def test_this_repository_tracks_no_junk():
    assert cr.tracked_junk(ROOT) == []


@pytest.mark.skipif(not HAS_GIT, reason="needs the real repository")
@pytest.mark.parametrize("path", [".nlp_smarty_optimization.log", "out/extension.js", "x/__pycache__/y.cpython-39.pyc", "pkg.vsix",
                                  ".smarty/index.db", ".claude/settings.local.json", ".claude/worktrees/w/a.py", ".mcp.json",
                                  "dist/a.whl", ".env", "node_modules/x/y.js", "a.smarty-bak", "run.log"])
def test_gitignore_covers_local_and_generated_state(path):
    assert subprocess.run(["git", "-C", ROOT, "check-ignore", "-q", path]).returncode == 0, path


# ── distributions ─────────────────────────────────────────────────────────────
def make_zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, "x")
    return str(path)


def make_sdist(path, names):
    with tarfile.open(path, "w:gz") as t:
        for n in names:
            data = b"x"
            info = tarfile.TarInfo(f"pkg-1.0/{n}")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return str(path)


def test_wheel_audit_flags_junk_tests_and_missing_license(tmp_path):
    good = make_zip(tmp_path / "g.whl", ["mypkg/__init__.py", "mypkg/core.py", "mypkg-1.0.dist-info/licenses/LICENSE", "mypkg-1.0.dist-info/METADATA"])
    r = cr.audit_python_dist(good)
    assert r["problems"] == [] and r["top_level"] == ["mypkg"] and cr.namespace_problems(r["top_level"]) == []
    bad = make_zip(tmp_path / "b.whl", ["mypkg/core.py", "mypkg/__pycache__/core.cpython-39.pyc", "tests/test_x.py", "run.log"])
    problems = " | ".join(cr.audit_python_dist(bad)["problems"])
    assert "__pycache__" in problems and "tests/test_x.py" in problems and "run.log" in problems and "LICENSE" in problems


def test_sdist_may_contain_egg_info_and_tests_but_not_junk(tmp_path):
    ok = make_sdist(tmp_path / "ok.tar.gz", ["LICENSE", "pyproject.toml", "mod.py", "tests/test_mod.py", "pkg.egg-info/PKG-INFO"])
    assert cr.audit_python_dist(ok)["problems"] == []
    bad = make_sdist(tmp_path / "bad.tar.gz", ["LICENSE", "mod.py", "mod.log", "NLP-start/bin/python"])
    assert len(cr.audit_python_dist(bad)["problems"]) == 2


def test_flat_modules_are_reported_as_a_namespace_problem(tmp_path):
    flat = make_zip(tmp_path / "f.whl", ["app.py", "registry.py", "protocol.py", "x-1.0.dist-info/licenses/LICENSE"])
    top = cr.audit_python_dist(flat)["top_level"]
    assert top == ["app", "protocol", "registry"]
    msg = cr.namespace_problems(top)[0]
    assert "3 top-level names" in msg and "'app'" in msg and "ONE package" in msg
    assert cr.main(["--root", ROOT, "--wheel", flat]) == 0  # the namespace finding is only a warning unless --strict-namespace
    assert cr.main(["--root", ROOT, "--wheel", flat, "--strict-namespace"]) == 1


# ── vsix ──────────────────────────────────────────────────────────────────────
GOOD_VSIX = ["[Content_Types].xml", "extension.vsixmanifest"] + [f"extension/{n}" for n in cr.VSIX_REQUIRED]


def test_vsix_audit(tmp_path):
    assert cr.audit_vsix(make_zip(tmp_path / "ok.vsix", GOOD_VSIX)) == []
    missing = cr.audit_vsix(make_zip(tmp_path / "m.vsix", [n for n in GOOD_VSIX if not n.endswith("contextgate_core/protocol.py")]))
    assert missing == ["vsix is missing contextgate_core/protocol.py"]
    extra = " | ".join(cr.audit_vsix(make_zip(tmp_path / "e.vsix", GOOD_VSIX + ["extension/src/extension.ts", "extension/tests/test_a.py",
                       "extension/gateway.py", "extension/.nlp_smarty_optimization.log", "extension/notes.txt"])))
    for n in ("src/extension.ts", "tests/test_a.py", "gateway.py", ".nlp_smarty_optimization.log"):
        assert f"must not contain {n}" in extra
    assert "unexpected file: notes.txt" in extra


PKG = os.path.join(ROOT, "contextgate_core")


def _local_imports(module):
    """(required, optional) package modules imported by contextgate_core/<module>.py, found with the AST: an import inside any `try` is
    optional. Handles `from . import x`, `from .x import y` and absolute `contextgate_core.x` imports."""
    import ast

    mods = {f[:-3] for f in os.listdir(PKG) if f.endswith(".py") and f != "__init__.py"}
    required, optional = set(), set()

    def walk(node, in_try):
        for child in ast.iter_child_nodes(node):
            names = []
            if isinstance(child, ast.ImportFrom) and child.level == 1:
                names = [a.name for a in child.names] if not child.module else [child.module.split(".")[0]]
            elif isinstance(child, ast.ImportFrom) and child.module and child.module.startswith("contextgate_core."):
                names = [child.module.split(".")[1]]
            for n in names:
                if n in mods and n != module:
                    (optional if in_try else required).add(n)
            walk(child, in_try or isinstance(child, ast.Try))

    walk(ast.parse(open(os.path.join(PKG, module + ".py"), encoding="utf-8").read()), False)
    return required, optional


def _required_closure(module, seen=None):
    seen = seen if seen is not None else set()
    seen.add(module)
    for dep in _local_imports(module)[0] - seen:
        _required_closure(dep, seen)
    return seen


def _allowlisted_files():
    return [m.group(1) for line in open(os.path.join(ROOT, ".vscodeignore")) if (m := re.match(r"!([\w/]+\.py)\s*$", line.strip()))]


def test_vsix_engine_allowlist_equals_the_bridges_required_import_closure():
    """If the bridge (or something it requires) gains a new unguarded local import, the packaged extension would crash on a
    user's machine unless the .vscodeignore allowlist and the audit list are updated too."""
    closure = _required_closure("bridge")
    allow = {f for f in _allowlisted_files() if f.startswith("contextgate_core/") and not f.endswith("__init__.py")}
    assert {f"contextgate_core/{m}.py" for m in closure} == allow
    assert closure == {"bridge", "pipeline", "data_contracts", "registry", "protocol"}
    assert set(_allowlisted_files()) == set(allow) | {"bridge_server.py", "contextgate_core/__init__.py"}
    assert set(_allowlisted_files()) <= set(cr.VSIX_REQUIRED)


def test_package_init_imports_nothing_heavy():
    """The hook sets SMARTY_LIGHT_TOKENS before the estimator would load, and the vsix ships no gate/gateway: __init__ may only
    import protocol eagerly."""
    import ast

    tree = ast.parse(open(os.path.join(PKG, "__init__.py"), encoding="utf-8").read())
    eager = [n.module for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 1]
    assert eager == ["protocol"]


def test_bridge_optional_imports_are_guarded_and_absent_from_the_package():
    """`downstream` is an optional consumer: the bridge must degrade to its built-in runner when it is missing, as in the .vsix."""
    required, optional = _local_imports("bridge")
    assert optional == {"downstream"} and "downstream" not in required
    assert "contextgate_core/downstream.py" not in cr.VSIX_REQUIRED
    assert any(re.search(p, "contextgate_core/downstream.py") for p in cr.VSIX_FORBIDDEN)


def test_vscodeignore_is_an_allowlist_and_ships_the_licence():
    lines = [l.strip() for l in open(os.path.join(ROOT, ".vscodeignore")) if l.strip() and not l.startswith("#")]
    assert lines[0] == "**" and "!LICENSE" in lines and "!out/extension.js" in lines and "!package.json" in lines
    assert not any(l.startswith("!") and ("test" in l or "src/" in l or l.endswith(".log")) for l in lines)


# ── metadata and workflow ─────────────────────────────────────────────────────
def test_publishing_metadata_is_present_and_consistent():
    pj = json.load(open(os.path.join(ROOT, "package.json")))
    assert pj["publisher"] and pj["license"] == "MIT" and pj["repository"]["url"].endswith("/NLP-Smarty.git")
    assert pj["bugs"]["url"].endswith("/issues") and pj["homepage"].startswith("https://github.com/")
    assert pj["scripts"]["vscode:prepublish"] == "npm run compile" and pj["scripts"]["compile"]
    py = open(os.path.join(ROOT, "pyproject.toml")).read()
    assert 'license = "MIT"' in py and 'license-files = ["LICENSE"]' in py and "setuptools>=77" in py
    assert "[project.urls]" in py and pj["repository"]["url"].replace(".git", "") in py
    assert open(os.path.join(ROOT, "LICENSE")).read().startswith("MIT License")


def test_release_workflow_is_hardened():
    text = open(os.path.join(ROOT, ".github", "workflows", "release.yml")).read()
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", text, re.M)
    assert uses and all(re.search(r"@[0-9a-f]{40}$", u) for u in uses), [u for u in uses if not re.search(r"@[0-9a-f]{40}$", u)]
    assert re.search(r"^permissions:\n  contents: read\n", text, re.M)  # least privilege by default
    assert text.count("id-token: write") == 1 and text.count("contents: write") == 1  # one job each, nothing broader
    assert re.findall(r"secrets\.(\w+)", text) == ["VSCE_PAT"]  # PyPI uses trusted publishing: no stored token
    assert "pull_request" not in text and "pull_request_target" not in text
    assert re.search(r"publish_pypi:\n(?:.*\n)*?\s+default: false", text)  # a manual run never publishes by default
    assert "vars.PUBLISH_PYPI == 'true'" in text and "environment:" in text and "--strict-namespace" in text
    assert "--verify-tag" in text and "tools/check_release.py" in text and "bridge_server.py" in text
    for job in ("verify", "build-python", "build-vsix", "github-release", "publish-pypi", "publish-marketplace"):
        assert re.search(rf"^  {job}:", text, re.M), job


def test_packaged_engine_smoke_test_runs_like_the_extension_does():
    """Regression: the first draft used `python -I`, which drops the script folder from sys.path, so `import pipeline` failed
    silently (stderr was discarded) and the smoke test could never have passed."""
    text = open(os.path.join(ROOT, ".github", "workflows", "release.yml")).read()
    assert "python3 -u bridge_server.py" in text and " -I " not in text
    assert "optimize" in text and "test ! -e .nlp_smarty_optimization.log" in text


def test_the_bridge_really_runs_from_a_bare_copy_of_the_allowlisted_files(tmp_path):
    """The packaged extension is just these files in a folder: prove the bridge starts and optimises from exactly that."""
    allow = _allowlisted_files()
    for f in allow:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, f), tmp_path / f)
    req = "\n".join(json.dumps(r) for r in ({"id": "1", "command": "ping"},
                                            {"id": "2", "command": "optimize", "text": "Could you please write a function to reverse a string?"})) + "\n"
    env = {k: v for k, v in os.environ.items() if k != "SMARTY_AUDIT_LOG"}
    p = subprocess.run([sys.executable, "-u", "bridge_server.py"], cwd=tmp_path, input=req, capture_output=True, text=True, env=env, timeout=60)
    out = [json.loads(l) for l in p.stdout.splitlines()]
    assert [o["id"] for o in out] == ["init", "1", "2"] and out[1]["message"] == "pong" and out[2]["status"] == "OK"
    produced = sorted(str(q.relative_to(tmp_path)) for q in tmp_path.rglob("*") if q.is_file() and "__pycache__" not in q.parts)
    assert produced == sorted(allow)  # no log file or any other stray output
