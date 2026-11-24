"""
Release hygiene checks (stdlib only). Run by the release workflow and by tests.

  python tools/check_release.py [--tag v0.6.0] [--wheel W] [--sdist S] [--vsix V] [--strict-namespace]

* versions   pyproject.toml, package.json, package-lock.json and protocol.__version__ must all agree (and match --tag)
* tracked    git must not track caches, logs, compiled output, local state or secrets
* wheel/sdist  no junk, license shipped; reports every top-level name the wheel installs into site-packages
* vsix       must contain the compiled extension plus exactly the Python engine the bridge needs, nothing else
Exit status 1 on any problem (the namespace finding is a warning unless --strict-namespace).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JUNK = [r"(^|/)__pycache__/", r"\.py[co]$", r"\.log$", r"(^|/)\.DS_Store$", r"\.vsix$", r"^out/", r"(^|/)node_modules/",
        r"\.egg-info/", r"^dist/", r"^build/", r"(^|/)\.smarty/", r"^\.claude/", r"^\.mcp\.json$", r"\.smarty-(bak|tmp)$",
        r"(^|/)\.env(\.|$)", r"(^|/)NLP-start/"]
# vsce renames README.md -> readme.md and LICENSE -> LICENSE.txt inside the package
VSIX_REQUIRED = ["package.json", "readme.md", "LICENSE.txt", "out/extension.js", "bridge_server.py", "pipeline.py",
                 "data_contracts.py", "registry.py", "protocol.py"]
VSIX_FORBIDDEN = [r"^src/", r"^tests?/", r"^tools/", r"^\.github/", r"^tsconfig", r"^pyproject", r"^pytest", r"\.map$",
                  r"^gateway\.py$", r"^gate", r"^ledger", r"^token_audit", r"^ab_compare", r"^repo_indexer", r"^mcp_server",
                  r"^smarty_", r"^app\.py$", r"^test_harness", r"^requirements"]


def read_versions(root: str) -> dict[str, str]:
    out: dict[str, str] = {}
    s = open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
    m = re.search(r'^\[project\]\s*$.*?^version\s*=\s*"([^"]+)"', s, re.M | re.S)
    out["pyproject.toml"] = m.group(1) if m else "?"
    pj = json.load(open(os.path.join(root, "package.json"), encoding="utf-8"))
    out["package.json"] = pj.get("version", "?")
    lock = json.load(open(os.path.join(root, "package-lock.json"), encoding="utf-8"))
    out["package-lock.json"] = lock.get("version", "?")
    out["package-lock.json (root package)"] = lock.get("packages", {}).get("", {}).get("version", "?")
    p = open(os.path.join(root, "protocol.py"), encoding="utf-8").read()
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', p, re.M)
    out["protocol.__version__"] = m.group(1) if m else "?"
    return out


def check_versions(root: str, tag: str | None = None) -> list[str]:
    v = read_versions(root)
    problems = []
    if len(set(v.values())) != 1:
        problems.append("version mismatch: " + ", ".join(f"{k}={x}" for k, x in v.items()))
    if tag:
        want = tag[1:] if tag.startswith("v") else tag
        bad = {k: x for k, x in v.items() if x != want}
        if bad:
            problems.append(f"tag {tag} does not match: " + ", ".join(f"{k}={x}" for k, x in bad.items()))
    return problems


def tracked_junk(root: str) -> list[str]:
    try:
        files = subprocess.run(["git", "-C", root, "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    return [f for f in files if any(re.search(p, f) for p in JUNK)]


def _names(path: str) -> list[str]:
    if path.endswith((".whl", ".vsix", ".zip")):
        return zipfile.ZipFile(path).namelist()
    with tarfile.open(path) as tf:
        names = tf.getnames()
    return [n.split("/", 1)[1] for n in names if "/" in n]  # strip the sdist's top directory


def top_level_names(wheel: str) -> list[str]:
    seen = set()
    for n in zipfile.ZipFile(wheel).namelist():
        head = n.split("/", 1)[0]
        if head.endswith((".dist-info", ".data")):
            continue
        seen.add(head[:-3] if head.endswith(".py") else head)
    return sorted(seen)


def audit_python_dist(path: str) -> dict:
    names = _names(path)
    sdist = path.endswith((".tar.gz", ".tgz"))
    junk = [p for p in JUNK if not (sdist and "egg-info" in p)]  # setuptools always puts <name>.egg-info/ in an sdist
    problems = [f"junk in {os.path.basename(path)}: {n}" for n in names if any(re.search(p, n) for p in junk)]
    if not sdist:  # tests are normal (and useful to distro packagers) in an sdist, but must not be installed from a wheel
        problems += [f"tests shipped in {os.path.basename(path)}: {n}" for n in names if re.match(r"(tests?)/", n)][:3]
    if not any(n.endswith("LICENSE") for n in names):
        problems.append(f"{os.path.basename(path)} does not contain the LICENSE file")
    return {"problems": problems, "top_level": top_level_names(path) if path.endswith(".whl") else [], "files": len(names)}


def namespace_problems(top_level: list[str]) -> list[str]:
    if len(top_level) > 1:
        return [f"the wheel installs {len(top_level)} top-level names into site-packages ({', '.join(top_level)}); generic names such as "
                f"'app', 'registry', 'protocol', 'pipeline' collide with other packages. Move the code under ONE package before publishing to PyPI."]
    return []


def audit_vsix(path: str) -> list[str]:
    """The package must be exactly: the compiled extension + the Python engine the bridge imports (+ docs/license)."""
    names = [n[len("extension/"):] for n in zipfile.ZipFile(path).namelist() if n.startswith("extension/") and not n.endswith("/")]
    problems = [f"vsix is missing {r}" for r in VSIX_REQUIRED if r not in names]
    allowed = set(VSIX_REQUIRED) | {"changelog.md"}
    for n in names:
        if n in allowed:
            continue
        forbidden = any(re.search(p, n) for p in VSIX_FORBIDDEN + JUNK)
        problems.append(f"vsix must not contain {n}" if forbidden else f"vsix contains an unexpected file: {n}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tag")
    ap.add_argument("--wheel")
    ap.add_argument("--sdist")
    ap.add_argument("--vsix")
    ap.add_argument("--strict-namespace", action="store_true")
    a = ap.parse_args(argv)
    problems: list[str] = []
    warnings: list[str] = []

    v = read_versions(a.root)
    print("versions: " + ", ".join(f"{k}={x}" for k, x in v.items()))
    problems += check_versions(a.root, a.tag)
    problems += [f"git tracks a file that must not be committed: {f}" for f in tracked_junk(a.root)]
    for dist in (a.wheel, a.sdist):
        if dist:
            r = audit_python_dist(dist)
            print(f"{os.path.basename(dist)}: {r['files']} files" + (f", top-level: {', '.join(r['top_level'])}" if r["top_level"] else ""))
            problems += r["problems"]
            if dist == a.wheel:
                (problems if a.strict_namespace else warnings).extend(namespace_problems(r["top_level"]))
    if a.vsix:
        print(f"{os.path.basename(a.vsix)}: {len(zipfile.ZipFile(a.vsix).namelist())} entries")
        problems += audit_vsix(a.vsix)
    for w in warnings:
        print(f"WARN: {w}")
    for p in problems:
        print(f"FAIL: {p}")
    print("release checks " + ("FAILED" if problems else "passed"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
