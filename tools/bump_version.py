"""Bump the version everywhere at once: python tools/bump_version.py 0.7.0   (then commit + tag v0.7.0)."""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_release  # noqa: E402


def bump(root: str, version: str) -> list[str]:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("version must look like 1.2.3")
    changed = []

    def sub(path: str, pattern: str, repl: str) -> None:
        full = os.path.join(root, path)
        s = open(full, encoding="utf-8").read()
        new, n = re.subn(pattern, repl, s, count=1, flags=re.M | re.S)
        if n != 1:
            raise ValueError(f"could not find the version in {path}")
        open(full, "w", encoding="utf-8").write(new)
        changed.append(path)

    sub("pyproject.toml", r'(^\[project\]\s*$.*?^version\s*=\s*")[^"]+(")', rf"\g<1>{version}\g<2>")
    sub("protocol.py", r'(^__version__\s*=\s*")[^"]+(")', rf"\g<1>{version}\g<2>")
    for path in ("package.json", "package-lock.json"):
        full = os.path.join(root, path)
        d = json.load(open(full, encoding="utf-8"))
        d["version"] = version
        if "packages" in d and "" in d["packages"]:
            d["packages"][""]["version"] = version
        open(full, "w", encoding="utf-8").write(json.dumps(d, indent=2) + "\n")
        changed.append(path)
    problems = check_release.check_versions(root, f"v{version}")
    if problems:
        raise RuntimeError("; ".join(problems))
    return changed


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        files = bump(ROOT, argv[0])
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"bumped to {argv[0]}: {', '.join(files)}\nnext: git commit -am 'release {argv[0]}' && git tag v{argv[0]} && git push origin v{argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
