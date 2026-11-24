# Restructure plan: from 17 flat modules to one importable package

## Problem
The wheel installs 17 top-level modules (`app`, `gate`, `registry`, `protocol`, ...) into `site-packages`. That collides with
other projects, so PyPI publishing is blocked on purpose (`tools/check_release.py --strict-namespace`). The PyPI name
`contextgate` is already taken, by an unrelated project whose wheel also installs a top-level `contextgate/` package (so reusing that import name would collide in `site-packages`).

## Decisions
| Topic | Decision | Reason |
|---|---|---|
| Import package | `contextgate_core` | The unrelated PyPI `contextgate` installs `contextgate/` too; a distinct import name avoids overwriting its files |
| PyPI distribution name | `contextgate-core` (chosen 2026-11-24; free then) | Matches the product, unambiguous next to the unrelated `contextgate` |
| Layout | one flat package `contextgate_core/`, not sub-packages | Smallest diff; every module keeps its name except four renames below |
| Renames | `smarty_cli`→`cli`, `smarty_hook`→`hook`, `bridge_server`→`bridge`, `app`→`downstream` | `app` is too generic; the others drop the old project name |
| Import cost | `contextgate_core/__init__.py` imports nothing eagerly (lazy public API) | The hook sets `SMARTY_LIGHT_TOKENS` before the heavy estimator would load; hook latency matters |
| Root launchers | keep thin scripts at the repo root (`smarty_hook.py`, `mcp_server.py`, `repo_indexer.py`, `bridge_server.py`, `gateway.py`, ...) that call the package | `smarty init` registers absolute script paths and recognises its own entries by filename; existing installs, the extension spawn path and documented commands keep working |
| Wheel | contains only `contextgate_core/` | No more namespace pollution; `--strict-namespace` passes |
| Public API | `contextgate_core.gate_text`, `gate_request`, `optimize`, `estimate_tokens`, `__version__` | The documented library surface |

## Status
- Phase 1 (package move): done.
- Phase 2 (installed mode for `smarty init`): done. `Launch` in `contextgate_core/cli.py` picks clone mode (root launchers present: registers `<home>/smarty_hook.py`) or installed mode (registers `python -m contextgate_core.hook`, `-m contextgate_core.repo_indexer`, `-m contextgate_core.mcp_server`). Markers recognise both forms, so entries written in one mode are found, kept idempotent and removed in the other. An explicit `--home` always means a complete clone. Verified with a built wheel in a clean venv: init, doctor (7/7), hook output, uninstall --purge leaves the repo untouched; hook latency ~40 ms in both modes.
- Rename to `contextgate_core` / `contextgate-core`: done (the console script `contextgate` was dropped for the same collision reason; `smarty*` remain).
- Phase 3 (public API docs/types): open. Phase 4 (publish): needs the pending publisher on pypi.org (see README Releasing). Phase 5: open.

## Phases
1. **Mechanical move (this branch).** `git mv` into `contextgate_core/`, relative imports, root launchers, `pyproject.toml`
   (`packages`, console scripts), tests import from the package, extension/`.vscodeignore`/`check_release.py`/CI updated.
   Acceptance: the full test suite, `test_harness.py`, bridge ping, packaged-vsix smoke test and `check_release.py` pass; the wheel
   passes `--strict-namespace`.
2. **Installed mode for `smarty init`.** When running from a pip install (no root launchers), register hooks and MCP as
   `python -m contextgate_core.hook` / `-m contextgate_core.mcp_server` and recognise them in `doctor`/`uninstall`. Clone mode is unchanged.
3. **Public API + docs.** Stabilise and document `contextgate_core.*`; README "Use as a library" section; typed return values.
4. **Name + PyPI.** Pick the distribution name, set it in `pyproject.toml`, enable trusted publishing (`PUBLISH_PYPI=true`).
5. **Extension decision.** Keep bundling the engine, or depend on the installed package; independent of the above.

## Risks
| Risk | Mitigation |
|---|---|
| Existing hook/MCP registrations point at root script paths | Root launchers remain, tests cover `init`/`doctor`/`uninstall` unchanged |
| Hook latency regression from package import | `__init__` imports nothing; hook test measures a cold run |
| Extension bundle misses a module | `.vscodeignore` allowlist + AST closure test + packaged-engine smoke test (already in CI) |
| Relative imports break `python path/to/module.py` | All entry points go through root launchers or `python -m` |
