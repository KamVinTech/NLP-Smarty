"""
Zero-LLM repository indexer + task-capsule compiler (Pre-Ingestion Repo Intelligence).

* Python via stdlib `ast` (regex fallback on SyntaxError); TS/JS and Markdown via structural regexes.
* Stored in `<root>/.smarty/index.db` (stdlib sqlite3 + FTS5 `bm25()`); incremental by mtime/size/sha1.
* Line spans are never trusted blindly: every query first refreshes changed files, so spans always
  match the file content hash recorded in the index.
* Only `git ls-files` (or a pruned walk) is indexed; secret-looking files are skipped and slices redacted.
* No network, no LLM, standard library only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 4
INDEX_DIR = ".smarty"
DB_NAME = "index.db"

LANG_BY_EXT = {".py": "python", ".ts": "ts", ".tsx": "ts", ".js": "ts", ".jsx": "ts", ".mjs": "ts", ".md": "markdown"}
IGNORE_DIRS = {
    "node_modules", "NLP-start", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "out", "dist",
    "build", ".git", ".smarty", ".idea", ".vscode", ".mypy_cache", ".tox", "site-packages",
}
IGNORE_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
SECRET_FILE_RE = re.compile(r"(^|/)(\.env(\..*)?|.*\.(pem|key|p12|pfx)|id_rsa.*|.*secrets?.*\.(json|ya?ml))$", re.I)
MAX_FILE_BYTES = 1_000_000

STOPWORDS = frozenset(
    "a an the and or of to in on for with without from by is are was were be been this that it its as at "
    "into over under please can could would should how what why when where which who i me my we our you "
    "your do does did done make get set use using add fix update change new code file files function "
    "so then also just but if than too very really some any".split()
)

# Retrieval tuning (defaults; to be tuned with eval_retrieval.py, not guessed)
BM25_WEIGHTS = (8.0, 3.0, 2.0, 1.5, 1.0)  # name, signature, doc, path, calls
RECENCY_BOOST = 1.35
TEST_PENALTY = 0.6  # demote tests/specs unless the query is about tests
_TEST_PATH_RE = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*$|[._]test\.[a-z]+$|[._]spec\.[a-z]+$")
MAX_PER_FILE = 2
NAME_MATCH_BOOST = 0.3         # rank bonus per distinct query term in a symbol's NAME (0 = pure BM25); +0.12 top-1 on an unseen TS repo, neutral here
# Graph / doc levers (set to 1.0 / 0.0 to disable; tuned by eval_retrieval.py --golden, not guessed)
GRAPH_CALLEE_BOOST = 1.25      # a hit that is called by another hit is probably the real implementation
IMPORT_NEIGHBOR_BOOST = 1.15   # a hit in a file imported by another matching file
CENTRALITY_WEIGHT = 0.08       # mild prior for files many others import: 1 + w*log1p(in-degree)
DOC_PENALTY = 0.85             # markdown sections rank below code unless the query is about docs
_DOC_QUERY_TERMS = frozenset({"readme", "doc", "documentation", "markdown", "changelog", "guide"})
MIN_CONFIDENCE = 0.34  # 0.25 let weak chatty-prompt matches through in real sessions


# ── token estimation ──────────────────────────────────────────────────────────
_ESTIMATOR = None


def _heuristic_tokens(text: str) -> int:
    n = 0
    for piece in re.findall(r"\w+|[^\w\s]", text):
        n += max(1, math.ceil(len(piece) / 6)) if piece[0].isalnum() or piece[0] == "_" else 1
    return n


def est_tokens(text: str) -> int:
    """Use the project's BPE-calibrated estimator when importable, else a local heuristic."""
    global _ESTIMATOR
    if _ESTIMATOR is None:
        _ESTIMATOR = _heuristic_tokens
        if not os.environ.get("SMARTY_LIGHT_TOKENS"):
            try:
                from pipeline import estimate_tokens  # noqa: WPS433

                _ESTIMATOR = estimate_tokens
            except Exception:
                pass
    return _ESTIMATOR(text)


# ── identifier tokenization (shared by index + query) ────────────────────────
_IDENT_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _stem(w: str) -> str:
    for suf in ("ion", "ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            if suf == "s" and w.endswith(("ss", "us", "is")):
                continue  # keep compress/class/status/analysis intact so stem(compression) == stem(compress)
            return w[: -len(suf)]
    return w


def tokenize(text: str, drop_stop: bool = False) -> list[str]:
    out: list[str] = []
    for word in re.findall(r"[A-Za-z0-9_]+", text or ""):
        parts = [p.lower() for p in _IDENT_PART.findall(word.replace("_", " "))]
        cand = parts + ([word.lower()] if len(parts) > 1 else [])
        for c in cand:
            if drop_stop and c in STOPWORDS:
                continue
            out.append(_stem(c))
    return out



# ── acronym / abbreviation aliases (query-side expansion, both directions) ────
# multi-word expansions only fire when ALL words are present in the query (avoids "context" -> mcp);
# single-word abbreviations (msg <-> message) alias in both directions unconditionally.
ACRONYMS: dict[str, tuple[str, ...]] = {
    "mmr": ("maximal", "marginal", "relevance"), "tfidf": ("term", "frequency", "inverse", "document"),
    "rag": ("retrieval", "augmented", "generation"), "bpe": ("byte", "pair", "encoding"),
    "ast": ("abstract", "syntax", "tree"), "mcp": ("model", "context", "protocol"),
    "llm": ("language", "model"), "nlp": ("natural", "language", "processing"), "hf": ("hugging", "face"),
    "cfg": ("config",), "conf": ("config",), "msg": ("message",), "req": ("request",), "resp": ("response",),
    "ctx": ("context",), "idx": ("index",), "impl": ("implementation",), "init": ("initialize",),
    "repo": ("repository",), "env": ("environment",), "auth": ("authentication",), "db": ("database",),
    "fn": ("function",), "param": ("parameter",), "arg": ("argument",), "cmd": ("command",), "err": ("error",),
}  # deliberately excludes ambiguous short tokens (doc, dir, util, sig): aliasing them added noise in evals
_ACR_PHRASES: list[tuple[str, tuple[str, ...]]] = []   # (acronym stem, expansion stems) for multi-word
_WORD_ALIAS: dict[str, set[str]] = {}                  # single-word, bidirectional


def _build_alias_tables() -> None:
    for acr, words in ACRONYMS.items():
        a = _stem(acr)
        ws = tuple(_stem(w) for w in words)
        if len(ws) > 1:
            _ACR_PHRASES.append((a, ws))
        else:
            _WORD_ALIAS.setdefault(a, set()).add(ws[0])
            _WORD_ALIAS.setdefault(ws[0], set()).add(a)


def expand_terms(terms: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """term -> alternative tokens. Returns (match_alias, count_alias).

    match_alias widens FTS candidate retrieval (e.g. acronym in query -> its words, which docstrings may spell out).
    count_alias is the strict subset allowed to satisfy the precision gate / coverage: the acronym->words direction is
    excluded because expansion words ('context', 'model') are common and caused false name matches (mcp -> context)."""
    present = set(terms)
    match: dict[str, set[str]] = {}
    count: dict[str, set[str]] = {}
    for t in terms:
        if t in _WORD_ALIAS:
            match.setdefault(t, set()).update(_WORD_ALIAS[t])
            count.setdefault(t, set()).update(_WORD_ALIAS[t])
    for a, ws in _ACR_PHRASES:
        if a in present:
            match.setdefault(a, set()).update(ws)
        if all(w in present for w in ws):
            for w in ws:
                match.setdefault(w, set()).add(a)
                count.setdefault(w, set()).add(a)
    for m in (match, count):
        for t, alts in m.items():
            alts.discard(t)
    return ({t: v for t, v in match.items() if v}, {t: v for t, v in count.items() if v})


def _term_hits(terms: list[str], alias: dict[str, set[str]], tokens: set[str]) -> int:
    return sum(1 for t in terms if t in tokens or (alias.get(t, set()) & tokens))


_build_alias_tables()


# ── symbol model ──────────────────────────────────────────────────────────────
def _sym(name: str, kind: str, start: int, end: int, sig: str = "", doc: str = "", calls: Iterable[str] = ()) -> dict:
    return {"name": name, "kind": kind, "start": start, "end": max(start, end), "sig": sig[:200],
            "doc": (doc or "").strip().split("\n")[0][:160], "calls": list(calls)[:12]}


def _call_names(node: ast.AST) -> list[str]:
    seen: dict[str, None] = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if nm and nm not in seen:
                seen[nm] = None
    return list(seen)


def _py_sig(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        fields = [s.target.id for s in node.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
        extra = f" fields: {', '.join(fields[:12])}" if fields else ""
        return f"class {node.name}({bases}){extra}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def _py_start(node: Any) -> int:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _parse_python(src: str) -> tuple[list[dict], list[str]]:
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return _parse_python_regex(src), []
    syms: list[dict] = []
    imports: list[str] = []

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                syms.append(_sym(prefix + node.name, "method" if in_class else "function", _py_start(node),
                                 node.end_lineno or node.lineno, _py_sig(node), ast.get_docstring(node) or "",
                                 _call_names(node)))
            elif isinstance(node, ast.ClassDef):
                syms.append(_sym(prefix + node.name, "class", _py_start(node), node.end_lineno or node.lineno,
                                 _py_sig(node), ast.get_docstring(node) or ""))
                visit(node.body, f"{prefix}{node.name}.", True)
            elif not in_class and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 1:
                        syms.append(_sym(t.id, "constant", node.lineno, node.end_lineno or node.lineno,
                                         f"{t.id} = ..."))
            elif not in_class and isinstance(node, ast.Import):
                imports.extend(a.name.split(".")[0] for a in node.names)
            elif not in_class and isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.append(node.module.split(".")[0])

    visit(tree.body, "", False)
    return syms, sorted(set(imports))


def _parse_python_regex(src: str) -> list[dict]:
    lines = src.split("\n")
    syms = []
    pat = re.compile(r"^(\s*)(async\s+def|def|class)\s+(\w+)")
    for i, line in enumerate(lines, 1):
        m = pat.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        end = i
        for j in range(i, len(lines)):
            nxt = lines[j]
            if j > i - 1 and nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent and j >= i:
                end = j
                break
            end = j + 1
        syms.append(_sym(m.group(3), "class" if m.group(2) == "class" else "function", i, end, line.strip()))
    return syms


_TS_DECL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?"
    r"(?P<kind>class|interface|enum|type|namespace)\s+(?P<name>\w+)")
_TS_FUNC = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s*(?P<name>\w+)\s*[(<]")
_TS_ARROW = re.compile(
    r"^(?:export\s+)?(?:const|let)\s+(?P<name>\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*(?::[^=]+?)?=>")
_MODS = r"(?:(?:public|private|protected|static|async|readonly|override|abstract|get|set)\s+)*"
_TS_MEMBER = re.compile(r"^\s*" + _MODS + r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:<[^>(]*>)?\s*\(")
_TS_PROP_FN = re.compile(
    r"^\s*" + _MODS + r"(?P<name>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?"
    r"(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+?)?=>")
_TS_TEST = re.compile(r"^\s*(?P<fn>test|it|describe|suite)(?:\.(?:describe|only|skip|serial|parallel|fixme|slow))*\s*\(\s*['\"`]")
_TS_TEST_TITLE = re.compile(r"""^\s*(?:test|it|describe|suite)(?:\.\w+)*\s*\(\s*(?P<q>['"`])(?P<title>.*?)(?P=q)""")
_TS_KEYWORDS = frozenset(
    "if for while switch catch return function else do try new await super typeof delete throw import require "
    "with yield".split())
_TS_CALL_STOP = _TS_KEYWORDS | frozenset(
    "push map filter join then log error warn info from keys values entries resolve reject some find includes split "
    "slice trim replace expect async Promise String Number Boolean Object Array Error JSON parse stringify forEach "
    "reduce get set has add delete test describe it".split())
_TS_IMPORT = re.compile(r"""(?:from\s+|require\()\s*['"]([^'"]+)['"]""")


def _mask_ts(src: str) -> str:
    """Blank out string/template/comment/regex contents (keeping newlines, braces of real code and quote delimiters)
    so brace/paren counting and structural regexes can't be fooled by literals. Output has the same length."""
    out: list[str] = []
    i, n = 0, len(src)
    tpl: list[int] = []          # brace depth per active `${ ... }` expression inside template literals
    mode = "code"                # code | sq | dq | tpl | line | block | regex
    prev = ""                    # last significant code char (regex-vs-division heuristic)
    in_class = False             # inside [...] of a regex literal

    def blank(c: str) -> str:
        return c if c == "\n" else " "

    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        vis = not tpl            # delimiters stay visible only for top-level literals
        if mode == "code":
            if c in "'\"":
                mode = "sq" if c == "'" else "dq"
                out.append(c if vis else " ")
            elif c == "`":
                mode = "tpl"
                out.append(c if vis else " ")
            elif c == "/" and nxt == "/":
                mode = "line"
                out.append("  ")
                i += 1
            elif c == "/" and nxt == "*":
                mode = "block"
                out.append("  ")
                i += 1
            elif c == "/" and (prev == "" or prev in "(,=:[!&|?{};"):
                mode, in_class = "regex", False
                out.append(c if vis else " ")
            elif tpl:
                if c == "{":
                    tpl[-1] += 1
                elif c == "}":
                    if tpl[-1] == 0:
                        tpl.pop()
                        mode = "tpl"
                    else:
                        tpl[-1] -= 1
                out.append(blank(c))
            else:
                out.append(c)
                if not c.isspace():
                    prev = c
        elif mode in ("sq", "dq"):
            q = "'" if mode == "sq" else '"'
            if c == "\\":
                out.append(blank(c) + (blank(nxt) if nxt else ""))
                i += 1
            elif c == q:
                mode, prev = "code", "a"
                out.append(c if vis else " ")
            elif c == "\n":
                mode = "code"
                out.append("\n")
            else:
                out.append(" ")
        elif mode == "tpl":
            if c == "\\":
                out.append(blank(c) + (blank(nxt) if nxt else ""))
                i += 1
            elif c == "`":
                mode, prev = "code", "a"
                out.append(c if vis else " ")
            elif c == "$" and nxt == "{":
                tpl.append(0)
                mode = "code"
                out.append("  ")
                i += 1
            else:
                out.append(blank(c))
        elif mode == "line":
            if c == "\n":
                mode = "code"
            out.append(blank(c))
        elif mode == "block":
            if c == "*" and nxt == "/":
                mode = "code"
                out.append("  ")
                i += 1
            else:
                out.append(blank(c))
        else:  # regex literal
            if c == "\\":
                out.append(blank(c) + (blank(nxt) if nxt else ""))
                i += 1
            elif c == "\n":
                mode = "code"
                out.append("\n")
            elif c == "[":
                in_class = True
                out.append(" ")
            elif c == "]":
                in_class = False
                out.append(" ")
            elif c == "/" and not in_class:
                mode, prev = "code", "a"
                out.append(c if vis else " ")
            else:
                out.append(" ")
        i += 1
    return "".join(out)


def _depths(masked: list[str]) -> tuple[list[int], list[int], list[int]]:
    """Per line: brace depth before/after the line, and paren+bracket depth before the line."""
    before, after, pbefore = [], [], []
    d = p = 0
    for ln in masked:
        before.append(d)
        pbefore.append(p)
        d += ln.count("{") - ln.count("}")
        p += ln.count("(") + ln.count("[") - ln.count(")") - ln.count("]")
        after.append(d)
    return before, after, pbefore


def _span_end(masked: list[str], before: list[int], after: list[int], i: int) -> int:
    """0-based index of the line where the construct starting at line i closes (depth returns to its start level)."""
    base = before[i]
    scan_from = i + 1
    if after[i] <= base:  # no opening brace here: maybe Allman style (`{` on the next line)
        j = i + 1
        while j < len(masked) and not masked[j].strip():
            j += 1
        if j < len(masked) and masked[j].lstrip().startswith("{") and ";" not in masked[i]:
            scan_from = j + 1
        else:
            return i
    for k in range(scan_from, min(len(masked), i + 4000)):
        if after[k] <= base:
            return k
    return i


def _ts_calls(masked: list[str], start: int, end: int) -> list[str]:
    text = "\n".join(masked[start:min(end + 1, start + 400)])
    names = re.findall(r"(?<![\w$.])([A-Za-z_$][\w$]*)\s*\(", text) + re.findall(r"\.([A-Za-z_$][\w$]*)\s*\(", text)
    return [n for n in dict.fromkeys(names) if len(n) >= 3 and n not in _TS_CALL_STOP][:12]


def _ts_doc(lines: list[str], idx: int) -> str:
    j = idx - 1
    if j >= 0 and lines[j].strip().endswith("*/"):
        while j >= 0 and "/**" not in lines[j]:
            j -= 1
        for k in range(max(j, 0), idx):
            t = lines[k].strip().lstrip("/*").strip().rstrip("*/").strip()
            if t and not t.startswith("@"):
                return t
    elif j >= 0 and lines[j].strip().startswith("//"):
        return lines[j].strip().lstrip("/").strip()
    return ""


def _parse_ts(src: str) -> tuple[list[dict], list[str]]:
    lines = src.split("\n")
    masked = _mask_ts(src).split("\n")
    before, after, pbefore = _depths(masked)
    imports = sorted(set(_TS_IMPORT.findall(src)))
    syms: list[dict] = []
    classes: list[tuple[str, int, int]] = []
    suites: list[tuple[str, int, int]] = []

    def add(name: str, kind: str, i: int, end: int, sig: str) -> None:
        syms.append(_sym(name, kind, i + 1, end + 1, sig, _ts_doc(lines, i), _ts_calls(masked, i, end)))

    for i, line in enumerate(masked):
        top = before[i] == 0 and pbefore[i] == 0
        m = _TS_DECL.match(line)
        if m:
            kind, name = m.group("kind"), m.group("name")
            end = _span_end(masked, before, after, i)
            add(name, kind, i, end, lines[i].strip().rstrip("{").strip())
            if kind == "class":
                classes.append((name, i, end))
            continue
        if top and (m := (_TS_FUNC.match(line) or _TS_ARROW.match(line))):
            add(m.group("name"), "function", i, _span_end(masked, before, after, i), lines[i].strip().rstrip("{").strip())
            continue
        if _TS_TEST.match(line) and (t := _TS_TEST_TITLE.match(lines[i])):
            fn, title = _TS_TEST.match(line).group("fn"), t.group("title").strip()
            end = _span_end(masked, before, after, i)
            owner = next((s for s in reversed(suites) if s[1] < i <= s[2]), None)
            is_suite = fn in ("describe", "suite") or ".describe" in line[:line.index("(")]  # test.describe(...)
            name = f"{owner[0]} > {title}" if owner else title
            add(name, "suite" if is_suite else "test", i, end, f"{fn}('{title}')")
            if is_suite:
                suites.append((name, i, end))
    for cname, cs, ce in classes:
        for idx in range(cs + 1, ce + 1):
            if before[idx] != before[cs] + 1 or pbefore[idx] != pbefore[cs]:
                continue  # only direct class members, never statements nested in method bodies
            m = _TS_MEMBER.match(masked[idx]) or _TS_PROP_FN.match(masked[idx])
            if not m or m.group("name") in _TS_KEYWORDS:
                continue
            add(f"{cname}.{m.group('name')}", "method", idx, _span_end(masked, before, after, idx),
                lines[idx].strip().rstrip("{").strip())
    return syms, imports


def _parse_markdown(src: str) -> tuple[list[dict], list[str]]:
    lines = src.split("\n")
    heads: list[tuple[int, int, str]] = []
    fence = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence and (m := re.match(r"^(#{1,6})\s+(.*\S)", line)):
            heads.append((i, len(m.group(1)), m.group(2)))
    syms = []
    for idx, (ln, lvl, title) in enumerate(heads):
        end = len(lines)
        for ln2, lvl2, _ in heads[idx + 1:]:
            if lvl2 <= lvl:
                end = ln2 - 1
                break
        body = next((l.strip() for l in lines[ln:end] if l.strip() and not l.startswith("#")), "")
        syms.append(_sym(title, "section", ln, end, "#" * lvl + " " + title, body))
    return syms, []


def parse_file(lang: str, src: str) -> tuple[list[dict], list[str]]:
    if lang == "python":
        return _parse_python(src)
    if lang == "ts":
        return _parse_ts(src)
    return _parse_markdown(src)


# ── file discovery ────────────────────────────────────────────────────────────
def _git(root: str, *args: str) -> list[str] | None:
    try:
        p = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.splitlines() if p.returncode == 0 else None


def iter_repo_files(root: str) -> Iterator[str]:
    """Yield repo-relative POSIX paths of indexable files (git-tracked + untracked-unignored, else a pruned walk)."""
    listed = _git(root, "ls-files", "--cached", "--others", "--exclude-standard") if os.path.isdir(
        os.path.join(root, ".git")) else None
    if listed is None:
        listed = []
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in IGNORE_DIRS]
            for fn in fns:
                listed.append(os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/"))
    for rel in listed:
        parts = rel.split("/")
        if any(p in IGNORE_DIRS for p in parts[:-1]) or parts[-1] in IGNORE_FILES:
            continue
        if os.path.splitext(rel)[1].lower() not in LANG_BY_EXT or SECRET_FILE_RE.search(rel):
            continue
        yield rel


# ── storage ───────────────────────────────────────────────────────────────────
def _connect(root: str) -> sqlite3.Connection:
    d = os.path.join(root, INDEX_DIR)
    os.makedirs(d, exist_ok=True)
    gi = os.path.join(d, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w") as f:
            f.write("*\n")
    con = sqlite3.connect(os.path.join(d, DB_NAME))
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, sha1 TEXT, mtime_ns INTEGER, size INTEGER,
            lang TEXT, tokens_raw INTEGER, tokens_skeleton INTEGER, imports TEXT);
        CREATE TABLE IF NOT EXISTS symbols(id INTEGER PRIMARY KEY, path TEXT, name TEXT, kind TEXT,
            start INTEGER, end INTEGER, sig TEXT, doc TEXT, calls TEXT);
        CREATE INDEX IF NOT EXISTS symbols_path ON symbols(path);
        CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(name, sig, doc, path, calls);
        """
    )
    row = con.execute("SELECT v FROM meta WHERE k='schema'").fetchone()
    if row and row["v"] != str(SCHEMA_VERSION):
        con.executescript("DELETE FROM files; DELETE FROM symbols; DELETE FROM symbols_fts;")
    con.execute("INSERT OR REPLACE INTO meta VALUES('schema', ?)", (str(SCHEMA_VERSION),))
    return con


def _drop_file(con: sqlite3.Connection, path: str) -> None:
    ids = [r[0] for r in con.execute("SELECT id FROM symbols WHERE path=?", (path,))]
    con.executemany("DELETE FROM symbols_fts WHERE rowid=?", [(i,) for i in ids])
    con.execute("DELETE FROM symbols WHERE path=?", (path,))
    con.execute("DELETE FROM files WHERE path=?", (path,))


def _store_file(con: sqlite3.Connection, rel: str, lang: str, src: str, sha1: str, st: os.stat_result) -> None:
    _drop_file(con, rel)
    syms, imports = parse_file(lang, src)
    skeleton = sum(est_tokens(s["sig"]) + 2 for s in syms)
    con.execute("INSERT INTO files VALUES(?,?,?,?,?,?,?,?)",
                (rel, sha1, st.st_mtime_ns, st.st_size, lang, est_tokens(src), skeleton, json.dumps(imports)))
    path_tokens = " ".join(tokenize(rel.replace("/", " ").replace(".", " ")))
    for s in syms:
        cur = con.execute("INSERT INTO symbols(path,name,kind,start,end,sig,doc,calls) VALUES(?,?,?,?,?,?,?,?)",
                          (rel, s["name"], s["kind"], s["start"], s["end"], s["sig"], s["doc"], json.dumps(s["calls"])))
        con.execute("INSERT INTO symbols_fts(rowid,name,sig,doc,path,calls) VALUES(?,?,?,?,?,?)",
                    (cur.lastrowid, " ".join(tokenize(s["name"].replace(".", " "))), " ".join(tokenize(s["sig"])),
                     " ".join(tokenize(s["doc"])), path_tokens, " ".join(tokenize(" ".join(s["calls"])))))


def refresh(root: str, con: sqlite3.Connection | None = None) -> dict[str, int]:
    """Incrementally sync the index with the working tree. Cheap when nothing changed (stat only)."""
    root = os.path.abspath(root)
    own = con is None
    con = con or _connect(root)
    existing = {r["path"]: r for r in con.execute("SELECT path, sha1, mtime_ns, size FROM files")}
    seen: set[str] = set()
    stats = {"scanned": 0, "reparsed": 0, "unchanged": 0, "removed": 0}
    for rel in iter_repo_files(root):
        full = os.path.join(root, rel)
        try:
            st = os.stat(full)
        except OSError:
            continue
        if st.st_size > MAX_FILE_BYTES:
            continue
        seen.add(rel)
        stats["scanned"] += 1
        old = existing.get(rel)
        if old and old["mtime_ns"] == st.st_mtime_ns and old["size"] == st.st_size:
            stats["unchanged"] += 1
            continue
        try:
            raw = open(full, "rb").read()
        except OSError:
            continue
        sha1 = hashlib.sha1(raw).hexdigest()
        if old and old["sha1"] == sha1:
            con.execute("UPDATE files SET mtime_ns=?, size=? WHERE path=?", (st.st_mtime_ns, st.st_size, rel))
            stats["unchanged"] += 1
            continue
        _store_file(con, rel, LANG_BY_EXT[os.path.splitext(rel)[1].lower()], raw.decode("utf-8", "replace"), sha1, st)
        stats["reparsed"] += 1
    for gone in set(existing) - seen:
        _drop_file(con, gone)
        stats["removed"] += 1
    con.commit()
    if own:
        con.close()
    return stats


# ── retrieval ─────────────────────────────────────────────────────────────────
def modified_files(root: str) -> set[str]:
    if not os.path.isdir(os.path.join(root, ".git")):
        return set()
    out: set[str] = set()
    for line in _git(root, "status", "--porcelain") or []:
        if len(line) > 3:
            out.add(line[3:].split(" -> ")[-1].strip().strip('"'))
    return out



# ── import / call graph ───────────────────────────────────────────────────────
def _resolve_deps(con: sqlite3.Connection) -> tuple[dict[str, set[str]], dict[str, int]]:
    """File-level import graph resolved against indexed files -> (deps: path -> imported paths, in-degree)."""
    files = con.execute("SELECT path, lang, imports FROM files").fetchall()
    paths = {f["path"] for f in files}
    by_base: dict[str, list[str]] = {}
    for p in paths:
        if p.endswith(".py"):
            base = os.path.basename(p)[:-3]
            by_base.setdefault(os.path.basename(os.path.dirname(p)) if base == "__init__" else base, []).append(p)
    deps: dict[str, set[str]] = {}
    indeg: dict[str, int] = {}
    for f in files:
        out: set[str] = set()
        try:
            imports = json.loads(f["imports"] or "[]")
        except ValueError:
            imports = []
        for spec in imports:
            if f["lang"] == "python":
                cands = by_base.get(spec, [])
                if len(cands) == 1:
                    out.add(cands[0])
            elif f["lang"] == "ts" and spec.startswith("."):
                base = os.path.normpath(os.path.join(os.path.dirname(f["path"]), spec)).replace(os.sep, "/")
                for ext in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"):
                    if base + ext in paths:
                        out.add(base + ext)
                        break
        out.discard(f["path"])
        deps[f["path"]] = out
        for d in out:
            indeg[d] = indeg.get(d, 0) + 1
    return deps, indeg


def _graph_multipliers(con: sqlite3.Connection, rows: list[sqlite3.Row], wants_tests: bool) -> dict[int, float]:
    """Per-symbol score multipliers from the call graph, import neighbours and file centrality."""
    if GRAPH_CALLEE_BOOST == 1.0 and IMPORT_NEIGHBOR_BOOST == 1.0 and CENTRALITY_WEIGHT == 0.0:
        return {}
    deps, indeg = _resolve_deps(con)
    mult: dict[int, float] = {r["id"]: 1.0 for r in rows}
    by_simple: dict[str, list[int]] = {}
    for r in rows:
        by_simple.setdefault(r["name"].split(".")[-1], []).append(r["id"])
    row_paths = {r["id"]: r["path"] for r in rows}
    matched_files = set(row_paths.values())
    callee_boosted: set[int] = set()
    for r in rows:
        try:
            calls = json.loads(r["calls"] or "[]")
        except ValueError:
            calls = []
        for c in calls:
            if len(c) < 4:
                continue
            for tid in by_simple.get(c, []):
                if tid != r["id"] and tid not in callee_boosted:
                    callee_boosted.add(tid)
                    mult[tid] *= GRAPH_CALLEE_BOOST
    imported_by_match = {d for p in matched_files for d in deps.get(p, ()) if d in matched_files}
    for r in rows:
        if not wants_tests and _TEST_PATH_RE.search(r["path"]):
            continue
        if r["path"] in imported_by_match:
            mult[r["id"]] *= IMPORT_NEIGHBOR_BOOST
        mult[r["id"]] *= 1.0 + CENTRALITY_WEIGHT * math.log1p(indeg.get(r["path"], 0))
    return mult


def search(root: str, query: str, k: int = 5, recency: bool = True, max_per_file: int = MAX_PER_FILE,
           con: sqlite3.Connection | None = None, auto_refresh: bool = True) -> dict[str, Any]:
    """Rank symbols for a query. Returns {hits, coverage, terms}."""
    root = os.path.abspath(root)
    own = con is None
    con = con or _connect(root)
    if auto_refresh:
        refresh(root, con)
    terms = list(dict.fromkeys(tokenize(query, drop_stop=True)))
    if not terms:
        if own:
            con.close()
        return {"hits": [], "coverage": 0.0, "terms": []}
    match_alias, alias = expand_terms(terms)  # `alias` (strict) drives gate/coverage; match_alias only widens retrieval
    match_terms = list(dict.fromkeys(terms + [a for t in terms for a in sorted(match_alias.get(t, ()))]))
    match = " OR ".join(f'"{t}"' for t in match_terms)
    w = ", ".join(str(x) for x in BM25_WEIGHTS)
    rows = con.execute(
        f"SELECT s.*, -bm25(symbols_fts, {w}) AS score, "
        "(symbols_fts.name||' '||symbols_fts.sig||' '||symbols_fts.doc||' '||symbols_fts.path||' '||symbols_fts.calls) "
        "AS fts_text, (symbols_fts.name||' '||symbols_fts.sig||' '||symbols_fts.path) AS strict_text, "
        "symbols_fts.name AS name_text "
        "FROM symbols_fts JOIN symbols s ON s.id=symbols_fts.rowid "
        "WHERE symbols_fts MATCH ? ORDER BY score DESC LIMIT 60", (match,)).fetchall()
    # Intent is judged on prose only: path-like/kebab tokens ("NexusAI-Test-Framework", "src/test/x") must not flip it.
    intent_terms = set(tokenize(re.sub(r"\S*[/\\-]\S*", " ", query), drop_stop=True))
    wants_tests = any(t.startswith("test") or t in ("spec", "pytest", "fixture") for t in intent_terms)
    wants_docs = any(t in _DOC_QUERY_TERMS for t in intent_terms)
    gmult = _graph_multipliers(con, rows, wants_tests)
    mods = modified_files(root) if recency else set()
    ranked = []
    name_hits: dict[int, int] = {}
    for r in rows:
        name_hits[r["id"]] = _term_hits(terms, alias, set(tokenize(r["name_text"])))
        score = r["score"] * (1.0 + NAME_MATCH_BOOST * name_hits[r["id"]]) * (RECENCY_BOOST if r["path"] in mods else 1.0) * gmult.get(r["id"], 1.0)
        if not wants_docs and r["kind"] == "section":
            score *= DOC_PENALTY
        if not wants_tests and _TEST_PATH_RE.search(r["path"]):
            score *= TEST_PENALTY
        ranked.append((score, r))
    ranked.sort(key=lambda t: -t[0])

    hits: list[dict] = []
    per_file: dict[str, int] = {}
    for score, r in ranked:
        if len(hits) >= k:
            break
        if per_file.get(r["path"], 0) >= max_per_file:
            continue
        if any(h["path"] == r["path"] and h["start"] <= r["start"] and r["end"] <= h["end"] for h in hits):
            continue  # nested inside an already-selected span
        per_file[r["path"]] = per_file.get(r["path"], 0) + 1
        hits.append({"path": r["path"], "name": r["name"], "kind": r["kind"], "start": r["start"], "end": r["end"],
                     "sig": r["sig"], "score": round(score, 4), "modified": r["path"] in mods,
                     "matched": name_hits[r["id"]],  # precision gate: NAME tokens only
                     "matched_any": _term_hits(terms, alias, set(tokenize(r["strict_text"]))),  # name+signature+path
                     "eligible": (wants_tests or not _TEST_PATH_RE.search(r["path"])) and (wants_docs or r["kind"] != "section"),
                     "_tokens": set(tokenize(r["fts_text"]))})
    top_tokens: set[str] = set()
    for h in hits[:3]:
        top_tokens |= h["_tokens"]
    coverage = _term_hits(terms, alias, top_tokens) / len(terms)
    for h in hits:
        h.pop("_tokens", None)
    # Precision gate inputs: how many DISTINCT query terms the best non-test hit matches in its NAME
    # (signature/path matches are too weak: every symbol in repo_indexer.py matches "repo")
    strong = max((h["matched"] for h in hits if h["eligible"]), default=0)
    needed = 1 if len(terms) == 1 else 2
    idents = [w.lower().strip(".") for w in re.findall(r"[A-Za-z_][\w.]*", query)
              if "_" in w or re.search(r"[a-z][A-Z]", w) or "." in w.strip(".")]
    ident_hit = any(h["eligible"] and (i == h["name"].lower() or i in h["name"].lower().split(".")
                                       or i == os.path.basename(h["path"]).lower())
                    for h in hits for i in idents)
    if own:
        con.close()
    return {"hits": hits, "coverage": round(coverage, 3), "terms": terms, "strong": strong, "needed": needed,
            "ident_hit": ident_hit}


# ── slices & capsule ──────────────────────────────────────────────────────────
_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"""(?i)\b((?:api[_-]?key|secret|token|passw(?:or)?d)\w*\s*[:=]\s*)(['"])[^'"\n]{8,}\2"""),
]


def redact_secrets(text: str) -> str:
    for i, pat in enumerate(_SECRET_PATTERNS):
        text = pat.sub((lambda m: m.group(1) + m.group(2) + "[REDACTED]" + m.group(2)) if i == 6 else "[REDACTED]", text)
    return text


def safe_path(root: str, rel: str) -> str:
    """Resolve `rel` under `root`; raise ValueError on traversal/absolute escape or secret files."""
    root_real = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_real, rel))
    if full != root_real and not full.startswith(root_real + os.sep):
        raise ValueError(f"path escapes workspace root: {rel!r}")
    if SECRET_FILE_RE.search(rel.replace(os.sep, "/")):
        raise ValueError(f"refusing to read secret-like file: {rel!r}")
    return full


def get_slice(root: str, path: str, start: int, end: int, context: int = 0, redact: bool = True) -> dict[str, Any]:
    full = safe_path(root, path)
    with open(full, encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")
    s, e = max(1, start - context), min(len(lines), end + context)
    body = "\n".join(f"{n}| {lines[n - 1]}" for n in range(s, e + 1))
    return {"path": path, "start": s, "end": e, "text": redact_secrets(body) if redact else body}


def _escape_tag(text: str) -> str:
    return text.replace("</retrieved_context", "<\\/retrieved_context")


def build_capsule(root: str, query: str, token_budget: int = 450, k: int = 5, body_k: int = 3,
                  max_body_lines: int = 60, min_confidence: float = MIN_CONFIDENCE,
                  exclude: Iterable[tuple[str, str]] = ()) -> dict[str, Any]:
    """Compile a budgeted <retrieved_context> capsule. Empty (with reason) when confidence is too low.
    `exclude` = (path, name) pairs already delivered this session (session ledger dedupe)."""
    res = search(root, query, k=k + len(list(exclude)))
    skip = set(exclude)
    hits = [h for h in res["hits"] if (h["path"], h["name"]) not in skip][:k]
    if not hits:
        return {"capsule": "", "tokens": 0, "confidence": 0.0, "hits": [], "reason": "no_match"}
    # Gate (all must hold): (a) >=2 distinct query terms hit one non-test symbol's NAME,
    # or the prompt names an exact identifier; (b) overall coverage of the top hits clears min_confidence.
    if not (res["ident_hit"] or res["strong"] >= res["needed"]) or res["coverage"] < min_confidence:
        return {"capsule": "", "tokens": 0, "confidence": res["coverage"], "hits": hits, "reason": "low_confidence"}
    hits = [h for h in hits if h["matched_any"] >= 1]  # drop zero-match noise from the symbol list
    if not hits:
        return {"capsule": "", "tokens": 0, "confidence": res["coverage"], "hits": [], "reason": "no_match"}

    head = (f'<retrieved_context source="nlp-smarty" confidence="{res["coverage"]}" '
            'note="lexical retrieval; may be incomplete - verify with grep if unsure">')
    parts = [head, "symbols:"]
    for h in hits:
        parts.append(f"- {h['path']}:L{h['start']}-{h['end']} {h['kind']} {h['sig'] or h['name']}")
    used = est_tokens("\n".join(parts)) + 8
    delivered = []
    # bodies: code before docs (stable sort keeps score order within each group)
    for h in sorted(hits, key=lambda x: x["kind"] == "section")[:body_k]:
        remaining = token_budget - used
        if remaining < 40:
            break
        sl = get_slice(root, h["path"], h["start"], min(h["end"], h["start"] + max_body_lines - 1))
        text = _escape_tag(sl["text"])
        lines = text.split("\n")
        keep: list[str] = []
        cost = est_tokens(f"[{h['path']}:L{h['start']}-{h['end']}]")
        for ln in lines:
            c = est_tokens(ln) + 1
            if cost + c > remaining:
                break
            keep.append(ln)
            cost += c
        if not keep:
            break
        if len(keep) < (h["end"] - h["start"] + 1):
            keep.append(f"... (truncated; full span L{h['start']}-{h['end']})")
        block = f"[{h['path']}:L{h['start']}-{h['end']}]\n" + "\n".join(keep)
        parts.append(block)
        used += cost
        delivered.append((h["path"], h["name"]))
    parts.append("</retrieved_context>")
    capsule = "\n".join(parts)
    return {"capsule": capsule, "tokens": est_tokens(capsule), "confidence": res["coverage"], "hits": hits,
            "delivered": delivered, "reason": "ok"}


def find_symbol(root: str, name: str, limit: int = 10, con: sqlite3.Connection | None = None) -> list[dict]:
    """Exact (case-insensitive) then prefix/suffix match on symbol name, e.g. 'optimize' or 'PromptCache.lookup'."""
    root = os.path.abspath(root)
    own = con is None
    con = con or _connect(root)
    refresh(root, con)
    n = name.strip()
    rows = con.execute(
        "SELECT path,name,kind,start,end,sig FROM symbols WHERE lower(name)=lower(?) "
        "OR lower(name) LIKE lower(?) ESCAPE '\\' OR lower(name) LIKE lower(?) ESCAPE '\\' "
        "ORDER BY (lower(name)=lower(?)) DESC, length(name), path LIMIT ?",
        (n, n.replace("%", r"\%").replace("_", r"\_") + "%", "%." + n.replace("%", r"\%").replace("_", r"\_"), n, limit),
    ).fetchall()
    out = [dict(r) for r in rows]
    if own:
        con.close()
    return out


MAX_SCAN_FILES = 5000
_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*")


def find_references(root: str, name: str, limit: int = 40, token_budget: int = 700, include_tests: bool = True,
                    con: sqlite3.Connection | None = None) -> dict[str, Any]:
    """Where is `name` used? Definitions first, then callers (call-graph), then other whole-word mentions as file:line
    snippets (code before tests before docs), all within a token budget. A grep replacement that is ranked, deduped and
    bounded. `name` may be dotted ('BasePage.click'); mentions are matched on the last part, definitions on the full name."""
    root = os.path.abspath(root)
    parts = name.strip().split(".")
    simple = parts[-1]
    if not _IDENT_RE.fullmatch(simple) or not all(_IDENT_RE.fullmatch(x) for x in parts):
        raise ValueError("name must be an identifier such as 'click' or 'BasePage.click'")
    own = con is None
    con = con or _connect(root)
    refresh(root, con)
    dotted = len(parts) > 1
    defs = [dict(r) for r in con.execute(
        "SELECT path,name,kind,start,end,sig FROM symbols WHERE name=? OR (? = 0 AND name LIKE ? ESCAPE '\\') "
        "ORDER BY path,start", (name.strip(), int(dotted), "%." + simple.replace("_", r"\_")))]
    def_ids = {(d["path"], d["start"]) for d in defs}
    pat = re.compile(r"(?<![\w$])" + re.escape(simple) + r"(?![\w$])")
    paths = [r["path"] for r in con.execute("SELECT path FROM files ORDER BY path")][:MAX_SCAN_FILES]

    def bucket(path: str) -> int:
        if path.lower().endswith(".md"):
            return 2
        return 1 if _TEST_PATH_RE.search(path) else 0
    call_pat = re.compile(r"(?<![\w$])" + re.escape(simple) + r"\s*\(")
    quoted_pat = re.compile(r"""['"`]""" + re.escape(simple) + r"""['"`]""")

    def form(line: str) -> int:
        """0 call site, 1 other code, 2 string literal, 3 comment: real usages first."""
        if re.match(r"\s*(//|/\*|\*|#)", line):
            return 3
        if call_pat.search(line):
            return 0
        return 2 if quoted_pat.search(line) else 1
    cls_pat = re.compile(r"(?<![\w$])" + re.escape(parts[-2]) + r"(?![\w$])") if dotted else None
    knows: dict[str, bool] = {}  # for 'Class.method': files that mention Class are far likelier to be real usages
    mentions: list[tuple[int, str, int, str, int]] = []
    for path in paths:
        if not include_tests and _TEST_PATH_RE.search(path):
            continue
        try:
            with open(os.path.join(root, path), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if not pat.search(text):
            continue
        knows[path] = bool(cls_pat and cls_pat.search(text))
        for i, line in enumerate(text.split("\n"), 1):
            if (path, i) not in def_ids and pat.search(line):
                mentions.append((bucket(path), path, i, redact_secrets(line.strip())[:140], form(line)))
    best = {}  # a file is ranked by its best usage; inside a file the best usages come first
    for b, pth, _, _, f in mentions:
        best[pth] = min(best.get(pth, 9), f)
    mentions.sort(key=lambda m: (m[0], 0 if (not dotted or knows.get(m[1])) else 1, best[m[1]], m[1], m[4], m[2]))
    # Callers = the innermost symbol around each code usage (calls and other code, not strings/comments). Derived from the
    # usage lines rather than the stored call lists, which drop common names (warn/log/error) and over-credit containers.
    spans: dict[str, list] = {}
    for pth in {m[1] for m in mentions}:
        spans[pth] = [dict(r) for r in con.execute("SELECT name,kind,start,end FROM symbols WHERE path=?", (pth,))]
    found: dict[tuple, dict] = {}
    for _, pth, ln, _snip, fm in mentions:
        if fm > 1 or pth.lower().endswith(".md"):
            continue  # strings/comments and prose are mentions, not callers
        around = [x for x in spans[pth] if x["start"] <= ln <= x["end"]]
        if not around:
            continue
        inner = min(around, key=lambda x: (x["end"] - x["start"], x["start"]))
        if (pth, inner["start"]) in def_ids:
            continue  # the usage is inside the definition itself (e.g. the method delegating to locator.click)
        key = (pth, inner["name"], inner["start"])
        c = found.setdefault(key, {"path": pth, "name": inner["name"], "kind": inner["kind"], "start": inner["start"],
                                   "end": inner["end"], "lines": 0})
        c["lines"] += 1
    callers = list(found.values())
    if not include_tests:
        callers = [c for c in callers if not _TEST_PATH_RE.search(c["path"])]
    callers.sort(key=lambda c: (bucket(c["path"]), c["path"], c["start"]))
    if own:
        con.close()
    counts = {"definitions": len(defs), "callers": len(callers), "mentions": len(mentions)}

    out: list[str] = []
    used = 0

    def emit(line: str) -> bool:
        nonlocal used
        c = est_tokens(line) + 1
        if used + c > token_budget and out:
            return False
        out.append(line)
        used += c
        return True
    truncated = False
    exhausted = False
    if defs:
        emit("definitions:")
        for d in defs:
            exhausted = exhausted or not emit(f"- {d['path']}:L{d['start']}-{d['end']} {d['kind']} {d['name']}")
    else:
        emit(f"definitions: none indexed for '{name}'")
    MAX_CALLERS = 8  # callers are structure; the line-level mentions below are the substance, so keep callers short
    if callers and not exhausted:
        shown_c = callers[:MAX_CALLERS]
        emit(f"callers (showing {len(shown_c)} of {len(callers)}):" if len(callers) > MAX_CALLERS else f"callers ({len(callers)}):")
        for c in shown_c:
            if not emit(f"- {c['path']}:L{c['start']}-{c['end']} {c['name']}"):
                exhausted = True
                break
        truncated |= len(callers) > MAX_CALLERS
    if mentions and not exhausted:
        emit(f"mentions ({len(mentions)}):")
        by_file: dict[str, list] = {}
        for m in mentions:  # already ranked: files by relevance, lines inside a file best-usage first
            by_file.setdefault(m[1], []).append(m)
        shown = 0
        for path, ms in by_file.items():
            if shown >= limit:
                break
            if not emit(f"{path}:"):
                exhausted = True
                break
            # pick the best usages first (<=4 per file so one noisy file can't eat the budget, and never more than the
            # global limit), THEN show them in line order
            chosen = sorted(ms[:min(4, limit - shown)], key=lambda m: m[2])
            for _, _, ln, snippet, _f in chosen:
                if not emit(f"  L{ln}: {snippet}"):
                    exhausted = True
                    break
                shown += 1
            if exhausted:
                break
        truncated |= shown < len(mentions)
    truncated |= exhausted
    if truncated:
        out.append("... (truncated; narrow the name, set include_tests=false, or use smarty_get_slice)")
    return {"text": "\n".join(out), "counts": counts, "truncated": truncated, "tokens": est_tokens("\n".join(out)),
            "definitions": defs, "callers": callers}


def repo_map(root: str, token_budget: int = 800) -> str:
    """Compact file -> top-level symbols map (modified files first). Stops at the token budget."""
    root = os.path.abspath(root)
    con = _connect(root)
    refresh(root, con)
    mods = modified_files(root)
    files = con.execute("SELECT path, lang, tokens_raw FROM files").fetchall()
    top = {}
    for r in con.execute("SELECT path,name,kind,start,end FROM symbols WHERE kind IN "
                         "('class','function','interface','enum','constant','type','suite') ORDER BY path, start"):
        top.setdefault(r["path"], []).append(r)
    con.close()
    order = sorted((f for f in files if f["lang"] != "markdown"), key=lambda f: (f["path"] not in mods, f["path"]))
    lines = ["# repo_map (nlp-smarty) - file: symbols L<start>-<end>; * = modified"]
    used = est_tokens(lines[0])
    for f in order:
        syms = top.get(f["path"], [])
        shown = ", ".join(f"{s['name']} L{s['start']}-{s['end']}" for s in syms[:14])
        more = f" (+{len(syms) - 14} more)" if len(syms) > 14 else ""
        line = f"{'*' if f['path'] in mods else ''}{f['path']}: {shown}{more}" if syms else f"{f['path']}"
        c = est_tokens(line) + 1
        if used + c > token_budget:
            lines.append(f"... (map truncated at {token_budget} tokens; use smarty_find_symbol / smarty_get_context)")
            break
        lines.append(line)
        used += c
    return "\n".join(lines)


def stats(root: str) -> dict[str, Any]:
    con = _connect(os.path.abspath(root))
    f = con.execute("SELECT COUNT(*) n, COALESCE(SUM(tokens_raw),0) raw, COALESCE(SUM(tokens_skeleton),0) sk FROM files").fetchone()
    n_sym = con.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    con.close()
    return {"files": f["n"], "symbols": n_sym, "tokens_raw": f["raw"], "tokens_skeleton": f["sk"],
            "skeleton_reduction_pct": round(100 * (1 - f["sk"] / f["raw"]), 1) if f["raw"] else 0.0}


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("SMARTY_LIGHT_TOKENS", "1")  # CLI/hook path: skip the heavy BPE estimator for speed + consistency
    ap = argparse.ArgumentParser(prog="repo_indexer", description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="build/refresh the index")
    sub.add_parser("stats", help="index statistics")
    q = sub.add_parser("query", help="search and print a capsule")
    q.add_argument("text")
    q.add_argument("--budget", type=int, default=450)
    q.add_argument("--k", type=int, default=5)
    s = sub.add_parser("slice", help="print a numbered file slice")
    s.add_argument("path")
    s.add_argument("start", type=int)
    s.add_argument("end", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "index":
        print(json.dumps({**refresh(a.root), **stats(a.root)}))
    elif a.cmd == "stats":
        print(json.dumps(stats(a.root)))
    elif a.cmd == "query":
        out = build_capsule(a.root, a.text, token_budget=a.budget, k=a.k)
        print(out["capsule"] or f"[no capsule: {out['reason']} confidence={out['confidence']}]")
        print(f"\n# tokens={out['tokens']} confidence={out['confidence']}", file=sys.stderr)
    elif a.cmd == "slice":
        print(get_slice(a.root, a.path, a.start, a.end)["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
