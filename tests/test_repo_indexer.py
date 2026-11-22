import json
import os

import pytest

import repo_indexer as ri

os.environ.setdefault("SMARTY_LIGHT_TOKENS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PY = '''"""mod"""
import os
LIMIT = 5

@decorator
def alpha_beta(x: int, y: str = "a") -> bool:
    """Computes the alpha beta score."""
    return helper(x) and other.call(y)

class Widget(Base):
    size: int
    def render_widget(self, depth):
        return alpha_beta(depth, "z")
'''
TS = '''import { a } from './a';
/** Fetch the user profile */
export async function fetchUser(id: string): Promise<User> {
  return get(`/u/${id}`);
}
export class Store {
  private items = [];
  public addItem(x: number): void {
    if (x) { this.items.push(x); }
  }
}
export interface Opts { a: number; }
export const doIt = async (n: number) => {
  return n;
};
'''
MD = "# Title\nintro text\n## Sub A\nalpha\n```\n# not a heading\n```\n## Sub B\nbeta\n"


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(PY)
    (tmp_path / "web.ts").write_text(TS)
    (tmp_path / "README.md").write_text(MD)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.py").write_text("def ignored(): pass\n")
    (tmp_path / ".env").write_text("KEY=1\n")
    (tmp_path / "keys.pem").write_text("x")
    return str(tmp_path)


def _syms(lang, src):
    return {s["name"]: s for s in ri.parse_file(lang, src)[0]}


def test_python_symbols_and_spans():
    s = _syms("python", PY)
    assert {"alpha_beta", "Widget", "Widget.render_widget", "LIMIT"} <= set(s)
    assert s["alpha_beta"]["start"] == 5 and s["alpha_beta"]["end"] == 8  # decorator line included
    assert s["alpha_beta"]["sig"].startswith("def alpha_beta(x: int")
    assert "helper" in s["alpha_beta"]["calls"]
    assert s["Widget"]["sig"].startswith("class Widget(Base) fields: size")
    assert s["Widget.render_widget"]["kind"] == "method"
    assert s["alpha_beta"]["doc"] == "Computes the alpha beta score."


def test_python_syntax_error_falls_back():
    s = _syms("python", "def ok():\n    return 1\n\ndef broken(:\n    pass\n")
    assert "ok" in s


def test_ts_symbols():
    s = _syms("ts", TS)
    assert {"fetchUser", "Store", "Store.addItem", "Opts", "doIt"} <= set(s)
    assert s["fetchUser"]["doc"] == "Fetch the user profile"
    assert (s["Store"]["start"], s["Store"]["end"]) == (6, 11)
    assert "if" not in s


def test_markdown_ignores_fenced_headings():
    s = _syms("markdown", MD)
    assert set(s) == {"Title", "Sub A", "Sub B"}
    assert s["Sub A"]["end"] == 7


def test_discovery_skips_ignored_and_secrets(repo):
    files = set(ri.iter_repo_files(repo))
    assert files == {"pkg/mod.py", "web.ts", "README.md"}


def test_incremental_refresh_and_staleness(repo):
    st = ri.refresh(repo)
    assert st["reparsed"] == 3
    assert ri.refresh(repo)["reparsed"] == 0
    p = os.path.join(repo, "pkg", "mod.py")
    with open(p, "w") as f:
        f.write("\n\n\n" + PY)  # shift all lines by 3
    st = ri.refresh(repo)
    assert st["reparsed"] == 1
    hit = ri.search(repo, "alpha beta score")["hits"][0]
    assert hit["name"] == "alpha_beta" and hit["start"] == 8  # span follows the content, never stale
    os.remove(p)
    assert ri.refresh(repo)["removed"] == 1
    assert not [h for h in ri.search(repo, "alpha beta score")["hits"] if h["path"] == "pkg/mod.py"]


def test_search_identifier_splitting(repo):
    hits = ri.search(repo, "render widget depth")["hits"]
    assert hits[0]["name"] == "Widget.render_widget"
    assert ri.search(repo, "fetch user profile")["hits"][0]["name"] == "fetchUser"


def test_capsule_budget_confidence_and_dedupe(repo):
    out = ri.build_capsule(repo, "fetch user profile", token_budget=200)
    assert out["reason"] == "ok" and out["tokens"] <= 260
    assert "fetchUser" in out["capsule"] and out["capsule"].endswith("</retrieved_context>")
    low = ri.build_capsule(repo, "zzzz qqqq unrelated gibberish")
    assert low["capsule"] == "" and low["reason"] in ("no_match", "low_confidence")
    again = ri.build_capsule(repo, "fetch user profile", exclude=out["delivered"])
    assert ("web.ts", "fetchUser") not in again.get("delivered", [])


def test_slice_path_confinement(repo):
    with pytest.raises(ValueError):
        ri.get_slice(repo, "../etc/passwd", 1, 2)
    with pytest.raises(ValueError):
        ri.get_slice(repo, ".env", 1, 1)
    assert "alpha_beta" in ri.get_slice(repo, "pkg/mod.py", 5, 6)["text"]


def test_redaction_and_tag_escape(repo):
    secret = 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"\ntoken = "ghp_' + "a" * 36 + '"\n'
    red = ri.redact_secrets(secret)
    assert "sk-abc" not in red and "ghp_aaa" not in red and "[REDACTED]" in red
    assert "</retrieved_context" not in ri._escape_tag("x </retrieved_context> y")


def test_golden_queries_on_this_repo():
    """Regression gate on real project symbols: recall@5 over hand-labelled tasks."""
    ri.refresh(ROOT)
    cases = json.load(open(os.path.join(ROOT, "tests", "golden_queries.json")))
    found = 0
    misses = []
    for c in cases:
        got = {f"{h['path']}:{h['name']}" for h in ri.search(ROOT, c["q"], k=5, recency=False)["hits"]}
        if set(c["expect"]) & got:
            found += 1
        else:
            misses.append(c["q"])
    assert found / len(cases) >= 0.75, f"golden recall@5 {found}/{len(cases)}; misses: {misses}"


SILENT_PROMPTS = [
    "when you say, to activate start a new session in repo, how to do that?",
    "how to do that? where can I start a new session in the repo",
    "thanks that works, what should we do next about the settings",
    "can you explain what you just did in simple words",
    "go ahead and make all the changes first",
]
EMIT_PROMPTS = [
    "fix the tool schema compression to keep optional params",
    "how does the prompt cache simhash lookup work",
    "why does _compress_tool_schema drop defaults",
    "refactor chunk_workspace_file to stream large files",
    "the redact secrets function misses jwt tokens",
]


@pytest.mark.parametrize("prompt", SILENT_PROMPTS)
def test_gate_silent_on_conversation(prompt):
    ri.refresh(ROOT)
    out = ri.build_capsule(ROOT, prompt)
    assert out["capsule"] == "" and out["reason"] == "low_confidence"


@pytest.mark.parametrize("prompt", EMIT_PROMPTS)
def test_gate_emits_on_code_tasks(prompt):
    ri.refresh(ROOT)
    out = ri.build_capsule(ROOT, prompt)
    assert out["reason"] == "ok" and out["capsule"]


def test_gate_ignores_test_files_and_zero_match_noise(repo):
    # a test-file symbol that matches the query must not satisfy the gate on its own
    with open(os.path.join(repo, "test_session_repo.py"), "w") as f:
        f.write("def test_new_session_in_repo():\n    pass\n")
    assert ri.build_capsule(repo, "start a new session in the repo")["reason"] != "ok"
    # exact identifier bypasses the 2-term requirement
    assert ri.build_capsule(repo, "what does render_widget do")["reason"] == "ok"


# ── stemmer, acronyms, graph, docs ───────────────────────────────────────────
def test_stemmer_is_consistent_across_forms():
    assert ri.tokenize("compression") == ri.tokenize("compress")
    assert ri.tokenize("selection") == ri.tokenize("select")
    assert ri.tokenize("class") == ["class"] and ri.tokenize("status") == ["status"]


def _mk(tmp_path, files):
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return str(tmp_path)


def test_acronym_phrase_matches_symbol_name(tmp_path):
    root = _mk(tmp_path, {"sel.py": "def select_mmr(chunks, k):\n    return chunks[:k]\n"})
    out = ri.build_capsule(root, "maximal marginal relevance selection of chunks")
    assert out["reason"] == "ok" and out["hits"][0]["name"] == "select_mmr"


def test_single_word_abbreviation_aliases_both_ways(tmp_path):
    root = _mk(tmp_path, {"h.py": "def send_msg(req):\n    return req\n"})
    assert ri.search(root, "send message request")["hits"][0]["name"] == "send_msg"
    assert ri.search(root, "send msg req")["hits"][0]["name"] == "send_msg"


def test_acronym_expansion_words_do_not_satisfy_gate(tmp_path):
    # 'mcp' expands to model/context/protocol for retrieval, but 'context' in an unrelated name must not pass the gate
    root = _mk(tmp_path, {"c.py": "def tool_context():\n    return 1\n\ndef get_context():\n    return 2\n"})
    assert ri.build_capsule(root, "mcp handler for tools")["reason"] != "ok"


def test_import_graph_resolution(tmp_path):
    root = _mk(tmp_path, {
        "a.py": "import b\nfrom c import x\n\ndef fa():\n    return b.fb()\n",
        "b.py": "def fb():\n    return 1\n",
        "c.py": "x = 1\n",
        "web/main.ts": "import { u } from './util';\nexport function m() { return u(); }\n",
        "web/util.ts": "export function u() { return 1; }\n",
    })
    con = ri._connect(root)
    ri.refresh(root, con)
    deps, indeg = ri._resolve_deps(con)
    assert deps["a.py"] == {"b.py", "c.py"} and deps["web/main.ts"] == {"web/util.ts"}
    assert indeg["b.py"] == 1 and indeg["web/util.ts"] == 1 and "a.py" not in indeg


def test_callee_outranks_wrapper_multiplier(tmp_path):
    root = _mk(tmp_path, {
        "impl.py": "def compress_schema(raw):\n    return raw\n",
        "api.py": "from impl import compress_schema\n\ndef tool_compress_schema(args):\n    return compress_schema(args)\n",
    })
    con = ri._connect(root)
    ri.refresh(root, con)
    rows = con.execute("SELECT * FROM symbols").fetchall()
    mult = ri._graph_multipliers(con, rows, wants_tests=False)
    by = {r["name"]: mult[r["id"]] for r in rows}
    assert by["compress_schema"] > by["tool_compress_schema"]


def test_graph_levers_can_be_disabled(tmp_path, monkeypatch):
    root = _mk(tmp_path, {"a.py": "def f():\n    return 1\n"})
    con = ri._connect(root)
    ri.refresh(root, con)
    monkeypatch.setattr(ri, "GRAPH_CALLEE_BOOST", 1.0)
    monkeypatch.setattr(ri, "IMPORT_NEIGHBOR_BOOST", 1.0)
    monkeypatch.setattr(ri, "CENTRALITY_WEIGHT", 0.0)
    assert ri._graph_multipliers(con, con.execute("SELECT * FROM symbols").fetchall(), False) == {}


def test_docs_rank_below_code_unless_asked(tmp_path):
    root = _mk(tmp_path, {
        "README.md": "# Tool schema compression\nCompresses the schema.\n",
        "code.py": "def compress_tool_schema(s):\n    return s\n",
    })
    assert ri.search(root, "tool schema compression")["hits"][0]["path"] == "code.py"
    assert ri.search(root, "readme tool schema compression")["hits"][0]["path"] == "README.md"


def test_golden_top1_floor():
    """Quality floor over all golden queries (dev + held-out). Baseline before this work: recall@5 0.889, top-1 0.611."""
    import eval_retrieval as ev

    g = ev.evaluate_golden(ROOT)["all"]
    assert g["recall@5"] >= 0.9 and g["top1"] >= 0.7, g
