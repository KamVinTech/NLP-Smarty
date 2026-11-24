# ContextGate

**A deterministic, zero-LLM gate in front of your LLM calls.** It compresses prose safely, checks that required inputs are present, grounds the prompt in a strict XML structure, audits the response, and records what it saved in a local ledger. Every transform is classical NLP/IR (regex, BM25, TF-IDF, MMR, SimHash); no model, no network call and no `tiktoken` is used inside the engine.

> **Naming:** the product and the VS Code extension are **ContextGate**. The GitHub repository is still `NLP-Smarty`; the command-line tool (`smarty`), the Python modules and the `SMARTY_*` environment variables keep their current names until the Python package is renamed.

## Contents

1. [What it is, and what it is not](#what-it-is-and-what-it-is-not)
2. [Architecture](#architecture)
3. [Quickstart](#quickstart)
4. [Surface 1: gateway proxy](#surface-1-gateway-proxy-smarty-serve)
5. [Surface 2: repo intelligence for coding agents](#surface-2-repo-intelligence-for-coding-agents-smarty-init)
6. [Surface 3: VS Code extension](#surface-3-vs-code-extension)
7. [The optimizer pipeline](#the-optimizer-pipeline-pipelinepy)
8. [Measured results](#measured-results-what-we-can-and-cannot-claim)
9. [Safety guarantees](#safety-guarantees)
10. [Project layout](#project-layout) · [Dependencies](#dependencies) · [Tests](#tests) · [Extending](#extending-the-registry) · [Releasing](#releasing)

---

## What it is, and what it is not

| It is | It is not |
|---|---|
| A deterministic pre-ingestion gate: same input, same output, no model in the loop | A way to cut an agent session's bill by large amounts on its own |
| Safe-by-default compression with verification and fail-open fallback | A tokenizer: token counts are **estimates** (median ~12% over `cl100k_base`) |
| A local, counts-only ledger that measures savings on **your** traffic | A guarantee of savings: on well-written text the strict profile saves about 1% |

Where the money is: in measured Claude Code sessions the fixed prefix (system prompt, tool definitions) was about 82% of weighted cost, appended context 7.6%, output 10.5% and the typed prompt 0.1%. Prompt-text compression can only touch the small remainder; avoiding whole API calls and shrinking bulky structure matter more. See [Measured results](#measured-results-what-we-can-and-cannot-claim).

---

## Architecture

One engine, three surfaces:

```
                         ┌───────────────────────────────────────────────┐
   OpenAI / Anthropic    │  gateway.py  (stdlib ThreadingHTTPServer)     │
   SDKs, Claude Code ───►│  route · parse · relay (SSE incremental)      │──► upstream API
   (base URL → :8787)    │        │                       │              │    (credentials forwarded
                         │        ▼                       ▼              │     untouched, never stored)
                         │  gate.py                  ledger.py           │
                         │  protected spans →        SQLite, counts only │
                         │  compress → verify →      est. vs provider    │
                         │  fall back to original    usage calibration   │
                         └───────────────────────────────────────────────┘

   Claude Code ─ UserPromptSubmit hook ─► smarty_hook.py ─┐
   Claude Code ─ MCP (optional) ────────► mcp_server.py ──┼─► repo_indexer.py  (symbols + line spans,
                                                          │        hash-checked, stdlib only)
   VS Code ── extension.ts ─ JSON-Lines ─► bridge_server.py ─► pipeline.py (9-step optimizer)
                                                  │                 ├─ data_contracts.py (typed contracts)
                                                  │                 └─ registry.py (tools, few-shots, guardrails, slots)
                                                  └─► downstream.py (optional downstream HF consumer / offline simulator)
```

| Component | File | Responsibility |
|---|---|---|
| Gate | `contextgate_core/gate.py` | Protected-span detection, deterministic prose compression (`strict`/`balanced`/`aggressive`), verification with fallback, OpenAI and Anthropic request walkers |
| Gateway | `contextgate_core/gateway.py` | Local proxy: request validation, gating, streaming relay, response audit, ledger rows |
| Ledger | `contextgate_core/ledger.py` | Counts-only SQLite ledger; estimator calibration against provider `usage`; `stats` report |
| Offline bench | `contextgate_core/gate_bench.py` | Replays recorded requests through the gate and reports per-profile, per-transform savings |
| Optimizer | `contextgate_core/pipeline.py` | 9-step prompt compiler (budgeting, schema compression, MMR/BM25 pruning, intent routing, cache, history pruning), post-LLM `audit_agent_response()` |
| Contracts | `contextgate_core/data_contracts.py`, `registry.py`, `protocol.py` | Typed dataclasses; static tools, few-shots, guardrails, required slots; protocol and version constants |
| Repo intelligence | `contextgate_core/repo_indexer.py`, `hook.py`, `mcp_server.py`, `cli.py` | Symbol index, Claude Code hook, MCP server, and the `smarty` CLI (`init`, `doctor`, `uninstall`, `audit`, `serve`, `stats`, `bench`) |
| VS Code | `src/extension.ts`, `contextgate_core/bridge.py` | Extension client plus persistent JSON-Lines engine sidecar |
| Measurement | `contextgate_core/token_audit.py`, `ab_compare.py`, `eval_retrieval.py` | Session cost audit, A/B tooling, retrieval-quality evaluation |
| Downstream demo | `contextgate_core/downstream.py` | HuggingFace consumer with offline simulator, used by the extension's compare view |

**Hard rules.** No LLM or network calls and no `tiktoken` in `pipeline.py`, `data_contracts.py` or `registry.py`. The default install has **zero runtime dependencies**; `spacy`, `rank_bm25`, `flashtext` and `scikit-learn` are optional, each with a pure-Python fallback.

---

## Quickstart

Python ≥ 3.9. No dependencies are needed for the gateway, hook, MCP server or CLI.

```bash
git clone https://github.com/KamVinTech/NLP-Smarty && cd NLP-Smarty
python3 smarty_cli.py --help
python3 test_harness.py | tail -n 30      # optimizer benchmark + regression suite
python3 -m pytest -q --tb=short tests     # unit and adversarial tests
```

Optional extras for the full optimizer: `pip install spacy rank_bm25 flashtext scikit-learn && python -m spacy download en_core_web_sm`.
`pip install .` installs the distribution `contextgate-core` (import package `contextgate_core`) and the commands `smarty`, `smarty-mcp`, `smarty-hook` and `smarty-gateway`.

---

## Use as a library

```python
from contextgate_core import RawRequest, estimate_tokens, gate_request, gate_text, optimize

# 1. Text: deterministic prose compression. Level 1-3 or a profile name ("strict", "balanced", "aggressive").
text, ok = gate_text("Could you please basically fix `get_user()`, do not use recursion.", "balanced")
assert ok and text == "Fix `get_user()`, do not use recursion."   # ok=False means: `text` is the ORIGINAL, unchanged

# 2. Requests: OpenAI-compatible or Anthropic Messages bodies (auto-detected). Never raises, never larger.
body = {"model": "m", "messages": [{"role": "user", "content": "Could you please basically explain this, in order to be safe?"}]}
result = gate_request(body, "strict")
print(result.body, result.saved, result.applied, result.protected_ok)   # estimated tokens saved; result.error names a failure

# 3. The 9-step prompt compiler: returns a strict XML payload for a downstream agent.
compiled = optimize(RawRequest(text="Could you please write a Python function to reverse a string without using slicing?"))
print(compiled.status, compiled.trace.query_token_savings_pct)          # "OK" | "NEEDS_CLARIFICATION" | "INSUFFICIENT_CONTEXT"

print(estimate_tokens("hello world"))                                    # an estimate (median ~12% over cl100k_base), not a provider count
```

| Name | Does | Notes |
|---|---|---|
| `gate_text(text, level=1)` | Compress prose, protected spans verbatim | Returns `(text, ok)`; `ValueError` only for an unknown level |
| `gate_request(body, profile="strict", api=None)` | Gate a chat request body | Returns `GateResult` (`body`, `raw_tokens`, `fwd_tokens`, `saved`, `applied`, `protected_ok`, `fallbacks`, `error`) |
| `optimize(RawRequest, few_shot_k=2)` | Compile an XML prompt | `payload_token_savings_pct` is vs a modeled baseline, see [the pipeline](#what-payload_token_savings_pct-means) |
| `estimate_tokens(text)` | Token estimate | Deterministic, pure Python |
| `audit_agent_response(text, trace)` | Score a response against the extracted negations and entities | Post-LLM, deterministic |
| `RawRequest`, `TokenBudget`, `CompiledPrompt`, `OptimizationTrace`, `GateResult`, `PROFILES` | Types and constants | |

The package is typed (`py.typed`). `import contextgate_core` is instant: the public names load lazily. The PyPI distribution is
`contextgate-core` (the unrelated `contextgate` on PyPI also installs an import package named `contextgate`, so we use `contextgate_core` to avoid
overwriting each other). Not published yet; install from a clone: `pip install .`. Console scripts: `smarty`, `smarty-gateway`, `smarty-mcp`, `smarty-hook`.
The public names are pinned by `tests/test_public_api.py`, which also executes the example above.

---

## Surface 1: gateway proxy (`smarty serve`)

An OpenAI- and Anthropic-compatible local proxy for any SDK or pipeline.

```bash
python3 smarty_cli.py serve --upstream https://api.openai.com --profile strict   # base_url=http://127.0.0.1:8787/v1
python3 smarty_cli.py serve --upstream https://api.anthropic.com                 # ANTHROPIC_BASE_URL=http://127.0.0.1:8787
python3 smarty_cli.py stats [--prices prices.json] [--since 24h]                 # measured savings ledger
python3 smarty_cli.py bench traffic.jsonl                                        # offline replay of recorded requests
```

Options (`python3 gateway.py -h`): `--host` (default `127.0.0.1`), `--port` (`8787`), `--profile`, `--ledger PATH` / `--no-ledger`, `--record FILE`, `--force-usage` (OpenAI streaming only), `--no-audit`, `--timeout`, `--verbose`, `--allow-remote`.

**Profiles**

| Profile | Does |
|---|---|
| `strict` (default) | Prose of user messages and tool-schema `description` strings only |
| `balanced` | + hedging words and causal duplicate collapse |
| `aggressive` | + articles in long prose |

**What is rewritten, per API**

- OpenAI: user message text and tool-schema descriptions. Tool schemas keep their exact JSON structure.
- Anthropic `/v1/messages`: only the user's plain-text blocks (not `<system-reminder>` wrappers) and tool/`input_schema` descriptions. `system`, signed `thinking`, `tool_use`, `tool_result` (strict) and `cache_control` are forwarded verbatim. Usage is normalised as fresh + cache read + cache write. `count_tokens` and batches are relayed ungated.
- Verified by tests against the documented formats and block shapes seen in real request recordings; **not yet verified against live Claude Code traffic**. Per the [Claude Code gateway docs](https://code.claude.com/docs/en/llm-gateway-connect), a base URL alone does not replace a subscription login, and the desktop app ignores `ANTHROPIC_BASE_URL`/`settings.json`; use the CLI or the VS Code extension setting.

**Ledger** (`~/.smarty/ledger.db`): counts only, never prompt text, responses, headers or credentials. Savings are original vs forwarded request size with the same estimator, calibrated against the provider's exact `usage`; USD uses a price table you supply. The row is written after the response is sent, so it adds no latency.

**Hardening (each has a test in `tests/test_security.py`)**: loopback bind unless `--allow-remote`; origin-form request targets only (no `..`, encoded dot-segments, CR/LF, NUL, backslash); validated `Content-Length` and chunk sizes (size limit checked before reading); `Transfer-Encoding` + `Content-Length` together rejected; 60 s client socket timeout; deeply nested JSON forwarded verbatim; bounded gate work (pathological input skips compression and forwards the original); `--record` file forced to mode 0600; log lines sanitised; warning for plain-http non-loopback upstreams; SIGTERM drains in-flight requests so no ledger row is lost.

**Measure before you claim.** Run `serve --record traffic.jsonl`, then `bench traffic.jsonl` on your own traffic to see which transforms apply. `--record` stores original prompts: keep the file private.

---

## Surface 2: repo intelligence for coding agents (`smarty init`)

Zero-LLM, stdlib-only. Indexes your repo (symbols with line spans, hash-checked so spans never go stale) and delivers a small, budgeted `<retrieved_context>` capsule to the agent through a Claude Code `UserPromptSubmit` hook and an optional MCP server.

```bash
python3 smarty_cli.py init --dry-run          # preview every change as a diff
python3 smarty_cli.py init --agents-md        # hooks + MCP + index (+ static CLAUDE.md/AGENTS.md block)
python3 smarty_cli.py doctor                  # verify end to end
python3 smarty_cli.py uninstall [--purge]     # remove only what init added
python3 smarty_cli.py audit --last 10         # where do your sessions' tokens go? (reads Claude Code transcripts)
```

- **Two install modes, same command.** From a clone, `init` registers the launcher scripts (`<clone>/smarty_hook.py`); from a pip install it registers `python -m contextgate_core.hook` / `-m contextgate_core.mcp_server` using the interpreter that has the package. `doctor` and `uninstall` recognise both.
- Hooks go to `.claude/settings.local.json` (machine-specific, git-excluded); `--scope project` shares them. `init` is idempotent, merges into existing JSON (aborts on unparseable files) and keeps one-time backups in `.smarty/backups/`.
- The hook is fail-open (always exits 0), stdlib-only, never injects the same symbol twice in a session, and stays silent below a confidence gate.
- MCP tools: `smarty_get_context`, `smarty_find_symbol`, `smarty_find_references`, `smarty_repo_map`, `smarty_get_slice`, plus the optimizer tools (`smarty_optimize`, `smarty_audit_response`, `smarty_compress_context`, `smarty_compress_tool_schema`). The server's agent-facing instructions are **off by default** (in a live test the agent never used the tools even with them); opt in with `SMARTY_MCP_INSTRUCTIONS=on` in the `env` of `.mcp.json`.
- Retrieval quality: `python3 eval_retrieval.py --golden` (hand-labelled queries) or without flags (git history). Cost audit: `smarty audit` (`token_audit.py`, weights `--w-cw/--w-cr/--w-out`). A/B tooling: `ab_compare.py` (`simulate` and `transcripts`).

---

## Surface 3: VS Code extension

Side-by-side comparison of a raw prompt against the optimized one, plus inline tools. Extension ID `KamVinTech-contextgate.contextgate`; requires VS Code ≥ 1.90.

```bash
npm install && npm run compile      # tsc -p ./ ; then press F5 for the Extension Development Host
```

| Command | Mac | Win/Linux | Does |
|---|---|---|---|
| `ContextGate: Optimize Prompt & Active File Context` | `Cmd+Alt+O` | `Ctrl+Alt+O` | Runs the optimizer on the selection or a typed prompt, copies the compiled XML, opens the telemetry view |
| `ContextGate: Open Prompt Diff` | `Cmd+Alt+D` | `Ctrl+Alt+D` | Native diff of old prompt vs optimized core prompt |
| `ContextGate: Compare Raw vs. Optimized LLM Output` | `Cmd+Alt+C` | `Ctrl+Alt+C` | Runs both prompts on `vscode.lm` (Copilot) or HuggingFace and audits both responses |
| `ContextGate: Compress Selected Prompt Inline` | `Cmd+Alt+I` | `Ctrl+Alt+I` | Replaces the selection with the compressed `<core_task>` |
| `ContextGate: Show Logs` | `Cmd+Alt+L` | `Ctrl+Alt+L` | Opens the Output channel |

Chat participant: `@contextgate /optimize <prompt>`, `@contextgate /compare <prompt>`, or `@contextgate <prompt>` (defaults to side-by-side comparison).

| Setting | Default | Description |
|---|---|---|
| `contextGate.pythonPath` | `python3` | Interpreter used for the engine sidecar |
| `contextGate.engineDirectory` | `""` | Directory containing `bridge_server.py`, `pipeline.py` (empty = the extension root, which bundles the engine) |
| `contextGate.downstreamBackend` | `auto` | `auto` \| `vscode-lm` \| `huggingface` |
| `contextGate.huggingFaceToken` / `huggingFaceModel` | `""` / `Qwen/Qwen2.5-Coder-32B-Instruct` | HuggingFace credentials and model |
| `contextGate.compressionIntensity` | `AUTO` | `AUTO` \| `LIGHT` \| `MEDIUM` \| `AGGRESSIVE` |
| `contextGate.maxTokenBudget` | `512` | Advisory `max_total_tokens` (not enforced as a total ceiling) |
| `contextGate.autoShowLogs` / `optimizeOnSave` | `true` / `false` | Reveal the Output panel; refresh the inspector on save |
| `contextGate.auditLogFile` | `""` | Optional file copy of the audit log. It contains **previews of your prompts**, so nothing is written to disk unless you set this (the Output stream is always on) |

**Bridge sidecar (`bridge_server.py`).** A persistent process speaking one JSON object per line on stdin/stdout. Commands: `optimize`, `compare_hf` (pipeline + Raw and Compiled prompts to `contextgate_core/downstream.py`; offline simulator without `HF_TOKEN`), `audit_pair` (audits responses produced by `vscode.lm`), `clear_cache`, `capabilities`, `ping`. Open workspace files are split into chunks of at most 45 words on blank lines and `def`/`class`/`function`/`export`/`interface` boundaries before MMR pruning. Smoke test: `echo '{"id":"1","command":"ping"}' | python3 bridge_server.py`.

The extension ships exactly the launcher `bridge_server.py`, the package files `contextgate_core/{__init__,bridge,pipeline,data_contracts,registry,protocol}.py` and the compiled extension (allowlist in `.vscodeignore`, enforced by a test).

**Reading the numbers in the UI.** "est. tokens" are the built-in estimator, not provider counts. "vs unpruned template" is measured against a *modeled baseline*, not against the request you would otherwise have sent (see below).

---

## The optimizer pipeline (`pipeline.py`)

`optimize(RawRequest) -> CompiledPrompt` runs a deterministic four-stage flow and compiles a fixed XML template.

```python
from data_contracts import RawRequest, TokenBudget
from pipeline import optimize

result = optimize(RawRequest(
    text="Could you please write a Python function to reverse a string without using slicing?",
    token_budget=TokenBudget(),        # per-section budgets
))
print(result.status, result.payload)   # payload: compiled XML for the downstream agent
print(result.trace)                    # OptimizationTrace
```

`status` is `OK`, `NEEDS_CLARIFICATION` (a required slot is missing, so no model call is needed) or `INSUFFICIENT_CONTEXT` (see `data_contracts.py`).

| Step | What it does |
|---|---|
| 1. Budgeting | `estimate_tokens` (pure-Python BPE-style estimate) and per-section budgets: `core_query_budget`, `context_budget`, `history_budget`, `tools_budget`, `few_shot_budget`. **`max_total_tokens` is advisory, not enforced as a total ceiling** (measured: the compiled payload stays ~629 est. tokens for caps 80–512) |
| 2. Tool schema compression | Verbose JSON Schema to compact signatures: `tool_name(param:str, opt?:int=5) // desc`. Measured saving 95–120 est. tokens per registry tool (median 97) |
| 3. MMR + BM25 + TF-IDF pruning | Relevance `0.65 × normalized BM25 + 0.35 × TF-IDF cosine`, then MMR for diversity. Applied to few-shots, tools, RAG chunks, history, workspace chunks |
| 4. Sentence compressor | `45% BM25 + 35% TF-IDF centroid + 20% positional`; negations, constraints, code, URLs, emails are never pruned. Active only above `core_query_budget` words |
| 5. Lexical normalization | NFKC, smart quotes, 14 contraction expansions, 10 verbose-idiom rewrites, deterministic stemmer |
| Stage 2. Intent router | Weighted keywords (multi-word phrases 2.5×, lead-verb bonus), secondary intents, `task_hint` override. Intents: `CODE_GENERATION`, `DATA_EXTRACTION`, `TOOL_EXECUTION`, `FACTUAL_QA`, `AMBIGUOUS_QUERY` |
| 6. Prompt cache | SHA-256 exact cache plus 64-bit SimHash near-duplicate cache (Hamming ≤ 2), gated on matching intent, negations, entities and context length; 256-entry LRU; `enable_cache=False` bypasses |
| 7. History pruning | Newest turn always kept; older turns ranked by recency-weighted MMR `(relevance + 0.15) × 0.85^steps_back` within `history_budget` |
| 8. Adaptive intensity | `AUTO` picks `LIGHT/MEDIUM/AGGRESSIVE` from token pressure; `MEDIUM` strips hedging adverbs; `AGGRESSIVE` strips articles above 25 words |
| 9. Telemetry | `OptimizationTrace` with query-level and payload-level token counts, savings and per-section detail |

**Pre-LLM slot check.** If a required parameter for the detected intent is missing (`REQUIRED_SLOTS`), the pipeline short-circuits before any model call. **Post-LLM audit.** `audit_agent_response()` scores a response against the extracted negations and entities.

### Defaults (`TokenBudget`)

| Field | Default | Controls |
|---|---|---|
| `max_total_tokens` | 512 | Advisory only |
| `core_query_budget` | 120 | Max words in the compressed query |
| `context_budget` | 160 | Max tokens of pruned RAG context |
| `history_budget` | 80 | Max tokens of pruned history |
| `tools_budget` | 60 | Max tokens of compressed tool schemas |
| `few_shot_budget` | 90 | Max tokens of selected few-shots |

### What `payload_token_savings_pct` means

It compares the compiled payload against a **modeled baseline**: the unpruned template containing every registry tool, every few-shot example, every guardrail and all context and history (`_estimate_unoptimized_baseline_tokens`). It is **not** savings versus the request you would otherwise have sent. For short prompts the compiled payload is *larger* than the raw prompt (9–31 raw tokens compile to roughly 350–430 tokens because of the template, guardrails, tools and examples). The prompt-text figure (`query_token_savings_pct`) is the honest per-prompt number.

---

## Measured results (what we can and cannot claim)

| Claim | Evidence |
|---|---|
| Strict gate on RAG built from real docs (synthetic framing) | ≈0.9–1.1% request savings |
| Claude Code hook pilot | ≈-20% weighted cost, 95% CI -44%..+4%, n=4 (**not significant**); -57..60% exploration calls |
| MCP tools in agent runs | used in 0 of 8 runs |
| Gate overhead | ≈1 ms mean, ≈5 ms p95 on RAG-sized requests |
| Optimizer payload vs raw request (8 chunks + 6 turns) | -64% (1868 → 677 est. tokens) |
| Optimizer payload vs raw short prompt | larger: the template adds ~350–430 tokens |
| Token estimator vs `cl100k_base` (151 samples) | median +12% over-count (code ~13–16%, JSON ~29%, prose ~7%); ~16% within 3% |
| Session cost structure | fixed prefix ≈82%, appended context 7.6%, output 10.5%, typed prompt 0.1%; one avoided API call ≈9.3% of a session; cold cache +76% |

Cost model used by `smarty audit`: weighted cost = input + 1.25×cache_write + 0.10×cache_read + 5×output (the weights are assumptions; override them on the command line).

---

## Safety guarantees

Each has tests.

| Mechanism | Prevents |
|---|---|
| Protected spans (fenced/inline code, URLs, emails, paths, identifiers, quoted strings, JSON, digit tokens, negation clauses) | The gate altering anything load-bearing; spans are re-verified after compression, with fallback to the original text |
| Deterministic, causal transforms | Breaking provider prompt-cache prefixes |
| Fail-open | A gate bug blocking or corrupting a request: the exact original bytes are forwarded |
| Never forward a larger request | Negative savings |
| Negation-safe compression; negation spans kept verbatim in `<task_metadata>` | Dropping `not`, `never`, `without`, `unless`, `neither`, `nor` |
| `<extracted_entities>` block | Mutation of exact dates, emails, URLs, quoted strings |
| Pre-LLM slot check | Spending tokens when a required parameter is missing |
| SimHash cache safety gate | Near-duplicate hits when intent, negations or entities differ |
| Abstention rule in the template (`STATUS: INSUFFICIENT_CONTEXT`) | The agent fabricating when data is missing |
| XML tag sanitization | User input injecting or overriding reserved template tags |
| Credentials forwarded untouched, never logged or stored; counts-only ledger | Secret and prompt leakage through the gateway |

---

## Project layout

```
NLP-Smarty/
├── contextgate_core/            # the importable package (this is all the wheel installs)
│   ├── gate.py  gateway.py  ledger.py  gate_bench.py           # gateway surface
│   ├── pipeline.py  data_contracts.py  registry.py  protocol.py     # optimizer + contracts
│   ├── repo_indexer.py  hook.py  mcp_server.py  cli.py        # repo-intelligence surface
│   ├── bridge.py  downstream.py                                 # VS Code sidecar, optional HF consumer
│   └── token_audit.py  ab_compare.py  eval_retrieval.py         # measurement
├── smarty_cli.py  smarty_hook.py  mcp_server.py  bridge_server.py  gateway.py  ...   # thin root launchers (see below)
├── src/extension.ts             # VS Code extension
├── test_harness.py  tests/      # benchmark + pytest suites
├── tools/  (bump_version.py, check_release.py)  ·  docs/RESTRUCTURE.md
├── .github/workflows/  (ci.yml, release.yml)
└── package.json  pyproject.toml  tsconfig.json  .vscodeignore
```

The root scripts are 10-line launchers that call the package (`python smarty_hook.py` runs `contextgate_core.hook`). They exist because `smarty init`
registers absolute script paths and recognises its own entries by filename, and the extension spawns `bridge_server.py`; they keep existing installs working.
They are not part of the wheel. Rationale and remaining phases: [`docs/RESTRUCTURE.md`](docs/RESTRUCTURE.md).

## Dependencies

| Package | Purpose | Fallback |
|---|---|---|
| none (stdlib) | Gateway, gate, ledger, hook, MCP, CLI, indexer | n/a |
| `spacy` + `en_core_web_sm` (NER disabled) | POS tagging, root verb extraction | Regex sentence splitting + `_KNOWN_TASK_VERBS` |
| `rank_bm25` | BM25 scoring | Built-in `_FastBM25Okapi` |
| `scikit-learn` | TF-IDF utilities | Built-in sparse TF-IDF + cosine |
| `flashtext` | Keyword matching | Weighted regex patterns |
| `huggingface_hub` | Live HF calls in `contextgate_core/downstream.py` | `urllib` HTTP, then offline simulator |

Extension build: `typescript ^5.4.0`, `@types/vscode ^1.90.0`, `@types/node ^20.0.0`.

## Tests

| Command | Covers |
|---|---|
| `python3 -m pytest -q --tb=short tests` | Gate fuzzing and prefix stability, gateway (OpenAI + Anthropic), adversarial inputs (`test_security.py`), ledger, hook, MCP, CLI, release hygiene, extension manifest, README claims guard (`test_claims.py`) |
| `python3 test_harness.py \| tail -n 30` | Optimizer benchmark (5 baseline prompts) and 7 regression checks: normalization, extractive compression, MMR context pruning, history pruning, schema compression, cache hit on paraphrase, missing-slot short-circuit |
| `python3 tools/check_release.py` | Version alignment, no junk tracked, packaging audit |
| `npm run compile` | Extension type-check |

## Extending the registry

Edit `registry.py`; `pipeline.py` needs no changes. The registry currently holds 4 tools, 6 few-shot examples and 3 guardrails.

```python
TOOL_REGISTRY.append({
    "id": "query_database",
    "trigger_keywords": ["database", "db", "table", "select", "insert"],
    "schema": json.dumps({
        "name": "query_database",
        "description": "Executes a read-only SQL query against the database.",
        "parameters": {"type": "object",
                       "properties": {"sql": {"type": "string", "description": "SQL SELECT statement"},
                                      "limit": {"type": "integer", "description": "Max rows to return", "default": 100}},
                       "required": ["sql"]},
    }),
})

FEW_SHOT_EXAMPLES.append({
    "id": "ex_code_gen_3", "intent": "CODE_GENERATION",
    "input": "Write a binary search function in Python",
    "output": "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    ...",
})
```

## Releasing

Versions live in four places (`pyproject.toml`, `protocol.py`, `package.json`, `package-lock.json`); never edit them by hand:

```bash
python3 tools/bump_version.py 0.7.0      # updates all four and verifies them
git commit -am "release 0.7.0" && git tag v0.7.0 && git push origin main v0.7.0
```

Pushing the tag runs [`release.yml`](.github/workflows/release.yml): verify (versions match the tag, no junk tracked, tests) → build wheel + sdist and the `.vsix` (audited by `tools/check_release.py`; the packaged engine is smoke-tested) → GitHub Release with all artifacts attached. **Actions → Release → Run workflow** is a dry run that builds and checks everything and publishes nothing.

One-time setup for publishing (nothing publishes until you do these):

- **VS Code Marketplace:** the publisher ID must equal `publisher` in `package.json` (`KamVinTech-contextgate`) and is permanent once created. Add a credential as the repository secret `VSCE_PAT` (scope *Marketplace > Manage*). Global Azure DevOps tokens retire on 2026-12-01; later releases need Microsoft Entra ID (`vsce publish --azure-credential`). The Marketplace step is **opt-in**: it runs only when the repository variable `PUBLISH_MARKETPLACE=true` is set (and `VSCE_PAT` exists); otherwise it is skipped and the `.vsix` is still attached to the GitHub Release.
- **PyPI (opt-in):** the distribution is `contextgate-core`. One-time setup on pypi.org: *Publishing → Add a pending publisher* (project `contextgate-core`, owner `KamVinTech`, repository `NLP-Smarty`, workflow `release.yml`, environment `pypi`); create the GitHub environment `pypi`; set the repository variable `PUBLISH_PYPI=true`. No API token is stored anywhere. The first tag after that publishes it.

## Privacy and license

The engine writes an audit log of prompt previews only when `SMARTY_AUDIT_LOG=<path>` is set; by default nothing is persisted. The gateway ledger stores counts only. License: MIT (see `LICENSE`).
