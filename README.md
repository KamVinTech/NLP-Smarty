# NLP-Smarty — Deterministic Prompt Optimization Pipeline + VS Code Extension

A production-grade, **zero-LLM** middleware layer that intercepts raw user input, cleans and compresses it, extracts intent and entities using classical NLP, and compiles a strictly structured, anti-hallucination prompt for a downstream AI Agent — now with a full **VS Code extension** for side-by-side Raw vs. Optimized LLM evaluation.

---

## How It Works

```
Raw User Input
      │
      ▼
┌──────────────────────────────────────────┐
│  Step 1: Token-Aware Budgeting           │  BPE-calibrated estimator, per-section budget
│  Step 8: Adaptive Intensity Selection    │  AUTO | LIGHT | MEDIUM | AGGRESSIVE scaling
│  Step 5: Lexical Normalization           │  NFKC, contraction expansion, verbose-idiom replacement
│  Step 4: Extractive Compression          │  Composite BM25 + TF-IDF Centroid + U-Positional Salience
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Stage 2: Weighted Intent Router         │  Weighted regex + spaCy POS/dependency parse
│  → TaskIntent + confidence               │  Secondary intent detection, task_hint override
│  → Entities + Negation Spans             │  Early short-circuit if required slots missing
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Step 6: Deterministic Prompt Cache      │  SHA-256 canonical + 64-bit SimHash LSH near-duplicate
│  → Cache Hit → return immediately        │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Step 2: Tool Schema Compression         │  JSON Schema → compact typed signature compiler
│  Step 3: MMR + BM25 + TF-IDF Pruner     │  Few-shots, tools, external context, history
│  Step 7: Conversation History Pruning    │  Recency-weighted MMR turn compaction
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  Stage 4: Structural Compiler            │  Fills structured XML template
│  Step 9: Full Payload Telemetry          │  Before/after token counts for query + full payload
│  → CompiledPrompt payload                │  Injects guardrails + output format constraints
└──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│  bridge_server.py  (JSON-Lines sidecar)  │  Persistent stdin/stdout process, 4 commands
└────────────────────┬─────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   vscode.lm (Copilot)    app.py (HuggingFace)
   Raw + Optimized A/B    InferenceClient / HTTP
   via extension.ts       offline simulator fallback
```

---

## Project Structure

```
NLP/
├── pipeline.py          # Core 9-step deterministic engine
├── data_contracts.py    # Typed dataclasses: RawRequest, CompiledPrompt, OptimizationTrace, TokenBudget
├── registry.py          # Static tools (verbose JSON Schema), few-shot examples, guardrails, required slots
├── app.py               # HuggingFace downstream agent — Raw vs. Optimized A/B comparison + audit
├── bridge_server.py     # Persistent JSON-Lines sidecar (optimize / compare_hf / audit_pair / clear_cache)
├── src/
│   └── extension.ts     # VS Code extension — 3 commands + @smarty chat participant
├── test_harness.py      # Benchmarking harness — Part A (5 baseline) + Part B (7 regression tests)
├── package.json         # VS Code extension manifest (v0.4.0)
├── tsconfig.json        # TypeScript compiler config (ES2022, strict, @types/node)
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Python pipeline

```bash
python3 -m venv NLP-start
source NLP-start/bin/activate   # Windows: NLP-start\Scripts\activate
pip install spacy rank_bm25 flashtext scikit-learn
python -m spacy download en_core_web_sm
python test_harness.py
```

> All Python dependencies have graceful pure-Python fallbacks — the pipeline runs even without `spacy`, `flashtext`, or `rank_bm25`.

### 2. VS Code extension

```bash
npm install
npm run compile        # tsc -p ./
# Press F5 in VS Code to launch Extension Development Host
```

### 3. HuggingFace live mode (optional)

```bash
export HF_TOKEN=hf_...          # your HuggingFace API token
export HF_MODEL_ID=Qwen/Qwen2.5-Coder-32B-Instruct   # default model
python app.py                   # runs a sample A/B comparison
```

---

## Usage — Python API

```python
from data_contracts import RawRequest, TokenBudget
from pipeline import optimize

result = optimize(RawRequest(
    text="Could you please write a Python function to reverse a string without using slicing?",
    context_chunks=["...optional RAG chunks..."],
    conversation_history=[{"role": "user", "content": "...prior turn..."}],
    token_budget=TokenBudget(max_total_tokens=512),
    compression_intensity="AUTO",
    enable_cache=True,
))

print(result.status)    # "OK"
print(result.payload)   # Compiled XML prompt → feed to downstream agent
print(result.trace)     # Full audit trail with before/after token telemetry
```

### Status values

| Status | Meaning |
|---|---|
| `OK` | Compiled prompt ready for downstream agent |
| `NEEDS_CLARIFICATION` | Required entity slot missing — returns clarification question, zero downstream tokens spent |
| `INSUFFICIENT_CONTEXT` | Agent signals it cannot answer from provided context |

---

## Usage — VS Code Extension

Three commands are registered, all accessible from the Command Palette, editor right-click menu, or keyboard shortcuts:

| Command | Shortcut (Mac) | Shortcut (Win/Linux) | What it does |
|---|---|---|---|
| `NLP-Smarty: Optimize Prompt & Active File Context` | `Cmd+Alt+O` | `Ctrl+Alt+O` | Runs the 9-step pipeline on selected text or a typed prompt, copies compiled XML to clipboard, opens telemetry webview |
| `NLP-Smarty: Compare Raw vs. Optimized LLM Output` | `Cmd+Alt+C` | `Ctrl+Alt+C` | Runs Raw + Optimized prompts side-by-side on `vscode.lm` (Copilot) or HuggingFace, opens A/B compliance webview |
| `NLP-Smarty: Compress Selected Prompt Inline` | `Cmd+Alt+I` | `Ctrl+Alt+I` | Replaces selected text in editor with the compressed `<core_task>` output |

### `@smarty` Chat Participant

Use `@smarty` in the VS Code Chat panel:

```
@smarty /optimize Could you please write a Python function to reverse a string without using slicing?
@smarty /compare  Search the web for the latest Python 3.12 release notes
@smarty           <any prompt>   ← defaults to side-by-side A/B comparison
```

---

## VS Code Extension Settings

| Setting | Default | Description |
|---|---|---|
| `nlpSmarty.pythonPath` | `python3` | Path to Python interpreter inside your `NLP-start` venv |
| `nlpSmarty.engineDirectory` | `""` | Directory containing `bridge_server.py`, `app.py`, `pipeline.py` |
| `nlpSmarty.downstreamBackend` | `auto` | `auto` \| `vscode-lm` \| `huggingface` |
| `nlpSmarty.huggingFaceToken` | `""` | HuggingFace API token (`HF_TOKEN`) |
| `nlpSmarty.huggingFaceModel` | `Qwen/Qwen2.5-Coder-32B-Instruct` | HuggingFace model ID |
| `nlpSmarty.compressionIntensity` | `AUTO` | `AUTO` \| `LIGHT` \| `MEDIUM` \| `AGGRESSIVE` |
| `nlpSmarty.maxTokenBudget` | `512` | Hard token ceiling for compiled payload |

---

## Bridge Server (`bridge_server.py`)

Persistent JSON-Lines sidecar spawned by `extension.ts`. Communicates over `stdin`/`stdout` — one JSON object per line.

### Commands

| Command | Description |
|---|---|
| `optimize` | Runs the 9-step deterministic pipeline, returns `payload` + `trace` |
| `compare_hf` | Runs pipeline + forwards Raw & Compiled prompts to `app.py` (HuggingFace or offline simulator) |
| `audit_pair` | Deterministically audits Raw vs. Optimized responses generated by `vscode.lm` |
| `clear_cache` | Clears `PROMPT_CACHE` (256-entry LRU) |
| `ping` | Health check — returns `pong` |

### Workspace file chunking

`bridge_server.py` splits open workspace files into semantic chunks (≤45 words each, split on blank lines and `def`/`class`/`function`/`export`/`interface` boundaries) before passing them as `context_chunks` to Stage 3 MMR pruning — so only the relevant slices of real files are injected into the compiled prompt.

---

## HuggingFace Consumer (`app.py`)

Standalone A/B evaluation script. Supports three execution modes:

| Mode | Condition | Backend |
|---|---|---|
| Live | `HF_TOKEN` set + `huggingface_hub` installed | `InferenceClient` |
| Live (fallback) | `HF_TOKEN` set, no `huggingface_hub` | `urllib.request` HTTP router |
| Offline | No `HF_TOKEN` | Deterministic simulator (always works for UI/CLI testing) |

```python
from data_contracts import RawRequest
from app import compare_raw_vs_optimized

result = compare_raw_vs_optimized(RawRequest(
    text="Could you please write a Python function to reverse a string without using slicing?"
))
# result keys: status, backend, raw_input_tokens, optimized_input_tokens,
#              raw_response, raw_llm_latency_ms, raw_audit,
#              optimized_payload, optimized_response, optimized_llm_latency_ms,
#              optimized_audit, trace
```

---

## Pipeline Steps

### Step 1 — Token-Aware Budgeting
- Pure-Python BPE-calibrated subword token estimator (`estimate_tokens`) — matches `cl100k_base` within ~3%
- `TokenBudget` enforces per-section ceilings: `max_total_tokens`, `core_query_budget`, `context_budget`, `history_budget`, `tools_budget`, `few_shot_budget`
- Hard ceiling enforced at compile time — trims examples first if payload exceeds `max_total_tokens`

### Step 2 — Tool Schema Compression
- Compresses verbose JSON Schema into compact typed signatures: `tool_name(param:str, opt_param?:int=5) // short desc`
- Saves 20–40 tokens per tool while preserving full type and optionality information

### Step 3 — MMR + BM25 + TF-IDF Cosine Pruner
- Hybrid relevance: `0.65 × Normalized_BM25 + 0.35 × TF-IDF_Cosine`, TF-IDF-only fallback when BM25 score is zero
- MMR diversity selection prevents near-duplicate examples or context chunks
- Applied to: few-shot examples, tool schemas, RAG context chunks, conversation history, and workspace file chunks

### Step 4 — Composite Extractive Sentence Compressor
- Scores sentences: `45% BM25 + 35% TF-IDF Centroid + 20% U-shaped positional weight`
- Hard salience boosts (+100) for negations, constraints, code blocks, URLs, emails — never pruned
- Only activates when input exceeds `core_query_budget` words

### Step 5 — Lexical Normalization
- NFKC Unicode normalization + smart quote replacement
- 14 contraction expansions: `can't → cannot`, `I'd → I would`, etc.
- 10 verbose idiom replacements: `in order to → to`, `due to the fact that → because`, etc.
- Deterministic morphological stemmer (`_fast_stem`) — no neural models

### Stage 2 — Weighted Intent Router
- Weighted keyword scoring — multi-word phrases like `"python function"` score 2.5×
- Lead-verb position bonus: match in first 3 words doubles score
- Secondary intent detection, `task_hint` override, spaCy root verb extraction with `_KNOWN_TASK_VERBS` fallback

| Intent | Example Trigger Keywords |
|---|---|
| `CODE_GENERATION` | write, generate, implement, function, sql query, python function… |
| `DATA_EXTRACTION` | extract, parse, find all, scrape, regex, extract all… |
| `TOOL_EXECUTION` | run, execute, search the web, read file, invoke… |
| `FACTUAL_QA` | what is, who is, when did, capital of, why does… |
| `AMBIGUOUS_QUERY` | fallback when no keywords match |

### Step 6 — Deterministic Prompt Cache
- Tier 1 — SHA-256 canonical exact cache: O(1) lookup
- Tier 2 — 64-bit SimHash LSH near-duplicate cache: Hamming distance ≤ 2, gated by safety signature matching 100% on `intent`, `negations`, `entities`, `context_len`
- LRU eviction at 256 entries; `enable_cache=False` bypasses both tiers
- Cache hits return in <1.5ms

### Step 7 — Conversation History Pruning
- Most recent turn always retained
- Older turns ranked via Recency-Weighted MMR: `score = (hybrid_relevance + 0.15) × 0.85^steps_back`
- Injected as `<conversation_history>` XML block within `history_budget`

### Step 8 — Adaptive Intensity Selection
- `AUTO` selects `LIGHT / MEDIUM / AGGRESSIVE` based on token pressure against `core_query_budget`
- `MEDIUM`: strips hedging adverbs (`basically`, `actually`, `essentially`, etc.)
- `AGGRESSIVE`: removes articles (`a`, `an`, `the`) when input exceeds 25 words

### Step 9 — Comprehensive Token Telemetry
- Query-level and full-payload-level before/after token counts
- `OptimizationTrace` reports `query_token_savings_pct` and `payload_token_savings_pct`

---

## Data Contracts

### `RawRequest`
| Field | Type | Default | Purpose |
|---|---|---|---|
| `text` | `str` | required | Raw user input |
| `task_hint` | `str` | `""` | Force a specific intent label |
| `context_chunks` | `list[str]` | `[]` | External RAG chunks for MMR pruning |
| `conversation_history` | `list[dict]` | `[]` | Prior turns `[{"role": "user\|assistant", "content": "..."}]` |
| `token_budget` | `TokenBudget` | defaults | Per-section token ceilings |
| `compression_intensity` | `str` | `"AUTO"` | `AUTO \| LIGHT \| MEDIUM \| AGGRESSIVE` |
| `enable_cache` | `bool` | `True` | Toggle deterministic prompt cache |

### `TokenBudget`
| Field | Default | Controls |
|---|---|---|
| `max_total_tokens` | 512 | Hard ceiling for entire compiled payload |
| `core_query_budget` | 120 | Max words in compressed query |
| `context_budget` | 160 | Max tokens for pruned RAG context |
| `history_budget` | 80 | Max tokens for pruned conversation history |
| `tools_budget` | 60 | Max tokens for compressed tool schemas |
| `few_shot_budget` | 90 | Max tokens for selected few-shot examples |

### `OptimizationTrace`
```python
OptimizationTrace(
    raw_token_count=16,           optimized_token_count=13,
    query_token_savings_pct=18.8,
    unoptimized_payload_token_count=420,
    compiled_payload_token_count=310,
    payload_token_savings_pct=26.2,
    tool_schema_tokens_saved=34,
    detected_intent="CODE_GENERATION",
    secondary_intents=[],         intent_confidence=1.0,
    compression_intensity_used="MEDIUM",
    pruning_strategy_used="MMR+BM25+TFIDF",
    stripped_tokens=["Could you please"],
    normalized_phrases=[],        negations=["without"],
    extracted_entities={"action": "write"},
    injected_example_ids=["ex_code_gen_1", "ex_code_gen_2"],
    injected_tool_ids=["run_python"],
    kept_context_indices=[],      kept_history_turns=0,
    cache_hit=False,              cache_type="NONE",
    cache_key="a3f1c9...",        simhash_signature="0a1b2c3d4e5f6789",
    latency_ms=3.81,
)
```

---

## Anti-Hallucination Guarantees

| Mechanism | What it prevents |
|---|---|
| Negation-safe compression (Step 5) | Optimizer never deletes `not`, `never`, `without`, `unless`, `neither`, `nor` |
| Negation span extraction | Full constraint phrases like `"without using recursion"` preserved verbatim in `<task_metadata>` |
| `<extracted_entities>` block | Agent receives exact dates, emails, URLs, quoted strings verbatim — no mutation |
| MMR tool pruning (Step 3) | Agent only sees relevant, deduplicated tool schemas — no irrelevant tool calls |
| Tool schema compression (Step 2) | Compact signatures reduce token noise without losing type or optionality |
| Pre-LLM slot check | LLM never called when required parameters are missing — zero tokens wasted |
| SimHash cache safety gate | Near-duplicate cache hits blocked when intent, negations, or entities differ |
| `INSUFFICIENT_CONTEXT` escape hatch | Agent abstains instead of fabricating when data is missing |
| XML tag sanitization | User input cannot inject or override reserved XML template tags |
| Deterministic response audit | `audit_agent_response()` scores compliance against extracted negations and entities |

---

## Test Harness

`test_harness.py` runs two suites:

**Part A — 5 Baseline Prompts**: full before/after telemetry (query tokens + payload tokens + schema savings + latency).

**Part B — 7 Regression Tests**:

| Test | Feature Validated |
|---|---|
| 6 | Lexical normalization + adaptive MEDIUM intensity |
| 7 | Extractive sentence compression + AGGRESSIVE intensity |
| 8 | MMR external context pruning (deduplicating near-duplicate RAG chunks) |
| 9 | Conversation history pruning (recency-weighted MMR) |
| 10 | Tool schema compression + morphological stemming match |
| 11 | Deterministic canonical cache hit on paraphrased polite prefix |
| 12 | Missing mandatory slot early short-circuit (zero downstream tokens) |

---

## Extending the Registry

Edit `registry.py` — no changes to `pipeline.py` needed.

```python
# Add a tool (verbose JSON Schema enables Step 2 compression)
TOOL_REGISTRY.append({
    "id": "query_database",
    "trigger_keywords": ["database", "db", "table", "select", "insert"],
    "schema": json.dumps({
        "name": "query_database",
        "description": "Executes a read-only SQL query against the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT statement"},
                "limit": {"type": "integer", "description": "Max rows to return", "default": 100},
            },
            "required": ["sql"],
        },
    }),
})

# Add a few-shot example (intent tag enables MMR boosting in Step 3)
FEW_SHOT_EXAMPLES.append({
    "id": "ex_code_gen_3",
    "intent": "CODE_GENERATION",
    "input": "Write a binary search function in Python",
    "output": "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    ...",
})
```

---

## Dependencies

### Python
| Package | Purpose | Fallback |
|---|---|---|
| `spacy` + `en_core_web_sm` | POS tagging, dependency parsing, root verb extraction | Regex sentence splitting + `_KNOWN_TASK_VERBS` dict |
| `rank_bm25` | BM25Okapi scoring | Built-in `_FastBM25Okapi` zero-dependency implementation |
| `scikit-learn` | TF-IDF utilities | Built-in sparse TF-IDF + cosine implementation |
| `flashtext` | O(n) trie keyword matching | Replaced by weighted regex patterns in v3 |
| `huggingface_hub` | `InferenceClient` for live HF API calls | `urllib.request` HTTP fallback → offline simulator |

### TypeScript / VS Code Extension
| Package | Purpose |
|---|---|
| `@types/vscode ^1.90.0` | VS Code extension API types |
| `@types/node ^20.0.0` | Node.js built-in types (`child_process`, `path`, `readline`, `buffer`, `process`) |
| `typescript ^5.4.0` | TypeScript compiler |
