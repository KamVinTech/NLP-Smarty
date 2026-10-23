# Deterministic Prompt Optimization Pipeline

A production-grade, **zero-LLM** middleware layer that intercepts raw user input, cleans and compresses it, extracts intent and entities using classical NLP, and compiles a strictly structured, anti-hallucination prompt for a downstream AI Agent.

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
│  Stage 2: Weighted Intent Router         │  Weighted FlashText + regex + spaCy POS/dependency parse
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
            Downstream AI Agent
```

---

## Project Structure

```
NLP/
├── pipeline.py          # Core 9-step deterministic engine
├── data_contracts.py    # Typed dataclasses: RawRequest, CompiledPrompt, OptimizationTrace, TokenBudget
├── registry.py          # Static tools (verbose JSON Schema), few-shot examples, guardrails, required slots
├── test_harness.py      # Benchmarking harness — Part A (5 baseline) + Part B (7 regression tests)
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Create & activate virtual environment

```bash
python3 -m venv NLP-start
source NLP-start/bin/activate   # Windows: NLP-start\Scripts\activate
```

### 2. Install dependencies

```bash
pip install spacy rank_bm25 flashtext scikit-learn
python -m spacy download en_core_web_sm
```

> All dependencies have graceful pure-Python fallbacks — the pipeline runs even if `spacy`, `flashtext`, or `rank_bm25` are unavailable.

### 3. Run the verification harness

```bash
python test_harness.py
```

---

## Usage

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

print(result.status)          # "OK"
print(result.payload)         # Compiled XML prompt → feed to your downstream agent
print(result.trace)           # Full audit trail with before/after token telemetry
```

### Status values

| Status | Meaning |
|---|---|
| `OK` | Compiled prompt ready for downstream agent |
| `NEEDS_CLARIFICATION` | Required entity slot missing — returns clarification question, zero downstream tokens spent |
| `INSUFFICIENT_CONTEXT` | Agent signals it cannot answer from provided context |

---

## Pipeline Steps

### Step 1 — Token-Aware Budgeting
- Pure-Python BPE-calibrated subword token estimator (`estimate_tokens`) — matches `cl100k_base` within ~3% with zero external dependencies
- `TokenBudget` dataclass enforces per-section ceilings: `max_total_tokens`, `core_query_budget`, `context_budget`, `history_budget`, `tools_budget`, `few_shot_budget`
- Hard ceiling enforced at compile time — progressively trims optional blocks (examples first) if payload exceeds `max_total_tokens`

### Step 2 — Tool Schema Compression
- Compresses verbose JSON Schema tool definitions into compact typed signatures: `tool_name(param:str, opt_param?:int=5) // short desc`
- Saves 20–40 tokens per tool schema while preserving full type and optionality information
- Integrated into Stage 3 tool selection — only matched tools are compressed and injected

### Step 3 — MMR + BM25 + TF-IDF Cosine Pruner
- Hybrid relevance scoring: `0.65 × Normalized_BM25 + 0.35 × TF-IDF_Cosine`, with smooth TF-IDF-only fallback when BM25 score is zero
- **Maximal Marginal Relevance (MMR)** diversity selection prevents injecting near-duplicate examples or context chunks
- Applied independently to: few-shot examples, tool schemas, external RAG context chunks, and conversation history turns
- All scoring uses stemmed tokens + character 4-gram features for robust partial-match coverage

### Step 4 — Composite Extractive Sentence Compressor
- Scores sentences using: `45% BM25 + 35% TF-IDF Centroid + 20% U-shaped positional weight`
- Hard salience boosts (+100) for sentences containing negations, constraints (`must`, `only`, `format`), code blocks, URLs, or emails — these are never pruned
- First sentence always gets a +25 boost to preserve the command framing
- Only activates when input exceeds `core_query_budget` words

### Step 5 — Lexical Normalization
- NFKC Unicode normalization + smart quote replacement
- Contraction expansion: `can't → cannot`, `I'd → I would`, etc. (14 patterns)
- Verbose idiom replacement: `in order to → to`, `due to the fact that → because`, `make use of → use`, etc. (10 patterns)
- Deterministic morphological stemmer (`_fast_stem`) for BM25/TF-IDF token normalization — no neural models

### Stage 2 — Weighted Intent Router
- Weighted keyword scoring (not binary) — multi-word phrases like `"python function"` score 2.5×, single keywords score 1.0–2.0×
- Lead-verb position bonus: keyword match in first 3 words doubles its score
- Secondary intent detection — reports intents scoring ≥50% of the top intent
- `task_hint` override: caller can force a specific intent label, bypassing scoring entirely
- spaCy dependency parse extracts root verb as `action` entity with `_KNOWN_TASK_VERBS` dictionary fallback

| Intent | Example Trigger Keywords |
|---|---|
| `CODE_GENERATION` | write, generate, implement, function, sql query, python function… |
| `DATA_EXTRACTION` | extract, parse, find all, scrape, regex, extract all… |
| `TOOL_EXECUTION` | run, execute, search the web, read file, invoke… |
| `FACTUAL_QA` | what is, who is, when did, capital of, why does… |
| `AMBIGUOUS_QUERY` | fallback when no keywords match |

### Step 6 — Deterministic Prompt Cache
- **Tier 1 — Canonical Exact Cache**: SHA-256 over `(normalized_query + intent + entities + context_len + budget)` — O(1) lookup
- **Tier 2 — SimHash Near-Duplicate Cache**: 64-bit SimHash LSH with Hamming distance ≤ 2, gated by a safety signature that must match 100% on `intent`, `negations`, `entities`, `context_len` — prevents false cache hits on semantically different queries
- LRU eviction at 256 entries; `enable_cache=False` on `RawRequest` bypasses both tiers
- Cache hits return in <1.5ms with `trace.cache_hit=True` and `trace.cache_type` set

### Step 7 — Conversation History Pruning
- Most recent turn is always retained for continuity
- Older turns are ranked via Recency-Weighted Hybrid MMR: `score = (hybrid_relevance + 0.15) × 0.85^steps_back`
- User turns are compressed via `_normalize_and_compress_text` before scoring
- Pruned history injected as `<conversation_history>` XML block within `history_budget`

### Step 8 — Adaptive Intensity Selection
- `AUTO` mode automatically selects `LIGHT / MEDIUM / AGGRESSIVE` based on token pressure against `core_query_budget`
- `MEDIUM`: additionally strips hedging adverbs (`basically`, `actually`, `essentially`, etc.)
- `AGGRESSIVE`: additionally removes articles (`a`, `an`, `the`) when input exceeds 25 words
- Caller can override with explicit `compression_intensity="AGGRESSIVE"` on `RawRequest`

### Step 9 — Comprehensive Token Telemetry
- Measures both **query-level** and **full-payload-level** before/after token counts
- Baseline payload is computed using all raw tools + all examples + full uncompressed context/history
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

### `OptimizationTrace` (Step 9 Telemetry)
```python
OptimizationTrace(
    # Query-level
    raw_token_count=16,
    optimized_token_count=13,
    query_token_savings_pct=18.8,
    # Payload-level
    unoptimized_payload_token_count=420,
    compiled_payload_token_count=310,
    payload_token_savings_pct=26.2,
    tool_schema_tokens_saved=34,
    # Routing
    detected_intent="CODE_GENERATION",
    secondary_intents=[],
    intent_confidence=1.0,
    compression_intensity_used="MEDIUM",
    pruning_strategy_used="MMR+BM25+TFIDF",
    # Transformations
    stripped_tokens=["Could you please"],
    normalized_phrases=[],
    negations=["without"],
    extracted_entities={"action": "write"},
    injected_example_ids=["ex_code_gen_1", "ex_code_gen_2"],
    injected_tool_ids=["run_python"],
    kept_context_indices=[],
    kept_history_turns=0,
    # Cache
    cache_hit=False,
    cache_type="NONE",
    cache_key="a3f1c9...",
    simhash_signature="0a1b2c3d4e5f6789",
    latency_ms=3.81,
)
```

---

## Registry

`registry.py` now uses **verbose JSON Schema** for tool definitions (enabling the Step 2 schema compressor) and **intent-tagged** few-shot examples (enabling intent-boosted MMR selection in Step 3).

### Adding a tool

```python
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
```

### Adding a few-shot example

```python
FEW_SHOT_EXAMPLES.append({
    "id": "ex_code_gen_3",
    "intent": "CODE_GENERATION",          # intent tag enables MMR boosting
    "input": "Write a binary search function in Python",
    "output": "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    ...",
})
```

---

## Anti-Hallucination Guarantees

| Mechanism | What it prevents |
|---|---|
| Negation-safe compression (Step 5) | Optimizer never deletes `not`, `never`, `without`, `unless`, `neither`, `nor` |
| Negation span extraction | Full constraint phrases like `"without using recursion"` preserved verbatim in `<task_metadata>` |
| `<extracted_entities>` block | Agent receives exact dates, emails, URLs, quoted strings verbatim — no mutation |
| MMR tool pruning (Step 3) | Agent only sees relevant, deduplicated tool schemas — no irrelevant tool calls |
| Tool schema compression (Step 2) | Compact signatures reduce token noise without losing type or optionality information |
| Pre-LLM slot check | LLM never called when required parameters are missing — zero tokens wasted |
| SimHash cache safety gate | Near-duplicate cache hits blocked when intent, negations, or entities differ |
| `INSUFFICIENT_CONTEXT` escape hatch | Agent abstains instead of fabricating when data is missing |
| XML tag sanitization | User input cannot inject or override reserved XML template tags |

---

## Test Harness

`test_harness.py` runs two suites:

**Part A — 5 Baseline Prompts**: original benchmark prompts with full before/after telemetry (query tokens + payload tokens + schema savings + latency).

**Part B — 7 Regression Tests**: validates each major enhancement with `assert_fn` checks:

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

## Dependencies

| Package | Purpose | Fallback |
|---|---|---|
| `spacy` + `en_core_web_sm` | POS tagging, dependency parsing, root verb extraction | Regex-based sentence splitting + `_KNOWN_TASK_VERBS` dict |
| `flashtext` | O(n) trie-based keyword matching | Not used in v3 (replaced by weighted regex patterns) |
| `rank_bm25` | BM25Okapi scoring | Built-in `_FastBM25Okapi` zero-dependency implementation |
| `scikit-learn` | TF-IDF utilities | Built-in sparse TF-IDF + cosine implementation |
