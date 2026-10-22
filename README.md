# Deterministic Prompt Optimization Pipeline

A production-grade, **zero-LLM** middleware layer that intercepts raw user input, cleans and compresses it, extracts intent and entities using classical NLP, and compiles a strictly structured, anti-hallucination prompt for a downstream AI Agent.

---

## How It Works

```
Raw User Input
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Intent Router     │  FlashText trie + spaCy POS/dependency parse
│  → TaskIntent + confidence  │  Early short-circuit if required slots missing
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: Compressor        │  Regex filler-strip (negation-safe)
│  → Cleaned, dense query     │  BM25 sentence pruning at token budget
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 3: BM25 Pruner       │  Top-K few-shot examples via BM25Okapi
│  → Relevant tools + shots   │  Keyword-matched tool schemas
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 4: Compiler          │  Fills structured XML template
│  → CompiledPrompt payload   │  Injects guardrails + output format constraints
└─────────────────────────────┘
             │
             ▼
   Downstream AI Agent
```

---

## Project Structure

```
NLP/
├── app.py               # HuggingFace downstream agent (consumer of compiled prompts)
├── pipeline.py          # Core 4-stage deterministic engine
├── data_contracts.py    # Typed dataclasses: RawRequest, CompiledPrompt, OptimizationTrace
├── registry.py          # Static tools, few-shot examples, guardrails, required slots
├── test_harness.py      # Benchmarking harness — before/after token stats + latency
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

### 3. Run the verification harness

```bash
python test_harness.py
```

---

## Usage

```python
from data_contracts import RawRequest
from pipeline import optimize

result = optimize(RawRequest(text="Could you please write a Python function to reverse a string without using slicing?"))

print(result.status)          # "OK"
print(result.payload)         # Compiled XML prompt → feed to your downstream agent
print(result.trace)           # Full audit trail
```

### Status values

| Status | Meaning |
|---|---|
| `OK` | Compiled prompt ready for downstream agent |
| `NEEDS_CLARIFICATION` | Required entity slot missing — returns clarification question instead |
| `INSUFFICIENT_CONTEXT` | Agent signals it cannot answer from provided context |

---

## Pipeline Stages

### Stage 1 — Intent Router
- Uses **FlashText** trie matching (O(n)) to score keyword hits across 4 intent classes
- Runs **spaCy** dependency parse to extract the root verb as the `action` entity
- Short-circuits immediately with a clarification prompt if a required slot is absent — no LLM call wasted

| Intent | Trigger Keywords |
|---|---|
| `CODE_GENERATION` | write, generate, create, implement, function, sql, query… |
| `DATA_EXTRACTION` | extract, parse, find all, scrape, regex, identify… |
| `TOOL_EXECUTION` | run, execute, search, send, call, invoke… |
| `FACTUAL_QA` | what is, who is, when did, define, explain, tell me… |
| `AMBIGUOUS_QUERY` | fallback when no keywords match |

### Stage 2 — Prompt Compressor
- Strips conversational filler (`"Could you please..."`, `"I was wondering if..."`) via regex
- **Negation-safe**: protects `not`, `never`, `without`, `except` and quoted strings behind placeholders before stripping
- Runs **BM25 sentence scoring** against query terms to prune long inputs to a 120-token budget

### Stage 3 — BM25 Pruner
- Scores all few-shot examples in `registry.py` against the cleaned query using `BM25Okapi`
- Returns only the **top-K=2** most relevant examples
- Matches tools by trigger keyword overlap — no irrelevant tool schemas injected

### Stage 4 — Structural Compiler
- Fills a rigid XML template with all extracted data
- Injects domain guardrails from `registry.py` verbatim
- Infers the required output format from the detected intent

---

## Audit Trail — `OptimizationTrace`

Every call produces a full trace object:

```python
OptimizationTrace(
    raw_token_count=16,
    optimized_token_count=13,
    detected_intent="CODE_GENERATION",
    intent_confidence=1.0,
    stripped_tokens=["Could you please"],
    negations=["without"],
    extracted_entities={"action": "write"},
    injected_example_ids=["ex_code_gen_1", "ex_code_gen_2"],
    injected_tool_ids=["run_python"],
    latency_ms=3.81
)
```

---

## Extending the Registry

Edit `registry.py` to add your own tools, few-shot examples, and guardrails — no code changes needed elsewhere.

```python
# Add a tool
TOOL_REGISTRY.append({
    "id": "query_database",
    "trigger_keywords": ["database", "db", "table", "select", "insert"],
    "schema": '{"name":"query_database","params":{"sql":"string"}}',
})

# Add a few-shot example
FEW_SHOT_EXAMPLES.append({
    "id": "ex_sql_1",
    "input": "Count all active users in the database",
    "output": "SELECT COUNT(*) FROM users WHERE status = 'active';",
})
```

---

## Anti-Hallucination Guarantees

| Mechanism | What it prevents |
|---|---|
| Negation-safe compression | Optimizer never deletes `not`, `never`, `without` constraints |
| `<extracted_entities>` block | Agent receives exact dates, IDs, emails verbatim — no mutation |
| BM25 tool pruning | Agent only sees relevant tool schemas — no irrelevant tool calls |
| Pre-LLM slot check | LLM never guesses missing required parameters |
| `INSUFFICIENT_CONTEXT` escape hatch | Agent abstains instead of fabricating when data is missing |

---

## Dependencies

| Package | Purpose |
|---|---|
| `spacy` + `en_core_web_sm` | POS tagging, dependency parsing, root verb extraction |
| `flashtext` | O(n) trie-based keyword matching for intent routing |
| `rank_bm25` | BM25Okapi scoring for few-shot and sentence pruning |
| `scikit-learn` | TF-IDF utilities (available for extension) |
