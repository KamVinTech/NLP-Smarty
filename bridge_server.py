"""
Persistent JSON-RPC / JSON-Lines Sidecar Bridge for VS Code Extension
=====================================================================
Why a persistent sidecar?
Spawning `python3` per click costs ~300-500ms of interpreter + spaCy startup time.
This persistent daemon loads `pipeline.py` and spaCy ONCE when VS Code activates,
then processes real-time editor prompts and workspace file chunks over stdin/stdout
in ~1-3ms per request.

Protocol:
  Request (1 line JSON on stdin):
    {
      "id": "req_1",
      "text": "Could you please help me...",
      "task_hint": "",
      "context_chunks": ["chunk 1...", "chunk 2..."],
      "workspace_files": [{"path": "app.py", "content": "..."}],
      "conversation_history": [{"role": "user", "content": "..."}],
      "compression_intensity": "AUTO",
      "max_total_tokens": 512,
      "core_query_budget": 120,
      "context_budget": 160,
      "history_budget": 80,
      "enable_cache": true
    }

  Response (1 line JSON on stdout):
    {
      "id": "req_1",
      "ok": true,
      "status": "OK",
      "clarification_request": "",
      "payload": "<agent_instructions>...",
      "trace": { ...full OptimizationTrace dict... }
    }
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import traceback
from typing import Any

from data_contracts import RawRequest, TokenBudget
from pipeline import PROMPT_CACHE, optimize


def chunk_workspace_file(path: str, content: str, max_words_per_chunk: int = 45) -> list[str]:
    """
    Splits a real workspace file (code, markdown, logs, JSON) into semantic chunks
    (by function/class/paragraph blocks or sliding window) so Stage 3 MMR can
    select only the relevant slices of real files.
    """
    if not content or not content.strip():
        return []

    # Split on double newlines or top-level def/class boundaries first
    raw_blocks = re.split(r"\n\s*\n|(?=\n(?:def |class |function |export |interface ))", content)
    chunks: list[str] = []
    current_words: list[str] = []

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        words = block.split()
        if len(current_words) + len(words) <= max_words_per_chunk:
            current_words.extend(words)
        else:
            if current_words:
                chunks.append(f"[{path}] " + " ".join(current_words))
                current_words = []
            # If a single block is huge, slice it into windows
            for i in range(0, len(words), max_words_per_chunk):
                slice_words = words[i : i + max_words_per_chunk]
                chunks.append(f"[{path}] " + " ".join(slice_words))

    if current_words:
        chunks.append(f"[{path}] " + " ".join(current_words))

    return chunks


def handle_request(req_data: dict[str, Any]) -> dict[str, Any]:
    req_id = req_data.get("id", "0")
    cmd = req_data.get("command", "optimize")

    if cmd == "clear_cache":
        PROMPT_CACHE.clear()
        return {"id": req_id, "ok": True, "message": "Cache cleared"}

    if cmd == "ping":
        return {"id": req_id, "ok": True, "message": "pong"}

    # Gather explicit context_chunks + auto-chunked workspace files from VS Code
    context_chunks: list[str] = list(req_data.get("context_chunks") or [])
    for wf in req_data.get("workspace_files") or []:
        wf_path = wf.get("path", "workspace_file")
        wf_content = wf.get("content", "")
        context_chunks.extend(chunk_workspace_file(wf_path, wf_content))

    budget = TokenBudget(
        max_total_tokens=int(req_data.get("max_total_tokens", 512)),
        core_query_budget=int(req_data.get("core_query_budget", 120)),
        context_budget=int(req_data.get("context_budget", 160)),
        history_budget=int(req_data.get("history_budget", 80)),
        tools_budget=int(req_data.get("tools_budget", 60)),
        few_shot_budget=int(req_data.get("few_shot_budget", 90)),
    )

    raw_req = RawRequest(
        text=str(req_data.get("text", "")),
        task_hint=str(req_data.get("task_hint", "")),
        context_chunks=context_chunks,
        conversation_history=list(req_data.get("conversation_history") or []),
        token_budget=budget,
        compression_intensity=str(req_data.get("compression_intensity", "AUTO")),
        enable_cache=bool(req_data.get("enable_cache", True)),
    )

    compiled = optimize(raw_req, few_shot_k=int(req_data.get("few_shot_k", 2)))
    return {
        "id": req_id,
        "ok": True,
        "status": compiled.status,
        "clarification_request": compiled.clarification_request,
        "payload": compiled.payload,
        "trace": dataclasses.asdict(compiled.trace),
    }


def main() -> None:
    # Signal readiness to the VS Code Extension host
    sys.stdout.write(json.dumps({"id": "init", "ok": True, "status": "READY"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = "unknown"
        try:
            payload = json.loads(line)
            req_id = payload.get("id", "unknown")
            response = handle_request(payload)
        except Exception as exc:
            response = {
                "id": req_id,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()