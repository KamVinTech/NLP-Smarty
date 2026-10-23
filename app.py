"""
HuggingFace Downstream Agent Consumer (`app.py`)
================================================
Consumes `CompiledPrompt` from `pipeline.py` and supports side-by-side A/B evaluation
of Raw vs. Deterministically Optimized prompts against HuggingFace Inference API
(or any OpenAI-compatible local/remote endpoint), with an offline deterministic
simulation mode when `HF_TOKEN` is not set.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from data_contracts import RawRequest
from pipeline import audit_agent_response, estimate_tokens, optimize

DEFAULT_HF_MODEL = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-Coder-32B-Instruct")
DEFAULT_HF_API_URL = os.environ.get(
    "HF_API_URL",
    f"https://router.huggingface.co/hf-inference/models/{DEFAULT_HF_MODEL}/v1/chat/completions",
)


def call_huggingface_llm(
    prompt_text: str,
    model_id: str | None = None,
    max_new_tokens: int = 350,
    temperature: float = 0.1,
) -> tuple[str, float, str]:
    """
    Calls HuggingFace Inference API (via `huggingface_hub.InferenceClient` if installed,
    or standard-library `urllib.request`) and returns (response_text, latency_ms, backend_used).
    Falls back to a deterministic offline evaluator if `HF_TOKEN` is not configured.
    """
    t0 = time.perf_counter()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
    target_model = model_id or DEFAULT_HF_MODEL

    if hf_token:
        # 1. Try huggingface_hub.InferenceClient if installed in user's venv
        try:
            from huggingface_hub import InferenceClient  # type: ignore
            client = InferenceClient(model=target_model, token=hf_token)
            res = client.chat_completion(
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            content = res.choices[0].message.content or ""
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return content.strip(), latency, f"huggingface_hub ({target_model})"
        except ImportError:
            pass
        except Exception as exc:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return f"[HF API Error: {exc}]", latency, f"huggingface_hub_error ({target_model})"

        # 2. Zero-dependency HTTP fallback via urllib.request
        try:
            req_body = json.dumps({
                "model": target_model,
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
            }).encode("utf-8")
            req = urllib.request.Request(
                DEFAULT_HF_API_URL,
                data=req_body,
                headers={
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                latency = round((time.perf_counter() - t0) * 1000, 2)
                return content.strip(), latency, f"hf_http_router ({target_model})"
        except Exception as exc:
            latency = round((time.perf_counter() - t0) * 1000, 2)
            return f"[HF HTTP Error: {exc}]", latency, f"hf_http_error ({target_model})"

    # 3. Offline local simulation when HF_TOKEN is not set (so UI/CLI testing always works)
    is_compiled_xml = "<agent_instructions>" in prompt_text and "<core_task>" in prompt_text
    if is_compiled_xml:
        sim_output = (
            "```python\n"
            "def reverse_string(s: str) -> str:\n"
            "    # Complies with <task_metadata> negation: without using slicing\n"
            "    chars = []\n"
            "    for idx in range(len(s) - 1, -1, -1):\n"
            "        chars.append(s[idx])\n"
            "    return ''.join(chars)\n"
            "```"
        )
    else:
        sim_output = (
            "Sure! Here is a quick Python function to reverse a string:\n"
            "def reverse_string(s):\n"
            "    return s[::-1]  # Note: unoptimized prompt often defaults to [::-1] slicing!"
        )
    latency = round((time.perf_counter() - t0) * 1000, 2)
    return sim_output, latency, "offline_simulator (set HF_TOKEN for live HuggingFace API)"


def compare_raw_vs_optimized(request: RawRequest, model_id: str | None = None) -> dict[str, Any]:
    """
    Runs a side-by-side A/B comparison:
      - Call A: Raw unoptimized prompt + unpruned context chunks
      - Call B: Deterministically compiled XML prompt (`CompiledPrompt.payload`)
    Audits both responses using `audit_agent_response()` from `pipeline.py`.
    """
    compiled = optimize(request)
    if compiled.status != "OK":
        return {
            "status": compiled.status,
            "clarification_request": compiled.clarification_request,
            "trace": dataclasses.asdict(compiled.trace),
        }

    # Build the unoptimized raw input (what a naive agent sends without NLP-Smarty)
    raw_parts = []
    if request.conversation_history:
        hist_str = "\n".join(f"{t.get('role', 'user')}: {t.get('content', '')}" for t in request.conversation_history)
        raw_parts.append(f"Conversation History:\n{hist_str}")
    if request.context_chunks:
        raw_parts.append("Context:\n" + "\n\n".join(request.context_chunks))
    raw_parts.append(request.text)
    unoptimized_input = "\n\n".join(raw_parts)

    raw_answer, raw_llm_ms, backend = call_huggingface_llm(unoptimized_input, model_id=model_id)
    opt_answer, opt_llm_ms, _ = call_huggingface_llm(compiled.payload, model_id=model_id)

    raw_audit = audit_agent_response(raw_answer, compiled.trace)
    opt_audit = audit_agent_response(opt_answer, compiled.trace)

    return {
        "status": "OK",
        "backend": backend,
        "raw_input_tokens": estimate_tokens(unoptimized_input),
        "optimized_input_tokens": compiled.trace.compiled_payload_token_count,
        "raw_response": raw_answer,
        "raw_llm_latency_ms": raw_llm_ms,
        "raw_audit": raw_audit,
        "optimized_payload": compiled.payload,
        "optimized_response": opt_answer,
        "optimized_llm_latency_ms": opt_llm_ms,
        "optimized_audit": opt_audit,
        "trace": dataclasses.asdict(compiled.trace),
    }


if __name__ == "__main__":
    sample_req = RawRequest(
        text="Could you please help me write a Python function to reverse a string without using slicing?"
    )
    comparison = compare_raw_vs_optimized(sample_req)
    print(json.dumps(comparison, indent=2))