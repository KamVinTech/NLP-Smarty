import copy
import json
import random

import pytest

from contextgate import gate

SPANS = {
    "fenced code": "```python\ndef f(x):\n    return  x  +  1\n```",
    "inline code": "`rm -rf /tmp/x`",
    "URL": "https://example.com/a/b?c=1&d=2",
    "email": "dev.team@example.com",
    "snake_case": "get_user_by_id",
    "CamelCase": "UserProfileService",
    "lowerCamel": "getUserProfile",
    "CONSTANT": "HTTP_TIMEOUT_MS",
    "dotted": "os.path.join",
    "path": "src/core/ConfigManager.ts",
    "filename": "settings.local.json",
    "negation: do not": "do not use recursion",
    "negation: never": "never modify the input",
    "negation: without": "without using slicing",
    "negation: except": "except for the header row",
    "date": "2026-11-24",
    "version": "v3.12.1",
    "percent": "42%",
    "quoted": '"exact phrase here"',
    "json": '{"a": 1, "b": [1, 2]}',
}
FILLER = ("Hey! I was wondering if you could please help me with a task. Basically, I would really appreciate it if you "
          "could kindly write the code, and honestly just make sure that you follow these details:\n")
OUTRO = "\nThank you so much in advance, that would be great!"
PROMPT = FILLER + "\n".join(f"- {v}" for v in SPANS.values()) + OUTRO


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("name", list(SPANS))
def test_every_protected_span_survives_at_every_level(name, level):
    out, ok = gate.gate_text(PROMPT, level)
    assert ok and SPANS[name] in out
    assert gate.estimate_tokens(out) < gate.estimate_tokens(PROMPT)  # and it still saves tokens


def test_verify_reports_missing_spans_and_numbers():
    assert gate.verify("use get_user_by_id with 42%", "use get_user_by_id with 42%") == []
    miss = gate.verify("use get_user_by_id with 42% of v3.1", "use get user by id with 4 of v3")
    assert "get_user_by_id" in miss and "42%" in miss and "3.1" in miss  # digit tokens are checked from the first digit


def test_failed_verification_falls_back_to_the_original(monkeypatch):
    monkeypatch.setattr(gate, "_compress_segments", lambda t, l: t.replace("get_user_by_id", "get user by id"))
    out, ok = gate.gate_text("please call get_user_by_id now", 1)
    assert not ok and out == "please call get_user_by_id now"


def test_no_unicode_normalisation_of_code_or_prose():
    text = "Please run `print(“smart”)` and keep ﬁnal Ａ as written; don’t touch “quotes”."
    out, ok = gate.gate_text(text, 3)
    assert ok and "“smart”" in out and "ﬁnal" in out and "Ａ" in out and "’" in out


def test_levels_are_progressive():
    t = "Basically, we should refactor the retry handler so that it is simple, for the purpose of clarity."
    l1, l2 = gate.gate_text(t, 1)[0], gate.gate_text(t, 2)[0]
    assert "Basically" in l1 and "Basically" not in l2 and "for the purpose of" not in l1  # phrase map at level 1, hedging at 2


def test_markdown_structure_and_indentation_are_kept():
    t = "Could you please check:\n\n    indented block\n  - nested   item\n\n\nAnd   then run it."
    out, ok = gate.gate_text(t, 1)
    assert ok and "    indented block" in out and "  - nested item" in out and "\n\n\n" in out


def test_capitalisation_after_removed_opener():
    out, _ = gate.gate_text("could you please fix the retry loop", 1)
    assert out == "fix the retry loop" or out[0].islower() is False or out.startswith("Fix")
    assert gate.gate_text("Could you please fix the retry loop", 1)[0].startswith("Fix")


def test_deterministic():
    assert gate.gate_text(PROMPT, 2) == gate.gate_text(PROMPT, 2)


def test_fuzz_protected_spans_always_survive():
    rng = random.Random(1234)
    prose = ["could you please", "I was wondering if", "basically", "in order to", "the", "a", "refactor", "this", "handler",
             "so that", "it", "works", "honestly", "just", "we need", "to", "make", "sure", "for the purpose of", "and"]
    prot = list(SPANS.values()) + ["not", "never touch it", "cannot be empty", "unless told", "x = {'k': [1, 2]}", "3.14", "10:30"]
    seps = [" ", " ", "  ", "\t", "\n", "\n\n", ", ", ". "]
    for _ in range(1500):
        parts = [rng.choice(prose if rng.random() < 0.6 else prot) for _ in range(rng.randint(1, 40))]
        text = "".join(p + rng.choice(seps) for p in parts)
        for level in (1, 2, 3):
            out, ok = gate.gate_text(text, level)
            assert ok, (level, text)
            assert gate.verify(text, out) == [], (level, text, out)


# ── structured requests ───────────────────────────────────────────────────────
TOOLS = [{"type": "function", "function": {
    "name": "get_user", "description": "Could you please use this tool in order to fetch the user record by get_user_by_id.",
    "parameters": {"type": "object", "properties": {
        "id": {"type": "string", "description": "Basically the unique id; do not pass emails", "enum": ["a", "b"]},
        "opts": {"type": "object", "properties": {"deep": {"type": "boolean", "description": "I would like to enable deep mode"}}}},
        "required": ["id"]}}}]


def _strip_descriptions(node):
    if isinstance(node, dict):
        return {k: _strip_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_strip_descriptions(x) for x in node]
    return node


def test_tool_schema_structure_is_identical_only_descriptions_change():
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}
    r = gate.gate_openai_request(body, "strict")
    assert r.applied and r.by_category["tool_schemas"][1] < r.by_category["tool_schemas"][0]
    assert _strip_descriptions(r.body["tools"]) == _strip_descriptions(TOOLS)  # types, enum, required untouched
    assert json.loads(json.dumps(r.body["tools"]))  # still valid JSON
    d = r.body["tools"][0]["function"]["description"]
    assert "get_user_by_id" in d and "in order to" not in d
    assert "do not pass emails" in r.body["tools"][0]["function"]["parameters"]["properties"]["id"]["description"]
    assert body["tools"] == TOOLS  # the caller's object is never mutated


def test_strict_profile_only_touches_user_messages():
    msgs = [{"role": "system", "content": "Could you please always be basically helpful."},
            {"role": "user", "content": "Could you please fix the bug in retry_handler.py?"},
            {"role": "assistant", "content": "I would like to help; could you please share the log?"},
            {"role": "tool", "tool_call_id": "t1", "content": "I was wondering if you could please see this"}]
    r = gate.gate_openai_request({"messages": copy.deepcopy(msgs)}, "strict")
    assert [m["content"] for i, m in enumerate(r.body["messages"]) if i != 1] == [m["content"] for i, m in enumerate(msgs) if i != 1]
    assert r.body["messages"][1]["content"] != msgs[1]["content"] and "retry_handler.py" in r.body["messages"][1]["content"]


def test_content_parts_text_is_compressed_other_parts_untouched():
    img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Could you please describe this picture?"}, img]}]}
    r = gate.gate_openai_request(body, "strict")
    parts = r.body["messages"][0]["content"]
    assert parts[1] == img and parts[0]["text"] != "Could you please describe this picture?"


def test_never_forwards_a_larger_request():
    body = {"messages": [{"role": "user", "content": "Refactor retry_handler.py."}]}
    r = gate.gate_openai_request(body, "strict")
    assert not r.applied and r.body is body and r.saved == 0


def test_fails_open_on_internal_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(gate, "_compress_segments", boom)
    body = {"messages": [{"role": "user", "content": "Could you please fix this?"}]}
    r = gate.gate_openai_request(body, "strict")
    assert r.body is body and not r.applied and "RuntimeError" in r.error


def _conversation(rng, n):
    big = "log line one two three four five six seven eight nine ten " * 40
    msgs = []
    for i in range(n):
        role = rng.choice(["user", "assistant", "tool", "user"])
        text = rng.choice([big, big, f"Could you please look at item {i}, basically?", f"result {i} " * 5])
        m = {"role": role, "content": text}
        if role == "tool":
            m["tool_call_id"] = f"t{i}"
        msgs.append(m)
    return msgs


@pytest.mark.parametrize("profile", ["strict", "balanced", "aggressive"])
def test_prefix_stability_transform_of_a_prefix_equals_prefix_of_transform(profile):
    """Providers cache by exact prefix: the transformed prefix must not depend on later messages."""
    rng = random.Random(99)
    for _ in range(25):
        msgs = _conversation(rng, rng.randint(2, 12))
        full = gate.gate_openai_request({"messages": copy.deepcopy(msgs), "tools": TOOLS}, profile).body["messages"]
        for k in range(1, len(msgs) + 1):
            part = gate.gate_openai_request({"messages": copy.deepcopy(msgs[:k]), "tools": TOOLS}, profile).body["messages"]
            assert part == full[:k], (profile, k)


def test_dedupe_is_causal_and_only_in_non_strict_profiles():
    big = "repeated tool output line " * 80
    msgs = [{"role": "user", "content": "run it"}, {"role": "tool", "tool_call_id": "a", "content": big},
            {"role": "assistant", "content": "ok"}, {"role": "tool", "tool_call_id": "b", "content": big}]
    strict = gate.gate_openai_request({"messages": copy.deepcopy(msgs)}, "strict")
    bal = gate.gate_openai_request({"messages": copy.deepcopy(msgs)}, "balanced")
    assert strict.body["messages"][3]["content"] == big  # strict never rewrites tool results
    assert bal.body["messages"][1]["content"] == big  # the first occurrence is kept
    assert bal.body["messages"][3]["content"] == "[identical to message #1]"
    assert bal.by_category["duplicates"][0] > bal.by_category["duplicates"][1] and bal.saved > 0


def test_accounting_and_latency_fields():
    body = {"messages": [{"role": "user", "content": FILLER + "- do not use recursion" + OUTRO}], "tools": TOOLS}
    r = gate.gate_openai_request(body, "strict")
    assert r.applied and r.raw_tokens > r.fwd_tokens and r.saved == r.raw_tokens - r.fwd_tokens
    assert r.protected_ok and r.fallbacks == 0 and 0 < r.ms < 100


# ── Anthropic /v1/messages ────────────────────────────────────────────────────
REMINDER = "<system-reminder>\nCodebase and user instructions: please always just be careful, basically.\n</system-reminder>"
SIG = "CAQSmgUKEAgUGAI4AUIIdGhpbmtpbmc" * 4
A_TOOLS = [{"name": "Bash", "description": "Could you please use this tool in order to run get_user_by_id style commands.",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string",
                             "description": "I would like you to provide the command; do not pass secrets", "enum": ["a"]}},
                             "required": ["command"]}, "cache_control": {"type": "ephemeral"}}]


def a_req():
    return {"model": "claude", "max_tokens": 100, "stream": True,
            "system": [{"type": "text", "text": "Could you please always be basically helpful.", "cache_control": {"type": "ephemeral"}}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": REMINDER},
                                             {"type": "text", "text": FILLER + "Keep `get_user_by_id` verbatim, do not use recursion."}]},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": SIG},
                                                  {"type": "text", "text": "Could you please share more? I would like to help."},
                                                  {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls  -la   src/"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                              "content": "Could you please see   this output I was wondering if"}]}],
            "tools": A_TOOLS}


def test_anthropic_only_the_users_plain_text_and_tool_descriptions_change():
    body = a_req()
    r = gate.gate_anthropic_request(body, "strict")
    assert r.applied and body == a_req()  # the caller's object is never mutated
    out = r.body
    blocks = out["messages"][0]["content"]
    assert blocks[0]["text"] == REMINDER  # system-reminder wrapper: instruction text, never rewritten
    assert blocks[1]["text"] != body["messages"][0]["content"][1]["text"] and "`get_user_by_id`" in blocks[1]["text"]
    assert "Could you please" not in blocks[1]["text"] and "do not use recursion" in blocks[1]["text"]
    assert out["system"] == body["system"]  # system prompt untouched (and its cache_control)
    assert out["messages"][1] == body["messages"][1]  # thinking signature, assistant text and tool_use verbatim
    assert out["messages"][2] == body["messages"][2]  # tool_result untouched in strict
    t = out["tools"][0]
    assert t["cache_control"] == {"type": "ephemeral"} and "in order to" not in t["description"] and "get_user_by_id" in t["description"]
    assert _strip_descriptions(out["tools"]) == _strip_descriptions(A_TOOLS)  # input_schema structure identical
    assert "do not pass secrets" in t["input_schema"]["properties"]["command"]["description"]
    assert r.by_category["user_text"][1] < r.by_category["user_text"][0] and r.by_category["tool_schemas"][1] < r.by_category["tool_schemas"][0]


def test_anthropic_string_content_and_non_prose_blocks():
    body = {"system": "Be helpful.", "messages": [{"role": "user", "content": FILLER + "Fix the retry_handler.py bug."},
                                                  {"role": "user", "content": REMINDER + " could you please"},
                                                  {"role": "user", "content": "<ide_selection>could you please</ide_selection>"}]}
    r = gate.gate_anthropic_request(body, "strict")
    msgs = r.body["messages"]
    assert "retry_handler.py" in msgs[0]["content"] and "Could you please" not in msgs[0]["content"]
    assert msgs[1]["content"] == body["messages"][1]["content"] and msgs[2]["content"] == body["messages"][2]["content"]


def test_anthropic_balanced_collapses_duplicate_tool_results_causally():
    big = "same tool output line " * 80
    def tr(i, content): return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": content}]}
    msgs = [tr(1, big), {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}, tr(2, big),
            tr(3, [{"type": "text", "text": big}])]
    strict = gate.gate_anthropic_request({"messages": copy.deepcopy(msgs)}, "strict")
    bal = gate.gate_anthropic_request({"messages": copy.deepcopy(msgs)}, "balanced")
    assert strict.body["messages"][2] == msgs[2]  # strict never rewrites tool results
    assert bal.body["messages"][0] == msgs[0]  # first occurrence kept
    assert bal.body["messages"][2]["content"][0]["content"] == "[identical to message #0]"
    assert bal.body["messages"][3]["content"][0]["content"] == "[identical to message #0]"  # list-of-text form too
    assert bal.body["messages"][2]["content"][0]["tool_use_id"] == "t2"  # pairing ids preserved
    assert bal.by_category["duplicates"][0] > bal.by_category["duplicates"][1]


@pytest.mark.parametrize("profile", ["strict", "balanced", "aggressive"])
def test_anthropic_prefix_stability(profile):
    rng = random.Random(5)
    big = "log line one two three four five six seven eight nine ten " * 40
    for _ in range(20):
        msgs = []
        for i in range(rng.randint(2, 10)):
            if rng.random() < 0.5:
                msgs.append({"role": "user", "content": rng.choice([f"Could you please look at item {i}, basically?", big])})
            else:
                msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": rng.choice([big, f"r{i}"])}]})
        full = gate.gate_anthropic_request({"messages": copy.deepcopy(msgs), "tools": A_TOOLS}, profile).body["messages"]
        for k in range(1, len(msgs) + 1):
            assert gate.gate_anthropic_request({"messages": copy.deepcopy(msgs[:k]), "tools": A_TOOLS}, profile).body["messages"] == full[:k]


def test_anthropic_never_larger_and_fails_open(monkeypatch):
    plain = {"system": "s", "messages": [{"role": "user", "content": "Refactor retry_handler.py."}]}
    r = gate.gate_anthropic_request(plain, "strict")
    assert not r.applied and r.body is plain and r.saved == 0
    monkeypatch.setattr(gate, "_compress_segments", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    r = gate.gate_anthropic_request(a_req(), "strict")
    assert not r.applied and "RuntimeError" in r.error and r.body == a_req()


def test_system_counts_toward_token_accounting_even_though_it_is_untouched():
    body = a_req()
    r = gate.gate_anthropic_request(body, "strict")
    without = gate._tok({"messages": body["messages"], "tools": body["tools"]})
    assert r.raw_tokens > without  # `system` is input the provider bills, so calibration against usage stays honest


def test_detect_api_and_dispatcher():
    assert gate.detect_api(a_req()) == "anthropic"
    assert gate.detect_api({"system": "s", "messages": []}) == "anthropic"
    assert gate.detect_api({"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}) == "openai"
    assert gate.detect_api({"tools": [{"type": "function", "function": {"name": "f"}}], "messages": []}) == "openai"
    assert gate.detect_api("junk") == "openai"
    assert gate.gate_request(a_req()).applied and gate.gate_request({"messages": [{"role": "user", "content": PROMPT}]}).applied
    assert gate.gate_request(a_req(), api="openai").body is not None  # explicit routing wins over detection
