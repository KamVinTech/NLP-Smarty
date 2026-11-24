"""The extension cannot be launched in CI, so these tests cross-check package.json against src/extension.ts: a renamed or
missing identifier would otherwise only show up when a user clicks something."""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PJ = json.load(open(os.path.join(ROOT, "package.json"), encoding="utf-8"))
TS = open(os.path.join(ROOT, "src", "extension.ts"), encoding="utf-8").read()
CONTRIB = PJ["contributes"]


def test_identity_is_contextgate():
    assert PJ["name"] == "contextgate" and re.fullmatch(r"[a-z0-9][a-z0-9\-]*", PJ["name"])  # VS Code requires a lowercase id
    assert PJ["displayName"] == "ContextGate"
    assert f"{PJ['publisher']}.{PJ['name']}".endswith(".contextgate")
    lock = json.load(open(os.path.join(ROOT, "package-lock.json")))
    assert lock["name"] == lock["packages"][""]["name"] == "contextgate"


def test_old_product_names_are_gone_from_the_extension():
    legacy = re.compile(r"nlp-smarty|nlpSmarty|NLP-Smarty|\bSmarty[A-Z]|@smarty\b|Smarty:", re.I)
    manifest = re.sub(r"https://github\.com/[\w-]+/NLP-Smarty[\w./#-]*", "", json.dumps(PJ))  # the repo URL keeps its name
    assert not legacy.findall(manifest), legacy.findall(manifest)
    assert not [m for m in legacy.findall(TS) if m.lower() != "smarty:" or "ContextGate" not in TS], legacy.findall(TS)
    assert not re.search(r"\bnlp_smarty_optimization\b", TS)  # the engine no longer writes that file by default


def test_every_declared_command_is_registered_and_nothing_else_is():
    declared = {c["command"] for c in CONTRIB["commands"]}
    registered = set(re.findall(r"registerCommand\('([^']+)'", TS))
    assert declared == registered and all(c.startswith("contextGate.") for c in declared)
    used = {k["command"] for k in CONTRIB.get("keybindings", [])}
    for entries in CONTRIB.get("menus", {}).values():
        used |= {m["command"] for m in entries if "command" in m}
    assert used <= declared, used - declared
    assert all(c["title"].startswith("ContextGate: ") for c in CONTRIB["commands"])


def test_settings_used_in_code_are_declared_and_namespaced():
    declared = set(CONTRIB["configuration"]["properties"])
    assert all(k.startswith("contextGate.") for k in declared)
    assert CONTRIB["configuration"].get("title") == "ContextGate"
    keys = set(re.findall(r"getConfiguration\('contextGate'\)\s*\.get(?:<[^>]*>)?\('(\w+)'", TS))
    keys |= set(re.findall(r"\bconfig\.get(?:<[^>]*>)?\('(\w+)'", TS))
    assert keys and {f"contextGate.{k}" for k in keys} <= declared, {f"contextGate.{k}" for k in keys} - declared
    assert "getConfiguration('nlpSmarty')" not in TS


def test_chat_participant_ids_agree_everywhere():
    part = CONTRIB["chatParticipants"][0]
    assert part["id"] == "contextgate.participant" and part["name"] == "contextgate" and part["fullName"] == "ContextGate"
    assert re.search(rf"createChatParticipant\(\s*'{re.escape(part['id'])}'", TS)  # the call spans several lines
    assert PJ["activationEvents"] == [f"onChatParticipant:{part['id']}"]
    assert "@contextgate /optimize" in TS and "@contextgate /compare" in TS


def test_audit_log_is_an_explicit_setting_not_a_hidden_file():
    prop = CONTRIB["configuration"]["properties"]["contextGate.auditLogFile"]
    assert prop["default"] == "" and "prompt" in prop["markdownDescription"].lower()
    assert "SMARTY_AUDIT_LOG: auditLogPath()" in TS
    assert TS.count("auditLogPath()") >= 4  # helper, spawn env, and both Show Logs paths
