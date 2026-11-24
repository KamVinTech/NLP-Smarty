"""
Persistent savings ledger (SQLite, stdlib). One row per proxied request. Stores COUNTS ONLY: never prompt text, response
text, headers or credentials.

Honest accounting:
  * saved_tokens = estimate(original request) - estimate(forwarded request), same estimator on both sides.
  * The provider reports exact `usage.prompt_tokens` for what was FORWARDED. The ratio actual/estimate calibrates the
    estimator, and calibrated savings = estimated savings x ratio. Both numbers are reported.
  * USD uses a price table YOU supply (--prices FILE); nothing is hard-coded. It ignores cache discounts: an estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import threading
import time
from typing import Any

SCHEMA = 1
DEFAULT_PATH = os.environ.get("SMARTY_LEDGER") or os.path.join(os.path.expanduser("~"), ".smarty", "ledger.db")
AUDIT_PASS = 80
COLS = ("ts", "route", "host", "model", "profile", "stream", "applied", "raw_tokens", "fwd_tokens", "up_prompt_tokens",
        "up_completion_tokens", "gate_ms", "upstream_ms", "status", "fallbacks", "protected_ok", "error", "audit_score",
        "neg_ok", "entity_ok", "api", "up_cache_read_tokens", "up_cache_write_tokens")
_ADDED_LATER = {"api": "TEXT", "up_cache_read_tokens": "INTEGER", "up_cache_write_tokens": "INTEGER"}  # migrated into old ledgers
DEFAULT_CACHE_MULT = {"read": 0.10, "write": 1.25}  # multipliers on the input price: ASSUMPTIONS, override per model in --prices


class Ledger:
    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if path != ":memory:":  # WAL: cheap commits off the request path, and `stats` can read while the gateway writes
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS requests(id INTEGER PRIMARY KEY, ts REAL, route TEXT, host TEXT, model TEXT, "
                "profile TEXT, stream INTEGER, applied INTEGER, raw_tokens INTEGER, fwd_tokens INTEGER, "
                "up_prompt_tokens INTEGER, up_completion_tokens INTEGER, gate_ms REAL, upstream_ms REAL, status INTEGER, "
                "fallbacks INTEGER, protected_ok INTEGER, error TEXT, audit_score INTEGER, neg_ok INTEGER, entity_ok INTEGER, "
                "api TEXT, up_cache_read_tokens INTEGER, up_cache_write_tokens INTEGER)")
            have = {r[1] for r in self._db.execute("PRAGMA table_info(requests)")}
            for col, typ in _ADDED_LATER.items():  # migrate ledgers created before these columns existed
                if col not in have:
                    self._db.execute(f"ALTER TABLE requests ADD COLUMN {col} {typ}")
            self._db.execute("CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts)")
            self._db.execute("INSERT OR REPLACE INTO meta VALUES('schema', ?)", (str(SCHEMA),))
            self._db.commit()

    def record(self, **row: Any) -> None:
        vals = {c: row.get(c) for c in COLS}
        vals["ts"] = vals["ts"] if vals["ts"] is not None else time.time()
        for b in ("stream", "applied", "protected_ok", "neg_ok", "entity_ok"):
            if vals[b] is not None:
                vals[b] = int(bool(vals[b]))
        vals["error"] = (vals["error"] or "")[:200] or None  # a class name / short reason, never content
        with self._lock:
            self._db.execute(f"INSERT INTO requests({','.join(COLS)}) VALUES({','.join('?' * len(COLS))})", [vals[c] for c in COLS])
            self._db.commit()

    def rows(self, since: float | None = None) -> list[sqlite3.Row]:
        with self._lock:
            if since is None:
                return self._db.execute("SELECT * FROM requests ORDER BY ts").fetchall()
            return self._db.execute("SELECT * FROM requests WHERE ts >= ? ORDER BY ts", (since,)).fetchall()

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _price(prices: dict | None, model: str | None) -> dict | None:
    if not prices:
        return None
    return prices.get(model or "") or prices.get("default")


def _pct(vals: list[float], q: float) -> float:
    return sorted(vals)[max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))] if vals else 0.0


def summarize(rows: list[Any], prices: dict | None = None) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"requests": 0}
    raw = sum(r["raw_tokens"] or 0 for r in rows)
    fwd = sum(r["fwd_tokens"] or 0 for r in rows)
    with_usage = [r for r in rows if r["up_prompt_tokens"] and r["fwd_tokens"]]
    ratio = (sum(r["up_prompt_tokens"] for r in with_usage) / sum(r["fwd_tokens"] for r in with_usage)) if with_usage else None
    saved = raw - fwd
    out: dict[str, Any] = {
        "requests": n, "gated": sum(1 for r in rows if r["applied"]), "passthrough": sum(1 for r in rows if not r["applied"]),
        "est_tokens_original": raw, "est_tokens_forwarded": fwd, "est_tokens_saved": saved,
        "est_saved_pct": round(100 * saved / raw, 2) if raw else 0.0,
        "calibration_ratio": round(ratio, 3) if ratio else None, "calibration_samples": len(with_usage),
        "calibrated_tokens_saved": round(saved * ratio) if ratio else None,
        "upstream_prompt_tokens": sum(r["up_prompt_tokens"] or 0 for r in rows),
        "upstream_completion_tokens": sum(r["up_completion_tokens"] or 0 for r in rows),
        "gate_ms_mean": round(statistics.mean(r["gate_ms"] or 0 for r in rows), 2),
        "gate_ms_p95": round(_pct([r["gate_ms"] or 0 for r in rows], 0.95), 2),
        "upstream_ms_mean": round(statistics.mean(r["upstream_ms"] or 0 for r in rows), 1),
        "fallback_to_original": sum(r["fallbacks"] or 0 for r in rows),
        "protected_ok_rate": round(sum(1 for r in rows if r["protected_ok"] in (None, 1)) / n, 4),
        "errors": sum(1 for r in rows if r["error"]), "upstream_errors": sum(1 for r in rows if (r["status"] or 0) >= 400),
    }
    aud = [r for r in rows if r["audit_score"] is not None]
    if aud:
        out["audited"] = len(aud)
        out["audit_mean_score"] = round(statistics.mean(r["audit_score"] for r in aud), 1)
        out["audit_pass_rate"] = round(sum(1 for r in aud if r["audit_score"] >= AUDIT_PASS) / len(aud), 4)
        # a check that did not apply (None) is not a pass: rates are over the responses where the check actually ran
        for key, col in (("negation_pass_rate", "neg_ok"), ("entity_pass_rate", "entity_ok")):
            ran = [r for r in aud if r[col] is not None]
            out[key] = round(sum(1 for r in ran if r[col] == 1) / len(ran), 4) if ran else None
            out[key.replace("pass_rate", "checked")] = len(ran)
    cr = sum(r["up_cache_read_tokens"] or 0 for r in rows)
    cw = sum(r["up_cache_write_tokens"] or 0 for r in rows)
    if cr or cw:
        total_prompt = out["upstream_prompt_tokens"]
        out.update({"upstream_cache_read_tokens": cr, "upstream_cache_write_tokens": cw,
                    "cache_read_share": round(cr / total_prompt, 4) if total_prompt else None})
    if prices:
        usd_saved = usd_spend = 0.0
        priced = unpriced = 0
        for r in rows:
            p = _price(prices, r["model"])
            if not p:
                unpriced += 1
                continue
            priced += 1
            k = ratio or 1.0
            prompt = r["up_prompt_tokens"] or 0
            eff = 1.0  # saved tokens are assumed to have the cache mix this request actually had
            if prompt and (r["up_cache_read_tokens"] or r["up_cache_write_tokens"]):
                rd, wr = r["up_cache_read_tokens"] or 0, r["up_cache_write_tokens"] or 0
                eff = ((prompt - rd - wr) + rd * p.get("cache_read_mult", DEFAULT_CACHE_MULT["read"])
                       + wr * p.get("cache_write_mult", DEFAULT_CACHE_MULT["write"])) / prompt
            usd_saved += ((r["raw_tokens"] or 0) - (r["fwd_tokens"] or 0)) * k * eff / 1e6 * p["input_per_mtok"]
            usd_spend += (r["up_prompt_tokens"] or 0) / 1e6 * p["input_per_mtok"] + \
                (r["up_completion_tokens"] or 0) / 1e6 * p.get("output_per_mtok", 0)
        out.update({"usd_input_saved_est": round(usd_saved, 6), "usd_spend_observed": round(usd_spend, 6),
                    "usd_priced_requests": priced, "usd_unpriced_requests": unpriced})
    by: dict[str, dict] = {}
    for key in ("model", "profile", "api"):
        grp: dict[str, list] = {}
        for r in rows:
            grp.setdefault(r[key] or "?", []).append(r)
        by[key] = {k: {"requests": len(v), "est_saved_pct": round(100 * (sum(x["raw_tokens"] or 0 for x in v) - sum(x["fwd_tokens"] or 0 for x in v))
                                                                  / max(1, sum(x["raw_tokens"] or 0 for x in v)), 2)} for k, v in grp.items()}
    out["by_model"], out["by_profile"], out["by_api"] = by["model"], by["profile"], by["api"]
    return out


def format_summary(s: dict[str, Any]) -> str:
    if not s.get("requests"):
        return "no requests recorded yet"
    L = [f"requests: {s['requests']}  (gated {s['gated']}, passed through unchanged {s['passthrough']})",
         f"tokens (estimated): {s['est_tokens_original']} original -> {s['est_tokens_forwarded']} forwarded = "
         f"{s['est_tokens_saved']} saved ({s['est_saved_pct']}%)"]
    if s.get("calibration_ratio"):
        L.append(f"calibration vs provider usage: x{s['calibration_ratio']} over {s['calibration_samples']} requests -> "
                 f"~{s['calibrated_tokens_saved']} provider tokens saved")
    else:
        L.append("calibration: no upstream usage seen yet (savings above are estimator-only)")
    if s.get("cache_read_share") is not None:
        L.append(f"provider cache: {100 * s['cache_read_share']:.1f}% of prompt tokens were cache reads "
                 f"({s['upstream_cache_read_tokens']} read, {s['upstream_cache_write_tokens']} written)")
    if "usd_input_saved_est" in s:
        L.append(f"USD (your price table; estimate, saved tokens priced at the cache mix actually observed): ~${s['usd_input_saved_est']:.6f} "
                 f"input saved of ${s['usd_spend_observed']:.6f} observed spend ({s['usd_priced_requests']} priced, {s['usd_unpriced_requests']} unpriced requests; "
                 f"cache read/write multipliers assumed x{DEFAULT_CACHE_MULT['read']}/x{DEFAULT_CACHE_MULT['write']} unless set per model)")
    L.append(f"gate overhead: {s['gate_ms_mean']} ms mean, {s['gate_ms_p95']} ms p95  (upstream mean {s['upstream_ms_mean']} ms)")
    L.append(f"safety: protected-span verification fallbacks {s['fallback_to_original']}, protected_ok {100 * s['protected_ok_rate']:.2f}%, "
             f"internal errors {s['errors']}, upstream errors {s['upstream_errors']}")
    if "audit_mean_score" in s:
        fmt = lambda v, n: f"{100 * v:.1f}% of {n}" if v is not None else "not applicable"  # noqa: E731
        L.append(f"post-audit ({s['audited']} responses): mean {s['audit_mean_score']}, pass rate {100 * s['audit_pass_rate']:.1f}%; "
                 f"negation check {fmt(s['negation_pass_rate'], s['negation_checked'])}, entity check "
                 f"{fmt(s['entity_pass_rate'], s['entity_checked'])}")
    return "\n".join(L)


def parse_since(text: str | None) -> float | None:
    if not text:
        return None
    m = re.fullmatch(r"(\d+)\s*([smhd])", text.strip().lower())
    if not m:
        raise ValueError("--since must look like 30m, 24h or 7d")
    return time.time() - int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stats", description="Savings ledger report")
    ap.add_argument("--ledger", default=DEFAULT_PATH)
    ap.add_argument("--since", help="e.g. 24h, 7d")
    ap.add_argument("--prices", help='JSON price table: {"model": {"input_per_mtok": 3, "output_per_mtok": 15}, "default": {...}}')
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        since = parse_since(a.since)
        prices = json.load(open(a.prices)) if a.prices else None
        if not os.path.exists(a.ledger):
            print(f"error: no ledger at {a.ledger} (run `serve` first)", file=sys.stderr)
            return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    led = Ledger(a.ledger)
    s = summarize(led.rows(since), prices)
    led.close()
    print(json.dumps(s, indent=2) if a.json else format_summary(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
