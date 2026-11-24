import json
import os

import pytest

import ledger as lg


def row(**kw):
    base = dict(route="/v1/chat/completions", host="api.example.com", model="m1", profile="strict", stream=False, applied=True,
                raw_tokens=1000, fwd_tokens=800, up_prompt_tokens=880, up_completion_tokens=50, gate_ms=2.0, upstream_ms=400.0,
                status=200, fallbacks=0, protected_ok=True, error=None, audit_score=100, neg_ok=True, entity_ok=True)
    base.update(kw)
    return base


@pytest.fixture
def led(tmp_path):
    l = lg.Ledger(str(tmp_path / "l.db"))
    yield l
    l.close()


def test_empty_ledger_reports_nothing():
    assert lg.summarize([]) == {"requests": 0}
    assert lg.format_summary({"requests": 0}) == "no requests recorded yet"


def test_token_accounting_and_calibration(led):
    led.record(**row())
    led.record(**row(raw_tokens=500, fwd_tokens=500, applied=False, up_prompt_tokens=550))
    s = lg.summarize(led.rows())
    assert s["requests"] == 2 and s["gated"] == 1 and s["passthrough"] == 1
    assert s["est_tokens_original"] == 1500 and s["est_tokens_forwarded"] == 1300 and s["est_tokens_saved"] == 200
    assert s["est_saved_pct"] == pytest.approx(13.33, abs=0.01)
    assert s["calibration_ratio"] == pytest.approx((880 + 550) / (800 + 500), abs=0.001)  # provider tokens per estimated token
    assert s["calibrated_tokens_saved"] == round(200 * (880 + 550) / 1300)
    assert s["upstream_prompt_tokens"] == 1430


def test_no_usage_means_no_calibration_claim(led):
    led.record(**row(up_prompt_tokens=None))
    s = lg.summarize(led.rows())
    assert s["calibration_ratio"] is None and s["calibrated_tokens_saved"] is None
    assert "no upstream usage" in lg.format_summary(s)


def test_usd_uses_only_the_price_table_you_supply(led):
    led.record(**row(model="m1"))
    led.record(**row(model="other"))
    assert "usd_input_saved_est" not in lg.summarize(led.rows())  # nothing hard-coded
    prices = {"m1": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}
    s = lg.summarize(led.rows(), prices)
    k = 880 / 800
    assert s["usd_priced_requests"] == 1 and s["usd_unpriced_requests"] == 1
    assert s["usd_input_saved_est"] == pytest.approx(200 * k / 1e6 * 3.0, abs=1e-4)
    assert lg.summarize(led.rows(), {"default": {"input_per_mtok": 1.0}})["usd_priced_requests"] == 2  # default row


def test_safety_and_audit_rates(led):
    led.record(**row())
    led.record(**row(fallbacks=2, protected_ok=False, audit_score=60, neg_ok=False))
    led.record(**row(error="RuntimeError", status=502))
    s = lg.summarize(led.rows())
    assert s["fallback_to_original"] == 2 and s["protected_ok_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["errors"] == 1 and s["upstream_errors"] == 1
    assert s["audited"] == 3 and s["audit_pass_rate"] == pytest.approx(2 / 3, abs=1e-3) and s["negation_pass_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["by_profile"]["strict"]["requests"] == 3 and "m1" in s["by_model"]


def test_since_filter_and_parse(led):
    led.record(**row(ts=1.0))
    led.record(**row())
    assert len(led.rows()) == 2 and len(led.rows(since=lg.parse_since("1h"))) == 1
    assert lg.parse_since(None) is None
    with pytest.raises(ValueError):
        lg.parse_since("yesterday")


def test_ledger_never_stores_content(tmp_path):
    secret = "SUPER-SECRET-PROMPT-TEXT-sk-abcdef"
    l = lg.Ledger(str(tmp_path / "x.db"))
    l.record(**row(error=f"RuntimeError {secret}" * 20))  # even if a caller misuses `error`, it is length-capped
    l.close()
    raw = open(tmp_path / "x.db", "rb").read()
    assert b"Authorization" not in raw and b"Bearer" not in raw
    cols = [c for c in lg.COLS]
    assert not any(c in cols for c in ("prompt", "messages", "content", "response", "headers", "api_key"))


def test_cli(led, tmp_path, capsys):
    led.record(**row())
    prices = tmp_path / "p.json"
    prices.write_text(json.dumps({"m1": {"input_per_mtok": 3.0}}))
    assert lg.main(["--ledger", led.path, "--prices", str(prices)]) == 0
    out = capsys.readouterr().out
    assert "tokens (estimated)" in out and "USD (your price table" in out and "gate overhead" in out
    assert lg.main(["--ledger", led.path, "--json"]) == 0 and json.loads(capsys.readouterr().out)["requests"] == 1
    assert lg.main(["--ledger", str(tmp_path / "missing.db")]) == 2
    assert lg.main(["--ledger", led.path, "--since", "bogus"]) == 2


def test_checks_that_did_not_run_are_not_counted_as_passes(led):
    led.record(**row(neg_ok=None, entity_ok=None))
    led.record(**row(neg_ok=True, entity_ok=None))
    led.record(**row(neg_ok=False, entity_ok=None))
    s = lg.summarize(led.rows())
    assert s["negation_checked"] == 2 and s["negation_pass_rate"] == 0.5  # the None row is excluded, not a free pass
    assert s["entity_checked"] == 0 and s["entity_pass_rate"] is None
    assert "entity check not applicable" in lg.format_summary(s)


def test_old_ledgers_are_migrated_in_place(tmp_path):
    import sqlite3

    old = str(tmp_path / "old.db")
    db = sqlite3.connect(old)
    db.execute("CREATE TABLE requests(id INTEGER PRIMARY KEY, ts REAL, route TEXT, host TEXT, model TEXT, profile TEXT, stream INTEGER, "
               "applied INTEGER, raw_tokens INTEGER, fwd_tokens INTEGER, up_prompt_tokens INTEGER, up_completion_tokens INTEGER, "
               "gate_ms REAL, upstream_ms REAL, status INTEGER, fallbacks INTEGER, protected_ok INTEGER, error TEXT, "
               "audit_score INTEGER, neg_ok INTEGER, entity_ok INTEGER)")
    db.execute("INSERT INTO requests(ts, raw_tokens, fwd_tokens, applied) VALUES (1.0, 100, 90, 1)")
    db.commit()
    db.close()
    led = lg.Ledger(old)
    led.record(**row(api="anthropic", up_cache_read_tokens=700, up_cache_write_tokens=100))
    rows = led.rows()
    assert len(rows) == 2 and rows[0]["raw_tokens"] == 100 and rows[0]["api"] is None  # the old row survived
    assert rows[1]["api"] == "anthropic" and rows[1]["up_cache_read_tokens"] == 700
    led.close()


def test_cache_usage_is_summarised_and_discounts_the_usd_estimate(led):
    led.record(**row(api="anthropic", up_prompt_tokens=1000, up_cache_read_tokens=800, up_cache_write_tokens=100))
    s = lg.summarize(led.rows(), {"m1": {"input_per_mtok": 10.0}})
    assert s["upstream_cache_read_tokens"] == 800 and s["cache_read_share"] == 0.8 and s["by_api"]["anthropic"]["requests"] == 1
    eff = ((1000 - 900) + 800 * 0.10 + 100 * 1.25) / 1000  # blended price multiplier of that request's actual cache mix
    k = 1000 / 800
    assert s["usd_input_saved_est"] == pytest.approx(200 * k * eff / 1e6 * 10.0, abs=1e-5)
    flat = lg.summarize(led.rows(), {"m1": {"input_per_mtok": 10.0, "cache_read_mult": 1.0, "cache_write_mult": 1.0}})
    assert flat["usd_input_saved_est"] == pytest.approx(200 * k / 1e6 * 10.0, abs=1e-5)  # overridable per model
    assert "provider cache: 80.0%" in lg.format_summary(s)
