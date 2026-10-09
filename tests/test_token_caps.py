"""Opt-in token caps: tool payload limits, memory counts and cache-aware stats."""

from types import SimpleNamespace

import pandas as pd

from cli.stats_handler import StatsCallbackHandler
from tradingagents.dataflows import config as df_config
from tradingagents.decision_log import TradingMemoryLog
from tradingagents.default_config import DEFAULT_CONFIG


def test_caps_default_to_unchanged_behaviour():
    assert DEFAULT_CONFIG["max_stock_data_rows"] == 0
    assert DEFAULT_CONFIG["max_indicator_look_back_days"] == 0
    assert DEFAULT_CONFIG["memory_same_ticker_entries"] == 5
    assert DEFAULT_CONFIG["memory_cross_ticker_entries"] == 3


def _frame(n):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({c: 1.0 for c in ("Open", "High", "Low", "Close", "Adj Close")}
                        | {"Volume": 10}, index=idx)


def _fetch(monkeypatch, rows, cap):
    import tradingagents.dataflows.vendors.yahoo.market as market

    monkeypatch.setattr(market.yf, "Ticker",
                        lambda s: SimpleNamespace(history=lambda **k: _frame(rows)))
    monkeypatch.setattr(market, "yf_retry", lambda fn: fn())
    monkeypatch.setattr(market, "_assert_ohlcv_not_stale", lambda *a, **k: None)
    monkeypatch.setattr(market, "get_config", lambda: {"max_stock_data_rows": cap})
    return market.get_YFin_data_online("NVDA", "2026-01-01", "2026-03-01")


def test_stock_data_keeps_only_the_newest_rows(monkeypatch):
    out = _fetch(monkeypatch, rows=50, cap=10)
    assert "Total records: 10" in out and "newest 10 of 50" in out
    rows = [l for l in out.splitlines() if l.startswith("2026-")]
    assert rows[0].startswith("2026-02-10") and rows[-1].startswith("2026-02-19")


def test_stock_data_uncapped_by_default(monkeypatch):
    out = _fetch(monkeypatch, rows=50, cap=0)
    assert "Total records: 50" in out and "omitted" not in out


def test_indicator_look_back_is_clamped(monkeypatch):
    import tradingagents.agents.tools as tools

    seen = []
    monkeypatch.setattr(tools, "route_to_vendor",
                        lambda name, sym, ind, date, days: seen.append(days) or "ok")
    monkeypatch.setattr(tools, "get_config", lambda: {"max_indicator_look_back_days": 20})
    tools.get_indicators.func("NVDA", "rsi", "2026-01-05", look_back_days=365, trade_date="")
    assert seen == [20]


def test_memory_entry_counts_are_respected(tmp_path):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    for i in range(4):
        log.store_decision("NVDA", f"2026-01-0{i + 1}", "Rating: Buy")
        log.update_with_outcome("NVDA", f"2026-01-0{i + 1}", 0.01, 0.0, 5, f"lesson {i}")
    text = log.get_past_context("NVDA", n_same=2, n_cross=0)
    assert text.count("lesson") == 2 and "lesson 3" in text


def test_stats_track_cache_and_reasoning_tokens():
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    msg = AIMessage(content="x", usage_metadata={
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "input_token_details": {"cache_read": 80, "cache_creation": 5},
        "output_token_details": {"reasoning": 7},
    })
    stats = StatsCallbackHandler()
    stats.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]))
    got = stats.get_stats()
    assert (got["cache_read_tokens"], got["cache_write_tokens"], got["reasoning_tokens"]) == (80, 5, 7)
