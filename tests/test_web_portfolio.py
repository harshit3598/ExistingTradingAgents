"""Portfolio tab, charts and the Q&A assistant: the HTTP surface and the maths."""

import datetime as dt
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import web.engine as engine
from web import chat, portfolio
from web.app import app
from web.security import TOKEN_ENV

QUOTES = {"NVDA": {"price": 120.0, "prev_close": 110.0, "as_of": "2026-10-08"},
          "SPY": {"price": 500.0, "prev_close": 505.0, "as_of": "2026-10-08"}}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.setenv(portfolio.PORTFOLIO_ENV, str(tmp_path / "portfolio.json"))
    monkeypatch.setattr(portfolio, "quote", lambda t: QUOTES.get(t))
    monkeypatch.setattr(portfolio, "profile",
                        lambda t: ("S&P 500 ETF", "ETFs") if t == "SPY" else ("NVIDIA", "Technology"))
    return TestClient(app, base_url="http://localhost")


BOOK = {"cash": 1000, "positions": [
    {"ticker": "nvda", "quantity": 10, "average_price": 100},
    {"ticker": "SPY", "quantity": 2},
]}


def test_empty_portfolio(client):
    body = client.get("/api/portfolio").json()
    assert body["empty"] is True and body["source"] == "none"


def test_save_and_value_the_book(client, tmp_path):
    body = client.put("/api/portfolio", json=BOOK).json()
    s = body["summary"]
    # 10 NVDA @120 + 2 SPY @500 + 1000 cash
    assert s["total_value"] == pytest.approx(3200)
    # NVDA +10 x10, SPY -5 x2
    assert s["day_change"] == pytest.approx(90)
    assert s["day_change_pct"] == pytest.approx(90 / (2200 - 90) * 100)
    # cost basis only where an average price is known
    assert s["all_time_return"] == pytest.approx(200)
    assert s["all_time_return_pct"] == pytest.approx(20)
    assert s["buying_power"] == 1000
    assert [h["ticker"] for h in body["holdings"]] == ["NVDA", "SPY"]
    assert sum(a["pct"] for a in body["allocation"]) == pytest.approx(100)
    assert {a["label"] for a in body["allocation"]} == {"Technology", "ETFs", "Cash"}
    assert oct((tmp_path / "portfolio.json").stat().st_mode)[-3:] == "600"


@pytest.mark.parametrize("book", [
    {"positions": [{"ticker": "NVDA", "quantity": 1}, {"ticker": "nvda", "quantity": 2}]},
    {"positions": [{"ticker": "NVDA", "quantity": 0}]},
    {"positions": [{"ticker": "../etc", "quantity": 1}]},
    {"positions": [{"ticker": "NVDA", "quantity": 1, "average_price": -3}]},
    {"cash": -5, "positions": []},
])
def test_bad_books_are_rejected(client, book):
    assert client.put("/api/portfolio", json=book).status_code == 400


def test_unpriced_holdings_are_flagged(client):
    client.put("/api/portfolio", json={"positions": [{"ticker": "ZZZZ", "quantity": 3}]})
    s = client.get("/api/portfolio").json()["summary"]
    assert s["unpriced"] == ["ZZZZ"]


def test_preview_order_never_trades_even_with_live_trading_enabled(client, monkeypatch):
    monkeypatch.setenv("ROBINHOOD_TRADING_ENABLED", "true")
    client.put("/api/portfolio", json=BOOK)
    res = client.post("/api/portfolio/preview-order", json={"ticker": "NVDA", "side": "sell"})
    assert res.status_code == 200
    result = res.json()["result"]
    assert result["dry_run"] is True
    assert result["order"]["quantity"] == 10 and result["broker_response"]["status"] == "dry_run"


@pytest.mark.parametrize("body", [{"ticker": "NVDA", "side": "short"}, {"ticker": "", "side": "buy"},
                                  {"ticker": "NVDA", "side": "buy", "price": -1}])
def test_preview_order_validates(client, body):
    assert client.post("/api/portfolio/preview-order", json=body).status_code == 400


def _closes(n=120):
    """The cache's real shape: a RangeIndex, with dates in a ``Date`` column."""
    close = pd.Series([100.0 + i for i in range(n)])
    frame = pd.DataFrame({"Date": pd.date_range("2026-04-01", periods=n, freq="D"),
                          "Open": close - 1, "High": close + 2, "Low": close - 2,
                          "Close": close, "Volume": 1000})
    return frame, close


def test_chart_ranges_and_moving_averages(client, monkeypatch):
    monkeypatch.setattr(portfolio, "load_closes", lambda t, d=None: _closes())
    body = client.get("/api/chart/NVDA", params={"range": "1M"}).json()
    assert body["interval"] == "1d" and 20 <= len(body["candles"]) <= 32
    last = body["candles"][-1]
    assert last["sma20"] == pytest.approx(209.5)  # mean of the closes 200..219
    assert body["candles"][0]["sma50"] is not None  # computed on the full history
    assert body["change"] > 0
    assert len(client.get("/api/chart/NVDA", params={"range": "ALL"}).json()["candles"]) == 120
    assert body["as_of"].startswith("2026-07-29")  # a date, not a row number


def test_quote_reads_dates_from_the_cache_frame(monkeypatch):
    monkeypatch.setattr(portfolio, "load_closes", lambda t, d=None: _closes())
    monkeypatch.setattr(portfolio, "market_status", lambda now=None: "closed")
    assert portfolio.quote("NVDA") == {"price": 219.0, "prev_close": 218.0,
                                       "as_of": "2026-07-29", "live": False}


def _bars(close, prev_close):
    return [{"t": "2026-10-09T10:35:00-04:00", "o": close, "h": close, "l": close, "c": close,
             "v": 1, "sma20": None, "sma50": None, "prev_close": prev_close}]


def test_quote_uses_the_intraday_bar_while_the_market_is_open(monkeypatch):
    monkeypatch.setattr(portfolio, "load_closes", lambda t, d=None: _closes())
    monkeypatch.setattr(portfolio, "market_status", lambda now=None: "open")
    monkeypatch.setattr(portfolio, "_intraday", lambda t: _bars(231.5, 229.0))
    assert portfolio.quote("NVDA") == {"price": 231.5, "prev_close": 229.0,
                                       "as_of": "2026-10-09T10:35:00-04:00", "live": True}


def test_quote_falls_back_to_daily_when_intraday_is_unavailable(monkeypatch):
    monkeypatch.setattr(portfolio, "load_closes", lambda t, d=None: _closes())
    monkeypatch.setattr(portfolio, "market_status", lambda now=None: "open")

    def down(t):
        raise engine.RequestError("unavailable")

    monkeypatch.setattr(portfolio, "_intraday", down)
    got = portfolio.quote("NVDA")
    assert got["live"] is False and got["price"] == 219.0


def test_equities_stay_on_daily_prices_when_the_market_is_closed(monkeypatch):
    monkeypatch.setattr(portfolio, "load_closes", lambda t, d=None: _closes())
    monkeypatch.setattr(portfolio, "market_status", lambda now=None: "closed")
    monkeypatch.setattr(portfolio, "_intraday", lambda t: pytest.fail("no intraday fetch when closed"))
    assert portfolio.quote("NVDA")["live"] is False


def test_chart_validates(client):
    assert client.get("/api/chart/NVDA", params={"range": "5Y"}).status_code == 400
    assert client.get("/api/chart/%2E%2E%2Fx", params={"range": "1M"}).status_code in (400, 404)


def test_market_status():
    ny = portfolio._NY
    assert portfolio.market_status(dt.datetime(2026, 10, 10, 11, 0, tzinfo=ny)) == "closed"  # Saturday
    assert portfolio.market_status(dt.datetime(2026, 10, 9, 10, 0, tzinfo=ny)) == "open"
    assert portfolio.market_status(dt.datetime(2026, 10, 9, 16, 30, tzinfo=ny)) == "closed"


def test_robinhood_sync_needs_credentials(client, monkeypatch):
    monkeypatch.delenv("ROBINHOOD_USERNAME", raising=False)
    res = client.post("/api/portfolio/sync-robinhood", json={})
    assert res.status_code == 400 and "ROBINHOOD_USERNAME" in res.json()["detail"]


# ---------------------------------------------------------------------------
# Q&A assistant
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,held,expected", [
    ("Analyze NVDA sentiment", set(), ["NVDA"]),
    ("what is my cat doing", set(), []),
    ("how is nvda doing", {"NVDA"}, ["NVDA"]),
    ("tell me about $abc", set(), ["ABC"]),
    ("WHAT IS AN ETF", set(), []),
    ("Compare AAPL and MSFT momentum", set(), ["AAPL", "MSFT"]),
])
def test_ticker_detection(text, held, expected):
    assert chat.mentioned_tickers(text, held) == expected


def test_chat_rejects_bad_requests(client, monkeypatch):
    assert client.post("/api/chat", json={"messages": []}).status_code == 400
    assert client.post("/api/chat", json={"messages": [
        {"role": "assistant", "content": "hi"}]}).status_code == 400
    monkeypatch.setattr(engine, "llm_key_available", lambda p: False)
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                         "settings": {"llm_provider": "openai"}})
    assert res.status_code == 400 and "API key" in res.json()["detail"]


def test_chat_streams_grounded_markdown(client, monkeypatch):
    import tradingagents.llm_clients.factory as factory
    import web.screener as screener

    client.put("/api/portfolio", json=BOOK)
    seen = {}

    class FakeLLM:
        def stream(self, messages):
            seen["system"] = messages[0].content
            for part in ["**NVDA** is ", "up today. <img src=x onerror=alert(1)>"]:
                yield SimpleNamespace(content=part)

    monkeypatch.setattr(engine, "llm_key_available", lambda p: True)
    monkeypatch.setattr(factory, "create_llm_client",
                        lambda **kw: SimpleNamespace(get_llm=lambda: FakeLLM()))
    monkeypatch.setattr(screener, "fetch_metrics", lambda t, d, r: {
        "name": "NVIDIA", "price": 120, "ret_5d_pct": 3.1, "ret_21d_pct": 8.0,
        "vol_ann_pct": 40.0, "dist_high_pct": -2.0, "above_20ma_pct": 4.0})
    res = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "Analyze NVDA sentiment"}],
        "settings": {"llm_provider": "openai", "quick_think_llm": "gpt-6-luna"}})
    assert res.status_code == 200
    events = [json.loads(line[6:]) for line in res.text.split("\n") if line.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[0] == "actions" and events[0]["tickers"] == ["NVDA"]
    assert types.count("delta") == 2 and types[-1] == "done"
    html = events[-1]["html"]
    assert "<strong>NVDA</strong>" in html and "<img" not in html
    # the prompt is grounded in the saved book and the ticker's metrics
    assert "Total value $3,200.00" in seen["system"] and "NVDA (NVIDIA): last 120" in seen["system"]


def test_chat_history_is_trimmed():
    msgs = [{"role": "user", "content": "x" * 10_000}] * 30
    clean = chat._clean_history(msgs)
    assert len(clean) == chat.MAX_TURNS and all(len(m["content"]) == chat.MAX_CHARS for m in clean)
