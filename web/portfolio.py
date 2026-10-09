"""The Portfolio tab: a saved book, valued from the cached data layer.

Holdings come from one JSON file on the server (``~/.tradingagents/portfolio.json``
by default). The user edits it in the UI or fills it from Robinhood with a
one-shot, read-only sync. Every read values the book from the same cached
daily prices the analysts use, so nothing here needs a broker session.

Charts use those daily bars too. Only the 1D view fetches intraday bars, and
it caches them for a minute.

Quick Buy/Sell actions are previews only: they run the execution pipeline
against the dry-run broker and never send an order.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import functools
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from cli.prompts import detect_asset_type, is_valid_ticker_input
from tradingagents.portfolio import PortfolioContext, Position
from web.engine import RequestError
from web.screener import NAMES, load_closes

logger = logging.getLogger("tradingagents.web.portfolio")

PORTFOLIO_ENV = "TRADINGAGENTS_PORTFOLIO_PATH"
_NY = ZoneInfo("America/New_York")
_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "SMH", "TQQQ", "SOXL", "XLK", "XLF",
         "XLE", "ARKK", "GLD", "SLV", "TLT", "HYG", "EEM", "VEA", "SCHD"}
RANGES = {"1D": None, "1W": 7, "1M": 31, "1Y": 366, "ALL": None}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def portfolio_path() -> Path:
    default = Path.home() / ".tradingagents" / "portfolio.json"
    return Path(os.environ.get(PORTFOLIO_ENV) or default)


def load_saved() -> dict:
    """``{"portfolio": PortfolioContext | None, "source", "updated_at", "names"}``."""
    path = portfolio_path()
    if not path.is_file():
        return {"portfolio": None, "source": "none", "updated_at": None, "names": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        book = PortfolioContext.model_validate(raw.get("portfolio") or {})
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("saved portfolio unreadable: %s", exc)
        return {"portfolio": None, "source": "none", "updated_at": None, "names": {},
                "error": "The saved portfolio file could not be read."}
    return {"portfolio": book, "source": raw.get("source") or "manual",
            "updated_at": raw.get("updated_at"), "names": raw.get("names") or {}}


def save(book: PortfolioContext, source: str = "manual", names: dict | None = None) -> None:
    """Write the book atomically, readable by this user only (it is account data)."""
    path = portfolio_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "portfolio": book.model_dump(mode="json"),
        "source": source,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "names": names or {},
    }
    tmp = path.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def parse_book(body: dict) -> PortfolioContext:
    """Validate a submitted book: known-good tickers, finite numbers, no duplicates."""
    if not isinstance(body, dict):
        raise RequestError("Expected a JSON object.")
    try:
        book = PortfolioContext.model_validate(body)
    except ValidationError as exc:
        raise RequestError(f"Portfolio is not usable: {exc.errors()[0]['msg']}") from None
    if book.cash is not None and (not math.isfinite(book.cash) or book.cash < 0):
        raise RequestError("Cash must be a non-negative number.")
    seen = set()
    for p in book.positions:
        p.ticker = p.ticker.strip().upper()
        if not p.ticker or not is_valid_ticker_input(p.ticker):
            raise RequestError(f"Not a usable ticker: {p.ticker!r}")
        if p.ticker in seen:
            raise RequestError(f"{p.ticker} is listed twice.")
        seen.add(p.ticker)
        if not math.isfinite(p.quantity) or p.quantity == 0:
            raise RequestError(f"{p.ticker}: quantity must be a non-zero number.")
        if p.average_price is not None and (not math.isfinite(p.average_price) or p.average_price <= 0):
            raise RequestError(f"{p.ticker}: average cost must be positive.")
    if len(book.positions) > 200:
        raise RequestError("200 positions at most.")
    return book


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

def market_status(now: dt.datetime | None = None) -> str:
    """``open`` during regular US hours on a weekday, else ``closed`` (holidays not modelled)."""
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(_NY)
    if now.weekday() >= 5:
        return "closed"
    return "open" if dt.time(9, 30) <= now.time() < dt.time(16, 0) else "closed"


def dated_closes(ticker: str):
    """``load_closes`` re-indexed by date (the cache keeps dates in a column)."""
    import pandas as pd

    frame, close = load_closes(ticker)
    date_col = next((c for c in frame.columns if str(c).lower() in ("date", "datetime")), None)
    if date_col is not None:
        idx = pd.DatetimeIndex(pd.to_datetime(frame.loc[close.index, date_col]))
        frame = frame.loc[close.index].set_index(idx)
        close = pd.Series(close.to_numpy(), index=idx)
    return frame, close


def quote(ticker: str) -> dict | None:
    """Best available price: the latest intraday bar while trading, else the daily close.

    ``live`` says which it is. Intraday bars come from Yahoo (delayed by up to
    ~15 minutes) and are cached for a minute, shared with the 1D chart.
    """
    try:
        _, close = dated_closes(ticker)
    except Exception as exc:  # offline / unknown symbol
        logger.debug("no quote for %s: %s", ticker, exc)
        return None
    if close.empty:
        return None
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    daily = {"price": last, "prev_close": prev, "as_of": str(close.index[-1])[:10], "live": False}

    if market_status() == "open" or detect_asset_type(ticker).value == "crypto":
        try:
            bars = _intraday(ticker)
        except RequestError:
            bars = []
        if bars:
            return {"price": bars[-1]["c"], "prev_close": bars[0].get("prev_close") or prev,
                    "as_of": bars[-1]["t"], "live": True}
    return daily


@functools.lru_cache(maxsize=512)
def profile(ticker: str) -> tuple[str, str]:
    """``(name, category)`` for allocation. Best-effort; cached for the process."""
    if detect_asset_type(ticker).value == "crypto":
        return NAMES.get(ticker, ticker.split("-")[0]), "Crypto"
    if ticker in _ETFS:
        return NAMES.get(ticker, ticker), "ETFs"
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or NAMES.get(ticker, ticker)
        if str(info.get("quoteType", "")).upper() == "ETF":
            return name, "ETFs"
        return name, info.get("sector") or "Stocks"
    except Exception:
        return NAMES.get(ticker, ticker), "Stocks"


def valuation() -> dict:
    """The Portfolio tab payload: summary, allocation and holdings."""
    saved = load_saved()
    book: PortfolioContext | None = saved["portfolio"]
    base = {"source": saved["source"], "updated_at": saved["updated_at"],
            "market_status": market_status(), "error": saved.get("error")}
    if book is None:
        return {**base, "empty": True, "holdings": [], "allocation": [], "summary": None}

    tickers = [p.ticker.upper() for p in book.positions]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        quotes = dict(zip(tickers, pool.map(quote, tickers)))
        profiles = dict(zip(tickers, pool.map(profile, tickers)))

    holdings, as_of, any_live = [], None, False
    total_value = day_change = cost_basis = gain = 0.0
    for p in book.positions:
        t = p.ticker.upper()
        q = quotes.get(t)
        name, category = profiles.get(t, (t, "Stocks"))
        name = saved["names"].get(t) or name
        row = {"ticker": t, "name": name, "category": category, "quantity": p.quantity,
               "average_price": p.average_price, "price": None, "day_change": None,
               "day_change_pct": None, "value": None, "gain": None, "gain_pct": None}
        if q:
            as_of = max(as_of or q["as_of"], q["as_of"])
            any_live = any_live or q.get("live", False)
            value = p.quantity * q["price"]
            change = p.quantity * (q["price"] - q["prev_close"])
            row.update(price=q["price"], value=value, day_change=change,
                       day_change_pct=_pct(q["price"] - q["prev_close"], q["prev_close"]))
            total_value += value
            day_change += change
            if p.average_price:
                g = p.quantity * (q["price"] - p.average_price)
                row.update(gain=g, gain_pct=_pct(q["price"] - p.average_price, p.average_price))
                gain += g
                cost_basis += abs(p.quantity) * p.average_price
        holdings.append(row)

    cash = book.cash or 0.0
    equity = total_value + cash
    holdings.sort(key=lambda r: r["value"] or 0, reverse=True)

    buckets: dict[str, float] = {}
    for row in holdings:
        if row["value"]:
            buckets[row["category"]] = buckets.get(row["category"], 0.0) + abs(row["value"])
    if cash:
        buckets["Cash"] = cash
    gross = sum(buckets.values()) or 1.0
    allocation = sorted(({"label": k, "value": v, "pct": v / gross * 100} for k, v in buckets.items()),
                        key=lambda a: a["value"], reverse=True)
    by_holding = [{"label": r["ticker"], "value": abs(r["value"]), "pct": abs(r["value"]) / gross * 100}
                  for r in holdings if r["value"]]
    if cash:
        by_holding.append({"label": "Cash", "value": cash, "pct": cash / gross * 100})

    return {
        **base,
        "empty": False,
        "as_of": as_of,
        "live": any_live,
        "currency": book.currency or "USD",
        "summary": {
            "total_value": equity,
            "positions_value": total_value,
            "day_change": day_change,
            "day_change_pct": _pct(day_change, total_value - day_change),
            "all_time_return": gain if cost_basis else None,
            "all_time_return_pct": _pct(gain, cost_basis) if cost_basis else None,
            "buying_power": book.cash,
            "unpriced": [r["ticker"] for r in holdings if r["price"] is None],
        },
        "allocation": allocation,
        "allocation_by_holding": by_holding,
        "holdings": holdings,
        "book": book.model_dump(mode="json"),
    }


def _pct(delta: float, base: float) -> float | None:
    return delta / base * 100 if base else None


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

_intraday_cache: dict[str, tuple[float, list[dict]]] = {}


def chart(ticker: str, range_key: str) -> dict:
    ticker = ticker.strip().upper()
    if not ticker or not is_valid_ticker_input(ticker):
        raise RequestError("Not a usable ticker.")
    range_key = range_key.upper()
    if range_key not in RANGES:
        raise RequestError("Range must be one of 1D, 1W, 1M, 1Y, ALL.")
    candles = _intraday(ticker) if range_key == "1D" else _daily(ticker, RANGES[range_key])
    if not candles:
        raise RequestError(f"No price history for {ticker}.")
    first, last = candles[0], candles[-1]
    base = first.get("prev_close") or first["o"]
    return {
        "ticker": ticker,
        "name": NAMES.get(ticker, ""),
        "range": range_key,
        "interval": "5m" if range_key == "1D" else "1d",
        "candles": candles,
        "last": last["c"],
        "change": last["c"] - base,
        "change_pct": _pct(last["c"] - base, base),
        "market_status": market_status(),
        "as_of": last["t"],
    }


def _bars(frame, close) -> list[dict]:
    cols = {str(c).lower(): c for c in frame.columns}
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    out = []
    for ts in close.index:
        row = frame.loc[ts]
        num = lambda key: _finite(row[cols[key]]) if key in cols else None  # noqa: E731
        out.append({
            "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "o": num("open") or float(close[ts]), "h": num("high") or float(close[ts]),
            "l": num("low") or float(close[ts]), "c": float(close[ts]),
            "v": num("volume") or 0, "sma20": _finite(sma20[ts]), "sma50": _finite(sma50[ts]),
        })
    return out


def _daily(ticker: str, days: int | None) -> list[dict]:
    try:
        frame, close = dated_closes(ticker)
    except Exception as exc:
        raise RequestError(f"No price history for {ticker}.") from exc
    bars = _bars(frame, close)  # moving averages see the full history
    if days is not None:
        cutoff = close.index[-1] - dt.timedelta(days=days)
        prior = [b for b, ts in zip(bars, close.index) if ts <= cutoff]
        bars = [b for b, ts in zip(bars, close.index) if ts > cutoff]
        if bars and prior:
            bars[0]["prev_close"] = prior[-1]["c"]
    return bars


def _intraday(ticker: str) -> list[dict]:
    cached = _intraday_cache.get(ticker)
    if cached and time.time() - cached[0] < 60:
        return cached[1]
    import pandas as pd
    import yfinance as yf

    from tradingagents.dataflows.symbols import normalize_symbol

    try:
        frame = yf.Ticker(normalize_symbol(ticker)).history(period="5d", interval="5m")
    except Exception as exc:
        raise RequestError(f"Intraday prices for {ticker} are unavailable right now.") from exc
    if frame is None or frame.empty:
        return []
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(_NY)
    sessions = sorted({ts.date() for ts in frame.index})
    prev_close = None
    if len(sessions) > 1:
        prev_close = float(frame[[ts.date() == sessions[-2] for ts in frame.index]]["Close"].iloc[-1])
    frame = frame[[ts.date() == sessions[-1] for ts in frame.index]]
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    bars = _bars(frame.loc[close.index], close)
    if bars and prev_close:
        bars[0]["prev_close"] = prev_close
    with _lock:
        _intraday_cache[ticker] = (time.time(), bars)
    return bars


def _finite(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Dry-run order preview
# ---------------------------------------------------------------------------

def preview_order(ticker: str, side: str, price: float | None = None) -> dict:
    """Walk risk → sizing → DryRunBroker for a quick Buy/Sell. Never trades."""
    from tradingagents.execution import create_orchestrator

    ticker = ticker.strip().upper()
    if not is_valid_ticker_input(ticker) or not ticker:
        raise RequestError("Not a usable ticker.")
    if side not in ("buy", "sell"):
        raise RequestError("side must be buy or sell.")
    if price is not None and not (0 < price < 1e7):
        raise RequestError("price must be a positive number.")
    if price is None:
        q = quote(ticker)
        price = q["price"] if q else None
    book = load_saved()["portfolio"]
    # trading_enabled is pinned False: this endpoint can never reach a broker.
    orchestrator = create_orchestrator({"trading_enabled": False, "default_price": price})
    result = orchestrator.execute("Buy" if side == "buy" else "Sell", ticker, book, decision_price=price)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Robinhood sync (read-only)
# ---------------------------------------------------------------------------

def _refuse_prompt(*_args, **_kwargs):
    raise RuntimeError("Robinhood asked for a verification code. Enter your 2FA code and sync again.")


def sync_robinhood(mfa_code: str | None = None, timeout: float = 60.0) -> dict:
    """Copy Robinhood positions and buying power into the saved book. Reads only."""
    if not (os.environ.get("ROBINHOOD_USERNAME") and os.environ.get("ROBINHOOD_PASSWORD")):
        raise RequestError("Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in .env to sync.")
    if mfa_code and not (mfa_code.isdigit() and len(mfa_code) <= 8):
        raise RequestError("The 2FA code must be up to 8 digits.")
    try:
        import robin_stocks.robinhood as rh
        import robin_stocks.robinhood.authentication as rh_auth
    except ImportError:
        raise RequestError('Install the extra first: pip install "tradingagents[robinhood]".') from None

    from tradingagents.execution import RobinhoodBroker

    def pull():
        # robin_stocks falls back to input() for challenges, which would block
        # a server thread on the terminal forever; fail fast instead.
        rh_auth.input = _refuse_prompt
        try:
            broker = RobinhoodBroker(mfa_code=mfa_code, dry_run=True)
            if not broker.login():
                raise RuntimeError("Robinhood login failed. Check the credentials and 2FA code.")
            holdings = rh.build_holdings() or {}
            profile_data = rh.load_account_profile() or {}
            return holdings, profile_data
        finally:
            rh_auth.__dict__.pop("input", None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(pull)
        try:
            holdings, profile_data = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise RequestError("Robinhood did not answer in time; try again.") from None
        except RuntimeError as exc:
            raise RequestError(str(exc)) from None

    positions, names = [], {}
    for symbol, h in holdings.items():
        try:
            qty = float(h.get("quantity") or 0)
        except (TypeError, ValueError):
            continue
        if qty and is_valid_ticker_input(symbol):
            avg = _finite(h.get("average_buy_price"))
            positions.append(Position(ticker=symbol.upper(), quantity=qty, average_price=avg or None))
            if h.get("name"):
                names[symbol.upper()] = str(h["name"])[:80]
    cash = _finite(profile_data.get("buying_power") or profile_data.get("cash"))
    book = PortfolioContext(cash=cash, currency="USD", positions=positions)
    save(book, source="robinhood", names=names)
    return {"positions": len(positions), "cash": cash}
