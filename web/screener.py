"""Stock suggestions from a goal: "what could give me +10% within a week".

Three stages, each publishing events into the run's log so the browser can
show progress like it does for analyses:

1. **Fetch** — recent price history for a universe of liquid tickers, through
   the project's cached data layer (``load_ohlcv``), in parallel.
2. **Score** — rank candidates on whether their typical movement over the
   requested horizon can even reach the target, plus momentum, trend
   strength and liquidity, all tuned by the requested risk appetite.
3. **Curate** — the quick-thinking LLM picks and explains the shortlist from
   those metrics *only* (it never invents numbers), falling back to
   metric-derived text when no LLM is available.

This is a research screen, not a promise: the result says so in as many
words, and the disclaimer travels with the payload.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import re
from typing import Any

from pydantic import BaseModel, Field

from cli.prompts import is_valid_ticker_input
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from web.engine import RequestError, _model_settings, config_with_model_choices, llm_key_available

logger = logging.getLogger("tradingagents.web.screener")

DISCLAIMER = (
    "Research screen, not investment advice. Momentum and volatility describe "
    "the past, not the future — short-horizon price targets are guesses, and "
    "most moves of this size are noise. Size positions as if you can be wrong."
)

# A liquid default universe: mega-caps for stability, high-beta names and
# leveraged ETFs where a +10%-in-a-week move is even plausible, plus majors
# from crypto. Users can replace it with their own list in the form.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "NFLX", "CRM", "ADBE", "ORCL", "QCOM", "MU", "SMCI", "ARM", "INTC",
    "JPM", "V", "BAC", "XOM", "CVX", "UNH", "JNJ", "LLY", "WMT", "COST",
    "KO", "PEP", "HD", "MCD", "NKE", "SBUX", "DIS", "BA", "CAT", "GE",
    "PLTR", "SOFI", "COIN", "MARA", "RIOT", "UBER", "SNAP",
    "SPY", "QQQ", "SMH", "TQQQ", "SOXL",
    "BTC-USD", "ETH-USD", "SOL-USD",
]

NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom",
    "AMD": "AMD", "NFLX": "Netflix", "CRM": "Salesforce", "ADBE": "Adobe",
    "ORCL": "Oracle", "QCOM": "Qualcomm", "MU": "Micron", "SMCI": "Super Micro",
    "ARM": "Arm Holdings", "INTC": "Intel", "JPM": "JPMorgan", "V": "Visa",
    "BAC": "Bank of America", "XOM": "Exxon Mobil", "CVX": "Chevron",
    "UNH": "UnitedHealth", "JNJ": "J&J", "LLY": "Eli Lilly", "WMT": "Walmart",
    "COST": "Costco", "KO": "Coca-Cola", "PEP": "PepsiCo", "HD": "Home Depot",
    "MCD": "McDonald's", "NKE": "Nike", "SBUX": "Starbucks", "DIS": "Disney",
    "BA": "Boeing", "CAT": "Caterpillar", "GE": "GE", "PLTR": "Palantir",
    "SOFI": "SoFi", "COIN": "Coinbase", "MARA": "MARA Holdings", "RIOT": "Riot Platforms",
    "UBER": "Uber", "SNAP": "Snap", "SPY": "S&P 500 ETF", "QQQ": "Nasdaq-100 ETF",
    "SMH": "Semiconductor ETF", "TQQQ": "3x Nasdaq-100 ETF", "SOXL": "3x Semis ETF",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
}

# Preferred ratio of horizon-sized volatility to the requested target,
# per risk appetite: how much room the candidate should give you.
RISK_BANDS = {
    "low": (0.6, 1.5),     # enough movement, but not a casino
    "medium": (1.0, 3.0),
    "high": (1.5, 6.0),    # wants plenty of room to run (and to fall)
}

# Worded horizons the free-text goal parser understands.
_PHRASE_HORIZONS = (
    ("tomorrow", 1), ("overnight", 1), ("next week", 7), ("this week", 7),
    ("a week", 7), ("one week", 7), ("next month", 30), ("a month", 30),
    ("one month", 30), ("next quarter", 90), ("two weeks", 14), ("2 weeks", 14),
)
_UNITS = {
    "d": 1, "day": 1, "days": 1,
    "w": 7, "wk": 7, "week": 7, "weeks": 7,
    "m": 30, "mo": 30, "month": 30, "months": 30,
}

_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
_DAYS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(day|days|d|week|weeks|wk|w|month|months|mo|m)\b",
                      re.IGNORECASE)


class StockPick(BaseModel):
    """One curated suggestion, explained only from the metrics provided."""

    ticker: str = Field(description="Ticker symbol, exactly as listed in the candidates.")
    rationale: str = Field(
        description="Why this candidate fits the goal, citing its provided metrics. "
                    "One or two sentences."
    )
    risks: str = Field(
        description="The main reason this call could be wrong, one sentence."
    )


class PickList(BaseModel):
    picks: list[StockPick] = Field(description="Best-first shortlist.")


# ---------------------------------------------------------------------------
# Goal parsing and request validation
# ---------------------------------------------------------------------------

def parse_goal(text: str) -> dict:
    """Extract (target %, horizon days) from free text. Missing parts omitted.

    ``"10% profit in next 1 week"`` → ``{"target_pct": 10.0, "horizon_days": 7}``.
    """
    text = str(text or "")
    found: dict[str, Any] = {}

    pct = _PCT_RE.search(text)
    if pct:
        found["target_pct"] = float(pct.group(1))

    days = _DAYS_RE.search(text)
    if days:
        found["horizon_days"] = int(round(float(days.group(1)) * _UNITS[days.group(2).lower()]))
    else:
        lowered = text.lower()
        for phrase, horizon in _PHRASE_HORIZONS:
            if phrase in lowered:
                found["horizon_days"] = horizon
                break
    return found


def _parse_universe(value) -> list[str]:
    """Accept a list or comma/newline-separated text; blank means default."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = [str(v) for v in value]
    else:
        raw = re.split(r"[,\n;]+", str(value))
    tickers = []
    for item in raw:
        item = item.strip().upper()
        if not item:
            continue
        if not is_valid_ticker_input(item):
            raise RequestError(f"Not a usable ticker: {item}")
        if item not in tickers:
            tickers.append(item)
    if len(tickers) > 120:
        raise RequestError("Universe is too large (120 tickers max).")
    return tickers


# Imported at module scope: engine only imports this module lazily (inside
# ``start_suggest``), so the import graph stays acyclic.
def normalize_suggestion_request(payload: dict) -> dict:
    """Validate the suggestions form; raises ``RequestError`` with a user message."""
    if not isinstance(payload, dict):
        raise RequestError("Expected a JSON object.")

    goal = str(payload.get("goal") or "").strip()[:300]
    parsed = parse_goal(goal)

    def _number(key, default, lo, hi, label, cast):
        raw = payload.get(key)
        if raw in (None, ""):
            raw = parsed.get(key, default)
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            raise RequestError(f"{label} must be a number.") from None
        if not (lo <= value <= hi):
            raise RequestError(f"{label} must be between {lo} and {hi}.")
        return value

    target_pct = _number("target_pct", 10.0, 0.5, 100.0, "Target gain", float)
    horizon_days = _number("horizon_days", 7, 1, 90, "Horizon", int)
    count = _number("count", 5, 1, 12, "Suggestion count", int)

    risk = str(payload.get("risk") or "medium").lower()
    if risk not in RISK_BANDS:
        raise RequestError("Risk appetite must be low, medium or high.")

    universe = _parse_universe(payload.get("universe")) or list(DEFAULT_UNIVERSE)

    return {
        "goal": goal,
        "target_pct": target_pct,
        "horizon_days": horizon_days,
        "count": count,
        "risk": risk,
        "universe": universe,
        **_model_settings(payload, default_provider=None),
    }


# ---------------------------------------------------------------------------
# Metrics and scoring
# ---------------------------------------------------------------------------

def _column(frame, name: str):
    return next((c for c in frame.columns if str(c).lower() == name), None)


def load_closes(ticker: str, curr_date: str | None = None):
    """``(frame, close series)`` from the cached data layer; raises when unusable."""
    import pandas as pd

    frame = load_ohlcv(ticker, curr_date or get_current_date(), fill_gaps=True)
    if frame is None or frame.empty:
        raise ValueError("no price history")
    close_col = _column(frame, "close")
    if close_col is None:
        raise ValueError("no close column")
    return frame, pd.to_numeric(frame[close_col], errors="coerce").dropna()


def fetch_metrics(ticker: str, curr_date: str, req: dict) -> dict:
    """Price-derived metrics for one ticker. Raises when the symbol is unusable."""
    import pandas as pd

    frame, close = load_closes(ticker, curr_date)
    if len(close) < 6:
        raise ValueError("not enough history")

    last = float(close.iloc[-1])
    horizon = req["horizon_days"]
    target = req["target_pct"]

    rets = close.pct_change().dropna()
    recent = rets.tail(60)
    vol_ann = float(recent.std() * math.sqrt(252) * 100) if len(recent) > 5 else None
    horizon_vol = vol_ann * math.sqrt(horizon / 252) if vol_ann else None

    ret_5d = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else None
    ret_21d = (last / float(close.iloc[-22]) - 1) * 100 if len(close) >= 22 else None
    dist_high = (last / float(close.tail(252).max()) - 1) * 100
    above_20 = (last / float(close.tail(20).mean()) - 1) * 100

    dollar_vol_m = None
    vol_col = _column(frame, "volume")
    if vol_col is not None:
        volume = pd.to_numeric(frame[vol_col], errors="coerce")
        tail = volume.tail(20)
        if len(tail) and tail.notna().any():
            dollar_vol_m = float((tail * close.tail(20)).mean()) / 1e6

    return {
        "ticker": ticker,
        "name": NAMES.get(ticker, ""),
        "price": round(last, 2),
        "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
        "ret_21d_pct": round(ret_21d, 2) if ret_21d is not None else None,
        "vol_ann_pct": round(vol_ann, 1) if vol_ann else None,
        "horizon_vol_pct": round(horizon_vol, 1) if horizon_vol else None,
        "dist_high_pct": round(dist_high, 1),
        "above_20ma_pct": round(above_20, 1),
        "dollar_vol_m": round(dollar_vol_m, 0) if dollar_vol_m else None,
        "target_pct": target,
        "horizon_days": horizon,
    }


def score_candidate(metrics: dict, risk: str) -> float:
    """0–100: can this candidate's typical horizon movement reach the target,
    is it already moving the right way, is it in a position of strength, and
    is it liquid enough to trade?"""
    target = max(metrics.get("target_pct") or 10.0, 0.5)
    horizon_vol = metrics.get("horizon_vol_pct")
    ret_5d = metrics.get("ret_5d_pct")
    dist_high = metrics.get("dist_high_pct")
    dollar_vol_m = metrics.get("dollar_vol_m")

    # Reach: horizon-sized (1σ) movement vs the requested gain.
    if horizon_vol is None:
        reach = 0.5
    else:
        ratio = horizon_vol / target
        lo, hi = RISK_BANDS[risk]
        if ratio < lo:
            reach = max(0.0, ratio / lo)          # too quiet to get there
        elif ratio > hi:
            reach = max(0.35, hi / ratio)         # roomy, but wilder than asked
        else:
            reach = 1.0

    # Momentum: recent movement in the direction we need, scaled by the goal.
    # Neutral (0.5) when flat or unknown; +target% over 5 sessions → 1.0.
    momentum = 0.5 if ret_5d is None else max(0.0, min(1.0, 0.5 + ret_5d / (2 * target)))

    # Strength: proximity to the 252-day high (0% below → 1.0, 40%+ → 0.0).
    strength = 0.5 if dist_high is None else max(0.0, min(1.0, (40 + dist_high) / 40))

    # Liquidity: log-scaled average daily dollar volume, in $M/day.
    # $50M → 0.23, $100M → 0.37, $500M → 0.80, $1B+ → 1.0.
    if dollar_vol_m:
        liquidity = max(0.0, min(1.0, math.log10(1 + dollar_vol_m / 50.0) / 1.3))
    else:
        liquidity = 0.5

    return round(100 * (0.40 * reach + 0.25 * momentum + 0.20 * strength + 0.15 * liquidity), 1)


def _metric_rationale(metrics: dict) -> tuple[str, str]:
    """Explanation built purely from the computed metrics (LLM-less fallback)."""
    parts = []
    if metrics.get("ret_5d_pct") is not None:
        parts.append(f"{metrics['ret_5d_pct']:+.1f}% over the last 5 sessions")
    if metrics.get("ret_21d_pct") is not None:
        parts.append(f"{metrics['ret_21d_pct']:+.1f}% over the last month")
    if metrics.get("horizon_vol_pct") is not None:
        parts.append(
            f"typically moves ±{metrics['horizon_vol_pct']:.1f}% over your "
            f"{metrics['horizon_days']}-day horizon (1σ)"
        )
    if metrics.get("dist_high_pct") is not None:
        parts.append(f"{metrics['dist_high_pct']:.1f}% from its 252-day high")
    rationale = "Quantitative screen: " + "; ".join(parts) + "."

    risks = (
        f"{metrics['vol_ann_pct']:.0f}% annualized volatility" if metrics.get("vol_ann_pct")
        else "Unknown volatility"
    ) + " — a move this size can just as easily go the other way."
    return rationale, risks


# ---------------------------------------------------------------------------
# LLM curation
# ---------------------------------------------------------------------------

def _llm_available(config: dict) -> bool:
    """Is there a key for the configured provider (or does it not need one)?"""
    return llm_key_available(config.get("llm_provider") or "")


def _build_prompt(req: dict, shortlist: list[dict]) -> str:
    header = (
        f"Goal: gain +{req['target_pct']}% within {req['horizon_days']} days "
        f"(risk appetite: {req['risk']})."
    )
    if req.get("goal"):
        header += f' Stated in the user\'s words: "{req["goal"]}".'
    rows = [
        (
            f"- {m['ticker']}: price {m['price']}, 5-session change "
            f"{m['ret_5d_pct']}%, 1-month change {m['ret_21d_pct']}%, "
            f"annualized vol {m['vol_ann_pct']}%, expected move over the "
            f"{req['horizon_days']}-day horizon ±{m['horizon_vol_pct']}%, "
            f"{m['dist_high_pct']}% from 252-day high, "
            f"distance from 20-day average {m['above_20ma_pct']}%, "
            f"avg daily dollar volume ${m['dollar_vol_m']}M, score {m['score']}."
        )
        for m in shortlist
    ]
    return (
        f"{header}\n\n"
        "Scored candidates (metrics computed from price history — treat them as "
        "the only evidence):\n" + "\n".join(rows) + "\n\n"
        f"Pick the {min(req['count'], len(shortlist))} best fits, best first. "
        "Only these tickers, only these numbers. For each, say in one or two "
        "sentences why it fits the goal (cite its metrics) and one sentence on "
        "the main risk. Do not promise returns; describe fit and risk."
    )


def curate_picks(config: dict, req: dict, shortlist: list[dict]) -> list[dict] | None:
    """LLM-ordered shortlist with rationale/risks, or None to fall back."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from tradingagents.agents.structured import bind_structured
    from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client

    if not shortlist or not _llm_available(config):
        return None
    try:
        client = create_llm_client(
            provider=config["llm_provider"],
            model=config.get("quick_think_llm") or config.get("deep_think_llm"),
            base_url=config.get("backend_url"),
            **build_llm_kwargs(config),
        )
        llm = client.get_llm()
        structured = bind_structured(llm, PickList, "StockScreener")
        if structured is None:
            return None
        messages = [
            SystemMessage(content=(
                "You are a quantitative screening assistant. You never invent "
                "numbers and never promise returns; you rank the provided "
                "candidates against the stated goal and state the risk."
            )),
            HumanMessage(content=_build_prompt(req, shortlist)),
        ]
        result = structured.invoke(messages)
        if result is None or not getattr(result, "picks", None):
            return None
        order = {m["ticker"].upper(): m for m in shortlist}
        picks = []
        for pick in result.picks:
            key = pick.ticker.strip().upper()
            if key in order:
                picks.append({
                    "ticker": order[key]["ticker"],
                    "rationale": pick.rationale.strip(),
                    "risks": pick.risks.strip(),
                })
        return picks or None
    except Exception as exc:  # any LLM failure falls back, never blocks the screen
        logger.warning("LLM curation unavailable: %s", exc)
        return None


def finalize_suggestions(ranked: list[dict], picks: list[dict] | None,
                         req: dict) -> list[dict]:
    """Merge LLM picks with the score ranking; metric text when uncurated."""
    by_ticker = {m["ticker"].upper(): dict(m) for m in ranked}
    ordered: list[dict] = []
    seen: set[str] = set()

    for pick in picks or []:
        row = by_ticker.get(pick["ticker"].upper())
        if row is None or row["ticker"].upper() in seen:
            continue
        seen.add(row["ticker"].upper())
        row.update(rationale=pick["rationale"], risks=pick["risks"], llm_curated=True)
        ordered.append(row)

    for row in ranked:
        if row["ticker"].upper() in seen or len(ordered) >= req["count"]:
            continue
        seen.add(row["ticker"].upper())
        rationale, risks = _metric_rationale(row)
        row.update(rationale=rationale, risks=risks, llm_curated=False)
        ordered.append(row)

    return ordered[: req["count"]]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_suggest(run) -> None:
    """The suggest run: fetch → score → curate, streaming progress events."""
    req = run.payload
    run.set_status("running")
    config = config_with_model_choices(req)
    curr_date = get_current_date()
    universe = req["universe"]

    run.emit(
        "message", kind="System",
        content=(f"Screening {len(universe)} tickers for +{req['target_pct']}% "
                 f"within {req['horizon_days']} day(s) · risk={req['risk']}"),
    )
    run.emit("progress", stage="fetch", done=0, total=len(universe))

    metrics: list[dict] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(fetch_metrics, ticker, curr_date, req): ticker
            for ticker in universe
        }
        for future in concurrent.futures.as_completed(futures):
            ticker = futures[future]
            try:
                metrics.append(future.result())
            except Exception as exc:  # one dead symbol must not kill the screen
                failures.append(ticker)
                logger.debug("no metrics for %s: %s", ticker, exc)
            run.emit("progress", stage="fetch", done=len(metrics) + len(failures),
                     total=len(universe))

    if not metrics:
        raise RuntimeError(
            "Could not fetch price data for any candidate "
            f"({len(failures)} symbols failed) — check connectivity or the universe list."
        )
    run.emit("message", kind="System",
             content=f"Priced {len(metrics)} candidates ({len(failures)} unavailable).")

    for row in metrics:
        row["score"] = score_candidate(row, req["risk"])
    ranked = sorted(metrics, key=lambda r: r["score"], reverse=True)
    shortlist = ranked[: max(req["count"] * 3, 12)]

    run.emit("message", kind="System",
             content=("Top of the screen: " +
                      ", ".join(f"{r['ticker']} {r['score']:.0f}" for r in shortlist[:5])))
    run.emit("progress", stage="score", done=len(metrics), total=len(metrics))

    picks = None
    if _llm_available(config):
        run.emit("message", kind="System",
                 content=(f"Curating the shortlist with {config['llm_provider']}/"
                          f"{config.get('quick_think_llm')}…"))
        picks = curate_picks(config, req, shortlist)
        if picks is None:
            run.emit("message", kind="System",
                     content="LLM curation unavailable — using the quantitative ranking as-is.")
    else:
        run.emit("message", kind="System",
                 content="No API key for the configured provider — using the quantitative ranking.")

    suggestions = finalize_suggestions(shortlist, picks, req)
    run.emit(
        "done",
        request={k: req[k] for k in
                 ("goal", "target_pct", "horizon_days", "count", "risk")},
        suggestions=suggestions,
        disclaimer=DISCLAIMER,
        universe_size=len(universe),
        scored=len(metrics),
        failed=len(failures),
        llm_used=picks is not None,
    )
    run.set_status("done")
