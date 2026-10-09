"""The Q&A assistant: a streaming chat over the user's book and price data.

Each question is answered by the quick-thinking model the form selects. The
prompt is grounded in facts computed here rather than in anything the model
fetches itself:

- the saved portfolio, valued like the Portfolio tab (which answers "top
  gainers today");
- price metrics for up to three tickers named in the question;
- the ratings of recent analyses in this session.

The assistant has no tools, so it cannot trade or change settings. Its reply
is rendered server-side with the same no-raw-HTML markdown as reports.

History is trimmed to the last few turns and characters, so a long
conversation does not grow the prompt without bound.
"""

from __future__ import annotations

import json
import logging
import re

from web import engine
from web.markdown_util import render_markdown

logger = logging.getLogger("tradingagents.web.chat")

MAX_TURNS = 12
MAX_CHARS = 4000
MAX_TICKERS = 3

SUGGESTIONS = [
    "What are my top gainers today?",
    "Analyze NVDA sentiment",
    "How concentrated is my portfolio?",
    "Compare AAPL and MSFT momentum",
]

_SYSTEM = (
    "You are the TradingAgents Q&A assistant inside a research app. Answer "
    "concisely in Markdown (short paragraphs, bullet lists, small tables). Use "
    "ONLY the facts in the CONTEXT block for numbers about the user's holdings "
    "and prices; if a fact is missing, say so and suggest running a full "
    "analysis from the New Analysis tab. Never invent prices, news or returns. "
    "You cannot place trades. This is research, not investment advice; say so "
    "briefly when the user asks what to buy or sell."
)

_TICKER_RE = re.compile(r"\$?\b[A-Za-z]{1,5}(?:[.-][A-Za-z]{1,4})?\b")
_STOPWORDS = {"I", "A", "AN", "AND", "THE", "MY", "ME", "IS", "IT", "TO", "OF", "IN", "ON",
              "FOR", "AI", "ETF", "USD", "CEO", "IPO", "EPS", "PE", "OK", "VS", "OR", "BE",
              "DO", "AM", "AT", "BY", "IF", "SO", "UP", "US", "WHAT", "HOW", "WHY", "ARE"}


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, default=str) + "\n\n"


def mentioned_tickers(text: str, held: set[str]) -> list[str]:
    """Tickers named in the question: held, well-known, or written as $TICKER."""
    from web.screener import NAMES

    found: list[str] = []
    for token in _TICKER_RE.findall(text):
        dollar = token.startswith("$")
        symbol = token.lstrip("$").upper()
        if dollar or symbol in held:
            accept = True
        elif token.isupper():  # "NVDA", but not the word "IS"
            accept = symbol in NAMES or (len(symbol) >= 2 and symbol not in _STOPWORDS)
        else:  # lower-case words ("my cat") are never guessed into tickers
            accept = False
        if accept and symbol not in found:
            found.append(symbol)
        if len(found) >= MAX_TICKERS:
            break
    return found


def build_context(question: str) -> tuple[str, list[str]]:
    """The CONTEXT block and the tickers it covers."""
    from web import portfolio
    from web.screener import fetch_metrics

    parts: list[str] = []
    held: set[str] = set()
    try:
        book = portfolio.valuation()
    except Exception:
        logger.debug("portfolio unavailable for chat", exc_info=True)
        book = {"empty": True}
    if not book.get("empty"):
        s = book["summary"]
        held = {h["ticker"] for h in book["holdings"]}
        lines = [f"Portfolio (prices as of {book.get('as_of')}, {book.get('currency', 'USD')}):",
                 f"- Total value {_money(s['total_value'])}; today {_signed(s['day_change'])} "
                 f"({_pct(s['day_change_pct'])}); buying power {_money(s['buying_power'])}"]
        if s.get("all_time_return") is not None:
            lines.append(f"- All-time return {_signed(s['all_time_return'])} ({_pct(s['all_time_return_pct'])})")
        for h in book["holdings"][:40]:
            lines.append(
                f"- {h['ticker']} ({h['name']}, {h['category']}): {h['quantity']:g} @ "
                f"{_money(h['price'])}, today {_pct(h['day_change_pct'])}, value "
                f"{_money(h['value'])}, gain {_pct(h['gain_pct'])}"
            )
        alloc = ", ".join(f"{a['label']} {a['pct']:.0f}%" for a in book["allocation"])
        lines.append(f"- Allocation: {alloc}")
        parts.append("\n".join(lines))
    else:
        parts.append("Portfolio: none saved (the user can add one on the Portfolio tab).")

    tickers = []
    for ticker in mentioned_tickers(question, held):
        try:
            m = fetch_metrics(ticker, None, {"horizon_days": 5, "target_pct": 5.0})
        except Exception:
            parts.append(f"{ticker}: no price data available.")
            continue
        tickers.append(ticker)
        parts.append(
            f"{ticker} ({m['name'] or 'name unknown'}): last {m['price']}, 5-session "
            f"{_pct(m['ret_5d_pct'])}, 1-month {_pct(m['ret_21d_pct'])}, annualized vol "
            f"{_pct(m['vol_ann_pct'])}, {m['dist_high_pct']}% from 52-week high, "
            f"{m['above_20ma_pct']}% vs 20-day average."
        )

    ratings = []
    for run in engine.manager.recent(limit=10):
        if run.kind == "analysis" and run.final_state:
            done = next((e for e in reversed(run.events_after(0)) if e["type"] == "done"), None)
            if done:
                ratings.append(f"{run.ticker} on {run.analysis_date}: {done.get('rating')}")
    if ratings:
        parts.append("Recent full analyses this session: " + "; ".join(ratings))
    return "\n\n".join(parts), tickers


def _clean_history(messages) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise engine.RequestError("Send at least one message.")
    clean = []
    for m in messages[-MAX_TURNS:]:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        text = str(m.get("content") or "").strip()[:MAX_CHARS]
        if text:
            clean.append({"role": m["role"], "content": text})
    if not clean or clean[-1]["role"] != "user":
        raise engine.RequestError("The last message must be the user's question.")
    return clean


def prepare(payload: dict) -> tuple[dict, list[dict]]:
    """Validate a chat request up front, so errors become HTTP 400s, not stream text."""
    if not isinstance(payload, dict):
        raise engine.RequestError("Expected a JSON object.")
    history = _clean_history(payload.get("messages"))
    config = engine.config_with_model_choices(
        engine._model_settings(payload.get("settings") or {}, default_provider=None)
    )
    if not engine.llm_key_available(config.get("llm_provider") or ""):
        raise engine.RequestError(
            f"No API key for {config.get('llm_provider')}. Add one in Settings to use the assistant."
        )
    return config, history


def stream(config: dict, history: list[dict]):
    """Yield SSE frames: ``actions`` (tickers), ``delta`` chunks, then ``done`` or ``error``."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client

    question = history[-1]["content"]
    try:
        context, tickers = build_context(question)
    except Exception:
        logger.exception("chat context failed")
        context, tickers = "No context available.", []
    if tickers:
        yield _sse({"type": "actions", "tickers": tickers})

    messages = [SystemMessage(content=f"{_SYSTEM}\n\nCONTEXT\n{context}")]
    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        messages.append(cls(content=m["content"]))

    text = ""
    try:
        llm = create_llm_client(
            provider=config["llm_provider"],
            model=config.get("quick_think_llm") or config.get("deep_think_llm"),
            base_url=config.get("backend_url"),
            **build_llm_kwargs(config),
        ).get_llm()
        for chunk in llm.stream(messages):
            delta = _chunk_text(chunk.content)
            if delta:
                text += delta
                yield _sse({"type": "delta", "text": delta})
    except Exception as exc:
        logger.warning("chat failed: %s", exc)
        yield _sse({"type": "error", "message": f"The assistant could not answer: {exc}"})
        return
    yield _sse({"type": "done", "html": render_markdown(text)})


def _chunk_text(content) -> str:
    """Stream chunks are a string, or (Anthropic, Gemini) a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                       if not isinstance(p, dict) or p.get("type") in (None, "text"))
    return ""


def _money(v) -> str:
    return "n/a" if v is None else f"${v:,.2f}"


def _signed(v) -> str:
    return "n/a" if v is None else f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"
