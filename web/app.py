"""The TradingAgents web app: a browser front-end over the same graph the CLI runs.

Endpoints are JSON under ``/api``; everything else is the single-page UI in
``web/static``. Runs execute on background threads and stream to the browser as
server-sent events, so the page can be refreshed mid-run and pick the stream up
again from where it left off.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cli.announcements import fetch_announcements
from cli.prefs import load_last_run, sanitize
from cli.prompts import _OPENROUTER_MAINSTREAM, _fetch_openrouter_models, _llm_provider_table
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.checkpointer import clear_all_checkpoints
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.model_catalog import get_model_options
from web import chat, engine, portfolio, screener
from web.apikeys import save_api_key
from web.markdown_util import render_markdown
from web.security import SecurityMiddleware, ensure_token, is_loopback

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="TradingAgents", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SecurityMiddleware)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

ANALYST_LABELS = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}

DEPTH_OPTIONS = [
    {"value": 1, "label": "Shallow", "hint": "Quick research, few debate rounds"},
    {"value": 3, "label": "Medium", "hint": "Moderate debate and strategy discussion"},
    {"value": 5, "label": "Deep", "hint": "Comprehensive, in-depth debate"},
]

LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", "Hindi", "Spanish",
    "Portuguese", "French", "German", "Arabic", "Russian", "Turkish",
    "Vietnamese", "Thai", "Indonesian",
]

THINKING_KNOBS = {
    "google": {
        "field": "google_thinking_level",
        "env": "TRADINGAGENTS_GOOGLE_THINKING_LEVEL",
        "label": "Gemini thinking mode",
        "options": [("high", "Enable thinking (recommended)"), ("minimal", "Minimal / disabled")],
    },
    "openai": {
        "field": "openai_reasoning_effort",
        "env": "TRADINGAGENTS_OPENAI_REASONING_EFFORT",
        "label": "Reasoning effort",
        "options": [("medium", "Medium (default)"), ("high", "High (more thorough)"), ("low", "Low (faster)")],
    },
    "anthropic": {
        "field": "anthropic_effort",
        "env": "TRADINGAGENTS_ANTHROPIC_EFFORT",
        "label": "Claude effort",
        "options": [("high", "High (recommended)"), ("medium", "Medium (balanced)"), ("low", "Low (faster)")],
    },
}

# Env vars that override the equivalent form field; surfaced so the UI can say
# why a submitted choice did not take effect.
ENV_LOCKS = {
    "TRADINGAGENTS_LLM_PROVIDER": "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM": "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM": "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL": "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE": "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS": "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED": "checkpoint_enabled",
    "TRADINGAGENTS_GOOGLE_THINKING_LEVEL": "google_thinking_level",
    "TRADINGAGENTS_OPENAI_REASONING_EFFORT": "openai_reasoning_effort",
    "TRADINGAGENTS_ANTHROPIC_EFFORT": "anthropic_effort",
}


def _results_dir() -> Path:
    return Path(DEFAULT_CONFIG["results_dir"]).resolve()


@functools.cache
def _model_catalog() -> dict[str, dict[str, list]]:
    """Static per-provider model menus (built once, not on every page load)."""
    models: dict[str, dict[str, list]] = {}
    for _, key, _ in _llm_provider_table():
        base = engine.base_provider(key)
        if base in ("azure", "openrouter"):
            continue  # free-text deployment / live-fetched model list
        try:
            models[base] = {
                "quick": [list(o) for o in get_model_options(base, "quick")],
                "deep": [list(o) for o in get_model_options(base, "deep")],
            }
        except KeyError:
            custom = [["Custom model ID", "custom"]]
            models[base] = {"quick": custom, "deep": custom}
    return models


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Meta: everything the form needs to render itself
# ---------------------------------------------------------------------------

@app.get("/api/meta")
def meta():
    providers = [
        {"key": key, "label": label, "url": url, "regional": False}
        for label, key, url in _llm_provider_table()
    ] + [
        {"key": key, "label": label, "url": url, "regional": True}
        for key, (label, url) in engine.REGIONAL_PROVIDERS.items()
    ]

    keys = {}
    for provider in providers:
        env_var = get_api_key_env(provider["key"])
        keys[provider["key"]] = {
            "env": env_var,
            "set": bool(env_var and os.environ.get(env_var)),
            "optional": engine.key_optional(provider["key"]),
        }

    env_locked = {
        config_key: DEFAULT_CONFIG[config_key]
        for env_var, config_key in ENV_LOCKS.items()
        if os.environ.get(env_var)
    }

    prefs = sanitize(load_last_run(), "stock")

    return {
        "providers": providers,
        "models": _model_catalog(),
        "analysts": [{"key": k, "label": v} for k, v in ANALYST_LABELS.items()],
        "depths": DEPTH_OPTIONS,
        "languages": LANGUAGES,
        "knobs": THINKING_KNOBS,
        "chat_suggestions": chat.SUGGESTIONS,
        "robinhood_configured": bool(
            os.environ.get("ROBINHOOD_USERNAME") and os.environ.get("ROBINHOOD_PASSWORD")
        ),
        "api_keys": keys,
        "env_locked": env_locked,
        "prefs": prefs,
        "defaults": {
            "date": datetime.date.today().isoformat(),
            "ticker": "SPY",
            "checkpoint": DEFAULT_CONFIG["checkpoint_enabled"],
            "results_dir": str(_results_dir()),
        },
    }


@app.get("/api/announcements")
def announcements():
    return fetch_announcements()


@app.get("/api/openrouter-models")
def openrouter_models():
    """Newest OpenRouter models, mirrored from the CLI's live fetch."""
    models = _fetch_openrouter_models()
    mainstream = [
        [name, mid] for name, mid in models
        if not mid.startswith("~") and mid.split("/", 1)[0] in _OPENROUTER_MAINSTREAM
    ]
    return {"models": (mainstream or models)[:5]}


@app.post("/api/apikey")
def set_api_key(body: dict):
    """Store a provider key in .env. ``saved_to`` is the file, never the key."""
    provider = str(body.get("provider") or "").lower()
    env_var = get_api_key_env(provider)
    if env_var is None:
        raise HTTPException(400, f"{provider} does not use an API key.")
    key = str(body.get("key") or "").strip()
    if not key:
        raise HTTPException(400, "Paste an API key first.")
    try:
        path = save_api_key(env_var, key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"env": env_var, "set": True, "saved_to": str(path)}


@app.post("/api/render")
def render(body: dict):
    return {"html": render_markdown(str(body.get("markdown") or ""))}


@app.post("/api/clear-checkpoints")
def clear_checkpoints():
    n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
    return {"cleared": n}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _start(starter, body: dict) -> dict:
    try:
        run = starter(body or {})
    except engine.RequestError as exc:
        raise HTTPException(400, str(exc)) from None
    except engine.BusyError as exc:
        raise HTTPException(409, str(exc)) from None
    return _run_view(run)


@app.post("/api/analyze")
def start_analysis(body: dict):
    return _start(engine.manager.start_analysis, body)


@app.post("/api/backtest")
def start_backtest(body: dict):
    return _start(engine.manager.start_backtest, body)


@app.post("/api/suggest")
def start_suggest(body: dict):
    """Screen for stock ideas matching a target gain over a horizon."""
    return _start(engine.manager.start_suggest, body)


@app.post("/api/suggest/parse")
def parse_suggestion_goal(body: dict):
    """Read a target % and horizon out of free text, live as the user types."""
    found = screener.parse_goal((body or {}).get("goal"))
    return {"target_pct": found.get("target_pct"), "horizon_days": found.get("horizon_days")}


@app.get("/api/runs")
def list_runs():
    return {"runs": [_run_view(r, brief=True) for r in engine.manager.recent()]}


def _get_run(run_id: str) -> engine.Run:
    run = engine.manager.get(run_id)
    if run is None:
        raise HTTPException(404, "No such run.")
    return run


def _run_view(run: engine.Run, brief: bool = False) -> dict:
    view = {
        "run_id": run.id,
        "kind": run.kind,
        "status": run.status,
        "ticker": run.ticker,
        "analysis_date": run.analysis_date,
        "created_at": run.created_at,
    }
    if not brief:
        view["payload"] = {k: v for k, v in run.payload.items() if k != "api_key"}
        view["events"] = run.events_after(0)
        view["has_report"] = bool(run.final_state)
    return view


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _run_view(_get_run(run_id))


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str, after: int = Query(0, ge=0)):
    run = _get_run(run_id)

    def stream():
        seq = after
        yield "retry: 2000\n\n"
        while True:
            batch = run.events_after(seq)
            if not batch:
                if run.finished:
                    # Everything already happened; say so instead of hanging.
                    yield "data: " + json.dumps({"type": "end"}) + "\n\n"
                    return
                batch = run.wait_after(seq, timeout=15.0)
                if not batch:
                    yield ": keep-alive\n\n"
                    continue
            for event in batch:
                seq = event["seq"]
                yield "data: " + json.dumps(event, default=str) + "\n\n"
                # The terminal event is always last, so the client never misses it.
                if event["type"] == "error" or (
                    event["type"] == "status"
                    and event.get("status") in engine.TERMINAL_STATUSES
                ):
                    return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Execution: risk → sizing → broker (tradingagents.execution)
# ---------------------------------------------------------------------------

_execute_lock = threading.Lock()


def _latest_close(ticker: str) -> float | None:
    """Latest close from the cached data layer, or None when unavailable."""
    try:
        _, close = screener.load_closes(ticker)
        return float(close.iloc[-1]) if len(close) else None
    except Exception:  # offline / no history — caller falls back to "no price"
        return None


@app.post("/api/runs/{run_id}/execute")
def execute_run(run_id: str, body: dict | None = None):
    """Run the multi-agent execution pipeline on a finished analysis decision.

    ``mode=preview`` (the default) walks RiskAgent → SizingAgent →
    DryRunBroker and never sends an order. ``mode=live`` places a real
    Robinhood limit order and is refused unless live trading is explicitly
    enabled (ROBINHOOD_TRADING_ENABLED plus credentials in .env), so a stray
    request can never trade the account.
    """
    run = _get_run(run_id)
    body = body or {}
    mode = str(body.get("mode") or "preview")
    if mode not in ("preview", "live"):
        raise HTTPException(400, "mode must be 'preview' or 'live'.")
    if run.kind != "analysis":
        raise HTTPException(400, "Only analysis runs carry a tradeable decision.")
    if run.status != "done" or not run.final_state:
        raise HTTPException(400, "The analysis has not finished yet.")
    if mode == "live":
        # A live order must be confirmed explicitly, and only once per run, so
        # a retried or replayed request can never double an order.
        if body.get("confirm") is not True:
            raise HTTPException(400, "A live order needs \"confirm\": true.")
        if run.live_order is not None:
            raise HTTPException(409, "A live order was already sent for this run.")

    from tradingagents.agents.rating import parse_rating
    from tradingagents.execution import create_orchestrator

    signal = parse_rating(str(run.final_state.get("final_trade_decision") or ""))
    ticker = str(run.ticker or "").strip().upper()
    if not ticker:
        raise HTTPException(400, "This run has no ticker to trade.")

    try:
        portfolio_text = run.payload.get("portfolio")
        portfolio = engine.load_portfolio_input(portfolio_text) if portfolio_text else None
    except Exception as exc:
        raise HTTPException(400, f"Portfolio input is not usable: {exc}") from None

    raw_price = body.get("price")
    try:
        price = float(raw_price) if raw_price not in (None, "") else None
    except (TypeError, ValueError):
        raise HTTPException(400, "price must be a number.") from None
    if price is not None and not (0 < price < 1e7):
        raise HTTPException(400, "price must be a positive number.")
    if price is None:
        price = _latest_close(ticker)

    with _execute_lock:
        if mode == "preview":
            orchestrator = create_orchestrator(
                {"trading_enabled": False, "default_price": price}
            )
        else:
            mfa_code = str(body.get("mfa_code") or "").strip() or None
            if mfa_code and not (mfa_code.isdigit() and len(mfa_code) <= 8):
                raise HTTPException(400, "The 2FA code must be up to 8 digits.")
            if run.live_order is not None:  # re-checked under the lock
                raise HTTPException(409, "A live order was already sent for this run.")
            orchestrator = create_orchestrator(
                {"default_price": price, "mfa_code": mfa_code}
            )
            if orchestrator.broker.dry_run:
                raise HTTPException(
                    409,
                    "Live trading is not enabled. Set ROBINHOOD_TRADING_ENABLED=true "
                    "and Robinhood credentials in .env, then retry.",
                )
            if not orchestrator.broker.username or not orchestrator.broker.password:
                raise HTTPException(
                    400,
                    "Robinhood credentials missing. Set ROBINHOOD_USERNAME and "
                    "ROBINHOOD_PASSWORD in .env, then retry.",
                )
        result = orchestrator.execute(signal, ticker, portfolio, decision_price=price)
        if mode == "live" and result.executed:
            run.live_order = result.model_dump(mode="json")

    return {
        "run_id": run.id,
        "mode": mode,
        "signal": signal,
        "result": result.model_dump(mode="json"),
    }


# ---------------------------------------------------------------------------
# Portfolio, charts and the Q&A assistant
# ---------------------------------------------------------------------------

def _bad_request(fn, *args):
    try:
        return fn(*args)
    except engine.RequestError as exc:
        raise HTTPException(400, str(exc)) from None


@app.get("/api/portfolio")
def get_portfolio():
    return portfolio.valuation()


@app.put("/api/portfolio")
def put_portfolio(body: dict):
    portfolio.save(_bad_request(portfolio.parse_book, body))
    return portfolio.valuation()


@app.post("/api/portfolio/sync-robinhood")
def sync_robinhood(body: dict | None = None):
    """Read-only: copy Robinhood positions and buying power into the saved book."""
    mfa = str((body or {}).get("mfa_code") or "").strip() or None
    synced = _bad_request(portfolio.sync_robinhood, mfa)
    return {**portfolio.valuation(), "synced": synced}


@app.post("/api/portfolio/preview-order")
def preview_order(body: dict):
    """Quick Buy/Sell from the holdings table: always the dry-run broker."""
    raw = body.get("price")
    try:
        price = float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        raise HTTPException(400, "price must be a number.") from None
    with _execute_lock:
        result = _bad_request(portfolio.preview_order, str(body.get("ticker") or ""),
                              str(body.get("side") or ""), price)
    return {"mode": "preview", "result": result}


@app.get("/api/chart/{ticker}")
def chart(ticker: str, range: str = Query("1M", max_length=4)):  # noqa: A002 - query name
    return _bad_request(portfolio.chart, ticker, range)


@app.post("/api/chat")
def chat_stream(body: dict):
    config, history = _bad_request(chat.prepare, body)
    return StreamingResponse(
        chat.stream(config, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/runs/{run_id}/save-report")
def save_report(run_id: str, body: dict | None = None):
    run = _get_run(run_id)
    try:
        report_file = engine.save_report(run, (body or {}).get("path"))
    except engine.RequestError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"path": str(report_file.parent), "file": report_file.name}


# ---------------------------------------------------------------------------
# Saved reports
# ---------------------------------------------------------------------------

def _safe_report_dir(rel: str) -> Path:
    root = _results_dir()
    resolved = (root / rel).resolve()
    if not resolved.is_relative_to(root):
        raise HTTPException(400, "Path is outside the results directory.")
    return resolved


@app.get("/api/reports")
def list_reports():
    root = _results_dir()
    candidates = []
    if root.is_dir():
        candidates += sorted(root.glob("reports/*"))
        candidates += sorted(root.glob("*/*/reports/*"))
    reports = []
    for directory in candidates:
        complete = directory / "complete_report.md"
        if not directory.is_dir() or not complete.is_file():
            continue
        try:
            rel = directory.relative_to(root).as_posix()
        except ValueError:
            continue
        reports.append({
            "name": directory.name,
            "path": rel,
            "mtime": directory.stat().st_mtime,
        })
    reports.sort(key=lambda r: r["mtime"], reverse=True)
    return {"reports": reports[:200], "results_dir": str(root)}


@app.get("/api/reports/files")
def report_files(path: str = Query(...)):
    directory = _safe_report_dir(path)
    if not directory.is_dir():
        raise HTTPException(404, "No such report.")
    files = []
    for md in sorted(directory.rglob("*.md")):
        rel = md.relative_to(directory).as_posix()
        markdown = md.read_text(encoding="utf-8", errors="replace")
        files.append({
            "name": rel,
            "markdown": markdown,
            "html": render_markdown(markdown),
        })
    if not files:
        raise HTTPException(404, "That report has no markdown files.")
    return {"path": path, "files": files}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """``tradingagents-web`` — serve the UI."""
    parser = argparse.ArgumentParser(prog="tradingagents-web", description="TradingAgents web UI")
    parser.add_argument("--host", default=os.environ.get("TRADINGAGENTS_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRADINGAGENTS_WEB_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="reload on code change (development)")
    args = parser.parse_args()

    import uvicorn

    token = ensure_token(args.host)
    shown = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    if token:
        print(f"TradingAgents web UI: http://{shown}:{args.port}/?token={token}", flush=True)
    else:
        print(f"TradingAgents web UI: http://{shown}:{args.port}/", flush=True)
    if not is_loopback(args.host):
        print("Serving beyond localhost — the token above is required to sign in.", flush=True)

    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
