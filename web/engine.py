"""Background execution for the web UI: analyses and backtests as event streams.

The CLI drives the graph from a rich ``Live`` view inside the terminal. Here the
same graph stream is forwarded into per-run event buffers, which the API replays
to the browser over server-sent events — so a run can be watched live and
re-opened afterwards without a console.

The streaming loop mirrors ``cli/run.py`` deliberately: same status transitions,
same report-section updates, same checkpoint handling, only the sink differs.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from cli.display import (
    ANALYST_ORDER,
    AnalystWallTimeTracker,
    MessageBuffer,
    classify_message_type,
    update_analyst_statuses,
)
from cli.prefs import save_last_run
from cli.prompts import (
    _llm_provider_table,
    detect_asset_type,
    is_valid_ticker_input,
    normalize_ticker_symbol,
    resolve_backend_url,
)
from cli.run import _build_run_config, _run_directory
from cli.stats_handler import StatsCallbackHandler
from tradingagents.agents.rating import is_review
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.decision_log import TradingMemoryLog
from tradingagents.default_config import _ENV_OVERRIDES, DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.portfolio import PortfolioContext, load_portfolio
from tradingagents.reporting import write_report_tree
from web.markdown_util import render_markdown

logger = logging.getLogger("tradingagents.web")

ANALYST_KEYS = ("market", "social", "news", "fundamentals")

# Regional endpoints the CLI asks about in a follow-up prompt; the web form
# offers them directly, with their own (non-interchangeable) API key.
REGIONAL_PROVIDERS = {
    "qwen-cn": ("Qwen — China (dashscope.aliyuncs.com)", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm-cn": ("GLM — BigModel China (open.bigmodel.cn)", "https://open.bigmodel.cn/api/paas/v4/"),
    "minimax-cn": ("MiniMax — China (api.minimaxi.com)", "https://api.minimaxi.com/v1"),
}

SECTION_TITLES = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Team Plan",
    "final_trade_decision": "Portfolio Management Decision",
}

TERMINAL_STATUSES = ("done", "error")


class RequestError(ValueError):
    """A bad request: told back to the client as HTTP 400."""


class BusyError(RuntimeError):
    """Another run holds the engine; told back as HTTP 409."""


def provider_keys() -> list[str]:
    """Every provider key the run form may submit (base + regional variants)."""
    return [key for _, key, _ in _llm_provider_table()] + list(REGIONAL_PROVIDERS)


def base_provider(provider: str) -> str:
    """Regional keys (``qwen-cn``) share the base provider's model catalog."""
    return provider.split("-cn")[0]


def key_optional(provider: str) -> bool:
    """True when the provider runs without an API key (Ollama, keyless relays…)."""
    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    if get_api_key_env(provider) is None:
        return True
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(str(provider).lower())
    return bool(spec is not None and spec.key_optional)


def llm_key_available(provider: str) -> bool:
    """Can a client for ``provider`` be built right now (key set or not needed)?"""
    env_var = get_api_key_env(provider)
    return key_optional(provider) or bool(env_var and os.environ.get(env_var))


def fmt_count(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else str(n)


def _coerce_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def load_portfolio_input(value: str):
    """Portfolio from pasted JSON or a ``.json`` file path, validated before the run."""
    text = str(value).strip()
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RequestError(f"Portfolio JSON is not valid: {exc}") from None
        try:
            return PortfolioContext.model_validate(data)
        except ValidationError as exc:
            raise RequestError(f"Portfolio JSON is not usable: {exc}") from None
    # Only .json paths: a request must not be able to probe arbitrary files
    # and read their contents back out of the validation error.
    if Path(text).suffix.lower() != ".json":
        raise RequestError("Portfolio must be pasted JSON or a path to a .json file.")
    try:
        return load_portfolio(Path(text).expanduser())
    except ValueError as exc:
        raise RequestError(str(exc)) from None


def _optional_portfolio(payload: dict) -> str | None:
    """The submitted portfolio text, validated now so a bad book fails before the run."""
    text = str(payload.get("portfolio") or "").strip() or None
    if text is not None:
        load_portfolio_input(text)
    return text


def _model_settings(payload: dict, default_provider: str | None) -> dict:
    """Provider, endpoint, models and reasoning knobs — shared by every run kind."""
    provider = str(payload.get("llm_provider") or default_provider or "").lower() or None
    if provider and provider not in provider_keys():
        raise RequestError(f"Unknown LLM provider: {provider}")

    menu_url = str(payload.get("backend_url") or "").strip() or None
    if menu_url and urlsplit(menu_url).scheme not in ("http", "https"):
        raise RequestError("Backend URL must start with http:// or https://.")
    backend_url = None
    if provider:
        if provider == "openai_compatible" and not (menu_url or DEFAULT_CONFIG["backend_url"]):
            raise RequestError(
                "An OpenAI-compatible endpoint needs a base URL (e.g. http://localhost:8000/v1)."
            )
        menu_url = menu_url or REGIONAL_PROVIDERS.get(provider, (None, None))[1]
        backend_url = resolve_backend_url(provider, menu_url, env_url=DEFAULT_CONFIG["backend_url"])

    def text(key):
        return str(payload.get(key) or "").strip() or None

    return {
        "llm_provider": provider,
        "backend_url": backend_url,
        "quick_think_llm": text("quick_think_llm"),
        "deep_think_llm": text("deep_think_llm"),
        "google_thinking_level": text("google_thinking_level"),
        "openai_reasoning_effort": text("openai_reasoning_effort"),
        "anthropic_effort": text("anthropic_effort"),
        "output_language": text("output_language"),
    }


def normalize_analysis(payload: dict) -> dict:
    """Validate a submitted run form and return the selections dict.

    Raises :class:`RequestError` with a message meant for the person who
    submitted the form.
    """
    if not isinstance(payload, dict):
        raise RequestError("Expected a JSON object.")

    raw_ticker = str(payload.get("ticker") or "").strip() or "SPY"
    if not is_valid_ticker_input(raw_ticker):
        raise RequestError(
            "Please enter a valid ticker symbol, e.g. AAPL, 000404.SZ, 0700.HK, GC=F."
        )
    ticker = normalize_ticker_symbol(raw_ticker) if raw_ticker else "SPY"

    date_str = str(payload.get("analysis_date") or "").strip()
    date_str = date_str or datetime.date.today().isoformat()
    try:
        parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise RequestError("Analysis date must be in YYYY-MM-DD format.") from None
    if parsed.date() > datetime.date.today():
        raise RequestError("Analysis date cannot be in the future.")

    asset_type = detect_asset_type(ticker).value
    analysts = [a for a in (payload.get("analysts") or []) if a in ANALYST_KEYS]
    if asset_type == "crypto":
        analysts = [a for a in analysts if a != "fundamentals"]
    if not analysts:
        raise RequestError(
            "Select at least one analyst"
            + (" (crypto runs have no Fundamentals Analyst)." if asset_type == "crypto" else ".")
        )

    try:
        research_depth = int(payload.get("research_depth") or 1)
    except (TypeError, ValueError):
        raise RequestError("Research depth must be 1, 3 or 5.") from None
    if research_depth not in (1, 3, 5):
        raise RequestError("Research depth must be 1, 3 or 5.")

    models = _model_settings(payload, default_provider="openai")
    if not models["quick_think_llm"] or not models["deep_think_llm"]:
        raise RequestError("Pick both a quick-thinking and a deep-thinking model.")
    models["output_language"] = models["output_language"] or "English"

    checkpoint = payload.get("checkpoint")
    if checkpoint not in (None, True, False):
        raise RequestError("Checkpoint must be true, false or null (env default).")

    return {
        "ticker": ticker,
        "asset_type": asset_type,
        "analysis_date": date_str,
        "analysts": analysts,
        "research_depth": research_depth,
        **models,
        "checkpoint": checkpoint,
        "portfolio": _optional_portfolio(payload),
    }


class Run:
    """One background job (an analysis or a backtest) and its event log."""

    def __init__(self, kind: str, payload: dict):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.payload = payload
        self.status = "queued"
        self.created_at = time.time()
        self.events: list[dict] = []
        self.final_state: dict | None = None
        self.config: dict | None = None
        self.live_order: dict | None = None  # set once a live order is sent
        self._cond = threading.Condition()

    # -- event log ---------------------------------------------------------
    def emit(self, event_type: str, **data) -> None:
        with self._cond:
            event = {
                "seq": len(self.events) + 1,
                "type": event_type,
                "at": round(time.time(), 3),
            }
            event.update(data)
            self.events.append(event)
            self._cond.notify_all()

    def set_status(self, status: str) -> None:
        self.status = status
        self.emit("status", status=status)

    def events_after(self, seq: int) -> list[dict]:
        with self._cond:
            return list(self.events[seq:])

    def wait_after(self, seq: int, timeout: float = 15.0) -> list[dict]:
        """Events newer than ``seq``, waiting up to ``timeout`` for one."""
        with self._cond:
            if len(self.events) <= seq:
                self._cond.wait(timeout)
            return list(self.events[seq:])

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def ticker(self) -> str | None:
        return self.payload.get("ticker")

    @property
    def analysis_date(self) -> str | None:
        return self.payload.get("analysis_date")


class EventBuffer(MessageBuffer):
    """A MessageBuffer that mirrors every change into the run's event log."""

    def __init__(self, run: Run):
        super().__init__()
        self._run = run

    def add_message(self, message_type, content):
        super().add_message(message_type, content)
        clock, kind, text = self.messages[-1]
        if text and str(text).strip():
            self._run.emit("message", kind=kind, content=str(text), clock=clock)

    def add_tool_call(self, tool_name, args):
        super().add_tool_call(tool_name, args)
        clock, name, raw = self.tool_calls[-1]
        clean = {str(k): str(v)[:500] for k, v in dict(raw).items()}
        self._run.emit("tool_call", name=name, args=clean, clock=clock)

    def update_agent_status(self, agent, status):
        before = self.agent_status.get(agent)
        super().update_agent_status(agent, status)
        if agent in self.agent_status and self.agent_status.get(agent) != before:
            self._run.emit("agents", status=dict(self.agent_status))

    def update_report_section(self, section_name, content):
        previous = self.report_sections.get(section_name)
        super().update_report_section(section_name, content)
        text = self.report_sections.get(section_name)
        if not text:
            return
        markdown = _coerce_content(text)
        if previous is not None and _coerce_content(previous) == markdown:
            return  # the graph re-sends unchanged sections; the client has them
        self._run.emit(
            "section",
            section=section_name,
            title=SECTION_TITLES.get(section_name, section_name),
            markdown=markdown,
            html=render_markdown(markdown),
        )


class RunManager:
    """Starts runs, keeps them, and guarantees one engine run at a time.

    The graph config is process-global (``dataflows.run_config`` swaps it in
    around tool execution), so two concurrent sweeps would fight over it.
    """

    def __init__(self):
        self._runs: dict[str, Run] = {}
        self._registry_lock = threading.Lock()
        self._engine_lock = threading.Lock()

    # -- registry ----------------------------------------------------------
    def register(self, run: Run) -> Run:
        with self._registry_lock:
            self._runs[run.id] = run
            # Keep memory bounded on a long-lived server.
            finished = [r for r in self._runs.values() if r.finished]
            for stale in sorted(finished, key=lambda r: r.created_at)[:-20:]:
                self._runs.pop(stale.id, None)
        return run

    def get(self, run_id: str) -> Run | None:
        with self._registry_lock:
            return self._runs.get(run_id)

    def active(self) -> Run | None:
        with self._registry_lock:
            return next((r for r in self._runs.values() if not r.finished), None)

    def recent(self, limit: int = 20) -> list[Run]:
        with self._registry_lock:
            runs = list(self._runs.values())
        return sorted(runs, key=lambda r: r.created_at, reverse=True)[:limit]

    # -- starting ----------------------------------------------------------
    def start_analysis(self, payload: dict) -> Run:
        return self._launch("analysis", payload, normalize_analysis, _execute_analysis)

    def start_backtest(self, payload: dict) -> Run:
        return self._launch("backtest", payload, _normalize_backtest, _execute_backtest)

    def start_suggest(self, payload: dict) -> Run:
        """Screen for stock suggestions (web/screener.py drives the run)."""
        from web import screener

        return self._launch("suggest", payload, screener.normalize_suggestion_request,
                            screener.execute_suggest)

    def _launch(self, kind: str, payload: dict, normalize, target) -> Run:
        """Validate, store a submitted key, take the engine and start the thread."""
        if not isinstance(payload, dict):
            raise RequestError("Expected a JSON object.")
        selections = normalize(payload)
        env_var = get_api_key_env(selections.get("llm_provider") or "")
        if payload.get("api_key") and env_var:
            from web.apikeys import save_api_key

            try:
                save_api_key(env_var, str(payload["api_key"]))
            except ValueError as exc:
                raise RequestError(f"API key not saved: {exc}") from None

        if not self._engine_lock.acquire(blocking=False):
            raise BusyError("Another run is already in progress.")
        run = self.register(Run(kind, selections))
        threading.Thread(
            target=self._guard, args=(run, target), daemon=True, name=f"ta-{kind}-{run.id}",
        ).start()
        return run

    def _guard(self, run: Run, target) -> None:
        try:
            target(run)
        except Exception as exc:  # never let a thread die silently
            logger.exception("run %s crashed", run.id)
            if not run.finished:
                run.set_status("error")
                run.emit("error", message=str(exc), traceback=traceback.format_exc()[-4000:])
        finally:
            if not run.finished:
                run.set_status("done")
            self._engine_lock.release()


def _normalize_backtest(payload: dict) -> dict:
    tickers = [t.strip().upper() for t in str(payload.get("tickers") or "").split(",") if t.strip()]
    if not tickers:
        raise RequestError("Enter at least one ticker, comma-separated (e.g. NVDA,AAPL).")
    for ticker in tickers:
        if not is_valid_ticker_input(ticker):
            raise RequestError(f"Not a usable ticker: {ticker}")

    def _date(key):
        raw = str(payload.get(key) or "").strip()
        try:
            return datetime.datetime.strptime(raw, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            raise RequestError("Start and end dates must be YYYY-MM-DD.") from None

    start, end = _date("start"), _date("end")
    if start > end:
        raise RequestError("Start date must be on or before the end date.")
    if end > datetime.date.today().isoformat():
        raise RequestError("Backtest dates cannot be in the future.")
    try:
        every = int(payload.get("every") or 7)
    except (TypeError, ValueError):
        raise RequestError("Days between analyses must be a whole number.") from None
    if every < 1:
        raise RequestError("Days between analyses must be at least 1.")

    submitted = payload.get("analysts")
    analysts = [a for a in (submitted or []) if a in ANALYST_KEYS]
    if submitted and not analysts:
        raise RequestError("Select at least one analyst.")
    if not analysts:
        analysts = list(ANALYST_KEYS)  # like the CLI: no --analysts means all four
    asset_type = str(payload.get("asset_type") or "stock").lower()
    if asset_type not in ("stock", "crypto"):
        raise RequestError("Asset type must be stock or crypto.")
    if asset_type == "crypto":
        analysts = [a for a in analysts if a != "fundamentals"]
        if not analysts:
            raise RequestError("Crypto backtests have no Fundamentals Analyst to run.")

    run_id = str(payload.get("run_id") or "").strip() or None
    if run_id and not is_valid_ticker_input(run_id.replace("_", "-")):
        raise RequestError("Run ID may only contain letters, digits, dots, dashes and underscores.")

    return {
        "tickers": tickers,
        "start": start,
        "end": end,
        "every": every,
        "analysts": analysts,
        "asset_type": asset_type,
        "run_id": run_id,
        "portfolio": _optional_portfolio(payload),
        # The same model settings the analysis form holds, so a sweep runs on
        # the provider the user picked instead of a silent env default.
        **_model_settings(payload, default_provider=None),
    }


_MODEL_CHOICE_KEYS = (
    "llm_provider", "backend_url", "quick_think_llm", "deep_think_llm",
    "google_thinking_level", "openai_reasoning_effort", "anthropic_effort",
    "output_language",
)
_ENV_FOR_KEY = {key: var for var, key in _ENV_OVERRIDES.items()}


def config_with_model_choices(sel: dict) -> dict:
    """DEFAULT_CONFIG with the form's model choices applied.

    Shared by backtests and stock suggestions, which run outside the graph's
    interactive selection flow but should still use the provider/models the
    form holds. An explicit ``TRADINGAGENTS_*`` environment variable still
    wins — the same precedence the analysis path applies — because
    DEFAULT_CONFIG already carries it.
    """
    config = DEFAULT_CONFIG.copy()
    for key in _MODEL_CHOICE_KEYS:
        value = sel.get(key)
        env_var = _ENV_FOR_KEY.get(key)
        if value is None or (env_var and os.environ.get(env_var)):
            continue  # nothing chosen, or the environment already set this one
        config[key] = value
    return config


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _execute_analysis(run: Run) -> None:
    sel = run.payload
    try:
        run.set_status("running")
        config = _build_run_config(sel, sel.get("checkpoint"))
        run.config = config

        portfolio = load_portfolio_input(sel["portfolio"]) if sel.get("portfolio") else None

        stats = StatsCallbackHandler()
        selected_set = set(sel["analysts"])
        selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
        plan = build_analyst_execution_plan(selected_analyst_keys)
        tracker = AnalystWallTimeTracker(plan)

        graph = TradingAgentsGraph(
            selected_analyst_keys, config=config, debug=True, callbacks=[stats],
        )
        buffer = EventBuffer(run)
        buffer.init_for_analysis(selected_analyst_keys)
        run.emit("agents", status=dict(buffer.agent_status))

        start_time = time.time()
        results_dir = _run_directory(config, sel["ticker"], sel["analysis_date"])
        (results_dir / "reports").mkdir(parents=True, exist_ok=True)

        buffer.add_message("System", f"Selected ticker: {sel['ticker']}")
        if sel["asset_type"] != "stock":
            buffer.add_message("System", f"Detected asset type: {sel['asset_type']}")
        buffer.add_message("System", f"Analysis date: {sel['analysis_date']}")
        buffer.add_message("System", f"Selected analysts: {', '.join(selected_analyst_keys)}")
        buffer.add_message(
            "System",
            f"Provider: {sel['llm_provider']} · quick={sel['quick_think_llm']} · "
            f"deep={sel['deep_think_llm']} · depth={sel['research_depth']}",
        )

        init_agent_state = graph.create_run_state(
            sel["ticker"], sel["analysis_date"], sel["asset_type"], portfolio
        )
        args = graph.propagator.get_graph_args(callbacks=[stats])

        checkpoint_tid = graph.begin_checkpoint(
            sel["ticker"], sel["analysis_date"], sel["asset_type"], portfolio
        )
        if checkpoint_tid is not None:
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_tid
            if getattr(graph, "_resuming", False):
                buffer.add_message(
                    "System", f"Resuming the saved run for {sel['ticker']} on {sel['analysis_date']}"
                )
            else:
                buffer.add_message(
                    "System", f"Starting fresh for {sel['ticker']} on {sel['analysis_date']}"
                )

        trace = []
        try:
            _stream_graph(run, graph, buffer, tracker, stats, args, init_agent_state,
                          start_time, trace)
            final_state: dict = {}
            for chunk in trace:
                final_state.update(chunk)
            # Clean run: log the decision, then drop this run's checkpoint.
            try:
                graph.record_decision(sel["ticker"], sel["analysis_date"], final_state)
                graph.clear_checkpoint_on_success(
                    sel["ticker"], sel["analysis_date"], sel["asset_type"], portfolio
                )
            except Exception as exc:  # the analysis itself succeeded
                buffer.add_message("System", f"Could not record the decision: {exc}")
        finally:
            graph.end_checkpoint()

        for agent in list(buffer.agent_status):
            buffer.update_agent_status(agent, "completed")
        buffer.add_message("System", f"Completed analysis for {sel['analysis_date']}")
        buffer.add_message("System", tracker.format_summary())

        for section in list(buffer.report_sections):
            if section in final_state and final_state[section]:
                buffer.update_report_section(section, final_state[section])

        signal = graph.process_signal(final_state.get("final_trade_decision", ""))
        run.final_state = final_state
        run.emit(
            "done",
            rating=signal,
            review=is_review(signal),
            elapsed=round(time.time() - start_time, 1),
            wall_time=tracker.format_summary(),
        )
        run.set_status("done")
        try:
            save_last_run(sel)  # next form load prefills what this run chose
        except Exception:
            logger.debug("could not remember run preferences", exc_info=True)
    except Exception as exc:
        logger.exception("analysis run %s failed", run.id)
        run.set_status("error")
        run.emit("error", message=str(exc), traceback=traceback.format_exc()[-4000:])


def _stream_graph(run, graph, buffer, tracker, stats, args, init_agent_state,
                  start_time, trace) -> None:
    """The graph stream, identical in behaviour to the CLI's live view."""
    last_stats = 0.0
    for chunk in graph.graph.stream(graph.checkpoint_input(init_agent_state), **args):
        for message in chunk.get("messages", []):
            msg_id = getattr(message, "id", None)
            if msg_id is not None:
                if msg_id in buffer._processed_message_ids:
                    continue
                buffer._processed_message_ids.add(msg_id)

            msg_type, content = classify_message_type(message)
            if content and content.strip():
                buffer.add_message(msg_type, content)

            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    if isinstance(tool_call, dict):
                        buffer.add_tool_call(tool_call["name"], tool_call["args"])
                    else:
                        buffer.add_tool_call(tool_call.name, tool_call.args)

        update_analyst_statuses(buffer, chunk, wall_time_tracker=tracker)

        # Research team — investment debate state
        if chunk.get("investment_debate_state"):
            debate_state = chunk["investment_debate_state"]
            bull_hist = debate_state.get("bull_history", "").strip()
            bear_hist = debate_state.get("bear_history", "").strip()
            judge = debate_state.get("judge_decision", "").strip()
            if bull_hist or bear_hist:
                for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                    buffer.update_agent_status(agent, "in_progress")
            if bull_hist:
                buffer.update_report_section(
                    "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                )
            if bear_hist:
                buffer.update_report_section(
                    "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                )
            if judge:
                buffer.update_report_section(
                    "investment_plan", f"### Research Manager Decision\n{judge}"
                )
                for agent in ("Bull Researcher", "Bear Researcher", "Research Manager"):
                    buffer.update_agent_status(agent, "completed")
                buffer.update_agent_status("Trader", "in_progress")

        # Trading team
        if chunk.get("trader_investment_plan"):
            buffer.update_report_section(
                "trader_investment_plan", chunk["trader_investment_plan"]
            )
            if buffer.agent_status.get("Trader") != "completed":
                buffer.update_agent_status("Trader", "completed")
                buffer.update_agent_status("Aggressive Analyst", "in_progress")

        # Risk management team — risk debate state
        if chunk.get("risk_debate_state"):
            risk_state = chunk["risk_debate_state"]
            agg_hist = risk_state.get("aggressive_history", "").strip()
            con_hist = risk_state.get("conservative_history", "").strip()
            neu_hist = risk_state.get("neutral_history", "").strip()
            judge = risk_state.get("judge_decision", "").strip()

            if agg_hist and buffer.agent_status.get("Aggressive Analyst") != "completed":
                buffer.update_agent_status("Aggressive Analyst", "in_progress")
                buffer.update_report_section(
                    "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                )
            if con_hist and buffer.agent_status.get("Conservative Analyst") != "completed":
                buffer.update_agent_status("Conservative Analyst", "in_progress")
                buffer.update_report_section(
                    "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                )
            if neu_hist and buffer.agent_status.get("Neutral Analyst") != "completed":
                buffer.update_agent_status("Neutral Analyst", "in_progress")
                buffer.update_report_section(
                    "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                )
            if judge and buffer.agent_status.get("Portfolio Manager") != "completed":
                buffer.update_agent_status("Portfolio Manager", "in_progress")
                buffer.update_report_section(
                    "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                )
                for agent in (
                    "Aggressive Analyst", "Conservative Analyst",
                    "Neutral Analyst", "Portfolio Manager",
                ):
                    buffer.update_agent_status(agent, "completed")

        now = time.time()
        if now - last_stats >= 1.0:
            last_stats = now
            run.emit("stats", **stats.get_stats(), elapsed=round(now - start_time, 1))

        trace.append(chunk)

    run.emit("stats", **stats.get_stats(), elapsed=round(time.time() - start_time, 1))


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

class _LogTap(logging.Handler):
    """Forward sweep warnings into the run's event stream as they happen."""

    def __init__(self, run: Run):
        super().__init__()
        self.run = run

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.WARNING:
                self.run.emit("message", kind="System",
                              content=self.format(record), clock="")
        except Exception:
            pass


def _execute_backtest(run: Run) -> None:
    sel = run.payload
    tap = _LogTap(run)
    backtest_logger = logging.getLogger("tradingagents.backtest")
    backtest_logger.addHandler(tap)
    poll_stop = threading.Event()
    try:
        run.set_status("running")
        config = config_with_model_choices(sel)
        dates = iter_grid(sel["start"], sel["end"], sel["every"])
        total = len(sel["tickers"]) * len(dates)
        run_id = sel["run_id"] or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(config["results_dir"]) / "backtest" / run_id / "trading_memory.md"
        run.payload["run_id"] = run_id

        run.emit("message", kind="System",
                 content=f"Backtest {run_id}: {len(sel['tickers'])} ticker(s) × "
                         f"{len(dates)} date(s) = up to {total} cells "
                         f"(every {sel['every']} day(s) from {sel['start']} to {sel['end']}).")
        run.emit("progress", done=0, total=total)

        threading.Thread(
            target=_poll_backtest_progress, args=(run, log_path, total, poll_stop),
            daemon=True, name=f"ta-bt-progress-{run.id}",
        ).start()

        portfolio = load_portfolio_input(sel["portfolio"]) if sel.get("portfolio") else None
        kwargs = {"asset_type": sel["asset_type"], "portfolio": portfolio, "run_id": run_id}
        if sel["analysts"]:
            kwargs["selected_analysts"] = sel["analysts"]

        stats = StatsCallbackHandler()
        start_time = time.time()
        result = run_backtest(sel["tickers"], dates, config, callbacks=[stats], **kwargs)
        usage = stats.get_stats()
        run.emit("message", kind="System",
                 content=(f"Token usage: {fmt_count(usage['tokens_in'])} in "
                          f"({fmt_count(usage['cache_read_tokens'])} cached) · "
                          f"{fmt_count(usage['tokens_out'])} out over "
                          f"{usage['llm_calls']} LLM calls."))

        # Final progress tick, so the bar reaches 100% even when the sweep
        # finishes between the poller's 3-second ticks.
        try:
            settled = len(TradingMemoryLog({"memory_log_path": str(log_path)}).load_entries())
            run.emit("progress", done=min(settled, total), total=total)
        except Exception:
            logger.debug("could not count backtest cells", exc_info=True)

        try:
            summary = summarize(result).render()
        except FileNotFoundError:
            summary = "No decisions were logged, so there is nothing to score yet."

        run.emit(
            "done",
            summary=summary,
            summary_html=render_markdown(summary),
            usage=usage,
            cells_run=result.cells_run,
            skipped=result.skipped,
            failures=[list(f) for f in result.failures],
            settlement_failures=[list(f) for f in result.settlement_failures],
            log_path=str(result.log_path),
            run_id=result.run_id,
            elapsed=round(time.time() - start_time, 1),
        )
        run.set_status("done")
    except Exception as exc:
        logger.exception("backtest run %s failed", run.id)
        run.set_status("error")
        run.emit("error", message=str(exc), traceback=traceback.format_exc()[-4000:])
    finally:
        poll_stop.set()
        backtest_logger.removeHandler(tap)


def _poll_backtest_progress(run: Run, log_path: Path, total: int, stop: threading.Event) -> None:
    """Count settled cells while the sweep runs, so the UI shows real progress."""
    last = -1
    while not stop.wait(3.0):
        try:
            done = len(TradingMemoryLog({"memory_log_path": str(log_path)}).load_entries())
        except Exception:
            continue
        if done != last:
            last = done
            run.emit("progress", done=min(done, total), total=total)


# ---------------------------------------------------------------------------
# Saving reports
# ---------------------------------------------------------------------------

def save_report(run: Run, save_path: str | None) -> Path:
    """Write a finished run's report tree; the path must sit in results_dir."""
    if run.kind != "analysis" or not run.final_state:
        raise RequestError("This run has no finished report to save.")
    results_dir = Path(run.config["results_dir"]).resolve()
    if save_path:
        candidate = Path(save_path).expanduser()
        resolved = (results_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not resolved.is_relative_to(results_dir):
            raise RequestError("Reports can only be saved under the results directory.")
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved = results_dir / "reports" / f"{run.ticker}_{stamp}"
    return write_report_tree(run.final_state, run.ticker, resolved)


manager = RunManager()
