"""The execute-trade endpoint: the web surface of the execution pipeline.

The multi-agent pipeline itself (risk rules, sizing maths, both brokers) is
covered in tests/test_execution.py. These tests pin the HTTP contract: which
runs are eligible, mode handling, input validation, and — most importantly —
that live trading stays refused unless it has been explicitly enabled.
"""

import time

import pytest
from fastapi.testclient import TestClient

import web.engine as engine
from web.app import app


@pytest.fixture()
def client():
    return TestClient(app, base_url="http://localhost")


def valid_payload(**overrides):
    payload = {
        "ticker": "NVDA",
        "analysis_date": "2026-01-05",
        "analysts": ["market", "news"],
        "research_depth": 3,
        "llm_provider": "openai",
        "quick_think_llm": "gpt-6-luna",
        "deep_think_llm": "gpt-6-sol",
        "output_language": "English",
        "checkpoint": None,
    }
    payload.update(overrides)
    return payload


def finished_analysis(monkeypatch, *, decision="Rating: Buy", **payload_overrides):
    """Start an analysis run whose stubbed runner finishes with ``decision``."""

    def fake(run):
        run.set_status("running")
        run.final_state = {"final_trade_decision": decision}
        run.emit("done", rating="Buy", review=False, elapsed=0.1, wall_time="")
        run.set_status("done")

    monkeypatch.setattr(engine, "_execute_analysis", fake)
    run = engine.manager.start_analysis(valid_payload(**payload_overrides))
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if run.status in engine.TERMINAL_STATUSES:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"run {run.id} never finished")
    # The guard releases the engine lock right after the status flips to done;
    # wait for that so the next start in a later test cannot see BusyError.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if engine.manager._engine_lock.acquire(blocking=False):
            engine.manager._engine_lock.release()
            return run
        time.sleep(0.02)
    pytest.fail(f"engine lock never released after run {run.id}")


def execute(client, run_id, **body):
    return client.post(f"/api/runs/{run_id}/execute", json=body)


# ---------------------------------------------------------------------------
# Preview (the default path)
# ---------------------------------------------------------------------------


def test_preview_walks_the_pipeline_against_the_dry_run_broker(client, monkeypatch):
    run = finished_analysis(monkeypatch)
    res = execute(client, run.id, mode="preview", price=173.5)
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "preview"
    assert data["signal"] == "Buy"
    result = data["result"]
    assert result["dry_run"] is True
    assert result["executed"] is True
    assert result["order"]["action"] == "buy"
    assert result["order"]["quantity"] > 0
    assert result["order"]["limit_price"] == 173.5
    assert result["risk"]["passed"] is True
    assert result["sizing"]["quantity"] == result["order"]["quantity"]


def test_mode_defaults_to_preview(client, monkeypatch):
    run = finished_analysis(monkeypatch)
    res = execute(client, run.id, price=10.0)
    assert res.status_code == 200
    assert res.json()["mode"] == "preview"
    assert res.json()["result"]["dry_run"] is True


def test_no_price_given_falls_back_to_the_latest_close(client, monkeypatch):
    import web.app as app_module

    monkeypatch.setattr(app_module, "_latest_close", lambda ticker: 99.25)
    run = finished_analysis(monkeypatch)
    res = execute(client, run.id, mode="preview")
    assert res.status_code == 200
    assert res.json()["result"]["order"]["limit_price"] == 99.25


def test_portfolio_context_is_loaded_for_sizing_and_risk(client, monkeypatch):
    run = finished_analysis(
        monkeypatch,
        portfolio='{"cash": 5000, "positions": []}',
    )
    res = execute(client, run.id, mode="preview", price=100.0)
    assert res.status_code == 200
    # 10% of 5,000 equity → $500 → 5 shares at $100.
    assert res.json()["result"]["order"]["quantity"] == 5


# ---------------------------------------------------------------------------
# Signals that must never become orders
# ---------------------------------------------------------------------------


def test_hold_signal_is_not_executed(client, monkeypatch):
    run = finished_analysis(monkeypatch, decision="Rating: Hold")
    res = execute(client, run.id, mode="preview", price=173.5)
    assert res.status_code == 200
    result = res.json()["result"]
    assert result["executed"] is False
    assert "no position change" in result["message"]


def test_unreadable_rating_is_not_tradable(client, monkeypatch):
    run = finished_analysis(monkeypatch, decision="The model rambled without a call.")
    res = execute(client, run.id, mode="preview", price=173.5)
    assert res.status_code == 200
    result = res.json()["result"]
    assert res.json()["signal"] == "REVIEW"
    assert result["executed"] is False
    assert "not tradeable" in result["message"]


# ---------------------------------------------------------------------------
# Live mode is opt-in
# ---------------------------------------------------------------------------


def test_live_mode_is_refused_when_trading_is_not_enabled(client, monkeypatch):
    monkeypatch.delenv("ROBINHOOD_TRADING_ENABLED", raising=False)
    run = finished_analysis(monkeypatch)
    res = execute(client, run.id, mode="live", price=173.5, confirm=True)
    assert res.status_code == 409
    assert "ROBINHOOD_TRADING_ENABLED" in res.json()["detail"]


def test_live_mode_without_credentials_is_refused_before_any_login(client, monkeypatch):
    monkeypatch.setenv("ROBINHOOD_TRADING_ENABLED", "true")
    monkeypatch.delenv("ROBINHOOD_USERNAME", raising=False)
    monkeypatch.delenv("ROBINHOOD_PASSWORD", raising=False)
    run = finished_analysis(monkeypatch)
    res = execute(client, run.id, mode="live", price=173.5, confirm=True)
    assert res.status_code == 400
    assert "ROBINHOOD_USERNAME" in res.json()["detail"]


def test_live_mode_forwards_mfa_code_and_price_to_the_factory(client, monkeypatch):
    import tradingagents.execution as execution_mod
    from tradingagents.execution import ExecutionResult, TradeAction

    captured = {}

    class FakeBroker:
        dry_run = False
        username = "u"
        password = "p"

    class FakeOrchestrator:
        broker = FakeBroker()

        def execute(self, signal, ticker, portfolio, decision_price=None):
            captured["execute"] = (signal, ticker, decision_price)
            return ExecutionResult(
                ticker=ticker, action=TradeAction.BUY, signal=signal,
                executed=True, dry_run=False, message="order submitted",
            )

    def fake_factory(config):
        captured["config"] = config
        return FakeOrchestrator()

    monkeypatch.setattr(execution_mod, "create_orchestrator", fake_factory)
    run = finished_analysis(monkeypatch)
    res = execute(client, run.id, mode="live", price=173.5, mfa_code="654321", confirm=True)
    assert res.status_code == 200
    assert captured["config"]["mfa_code"] == "654321"
    assert captured["config"]["default_price"] == 173.5
    assert captured["execute"][0] == "Buy"
    body = res.json()
    assert body["mode"] == "live"
    assert body["result"]["dry_run"] is False
    assert body["result"]["executed"] is True


# ---------------------------------------------------------------------------
# Eligibility and input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["preview", "live"])
def test_unfinished_run_cannot_execute(client, mode):
    run = engine.manager.register(engine.Run("analysis", valid_payload()))
    res = execute(client, run.id, mode=mode, price=10.0)
    assert res.status_code == 400
    assert "not finished" in res.json()["detail"]
    run.set_status("done")  # never leave an unfinished run in the registry


def test_only_analysis_runs_can_execute(client):
    run = engine.manager.register(engine.Run("backtest", {"ticker": "NVDA"}))
    run.set_status("done")
    run.final_state = {"final_trade_decision": "Rating: Buy"}
    res = execute(client, run.id, mode="preview", price=10.0)
    assert res.status_code == 400
    assert "Only analysis runs" in res.json()["detail"]


def test_unknown_run_is_404(client):
    assert execute(client, "nope", mode="preview").status_code == 404


@pytest.mark.parametrize("body", [
    {"mode": "yolo", "price": 10.0},
    {"mode": "preview", "price": "abc"},
    {"mode": "preview", "price": -5},
    {"mode": "preview", "price": 0},
])
def test_bad_input_is_rejected(client, monkeypatch, body):
    run = finished_analysis(monkeypatch)
    res = execute(client, run.id, **body)
    assert res.status_code == 400


def test_unusable_portfolio_is_rejected_before_trading(client, monkeypatch):
    run = finished_analysis(monkeypatch)
    run.payload["portfolio"] = "{this is not json"
    res = execute(client, run.id, mode="preview", price=10.0)
    assert res.status_code == 400
    assert "Portfolio" in res.json()["detail"]
