"""The web UI: form validation, run streaming, and report browsing.

These cover the HTTP surface and the engine's request handling; the graph
itself is exercised elsewhere, and any test that would start a real run swaps
``engine._execute_analysis`` for a stub that emits canned events.
"""

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web.apikeys
import web.engine as engine
from web.app import DEFAULT_CONFIG, app
from web.engine import RequestError, normalize_analysis


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


def stub_run(monkeypatch, body=None):
    """Replace the real runner with one that finishes immediately."""

    def fake(run):
        run.set_status("running")
        run.emit("message", kind="System", content="stubbed run")
        for event in body or []:
            event = dict(event)
            run.emit(event.pop("type"), **event)
        run.emit("done", rating="Hold", review=False, elapsed=0.1, wall_time="")
        run.set_status("done")

    monkeypatch.setattr(engine, "_execute_analysis", fake)
    return fake


def wait_finished(client, run_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in engine.TERMINAL_STATUSES:
            return run
        time.sleep(0.05)
    pytest.fail(f"run {run_id} never finished")


# ---------------------------------------------------------------------------
# Pages and meta
# ---------------------------------------------------------------------------

def test_index_and_static_assets_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "TradingAgents" in page.text
    for asset in ("/static/app.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200


def test_meta_describes_the_form_without_leaking_keys(client):
    res = client.get("/api/meta")
    assert res.status_code == 200
    meta = res.json()

    provider_keys = {p["key"] for p in meta["providers"]}
    assert {"openai", "google", "ollama", "openai_compatible", "qwen-cn"} <= provider_keys
    assert [a["key"] for a in meta["analysts"]] == ["market", "social", "news", "fundamentals"]
    assert {d["value"] for d in meta["depths"]} == {1, 3, 5}
    assert meta["models"]["openai"]["quick"] and meta["models"]["openai"]["deep"]
    assert meta["api_keys"]["ollama"]["optional"] is True
    assert meta["api_keys"]["openai"]["env"] == "OPENAI_API_KEY"
    # never a key value, even though the fixture env sets them
    assert "placeholder" not in res.text


def test_announcements_endpoint_is_optional(client, monkeypatch):
    from web import app as app_module

    monkeypatch.setattr(
        app_module, "fetch_announcements",
        lambda *a, **k: {"announcements": ["hello"], "require_attention": False},
    )
    assert client.get("/api/announcements").json()["announcements"] == ["hello"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    {"ticker": "../../etc/passwd"},
    {"analysis_date": "2199-01-01"},
    {"analysis_date": "05/01/2026"},
    {"analysts": []},
    {"research_depth": 2},
    {"llm_provider": "not-a-provider"},
    {"quick_think_llm": ""},
    {"llm_provider": "openai_compatible", "backend_url": None},
])
def test_analyze_rejects_bad_input(client, overrides):
    res = client.post("/api/analyze", json=valid_payload(**overrides))
    assert res.status_code == 400
    assert res.json()["detail"]


def test_crypto_ticker_without_fundamentals_still_requires_analysts():
    with pytest.raises(RequestError):
        normalize_analysis(valid_payload(ticker="BTC-USD", analysts=["fundamentals"]))


def test_crypto_ticker_keeps_other_analysts():
    selections = normalize_analysis(
        valid_payload(ticker="BTC-USD", analysts=["market", "fundamentals"])
    )
    assert selections["analysts"] == ["market"]
    assert selections["asset_type"] == "crypto"


def test_backend_url_falls_back_to_the_provider_default():
    selections = normalize_analysis(valid_payload(backend_url=None))
    assert selections["backend_url"].startswith("https://")


# ---------------------------------------------------------------------------
# Starting and streaming a run
# ---------------------------------------------------------------------------

def test_analyze_streams_events_to_completion(client, monkeypatch):
    stub_run(monkeypatch, body=[
        {"type": "message", "kind": "Agent", "content": "thinking"},
        {"type": "agents", "status": {"Market Analyst": "completed"}},
    ])
    started = client.post("/api/analyze", json=valid_payload())
    assert started.status_code == 200
    run_id = started.json()["run_id"]

    run = wait_finished(client, run_id)
    kinds = [e["type"] for e in run["events"]]
    assert "message" in kinds and "agents" in kinds and "done" in kinds
    assert run["payload"]["ticker"] == "NVDA"
    assert "api_key" not in run["payload"]

    # SSE replays the whole log and closes on the terminal status event.
    with client.stream("GET", f"/api/runs/{run_id}/events?after=0") as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in res.iter_lines()
                  if line.startswith("data: ")]
    assert [e["type"] for e in events if e["type"] in ("message", "done")]
    assert events[-1]["type"] == "status"
    assert events[-1]["status"] == "done"


def test_sse_resumes_from_after(client, monkeypatch):
    stub_run(monkeypatch)
    run_id = client.post("/api/analyze", json=valid_payload()).json()["run_id"]
    run = wait_finished(client, run_id)
    last_seq = run["events"][-1]["seq"]

    with client.stream("GET", f"/api/runs/{run_id}/events?after={last_seq}") as res:
        events = [json.loads(line[6:]) for line in res.iter_lines()
                  if line.startswith("data: ")]
    assert events == [{"type": "end"}]


def test_only_one_run_at_a_time(client, monkeypatch):
    stub_run(monkeypatch)
    assert engine.manager._engine_lock.acquire(blocking=False)
    try:
        res = client.post("/api/analyze", json=valid_payload())
        assert res.status_code == 409
    finally:
        engine.manager._engine_lock.release()


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.get("/api/runs/does-not-exist/events").status_code == 404


def test_save_report_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path))

    def fake(run):
        run.set_status("running")
        run.final_state = {"final_trade_decision": "## Decision\n\nBuy."}
        run.config = {"results_dir": str(tmp_path)}
        run.emit("done", rating="Buy", review=False, elapsed=1, wall_time="")
        run.set_status("done")

    monkeypatch.setattr(engine, "_execute_analysis", fake)
    run_id = client.post("/api/analyze", json=valid_payload()).json()["run_id"]
    wait_finished(client, run_id)

    saved = client.post(f"/api/runs/{run_id}/save-report", json={})
    assert saved.status_code == 200
    report_dir = Path(saved.json()["path"])
    assert report_dir.parent == tmp_path / "reports"
    assert (report_dir / "complete_report.md").is_file()

    # the same run refuses to write outside the results directory
    bad = client.post(f"/api/runs/{run_id}/save-report", json={"path": "/tmp/evil"})
    assert bad.status_code == 400


# ---------------------------------------------------------------------------
# Saving and browsing reports
# ---------------------------------------------------------------------------

def _finished_run(tmp_path):
    run = engine.Run("analysis", {"ticker": "NVDA"})
    run.final_state = {
        "final_trade_decision": "## Decision\n\nHold NVDA.",
        "risk_debate_state": {"judge_decision": "Hold NVDA."},
    }
    run.config = {"results_dir": str(tmp_path)}
    return run


def test_save_report_writes_under_results_dir(tmp_path):
    run = _finished_run(tmp_path)
    path = engine.save_report(run, None)
    assert path.name == "complete_report.md"
    assert path.parent.parent == tmp_path / "reports"
    assert path.is_file()
    assert "Hold NVDA" in path.read_text(encoding="utf-8")


def test_save_report_refuses_paths_outside_results_dir(tmp_path):
    run = _finished_run(tmp_path)
    with pytest.raises(RequestError):
        engine.save_report(run, "/etc/passwd")
    with pytest.raises(RequestError):
        engine.save_report(run, "../../elsewhere")


def test_reports_listing_and_reading(client, monkeypatch, tmp_path):
    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path))
    report_dir = tmp_path / "reports" / "NVDA_20260101_000000"
    (report_dir / "3_trading").mkdir(parents=True)
    (report_dir / "complete_report.md").write_text("# Report\n\n**NVDA**", encoding="utf-8")
    (report_dir / "3_trading" / "trader.md").write_text("Trader says hold.", encoding="utf-8")

    listed = client.get("/api/reports").json()
    assert [r["name"] for r in listed["reports"]] == ["NVDA_20260101_000000"]

    files = client.get("/api/reports/files", params={"path": "reports/NVDA_20260101_000000"}).json()
    names = [f["name"] for f in files["files"]]
    assert "complete_report.md" in names and "3_trading/trader.md" in names
    complete = next(f for f in files["files"] if f["name"] == "complete_report.md")
    assert "<strong>NVDA</strong>" in complete["html"]

    assert client.get("/api/reports", params={"path": "../../"}).status_code in (200, 400)
    assert client.get(
        "/api/reports/files", params={"path": "../../etc"}
    ).status_code == 400


def test_render_endpoint(client):
    html = client.post(
        "/api/render", json={"markdown": "| a |\n|---|\n| 1 |"}
    ).json()["html"]
    assert "<table>" in html


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

def test_apikey_is_saved_and_never_echoed(client, monkeypatch, tmp_path):
    monkeypatch.setattr(web.apikeys, "dotenv_path", lambda: tmp_path / ".env")
    previous = __import__("os").environ.get("OPENAI_API_KEY")
    try:
        res = client.post("/api/apikey", json={"provider": "openai", "key": "sk-secret-123"})
        assert res.status_code == 200
        assert res.json()["env"] == "OPENAI_API_KEY"
        assert "sk-secret-123" not in res.text
        assert "sk-secret-123" in (tmp_path / ".env").read_text(encoding="utf-8")
        assert oct((tmp_path / ".env").stat().st_mode)[-3:] == "600"

        assert client.post("/api/apikey", json={"provider": "openai", "key": "  "}).status_code == 400
        assert client.post("/api/apikey", json={"provider": "ollama", "key": "x"}).status_code == 400
    finally:
        import os

        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def test_backtest_validates_its_grid(client):
    res = client.post("/api/backtest", json={"tickers": "", "start": "2026-01-01", "end": "2026-02-01"})
    assert res.status_code == 400

    res = client.post("/api/backtest", json={
        "tickers": "NVDA", "start": "2026-02-01", "end": "2026-01-01", "every": 7,
    })
    assert res.status_code == 400

    res = client.post("/api/backtest", json={
        "tickers": "NVDA", "start": "2026-01-01", "end": "2199-01-01", "every": 7,
    })
    assert res.status_code == 400


def test_backtest_runs_to_a_summary(client, monkeypatch):
    def fake(run):
        run.set_status("running")
        run.emit("progress", done=2, total=2)
        run.emit("done", summary="Resolved cells: 2", cells_run=2, skipped=0,
                 failures=[], settlement_failures=[], log_path="/tmp/log", run_id="x", elapsed=1)
        run.set_status("done")

    monkeypatch.setattr(engine, "_execute_backtest", fake)
    started = client.post("/api/backtest", json={
        "tickers": "NVDA,AAPL", "start": "2026-01-01", "end": "2026-01-15", "every": 7,
    })
    assert started.status_code == 200
    run = wait_finished(client, started.json()["run_id"])
    kinds = [e["type"] for e in run["events"]]
    assert "progress" in kinds and "done" in kinds
    assert run["payload"]["tickers"] == ["NVDA", "AAPL"]


def test_backtest_uses_the_forms_model_settings(client, monkeypatch):
    monkeypatch.setattr(engine, "_execute_backtest", lambda run: run.set_status("done"))
    started = client.post("/api/backtest", json={
        "tickers": "NVDA", "start": "2026-01-01", "end": "2026-01-08", "every": 7,
        "analysts": ["market"],
        "llm_provider": "google", "quick_think_llm": "gemini-3.1-flash-lite",
        "deep_think_llm": "gemini-3.5-flash", "output_language": "German",
    })
    assert started.status_code == 200
    run = wait_finished(client, started.json()["run_id"])
    assert run["payload"]["llm_provider"] == "google"
    assert run["payload"]["deep_think_llm"] == "gemini-3.5-flash"
    assert run["payload"]["backend_url"] is None  # google has no fixed base URL


def test_backtest_rejects_unknown_provider():
    with pytest.raises(RequestError):
        engine._normalize_backtest({
            "tickers": "NVDA", "start": "2026-01-01", "end": "2026-01-08",
            "llm_provider": "skynet",
        })


def test_model_choice_config_applies_form_choices(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_DEEP_THINK_LLM", raising=False)
    config = engine.config_with_model_choices({
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-opus-5-5",
        "quick_think_llm": "claude-sonnet-5",
        "backend_url": None,
        "output_language": "Japanese",
    })
    assert config["llm_provider"] == "anthropic"
    assert config["deep_think_llm"] == "claude-opus-5-5"
    assert config["output_language"] == "Japanese"
    assert config["results_dir"] == engine.DEFAULT_CONFIG["results_dir"]


def test_model_choice_config_env_still_wins(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "ollama")
    config = engine.config_with_model_choices({"llm_provider": "anthropic"})
    # the environment set the provider, so the form's choice does not apply
    assert config["llm_provider"] != "anthropic"
