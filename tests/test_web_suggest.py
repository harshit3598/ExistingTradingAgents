"""Stock suggestions: goal parsing, request validation, scoring, and the API.

The network stage is stubbed wherever a test would fetch prices or call an
LLM; scoring and validation themselves are pure and tested directly.
"""

import pytest

import web.engine as engine
import web.screener as screener
from web.screener import (
    DEFAULT_UNIVERSE,
    finalize_suggestions,
    normalize_suggestion_request,
    parse_goal,
    score_candidate,
)

# ---------------------------------------------------------------------------
# Goal parsing
# ---------------------------------------------------------------------------

def test_parse_the_example_goal():
    found = parse_goal("Give me certain stocks which can give 10% profit in next 1 week")
    assert found == {"target_pct": 10.0, "horizon_days": 7}


@pytest.mark.parametrize("text,expected", [
    ("20 percent within 30 days", {"target_pct": 20.0, "horizon_days": 30}),
    ("15% next month", {"target_pct": 15.0, "horizon_days": 30}),
    ("a quick 5% in 3 days", {"target_pct": 5.0, "horizon_days": 3}),
    ("50% in 2 weeks", {"target_pct": 50.0, "horizon_days": 14}),
    ("something by tomorrow", {"horizon_days": 1}),
    ("nothing usable here", {}),
])
def test_parse_variants(text, expected):
    assert parse_goal(text) == expected


def test_parse_is_case_and_word_tolerant():
    assert parse_goal("Give me 12% within NEXT WEEK") == {"target_pct": 12.0, "horizon_days": 7}


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def test_defaults_when_no_goal_is_given():
    req = normalize_suggestion_request({})
    assert req["target_pct"] == 10.0
    assert req["horizon_days"] == 7
    assert req["count"] == 5
    assert req["risk"] == "medium"
    assert req["universe"] == DEFAULT_UNIVERSE


def test_goal_text_drives_the_numbers():
    req = normalize_suggestion_request({"goal": "25% in 2 weeks"})
    assert req["target_pct"] == 25.0
    assert req["horizon_days"] == 14


def test_explicit_fields_beat_the_parsed_goal():
    req = normalize_suggestion_request({
        "goal": "5% in 3 days", "target_pct": 12, "horizon_days": 5,
    })
    assert req["target_pct"] == 12
    assert req["horizon_days"] == 5


def test_custom_universe_is_parsed_and_deduped():
    req = normalize_suggestion_request({"universe": "nvda, AAPL\nnvda;"})
    assert req["universe"] == ["NVDA", "AAPL"]


@pytest.mark.parametrize("payload", [
    {"horizon_days": 0},
    {"horizon_days": 400},
    {"target_pct": 0},
    {"target_pct": 500},
    {"count": 50},
    {"risk": "yolo"},
    {"universe": "../../etc/passwd"},
    {"llm_provider": "skynet"},
])
def test_invalid_requests_are_rejected(payload):
    with pytest.raises(screener.RequestError):
        normalize_suggestion_request(payload)


# ---------------------------------------------------------------------------
# Metrics and scoring
# ---------------------------------------------------------------------------

def _metrics(**overrides):
    base = {
        "ticker": "NVDA", "name": "NVIDIA", "price": 180.0,
        "ret_5d_pct": 4.0, "ret_21d_pct": 9.0,
        "vol_ann_pct": 55.0, "horizon_vol_pct": 6.5,
        "dist_high_pct": -2.0, "above_20ma_pct": 3.0,
        "dollar_vol_m": 3000.0,
        "target_pct": 10.0, "horizon_days": 7,
    }
    base.update(overrides)
    return base


def test_score_prefers_reachable_momentum_and_liquidity():
    strong = score_candidate(_metrics(), "medium")
    quiet = score_candidate(_metrics(horizon_vol_pct=1.0, ret_5d_pct=-3.0,
                                     dist_high_pct=-35.0, dollar_vol_m=8.0), "medium")
    assert strong > quiet
    assert 0 <= quiet < strong <= 100


def test_low_risk_penalises_absurd_volatility():
    wild = score_candidate(_metrics(horizon_vol_pct=60.0), "low")   # 6x the target
    calm = score_candidate(_metrics(horizon_vol_pct=12.0), "low")   # inside the band
    assert calm > wild


def test_high_risk_rewards_room_to_run():
    roomy = score_candidate(_metrics(horizon_vol_pct=35.0), "high")
    tight = score_candidate(_metrics(horizon_vol_pct=4.0), "high")
    assert roomy > tight


def test_missing_metrics_score_neutral_not_crash():
    score = score_candidate({"target_pct": 10.0, "horizon_days": 7}, "medium")
    assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# Final assembly (LLM optional)
# ---------------------------------------------------------------------------

def test_finalize_falls_back_to_metric_text_without_llm():
    ranked = [dict(_metrics(ticker="AAA"), score=88.0),
              dict(_metrics(ticker="BBB"), score=70.0)]
    req = {"count": 2}
    result = finalize_suggestions(ranked, None, req)
    assert [r["ticker"] for r in result] == ["AAA", "BBB"]
    assert all(not r["llm_curated"] for r in result)
    assert "Quantitative screen" in result[0]["rationale"]
    assert result[0]["risks"]


def test_finalize_uses_llm_order_and_fills_from_ranking():
    ranked = [dict(_metrics(ticker="AAA"), score=88.0),
              dict(_metrics(ticker="BBB"), score=70.0),
              dict(_metrics(ticker="CCC"), score=60.0)]
    picks = [{"ticker": "BBB", "rationale": "Fits the goal.", "risks": "Beta."}]
    result = finalize_suggestions(ranked, picks, {"count": 2})
    assert [r["ticker"] for r in result] == ["BBB", "AAA"]
    assert result[0]["llm_curated"] is True
    assert result[0]["rationale"] == "Fits the goal."
    assert result[1]["llm_curated"] is False


def test_finalize_ignores_picks_outside_the_ranking():
    ranked = [dict(_metrics(ticker="AAA"), score=88.0)]
    picks = [{"ticker": "ZZZ", "rationale": "hallucinated", "risks": "—"}]
    result = finalize_suggestions(ranked, picks, {"count": 3})
    assert [r["ticker"] for r in result] == ["AAA"]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_parse_endpoint_extracts_the_goal(client):
    res = client.post("/api/suggest/parse",
                      json={"goal": "10% profit in next 1 week"})
    assert res.status_code == 200
    assert res.json() == {"target_pct": 10.0, "horizon_days": 7}

    res = client.post("/api/suggest/parse", json={"goal": "no numbers"})
    assert res.json() == {"target_pct": None, "horizon_days": None}


def test_suggest_rejects_bad_requests(client):
    assert client.post("/api/suggest", json={"horizon_days": 0}).status_code == 400
    assert client.post("/api/suggest", json={"risk": "wild"}).status_code == 400


def test_suggest_streams_a_done_payload(client, monkeypatch):
    def fake(run):
        run.set_status("running")
        run.emit("progress", stage="fetch", done=3, total=3)
        run.emit("done", request={"target_pct": 10.0, "horizon_days": 7},
                 suggestions=[{"ticker": "NVDA", "score": 88.0,
                               "rationale": "r", "risks": "k"}],
                 disclaimer="d", universe_size=3, scored=3, failed=0,
                 llm_used=False)
        run.set_status("done")

    monkeypatch.setattr(screener, "execute_suggest", fake)
    started = client.post("/api/suggest", json={"goal": "10% in 1 week"})
    assert started.status_code == 200
    assert started.json()["kind"] == "suggest"

    run = wait_finished(client, started.json()["run_id"])
    kinds = [e["type"] for e in run["events"]]
    assert "progress" in kinds and "done" in kinds
    done = next(e for e in run["events"] if e["type"] == "done")
    assert done["suggestions"][0]["ticker"] == "NVDA"
    assert run["payload"]["target_pct"] == 10.0
    assert run["payload"]["universe"] == DEFAULT_UNIVERSE


def test_suggest_reports_runtime_failure_as_error(client, monkeypatch):
    def boom(run):
        run.set_status("running")
        raise RuntimeError("Could not fetch price data for any candidate")

    monkeypatch.setattr(screener, "execute_suggest", boom)
    started = client.post("/api/suggest", json={})
    run = wait_finished(client, started.json()["run_id"])
    assert run["status"] == "error"
    error = next(e for e in run["events"] if e["type"] == "error")
    assert "price data" in error["message"]


def wait_finished(client, run_id, timeout=5.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in engine.TERMINAL_STATUSES:
            return run
        time.sleep(0.05)
    pytest.fail(f"run {run_id} never finished")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from web.app import app

    return TestClient(app, base_url="http://localhost")
