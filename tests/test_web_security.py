"""Request hardening for the web UI: host, CSRF, token auth, headers, XSS.

The UI can store API keys and place live orders, so a hostile web page the
user happens to have open must not be able to drive it.
"""

import pytest
from fastapi.testclient import TestClient

import web.engine as engine
from web.app import app
from web.markdown_util import render_markdown
from web.security import TOKEN_ENV, ensure_token
from tradingagents.execution import TradeAction, TradeOrchestrator


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    return TestClient(app, base_url="http://localhost")


def test_security_headers_on_every_response(client):
    res = client.get("/")
    csp = res.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp and "script-src 'self'" in csp
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-content-type-options"] == "nosniff"


def test_foreign_host_is_refused(monkeypatch):
    """DNS rebinding: evil.example resolving to 127.0.0.1 still says Host: evil.example."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    evil = TestClient(app, base_url="http://evil.example:8000")
    assert evil.get("/api/meta").status_code == 400


def test_extra_hosts_can_be_allowed(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    monkeypatch.setenv("TRADINGAGENTS_WEB_ALLOWED_HOSTS", "box.lan")
    assert TestClient(app, base_url="http://box.lan:8000").get("/").status_code == 200


def test_cross_origin_post_is_refused(client):
    res = client.post("/api/suggest/parse", json={"goal": "5% in 2 weeks"},
                      headers={"Origin": "https://evil.example"})
    assert res.status_code == 403


def test_same_origin_post_is_allowed(client):
    res = client.post("/api/suggest/parse", json={"goal": "5% in 2 weeks"},
                      headers={"Origin": "http://localhost"})
    assert res.status_code == 200


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", None])
def test_non_json_post_is_refused(client, content_type):
    """Simple (preflight-free) cross-site requests cannot carry application/json."""
    headers = {"Content-Type": content_type} if content_type else {}
    res = client.post("/api/apikey", content=b'{"provider":"openai","key":"sk-x"}',
                      headers=headers)
    assert res.status_code == 415


def test_token_auth_flow(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "s3cret-token")
    c = TestClient(app, base_url="http://192.168.1.20:8000")
    assert c.get("/api/meta").status_code == 401
    assert c.get("/?token=wrong", follow_redirects=False).status_code == 401
    login = c.get("/?token=s3cret-token", follow_redirects=False)
    assert login.status_code == 303 and "httponly" in login.headers["set-cookie"].lower()
    assert c.get("/api/meta").status_code == 200  # cookie now carried
    bearer = TestClient(app, base_url="http://192.168.1.20:8000")
    assert bearer.get("/api/meta", headers={"Authorization": "Bearer s3cret-token"}).status_code == 200


def test_token_is_minted_only_for_non_loopback_binds(monkeypatch):
    # setenv (not delenv) so monkeypatch restores the variable ensure_token writes.
    monkeypatch.setenv(TOKEN_ENV, "")
    assert ensure_token("127.0.0.1") is None
    token = ensure_token("0.0.0.0")
    assert token and len(token) >= 24


def test_markdown_never_renders_raw_html():
    html = render_markdown('<img src=x onerror="alert(1)">\n\n[x](javascript:alert(1))')
    assert "<img" not in html and "&lt;img" in html
    assert "href=\"javascript" not in html


def test_api_key_rejects_control_characters(client, monkeypatch, tmp_path):
    import web.apikeys

    monkeypatch.setattr(web.apikeys, "dotenv_path", lambda: tmp_path / ".env")
    res = client.post("/api/apikey", json={"provider": "openai", "key": "sk-a\nEVIL=1"})
    assert res.status_code == 400


def test_portfolio_paths_must_be_json_files():
    with pytest.raises(engine.RequestError):
        engine.load_portfolio_input("/etc/passwd")


def test_backend_url_must_be_http():
    with pytest.raises(engine.RequestError):
        engine._model_settings({"llm_provider": "openai", "backend_url": "file:///etc"}, None)


def test_backtest_keeps_the_portfolio():
    sel = engine._normalize_backtest({
        "tickers": "NVDA", "start": "2026-01-01", "end": "2026-01-10",
        "portfolio": '{"cash": 1000, "positions": []}',
    })
    assert sel["portfolio"]


class _LiveBroker:
    """A live broker whose reads only work after login, like Robinhood."""

    dry_run = False

    def __init__(self, login_ok=True, buying_power=10_000.0):
        self.logged_in = False
        self.login_ok = login_ok
        self.bp = buying_power
        self.orders = []

    def login(self):
        self.logged_in = self.login_ok
        return self.login_ok

    def get_quote(self, ticker):
        return 100.0 if self.logged_in else None

    def get_buying_power(self):
        return self.bp if self.logged_in else None

    def get_open_positions(self):
        return 0 if self.logged_in else None

    def place_order(self, order):
        self.orders.append(order)
        return {"status": "submitted", "order_id": "1"}


def test_live_orchestrator_logs_in_before_reading_the_account():
    broker = _LiveBroker()
    result = TradeOrchestrator(broker=broker).execute("Buy", "NVDA")
    assert result.executed and result.order.limit_price == 100.0
    assert result.order.action == TradeAction.BUY


def test_live_orchestrator_refuses_when_login_fails():
    broker = _LiveBroker(login_ok=False)
    result = TradeOrchestrator(broker=broker).execute("Buy", "NVDA", decision_price=50.0)
    assert not result.executed and not broker.orders


def test_live_buy_without_buying_power_is_refused():
    broker = _LiveBroker(buying_power=None)
    result = TradeOrchestrator(broker=broker).execute("Buy", "NVDA")
    assert not result.executed and "buying power" in result.message
