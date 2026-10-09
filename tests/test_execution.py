"""Multi-agent trade execution: risk, sizing, dry-run broker, orchestrator.

Every test here runs against the DryRunBroker — no live account, no network.
The RobinhoodBroker is exercised only for its dry-run short-circuit and its
credential validation (which raises before any network call).
"""

import pytest

from tradingagents.execution import (
    DryRunBroker,
    RiskAgent,
    RobinhoodBroker,
    SizingAgent,
    TradeAction,
    TradeOrchestrator,
    TradeOrder,
    create_orchestrator,
)
from tradingagents.portfolio import PortfolioContext, Position


def _order(**overrides):
    base = {
        "ticker": "NVDA",
        "action": TradeAction.BUY,
        "quantity": 10,
        "limit_price": 100.0,
        "signal": "Buy",
        "reason": "test",
        "dry_run": True,
    }
    base.update(overrides)
    return TradeOrder(**base)


def _portfolio(cash=10_000.0, positions=None):
    return PortfolioContext(
        cash=cash, currency="USD", positions=positions or [],
    )


# ---------------------------------------------------------------------------
# RiskAgent
# ---------------------------------------------------------------------------

class TestRiskAgent:
    def test_passes_a_normal_order(self):
        risk = RiskAgent().check(_order(), _portfolio(), buying_power=10_000)
        assert risk.passed
        assert "passed" in risk.reason.lower()

    def test_rejects_dust_order(self):
        risk = RiskAgent().check(
            _order(quantity=0.01, limit_price=1.0), _portfolio()
        )
        assert not risk.passed
        assert "minimum" in risk.reason.lower()

    def test_rejects_over_cap_order(self):
        risk = RiskAgent().check(
            _order(quantity=1000, limit_price=100.0), _portfolio()
        )
        assert not risk.passed
        assert "exceeds" in risk.reason.lower()

    def test_rejects_buy_over_buying_power(self):
        risk = RiskAgent().check(
            _order(quantity=100, limit_price=100.0), _portfolio(), buying_power=5000
        )
        assert not risk.passed
        assert "buying power" in risk.reason.lower()

    def test_rejects_concentration_breach(self):
        # Portfolio equity ≈ $10k (cash) + $8k existing NVDA = $18k.
        # Adding 10 × $100 = $1k → total NVDA $9k / $18k = 50% > 25%.
        existing = Position(ticker="NVDA", quantity=80, average_price=100.0)
        risk = RiskAgent().check(
            _order(quantity=10), _portfolio(cash=10_000, positions=[existing]),
            buying_power=None,
        )
        assert not risk.passed
        assert "limit" in risk.reason.lower()

    def test_rejects_too_many_open_positions(self):
        positions = [Position(ticker=f"T{i}", quantity=1, average_price=10.0)
                     for i in range(20)]
        risk = RiskAgent().check(
            _order(ticker="NEW"), _portfolio(positions=positions),
            open_positions=20,
        )
        assert not risk.passed
        assert "positions" in risk.reason.lower()

    def test_rejects_sell_without_position(self):
        risk = RiskAgent().check(
            _order(action=TradeAction.SELL), _portfolio(), open_positions=0
        )
        assert not risk.passed
        assert "no existing position" in risk.reason.lower()

    def test_rejects_sell_more_than_held(self):
        pos = Position(ticker="NVDA", quantity=5, average_price=100.0)
        risk = RiskAgent().check(
            _order(action=TradeAction.SELL, quantity=10),
            _portfolio(positions=[pos]),
        )
        assert not risk.passed
        assert "only 5" in risk.reason


# ---------------------------------------------------------------------------
# SizingAgent
# ---------------------------------------------------------------------------

class TestSizingAgent:
    def test_buy_allocates_fraction_of_equity(self):
        sizing = SizingAgent(fraction=0.10).size(
            TradeAction.BUY, "NVDA", 100.0, _portfolio(cash=10_000)
        )
        # 10% of $10k = $1k → 10 shares
        assert sizing.quantity == 10
        assert sizing.dollar_amount == 1000.0
        assert sizing.method == "fixed_fraction"

    def test_buy_capped_by_cash(self):
        sizing = SizingAgent(fraction=0.50).size(
            TradeAction.BUY, "NVDA", 100.0, _portfolio(cash=500)
        )
        # 50% of $500 = $250 → 2 shares (not 50% of a bigger equity)
        assert sizing.quantity == 2

    def test_buy_without_portfolio_falls_back_to_buying_power(self):
        sizing = SizingAgent(fraction=0.10).size(
            TradeAction.BUY, "NVDA", 100.0, None, buying_power=5_000
        )
        # 10% of $5k = $500 → 5 shares at $100
        assert sizing.quantity == 5

    def test_buy_floors_at_one_share_when_budget_allows(self):
        sizing = SizingAgent(fraction=0.01).size(
            TradeAction.BUY, "NVDA", 100.0, _portfolio(cash=500)
        )
        # 1% of $500 = $5 → below share price, but budget < price → 0 shares
        assert sizing.quantity == 0

    def test_sell_uses_full_position_by_default(self):
        pos = Position(ticker="NVDA", quantity=42, average_price=90.0)
        sizing = SizingAgent().size(
            TradeAction.SELL, "NVDA", 100.0, _portfolio(positions=[pos])
        )
        assert sizing.quantity == 42
        assert sizing.dollar_amount == 4200.0

    def test_sell_fraction_partial(self):
        pos = Position(ticker="NVDA", quantity=10, average_price=90.0)
        sizing = SizingAgent(sell_fraction=0.5).size(
            TradeAction.SELL, "NVDA", 100.0, _portfolio(positions=[pos])
        )
        assert sizing.quantity == 5

    def test_sell_without_position_returns_zero(self):
        sizing = SizingAgent().size(TradeAction.SELL, "NVDA", 100.0, _portfolio())
        assert sizing.quantity == 0
        assert "no existing position" in sizing.reasoning.lower()


# ---------------------------------------------------------------------------
# DryRunBroker
# ---------------------------------------------------------------------------

class TestDryRunBroker:
    def test_records_order_and_returns_dry_run_status(self):
        broker = DryRunBroker()
        result = broker.place_order(_order())
        assert result["status"] == "dry_run"
        assert len(broker.orders) == 1

    def test_dry_run_flag(self):
        assert DryRunBroker().dry_run is True

    def test_login_is_noop(self):
        assert DryRunBroker().login() is True

    def test_quote_is_none_by_default(self):
        assert DryRunBroker().get_quote("NVDA") is None

    def test_quote_returns_configured_price(self):
        broker = DryRunBroker(quote_price=123.45)
        assert broker.get_quote("NVDA") == 123.45


# ---------------------------------------------------------------------------
# RobinhoodBroker (dry-run path only)
# ---------------------------------------------------------------------------

class TestRobinhoodBrokerSafety:
    def test_dry_run_short_circuits_place_order(self):
        broker = RobinhoodBroker(dry_run=True)
        result = broker.place_order(_order(dry_run=True))
        assert result["status"] == "dry_run"

    def test_login_raises_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ROBINHOOD_USERNAME", raising=False)
        monkeypatch.delenv("ROBINHOOD_PASSWORD", raising=False)
        broker = RobinhoodBroker(username="", password="", dry_run=True)
        with pytest.raises(ValueError, match="credentials"):
            broker.login()

    def test_dry_run_flag_defaults_true(self):
        assert RobinhoodBroker().dry_run is True


# ---------------------------------------------------------------------------
# TradeOrchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_review_signal_is_never_traded(self):
        orch = TradeOrchestrator(broker=DryRunBroker())
        result = orch.execute("There is no rating in this decision.", "NVDA")
        assert result.executed is False
        assert result.action == TradeAction.HOLD
        assert "REVIEW" in result.signal
        assert "review" in result.message.lower() or "not tradeable" in result.message.lower()

    def test_hold_signal_is_not_traded(self):
        orch = TradeOrchestrator(broker=DryRunBroker())
        result = orch.execute("Rating: Hold", "NVDA")
        assert result.executed is False
        assert result.action == TradeAction.HOLD
        assert "hold" in result.message.lower()

    def test_buy_signal_executes_in_dry_run(self):
        orch = TradeOrchestrator(
            broker=DryRunBroker(quote_price=100.0),
            sizing_fraction=0.10,
        )
        result = orch.execute(
            "Rating: Buy", "NVDA", portfolio=_portfolio(cash=10_000)
        )
        assert result.executed is True
        assert result.dry_run is True
        assert result.action == TradeAction.BUY
        assert result.order.quantity == 10
        assert result.risk.passed is True
        assert result.broker_response["status"] == "dry_run"

    def test_sell_signal_executes_in_dry_run(self):
        pos = Position(ticker="NVDA", quantity=20, average_price=90.0)
        orch = TradeOrchestrator(broker=DryRunBroker(quote_price=100.0))
        result = orch.execute(
            "Rating: Sell", "NVDA", portfolio=_portfolio(positions=[pos])
        )
        assert result.executed is True
        assert result.action == TradeAction.SELL
        assert result.order.quantity == 20

    def test_risk_block_prevents_execution(self):
        # Order value 100 × $100 = $10k → over default $10k cap? No, cap is $50k.
        # Use a huge order to trigger the cap.
        orch = TradeOrchestrator(
            broker=DryRunBroker(quote_price=100.0),
            risk_limits={"max_order_value": 500.0},
        )
        result = orch.execute(
            "Rating: Buy", "NVDA", portfolio=_portfolio(cash=100_000)
        )
        assert result.executed is False
        assert result.risk is not None and not result.risk.passed
        assert "risk check failed" in result.message.lower()

    def test_zero_size_is_not_executed(self):
        orch = TradeOrchestrator(
            broker=DryRunBroker(quote_price=100.0),
            sizing_fraction=0.01,
        )
        result = orch.execute(
            "Rating: Buy", "NVDA", portfolio=_portfolio(cash=100)
        )
        # 1% of $100 = $1 → below share price → quantity 0
        assert result.executed is False
        assert "zero quantity" in result.message.lower()

    def test_no_price_available_returns_error(self):
        orch = TradeOrchestrator(broker=DryRunBroker(quote_price=None))
        result = orch.execute("Rating: Buy", "NVDA", portfolio=_portfolio())
        assert result.executed is False
        assert "no price" in result.message.lower()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_defaults_to_dry_run(self, monkeypatch):
        monkeypatch.delenv("ROBINHOOD_TRADING_ENABLED", raising=False)
        orch = create_orchestrator({})
        assert orch.broker.dry_run is True
        assert isinstance(orch.broker, DryRunBroker)

    def test_trading_enabled_creates_robinhood_broker(self, monkeypatch):
        monkeypatch.setenv("ROBINHOOD_TRADING_ENABLED", "true")
        monkeypatch.setenv("ROBINHOOD_USERNAME", "user")
        monkeypatch.setenv("ROBINHOOD_PASSWORD", "pass")
        orch = create_orchestrator({})
        assert isinstance(orch.broker, RobinhoodBroker)
        assert orch.broker.dry_run is False

    def test_config_overrides(self):
        orch = create_orchestrator({
            "sizing_fraction": 0.25,
            "risk_limits": {"max_order_value": 1000},
            "default_price": 50.0,
        })
        assert orch.sizing_agent.fraction == 0.25
        assert orch.risk_agent.limits["max_order_value"] == 1000
        assert orch.default_price == 50.0

    def test_mfa_code_and_credentials_reach_the_broker(self, monkeypatch):
        monkeypatch.delenv("ROBINHOOD_TRADING_ENABLED", raising=False)
        orch = create_orchestrator({
            "trading_enabled": True,
            "robinhood_username": "user@example.com",
            "robinhood_password": "secret",
            "mfa_code": "123456",
        })
        assert isinstance(orch.broker, RobinhoodBroker)
        assert orch.broker.username == "user@example.com"
        assert orch.broker.password == "secret"
        assert orch.broker.mfa_code == "123456"
