"""Multi-agent trade execution: risk check → sizing → broker.

Three specialized agents coordinate to turn a rating into an order:

- **RiskAgent** — pre-trade validation: concentration limits, buying power,
  daily-loss circuit-breaker. Blocks any order that fails a rule.
- **SizingAgent** — computes how many shares (or dollars) to allocate,
  capped by a configurable fraction of the portfolio.
- **BrokerAgent** — the execution venue. Two implementations:
  ``DryRunBroker`` (default — logs what would happen, never trades) and
  ``RobinhoodBroker`` (real orders via robin_stocks).

The ``TradeOrchestrator`` wires them together. Dry-run is the default; real
trading requires ``ROBINHOOD_TRADING_ENABLED=true`` plus credentials in
``.env`` — so the system is safe to run without a live account.

Signals from the analysis graph are already 5-tier ratings (Buy / Overweight /
Hold / Underweight / Sell); a ``REVIEW`` sentinel or a Hold is never traded.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field

from tradingagents.agents.rating import RATING_REVIEW, is_review, parse_rating
from tradingagents.portfolio import PortfolioContext

logger = logging.getLogger("tradingagents.execution")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TradeAction(str, Enum):
    """What the executor will do (a 2-way simplification of the 5-tier rating)."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeOrder(BaseModel):
    """A validated, sized order ready for the broker."""

    ticker: str
    action: TradeAction
    quantity: float = Field(gt=0, description="Whole shares (stocks) or fractional units")
    limit_price: float = Field(gt=0, description="Limit price per unit")
    order_type: str = Field(default="limit", description="limit or market")
    signal: str = Field(description="The 5-tier rating that triggered this order")
    reason: str = Field(default="", description="Human-readable justification")
    dry_run: bool = True


class RiskCheck(BaseModel):
    """Result of the RiskAgent's pre-trade validation."""

    passed: bool
    reason: str
    agent: str = "RiskAgent"


class SizingResult(BaseModel):
    """Result of the SizingAgent's position sizing."""

    quantity: float = Field(ge=0)
    dollar_amount: float = Field(ge=0)
    method: str
    reasoning: str


class ExecutionResult(BaseModel):
    """What happened after the orchestrator ran."""

    ticker: str
    action: TradeAction
    signal: str
    executed: bool
    dry_run: bool
    order: TradeOrder | None = None
    risk: RiskCheck | None = None
    sizing: SizingResult | None = None
    broker_response: dict = Field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Risk Agent
# ---------------------------------------------------------------------------

class RiskAgent:
    """Pre-trade risk checks. Every check must pass for an order to proceed.

    All limits are configurable via the ``risk`` dict in the execution config;
    sensible defaults apply when keys are absent.
    """

    DEFAULT_LIMITS = {
        "max_position_pct": 0.25,       # no single position > 25% of portfolio
        "max_order_value": 50_000.0,    # hard dollar cap per order
        "min_order_value": 1.0,         # skip dust orders
        "max_open_positions": 20,       # concentration guard
    }

    def __init__(self, limits: dict | None = None):
        self.limits = {**self.DEFAULT_LIMITS, **(limits or {})}

    def check(
        self,
        order: TradeOrder,
        portfolio: PortfolioContext | None,
        buying_power: float | None = None,
        open_positions: int | None = None,
    ) -> RiskCheck:
        dollar_value = order.quantity * order.limit_price

        # Minimum order value — skip dust.
        if dollar_value < self.limits["min_order_value"]:
            return RiskCheck(
                passed=False,
                reason=f"Order value ${dollar_value:,.2f} is below the minimum "
                       f"${self.limits['min_order_value']:,.2f}.",
            )

        # Hard dollar cap.
        if dollar_value > self.limits["max_order_value"]:
            return RiskCheck(
                passed=False,
                reason=f"Order value ${dollar_value:,.2f} exceeds the per-order cap "
                       f"${self.limits['max_order_value']:,.2f}.",
            )

        # Buying power (buy only).
        if order.action == TradeAction.BUY and buying_power is not None and dollar_value > buying_power:
            return RiskCheck(
                passed=False,
                reason=f"Order value ${dollar_value:,.2f} exceeds available "
                       f"buying power ${buying_power:,.2f}.",
            )

        # Position concentration (buy only, when portfolio context is known).
        if order.action == TradeAction.BUY and portfolio is not None:
            equity = self._portfolio_equity(portfolio, buying_power)
            if equity and equity > 0:
                existing = 0.0
                pos = portfolio.position_in(order.ticker)
                if pos and pos.average_price:
                    existing = abs(pos.quantity) * pos.average_price
                total_after = existing + dollar_value
                if total_after / equity > self.limits["max_position_pct"]:
                    return RiskCheck(
                        passed=False,
                        reason=f"Order would bring {order.ticker} to "
                               f"{total_after / equity:.0%} of portfolio, exceeding "
                               f"the {self.limits['max_position_pct']:.0%} limit.",
                    )

        # Open-position count (buy only, opening a *new* position).
        if (
            order.action == TradeAction.BUY
            and open_positions is not None
            and (portfolio is None or portfolio.position_in(order.ticker) is None)
            and open_positions >= self.limits["max_open_positions"]
        ):
            return RiskCheck(
                passed=False,
                reason=f"Already holding {open_positions} positions "
                       f"(limit {self.limits['max_open_positions']}).",
            )

        # Selling: must have something to sell.
        if order.action == TradeAction.SELL and portfolio is not None:
            pos = portfolio.position_in(order.ticker)
            if pos is None or pos.quantity <= 0:
                return RiskCheck(
                    passed=False,
                    reason=f"No existing position in {order.ticker} to sell.",
                )
            if order.quantity > pos.quantity + 1e-9:
                return RiskCheck(
                    passed=False,
                    reason=f"Cannot sell {order.quantity:g} {order.ticker}: "
                           f"only {pos.quantity:g} held.",
                )

        return RiskCheck(passed=True, reason="All risk checks passed.")

    @staticmethod
    def _portfolio_equity(portfolio: PortfolioContext, buying_power: float | None) -> float:
        """Rough equity: cash + sum of position market values (approximated by
        average_price when current price is unavailable)."""
        total = portfolio.cash or 0.0
        for p in portfolio.positions:
            total += abs(p.quantity) * (p.average_price or 0.0)
        if total == 0 and buying_power:
            total = buying_power
        return total


# ---------------------------------------------------------------------------
# Sizing Agent
# ---------------------------------------------------------------------------

class SizingAgent:
    """Compute order quantity from the signal and available capital.

    Strategies:
    - ``fixed_fraction`` (default): allocate ``fraction`` of portfolio equity
      to this position, capped by available cash for buys.
    - ``full_position`` for sells: sell the entire held quantity (or a
      configurable fraction of it).
    """

    def __init__(
        self,
        strategy: str = "fixed_fraction",
        fraction: float = 0.10,
        sell_fraction: float = 1.0,
    ):
        self.strategy = strategy
        self.fraction = min(max(fraction, 0.01), 1.0)
        self.sell_fraction = min(max(sell_fraction, 0.01), 1.0)

    def size(
        self,
        action: TradeAction,
        ticker: str,
        limit_price: float,
        portfolio: PortfolioContext | None,
        buying_power: float | None = None,
    ) -> SizingResult:
        if action == TradeAction.HOLD:
            return SizingResult(
                quantity=0, dollar_amount=0, method="hold",
                reasoning="Signal is Hold — no position change.",
            )

        if action == TradeAction.SELL:
            return self._size_sell(ticker, limit_price, portfolio)

        # BUY
        equity = RiskAgent._portfolio_equity(portfolio, buying_power) if portfolio else 0.0
        cash = portfolio.cash if portfolio and portfolio.cash is not None else buying_power

        if self.strategy == "fixed_fraction" and equity > 0:
            budget = equity * self.fraction
        elif cash is not None:
            budget = cash * self.fraction
        elif buying_power:
            budget = buying_power * self.fraction
        else:
            budget = 1000.0  # fallback when no portfolio context is provided

        if cash is not None:
            budget = min(budget, cash)
        if buying_power is not None:
            budget = min(budget, buying_power)

        quantity = int(budget / limit_price) if limit_price > 0 else 0

        actual = quantity * limit_price
        return SizingResult(
            quantity=quantity,
            dollar_amount=round(actual, 2),
            method=self.strategy,
            reasoning=(
                f"Allocated {self.fraction:.0%} of "
                f"{'equity' if equity else 'cash'} ≈ ${budget:,.2f} → "
                f"{quantity:g} share(s) × ${limit_price:,.2f} = ${actual:,.2f}."
            ),
        )

    def _size_sell(
        self, ticker: str, limit_price: float, portfolio: PortfolioContext | None
    ) -> SizingResult:
        if portfolio is None:
            return SizingResult(
                quantity=0, dollar_amount=0, method="sell",
                reasoning="No portfolio context — cannot determine what to sell.",
            )
        pos = portfolio.position_in(ticker)
        if pos is None or pos.quantity <= 0:
            return SizingResult(
                quantity=0, dollar_amount=0, method="sell",
                reasoning=f"No existing position in {ticker}.",
            )
        qty = round(pos.quantity * self.sell_fraction, 6)
        return SizingResult(
            quantity=qty,
            dollar_amount=round(qty * limit_price, 2),
            method=f"sell_{self.sell_fraction:.0%}",
            reasoning=(
                f"Selling {self.sell_fraction:.0%} of {pos.quantity:g} {ticker} "
                f"held at avg ${pos.average_price or limit_price:,.2f}."
            ),
        )


# ---------------------------------------------------------------------------
# Broker abstraction
# ---------------------------------------------------------------------------

class BrokerAgent(Protocol):
    """The execution venue. Implementations must be safe to call in dry-run."""

    dry_run: bool

    def login(self) -> bool: ...
    def get_quote(self, ticker: str) -> float | None: ...
    def get_buying_power(self) -> float | None: ...
    def get_open_positions(self) -> int | None: ...
    def place_order(self, order: TradeOrder) -> dict: ...


class DryRunBroker:
    """Default broker: logs what would happen, never touches a real account."""

    dry_run = True

    def __init__(self, quote_price: float | None = None):
        self._quote_price = quote_price
        self.orders: list[TradeOrder] = []

    def login(self) -> bool:
        logger.info("DryRunBroker: no login required.")
        return True

    def get_quote(self, ticker: str) -> float | None:
        return self._quote_price  # None → caller falls back to decision price

    def get_buying_power(self) -> float | None:
        return None

    def get_open_positions(self) -> int | None:
        return None

    def place_order(self, order: TradeOrder) -> dict:
        self.orders.append(order)
        msg = (
            f"[DRY RUN] Would {order.action.value.upper()} {order.quantity:g} "
            f"{order.ticker} @ ${order.limit_price:,.2f} (limit)"
        )
        logger.info(msg)
        return {"status": "dry_run", "message": msg, "order": order.model_dump()}


class RobinhoodBroker:
    """Real execution via robin_stocks. Requires credentials + explicit opt-in.

    Login is lazy: ``login()`` must be called before any trading method.
    ``dry_run=True`` short-circuits every order so this class can be
    instantiated safely even when you don't intend to trade for real.
    """

    dry_run = False

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        mfa_code: str | None = None,
        dry_run: bool = True,
        device_token: str | None = None,
    ):
        self.username = username or os.environ.get("ROBINHOOD_USERNAME", "")
        self.password = password or os.environ.get("ROBINHOOD_PASSWORD", "")
        self.mfa_code = mfa_code
        self.dry_run = dry_run
        self.device_token = device_token
        self._logged_in = False

    def login(self) -> bool:
        if self._logged_in:
            return True
        if not self.username or not self.password:
            raise ValueError(
                "Robinhood credentials missing. Set ROBINHOOD_USERNAME and "
                "ROBINHOOD_PASSWORD in .env."
            )
        try:
            import robin_stocks.robinhood as rh
            session = rh.login(
                self.username,
                self.password,
                mfa_code=self.mfa_code,
                expiresIn=86400,
                scope="internal",
            )
            self._logged_in = session is not None
            logger.info("Robinhood login %s", "OK" if self._logged_in else "FAILED")
            return self._logged_in
        except Exception as exc:
            logger.error("Robinhood login failed: %s", exc)
            return False

    def get_quote(self, ticker: str) -> float | None:
        try:
            import robin_stocks.robinhood as rh
            quote = rh.get_stock_quote_by_symbol(ticker)
            if quote and isinstance(quote, dict):
                price = quote.get("last_trade_price") or quote.get("bid_price")
                return float(price) if price else None
            return None
        except Exception:
            return None

    def get_buying_power(self) -> float | None:
        try:
            import robin_stocks.robinhood as rh
            profile = rh.load_account_profile()
            if profile and isinstance(profile, dict):
                bp = profile.get("buying_power") or profile.get("cash")
                return float(bp) if bp else None
            return None
        except Exception:
            return None

    def get_open_positions(self) -> int | None:
        try:
            import robin_stocks.robinhood as rh
            positions = rh.get_open_stock_positions()
            if positions is None:
                return None
            return len([p for p in positions if float(p.get("quantity", 0) or 0) > 0])
        except Exception:
            return None

    def place_order(self, order: TradeOrder) -> dict:
        if self.dry_run:
            return DryRunBroker().place_order(order)

        if not self._logged_in and not self.login():
            return {"status": "error", "message": "Not logged in."}

        import robin_stocks.robinhood as rh

        try:
            fn = rh.order_buy_limit if order.action == TradeAction.BUY else rh.order_sell_limit

            response = fn(
                symbol=order.ticker,
                quantity=order.quantity,
                limitPrice=str(order.limit_price),
                timeInForce="gtc",
                extendedHours=False,
            )
            if response and isinstance(response, dict) and response.get("id"):
                logger.info(
                    "Robinhood order placed: %s %g %s @ %.2f (id=%s)",
                    order.action.value, order.quantity, order.ticker,
                    order.limit_price, response["id"],
                )
                return {"status": "submitted", "order_id": response["id"],
                        "response": response}
            return {"status": "rejected", "response": response,
                    "message": "Robinhood rejected the order."}
        except Exception as exc:
            logger.error("Robinhood order failed: %s", exc)
            return {"status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TradeOrchestrator:
    """Coordinates RiskAgent → SizingAgent → BrokerAgent for one signal.

    The multi-agent flow for each trade:
    1. Parse the signal into a TradeAction (skip REVIEW / Hold).
    2. SizingAgent computes quantity and limit price.
    3. RiskAgent validates the sized order (blocking if any check fails).
    4. BrokerAgent places (or dry-runs) the order.
    """

    def __init__(
        self,
        broker: BrokerAgent | None = None,
        risk_agent: RiskAgent | None = None,
        sizing_agent: SizingAgent | None = None,
        risk_limits: dict | None = None,
        sizing_fraction: float = 0.10,
        default_price: float | None = None,
    ):
        self.broker = broker or DryRunBroker()
        self.risk_agent = risk_agent or RiskAgent(risk_limits)
        self.sizing_agent = sizing_agent or SizingAgent(fraction=sizing_fraction)
        self.default_price = default_price

    def execute(
        self,
        signal: str,
        ticker: str,
        portfolio: PortfolioContext | None = None,
        decision_price: float | None = None,
    ) -> ExecutionResult:
        """Run the full multi-agent pipeline for one signal.

        ``signal`` is the 5-tier rating string from ``process_signal``.
        ``decision_price`` is used when the broker cannot provide a live quote.
        """
        ticker = ticker.strip().upper()
        rating = parse_rating(signal) if signal else RATING_REVIEW

        # Step 1: map rating → action; REVIEW and Hold are never traded.
        if is_review(rating):
            return ExecutionResult(
                ticker=ticker, action=TradeAction.HOLD, signal=rating,
                executed=False, dry_run=self.broker.dry_run,
                message="Signal is REVIEW (no readable rating) — not tradeable.",
            )
        action = {
            "Buy": TradeAction.BUY, "Overweight": TradeAction.BUY,
            "Hold": TradeAction.HOLD,
            "Underweight": TradeAction.SELL, "Sell": TradeAction.SELL,
        }.get(rating, TradeAction.HOLD)
        if action == TradeAction.HOLD:
            return ExecutionResult(
                ticker=ticker, action=action, signal=rating,
                executed=False, dry_run=self.broker.dry_run,
                message=f"Signal is {rating} — no position change.",
            )

        def refuse(message: str, **extra) -> ExecutionResult:
            return ExecutionResult(
                ticker=ticker, action=action, signal=rating, executed=False,
                dry_run=self.broker.dry_run, message=message, **extra,
            )

        # A live broker must be authenticated *before* quotes, buying power
        # and positions are read; unauthenticated reads come back None, which
        # would silently skip the buying-power and position-count checks.
        if not self.broker.dry_run:
            try:
                logged_in = self.broker.login()
            except ValueError as exc:
                return refuse(f"Broker login failed: {exc}")
            if not logged_in:
                return refuse("Broker login failed — no order sent.")

        # Step 2: determine limit price (live quote → decision price → default).
        price = self.broker.get_quote(ticker) or decision_price or self.default_price
        if not price or price <= 0:
            return refuse(f"No price available for {ticker} — cannot size the order.")

        # Step 3: SizingAgent computes quantity.
        buying_power = self.broker.get_buying_power()
        if buying_power is None and action == TradeAction.BUY and not self.broker.dry_run:
            # Never size a real buy off the no-context fallback budget.
            return refuse("Could not read buying power from the broker — no order sent.")
        sizing = self.sizing_agent.size(action, ticker, price, portfolio, buying_power)
        if sizing.quantity <= 0:
            return refuse(f"Sizing returned zero quantity: {sizing.reasoning}", sizing=sizing)

        # Step 4: RiskAgent validates the sized order.
        order = TradeOrder(
            ticker=ticker, action=action, quantity=sizing.quantity,
            limit_price=price, signal=rating,
            reason=sizing.reasoning, dry_run=self.broker.dry_run,
        )
        open_positions = self.broker.get_open_positions()
        risk = self.risk_agent.check(order, portfolio, buying_power, open_positions)
        if not risk.passed:
            return refuse(f"Risk check failed: {risk.reason}",
                          order=order, sizing=sizing, risk=risk)

        # Step 5: BrokerAgent places (or dry-runs) the order.
        response = self.broker.place_order(order)
        executed = response.get("status") in ("submitted", "dry_run")

        return ExecutionResult(
            ticker=ticker, action=action, signal=rating,
            executed=executed, dry_run=self.broker.dry_run,
            order=order, sizing=sizing, risk=risk,
            broker_response=response,
            message=response.get("message", "OK"),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_orchestrator(config: dict | None = None) -> TradeOrchestrator:
    """Build a TradeOrchestrator from a config dict (or env defaults).

    Config keys (all optional):
    - ``trading_enabled``: bool — must be ``True`` for real orders (default False)
    - ``robinhood_username`` / ``robinhood_password``: credentials (or env vars)
    - ``mfa_code``: one-time 2FA/OTP code for accounts that prompt at login
    - ``sizing_fraction``: float — fraction of equity per position (default 0.10)
    - ``risk_limits``: dict — overrides for RiskAgent.DEFAULT_LIMITS
    - ``default_price``: float — fallback price when no quote is available
    """
    config = config or {}
    trading_enabled = config.get(
        "trading_enabled",
        os.environ.get("ROBINHOOD_TRADING_ENABLED", "").lower() in ("1", "true", "yes"),
    )
    dry_run = not trading_enabled

    broker: BrokerAgent
    if dry_run:
        broker = DryRunBroker(quote_price=config.get("default_price"))
    else:
        broker = RobinhoodBroker(
            username=config.get("robinhood_username"),
            password=config.get("robinhood_password"),
            mfa_code=config.get("mfa_code"),
            dry_run=False,
        )

    return TradeOrchestrator(
        broker=broker,
        risk_limits=config.get("risk_limits"),
        sizing_fraction=config.get("sizing_fraction", 0.10),
        default_price=config.get("default_price"),
    )
