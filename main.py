import os

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
# propagate() rejects a future trade_date, so default to today and let the
# environment override it for a specific point-in-time run.
ticker = os.environ.get("TRADINGAGENTS_TICKER", "NVDA")
trade_date = os.environ.get("TRADINGAGENTS_TRADE_DATE") or get_current_date()
print(f"Analyzing {ticker} as of {trade_date}...")
_, decision = ta.propagate(ticker, trade_date)
print(decision)

# Memorize mistakes and reflect
