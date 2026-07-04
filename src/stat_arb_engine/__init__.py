"""StatArbX: a minimal statistical arbitrage pairs-trading engine."""

from .backtest import BacktestResult, run_pairs_backtest
from .data import generate_synthetic_pair, load_yahoo_prices
from .metrics import performance_summary
from .strategy import estimate_hedge_ratio, generate_pair_signals

__all__ = [
    "BacktestResult",
    "estimate_hedge_ratio",
    "generate_pair_signals",
    "generate_synthetic_pair",
    "load_yahoo_prices",
    "performance_summary",
    "run_pairs_backtest",
]

