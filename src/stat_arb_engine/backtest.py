from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import performance_summary
from .strategy import estimate_hedge_ratio, generate_pair_signals


@dataclass(frozen=True)
class BacktestResult:
    pair: tuple[str, str]
    hedge_ratio: float
    signals: pd.DataFrame
    returns: pd.Series
    equity_curve: pd.Series
    summary: dict[str, float]


def run_pairs_backtest(
    prices: pd.DataFrame,
    asset_x: str,
    asset_y: str,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost_bps: float = 2.0,
) -> BacktestResult:
    """Backtest a market-neutral pairs trade on two price columns."""
    x = prices[asset_x].dropna()
    y = prices[asset_y].dropna()
    aligned = pd.concat([x, y], axis=1).dropna()
    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]

    hedge_ratio = estimate_hedge_ratio(x, y)
    signals = generate_pair_signals(x, y, hedge_ratio, window, entry_z, exit_z)

    x_ret = x.pct_change().fillna(0.0)
    y_ret = y.pct_change().fillna(0.0)

    shifted_position = signals["position"].shift(1).fillna(0.0)
    gross_returns = shifted_position * (y_ret - hedge_ratio * x_ret)

    turnover = signals["position"].diff().abs().fillna(0.0)
    costs = turnover * (transaction_cost_bps / 10_000)
    strategy_returns = gross_returns - costs
    equity_curve = (1.0 + strategy_returns).cumprod()

    summary = performance_summary(strategy_returns)
    return BacktestResult(
        pair=(asset_x, asset_y),
        hedge_ratio=hedge_ratio,
        signals=signals,
        returns=strategy_returns,
        equity_curve=equity_curve,
        summary=summary,
    )

