from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return the worst peak-to-trough decline in an equity curve."""
    equity_curve = equity_curve.dropna()
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> dict[str, float]:
    """Compute annualized return, risk, drawdown, and trading-efficiency metrics.

    ``risk_free_rate`` is expressed as an annual decimal rate. ``turnover`` and
    ``costs`` are optional per-period series produced by the backtester.
    """
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_turnover": 0.0,
            "annualized_turnover": 0.0,
            "total_transaction_cost": 0.0,
        }

    equity_curve = (1.0 + returns).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1.0)
    annualized_return = float(equity_curve.iloc[-1] ** (periods_per_year / len(returns)) - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))

    daily_risk_free_rate = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - daily_risk_free_rate
    return_std = float(returns.std(ddof=1))
    sharpe = (
        float(excess_returns.mean() / return_std * np.sqrt(periods_per_year))
        if return_std > 0
        else 0.0
    )

    downside_returns = np.minimum(excess_returns.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))
    sortino = (
        float(excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year))
        if downside_deviation > 0
        else 0.0
    )

    drawdown = max_drawdown(equity_curve)
    calmar = float(annualized_return / abs(drawdown)) if drawdown < 0 else 0.0

    active_returns = returns[returns != 0.0]
    win_rate = float((active_returns > 0.0).mean()) if not active_returns.empty else 0.0
    gross_profit = float(returns[returns > 0.0].sum())
    gross_loss = float(-returns[returns < 0.0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    total_turnover = float(turnover.fillna(0.0).sum()) if turnover is not None else 0.0
    annualized_turnover = total_turnover * periods_per_year / len(returns)
    total_transaction_cost = float(costs.fillna(0.0).sum()) if costs is not None else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "total_turnover": total_turnover,
        "annualized_turnover": float(annualized_turnover),
        "total_transaction_cost": total_transaction_cost,
    }
