from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    equity_curve = (1.0 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1.0
    annualized_return = equity_curve.iloc[-1] ** (periods_per_year / len(returns)) - 1.0
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown(equity_curve),
    }

