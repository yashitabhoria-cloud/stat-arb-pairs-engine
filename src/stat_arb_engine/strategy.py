from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    """Estimate beta in y = alpha + beta*x using ordinary least squares."""
    aligned = pd.concat([x, y], axis=1).dropna()
    x_values = aligned.iloc[:, 0].to_numpy()
    y_values = aligned.iloc[:, 1].to_numpy()
    beta, _alpha = np.polyfit(x_values, y_values, deg=1)
    return float(beta)


def compute_spread(x: pd.Series, y: pd.Series, hedge_ratio: float) -> pd.Series:
    return y - hedge_ratio * x


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def generate_pair_signals(
    x: pd.Series,
    y: pd.Series,
    hedge_ratio: float,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Generate long/short spread positions using z-score thresholds."""
    spread = compute_spread(x, y, hedge_ratio)
    zscore = rolling_zscore(spread, window)
    position = pd.Series(0.0, index=spread.index)

    current = 0.0
    for date, z in zscore.items():
        if np.isnan(z):
            position.loc[date] = 0.0
            continue
        if current == 0.0:
            if z > entry_z:
                current = -1.0
            elif z < -entry_z:
                current = 1.0
        elif abs(z) < exit_z:
            current = 0.0
        position.loc[date] = current

    return pd.DataFrame({"spread": spread, "zscore": zscore, "position": position})

