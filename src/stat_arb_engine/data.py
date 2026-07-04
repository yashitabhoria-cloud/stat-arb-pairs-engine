from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_pair(n_days: int = 756, beta: float = 1.2, seed: int = 42) -> pd.DataFrame:
    """Generate two synthetic cointegrated price series for a zero-dependency demo."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    x_returns = rng.normal(loc=0.0003, scale=0.012, size=n_days)
    x = 100 * np.exp(np.cumsum(x_returns))

    spread = np.zeros(n_days)
    noise = rng.normal(loc=0.0, scale=0.6, size=n_days)
    for i in range(1, n_days):
        spread[i] = 0.92 * spread[i - 1] + noise[i]

    y = beta * x + spread + 5.0
    return pd.DataFrame({"SYN_A": x, "SYN_B": y}, index=dates)


def load_yahoo_prices(tickers: list[str], start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """Load adjusted close prices from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance to load market data: pip install yfinance") from exc

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if "Close" in data:
        prices = data["Close"]
    else:
        prices = data
    prices = prices.dropna(how="all").ffill().dropna()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices

