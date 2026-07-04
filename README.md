# StatArbX: Statistical Arbitrage Pairs-Trading Engine

An end-to-end Python research project for cointegration-based statistical arbitrage. The engine screens equity pairs, estimates hedge ratios, generates mean-reversion signals, and evaluates a market-neutral pairs-trading strategy with transaction costs.

This is intentionally a clean v1: small enough to understand, but structured like a real quant research repo.

## Features

- Synthetic cointegrated data demo that runs without API keys
- Optional Yahoo Finance data loader via `yfinance`
- OLS hedge-ratio estimation
- Spread and rolling z-score signal generation
- Long/short pairs-trading backtest
- Transaction cost support
- Performance metrics: total return, annualized return, volatility, Sharpe, max drawdown
- Modular package layout with tests

## Project Structure

```text
stat-arb-pairs-engine/
├── examples/run_demo.py
├── src/stat_arb_engine/
│   ├── backtest.py
│   ├── data.py
│   ├── metrics.py
│   └── strategy.py
├── tests/test_engine.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Quickstart

```bash
git clone <your-repo-url>
cd stat-arb-pairs-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python examples/run_demo.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python examples/run_demo.py
```

## Example Output

```text
Pair: SYN_A / SYN_B
Hedge ratio: 1.21
Total return: 12.4%
Annualized return: 6.1%
Annualized volatility: 8.7%
Sharpe ratio: 0.70
Max drawdown: -5.3%
```

## Methodology

For two assets \(X_t\) and \(Y_t\), the engine estimates:

```text
Y_t = alpha + beta X_t + epsilon_t
```

The residual spread is:

```text
spread_t = Y_t - beta X_t
```

A rolling z-score is computed:

```text
z_t = (spread_t - rolling_mean) / rolling_std
```

Trading logic:

- If `z > entry_z`, short the spread
- If `z < -entry_z`, long the spread
- If `|z| < exit_z`, close position

## Roadmap

- Engle-Granger cointegration screening
- ADF test on residual spread
- Rolling OLS hedge ratio
- Kalman filter hedge ratio
- Walk-forward validation
- Streamlit dashboard
- FastAPI backend
- C++/pybind11 speedup for signal generation

## Resume Pitch

Built a statistical arbitrage pairs-trading engine in Python that estimates hedge ratios, generates z-score-based mean-reversion signals, and backtests market-neutral long/short strategies with transaction costs, Sharpe ratio, drawdown, and return analytics.

