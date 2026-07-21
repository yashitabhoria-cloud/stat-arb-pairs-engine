import unittest

import pandas as pd

from stat_arb_engine.metrics import max_drawdown, performance_summary


class MetricsTests(unittest.TestCase):
    def test_max_drawdown(self):
        equity = pd.Series([1.0, 1.2, 0.9, 1.1])
        self.assertAlmostEqual(max_drawdown(equity), -0.25)

    def test_summary_contains_risk_and_efficiency_metrics(self):
        returns = pd.Series([0.01, -0.005, 0.0, 0.02, -0.01])
        turnover = pd.Series([0.0, 1.0, 0.0, 2.0, 0.0])
        costs = turnover * 0.0002
        summary = performance_summary(returns, turnover=turnover, costs=costs)

        expected = {
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "total_turnover",
            "annualized_turnover",
            "total_transaction_cost",
        }
        self.assertTrue(expected.issubset(summary))
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertAlmostEqual(summary["total_turnover"], 3.0)
        self.assertAlmostEqual(summary["total_transaction_cost"], 0.0006)

    def test_empty_returns_are_handled(self):
        summary = performance_summary(pd.Series(dtype=float))
        self.assertTrue(all(value == 0.0 for value in summary.values()))


if __name__ == "__main__":
    unittest.main()
