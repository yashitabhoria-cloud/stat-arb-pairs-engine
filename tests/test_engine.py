import unittest

from stat_arb_engine import generate_synthetic_pair, run_pairs_backtest
from stat_arb_engine.strategy import estimate_hedge_ratio


class EngineTests(unittest.TestCase):
    def test_synthetic_pair_shape(self):
        prices = generate_synthetic_pair(n_days=100)
        self.assertEqual(prices.shape, (100, 2))
        self.assertTrue(prices.notna().all().all())

    def test_hedge_ratio_is_reasonable_on_synthetic_pair(self):
        prices = generate_synthetic_pair(n_days=300, beta=1.2)
        beta = estimate_hedge_ratio(prices["SYN_A"], prices["SYN_B"])
        self.assertGreater(beta, 1.1)
        self.assertLess(beta, 1.35)

    def test_backtest_runs_end_to_end(self):
        prices = generate_synthetic_pair(n_days=300)
        result = run_pairs_backtest(prices, "SYN_A", "SYN_B")
        self.assertGreater(result.hedge_ratio, 0)
        self.assertEqual(len(result.returns), len(prices))
        self.assertEqual(len(result.equity_curve), len(prices))
        self.assertIn("sharpe_ratio", result.summary)


if __name__ == "__main__":
    unittest.main()
