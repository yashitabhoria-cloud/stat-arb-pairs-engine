from stat_arb_engine import generate_synthetic_pair, run_pairs_backtest


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    prices = generate_synthetic_pair()
    result = run_pairs_backtest(
        prices=prices,
        asset_x="SYN_A",
        asset_y="SYN_B",
        window=60,
        entry_z=2.0,
        exit_z=0.5,
        transaction_cost_bps=2.0,
    )

    print(f"Pair: {result.pair[0]} / {result.pair[1]}")
    print(f"Hedge ratio: {result.hedge_ratio:.3f}")
    print(f"Total return: {pct(result.summary['total_return'])}")
    print(f"Annualized return: {pct(result.summary['annualized_return'])}")
    print(f"Annualized volatility: {pct(result.summary['annualized_volatility'])}")
    print(f"Sharpe ratio: {result.summary['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {pct(result.summary['max_drawdown'])}")


if __name__ == "__main__":
    main()

