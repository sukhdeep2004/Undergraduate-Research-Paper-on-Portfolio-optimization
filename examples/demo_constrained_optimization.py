import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.constrained_optimization import (
    constrained_efficient_frontier,
    max_return_target_volatility,
    plot_frontier_comparison,
    solve_min_variance,
)
from portfolio.covariance import portfolio_volatility, sample_covariance
from portfolio.data import download_prices, prices_to_returns
from portfolio.efficient_frontier import historical_expected_returns, portfolio_return

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
START = "2019-01-01"
END = "2024-12-31"
MAX_WEIGHT = 0.25
TARGET_VOL = 0.28


def show(label: str, weights: pd.Series, cov: pd.DataFrame, mu: pd.Series | None = None) -> None:
    vol = portfolio_volatility(weights, cov)
    print(f"\n{label}")
    print(weights.round(4).to_string())
    print(f"  sum of weights      : {weights.sum():.4f}")
    print(f"  min / max weight    : {weights.min():.4f} / {weights.max():.4f}")
    print(f"  volatility (annual) : {vol:.2%}")
    if mu is not None:
        print(f"  return (annual)     : {portfolio_return(weights, mu):.2%}")


def main() -> None:
    print(f"Downloading {TICKERS} from Yahoo Finance ({START} to {END})...")
    prices = download_prices(TICKERS, start=START, end=END)
    returns = prices_to_returns(prices)
    print(f"{len(returns)} daily returns, {len(returns.columns)} assets")
    cov = sample_covariance(returns, annualize=True)
    mu = historical_expected_returns(returns)
    print(f"\nConstrained minimum variance (max_weight = {MAX_WEIGHT})")
    print("=" * 50)
    show("Case 1 - unconstrained (shorts allowed)", solve_min_variance(cov), cov)
    show("Case 2 - long-only", solve_min_variance(cov, long_only=True), cov)
    show(
        f"Case 3 - max weight {MAX_WEIGHT} (shorts allowed)",
        solve_min_variance(cov, max_weight=MAX_WEIGHT),
        cov,
    )
    show(
        f"Case 4 - long-only + max weight {MAX_WEIGHT}",
        solve_min_variance(cov, long_only=True, max_weight=MAX_WEIGHT),
        cov,
    )
    print("\n" + "=" * 50)
    print(f"Target volatility (max return, vol <= {TARGET_VOL:.0%})")
    print("=" * 50)
    print("\nAnnualized expected returns (historical mean):")
    print((mu * 100).round(2).astype(str) + "%")
    try:
        show(
            f"Case 5 - max return, long-only, vol <= {TARGET_VOL:.0%}",
            max_return_target_volatility(cov, mu, TARGET_VOL, long_only=True),
            cov,
            mu,
        )
    except RuntimeError as exc:
        print(f"\nCase 5 - infeasible: {exc}")
    print("\n" + "=" * 50)
    print("Efficient frontier: unconstrained vs constrained")
    print("=" * 50)
    frontiers = {
        "Unconstrained": constrained_efficient_frontier(cov, mu, n_points=40),
        "Long-only": constrained_efficient_frontier(cov, mu, long_only=True, n_points=40),
        f"Long-only + cap {MAX_WEIGHT}": constrained_efficient_frontier(
            cov, mu, long_only=True, max_weight=MAX_WEIGHT, n_points=40
        ),
    }
    for label, frontier in frontiers.items():
        print(
            f"  {label:<24} {len(frontier)} pts, vol {frontier['volatility'].min():.2%}-{frontier['volatility'].max():.2%}, max return {frontier['expected_return'].max():.2%}"
        )
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_path = figures_dir / "week2_frontier_comparison.png"
    plot_frontier_comparison(
        frontiers,
        expected_returns=mu,
        cov=cov,
        title=f"Efficient frontier: unconstrained vs constrained ({START} to {END})",
        save_path=str(save_path),
    )
    print(f"\nSaved frontier comparison plot to {save_path}")


if __name__ == "__main__":
    main()
