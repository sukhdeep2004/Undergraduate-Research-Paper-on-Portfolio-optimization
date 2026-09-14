import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.covariance import sample_covariance
from portfolio.data import download_prices, prices_to_returns
from portfolio.efficient_frontier import (
    efficient_frontier,
    historical_expected_returns,
    plot_efficient_frontier,
)
from portfolio.expected_returns import capm_expected_returns, estimate_betas
from portfolio.minimum_variance import minimum_variance_weights

TRADING_DAYS = 252
DEFAULT_ANNUAL_RF = 0.02
MARKET_TICKER = "SPY"


def capm_expected_returns_from_data(
    returns: pd.DataFrame,
    *,
    market_ticker: str = MARKET_TICKER,
    annual_rf: float = DEFAULT_ANNUAL_RF,
    start: str,
    end: str | None,
) -> tuple[pd.Series, pd.Series]:
    spy_prices = download_prices(market_ticker, start=start, end=end)
    market_returns = prices_to_returns(spy_prices)[market_ticker]
    daily_rf = annual_rf / TRADING_DAYS
    betas = estimate_betas(returns, market_returns, risk_free=daily_rf)
    annual_market = float(market_returns.mean() * TRADING_DAYS)
    mu = capm_expected_returns(annual_rf, betas, annual_market)
    return (mu, betas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efficient frontier demo (Yahoo Finance).")
    parser.add_argument(
        "--capm",
        action="store_true",
        help="Use CAPM expected returns (SPY market) instead of historical means.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    start = "2019-01-01"
    end = "2024-12-31"
    print(f"Downloading {tickers} from Yahoo Finance ({start} to {end})...")
    prices = download_prices(tickers, start=start, end=end)
    returns = prices_to_returns(prices)
    cov = sample_covariance(returns, annualize=True)
    if args.capm:
        print(f"\nUsing CAPM (market={MARKET_TICKER}, R_f={DEFAULT_ANNUAL_RF:.2%} annual)...")
        (mu, betas) = capm_expected_returns_from_data(returns, start=start, end=end)
        mu_label = "CAPM expected return"
        betas_line = f"\nCAPM betas vs {MARKET_TICKER}:\n{betas.round(3)}"
        plot_suffix = "CAPM"
        figure_name = "efficient_frontier_capm.png"
    else:
        mu = historical_expected_returns(returns)
        mu_label = "historical mean"
        betas_line = ""
        plot_suffix = "historical mean"
        figure_name = "efficient_frontier.png"
    print(f"\n{len(returns)} daily returns, {len(tickers)} assets")
    if len(returns) < TRADING_DAYS:
        print(
            f"Warning: only {len(returns)} return rows — results may be unstable. Check tickers and overlapping history on Yahoo Finance."
        )
    print(f"\nAnnualized expected returns ({mu_label}):")
    print((mu * 100).round(2).astype(str) + "%")
    if betas_line:
        print(betas_line)
    frontier = efficient_frontier(cov, mu, n_points=35)
    gmv = minimum_variance_weights(cov)
    print("\nGlobal minimum-variance weights:")
    print(gmv.round(4))
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_path = figures_dir / figure_name
    plot_efficient_frontier(
        frontier,
        expected_returns=mu,
        cov=cov,
        gmv_weights=gmv,
        title=f"Efficient frontier ({plot_suffix}, {start} to {end})",
        save_path=str(save_path),
    )
    print(f"\nSaved plot to {save_path}")


if __name__ == "__main__":
    main()
