import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.covariance import portfolio_volatility, sample_covariance


def main() -> None:
    rng = np.random.default_rng(7)
    n_periods = 200
    dates = pd.bdate_range("2019-01-01", periods=n_periods)
    z = rng.normal(0, 0.01, (n_periods, 2))
    asset_a = z[:, 0]
    asset_b = 0.6 * z[:, 0] + 0.8 * z[:, 1]
    returns = pd.DataFrame({"AssetA": asset_a, "AssetB": asset_b}, index=dates)
    cov_daily = sample_covariance(returns)
    cov_annual = sample_covariance(returns, annualize=True)
    equal_weights = pd.Series([0.5, 0.5], index=returns.columns)
    vol_annual = portfolio_volatility(equal_weights, cov_annual)
    print("Daily covariance matrix:")
    print(cov_daily.round(6))
    print("\nAnnualized covariance matrix (x252):")
    print(cov_annual.round(4))
    print(f"\nEqual-weight portfolio volatility (annualized): {vol_annual:.2%}")


if __name__ == "__main__":
    main()
