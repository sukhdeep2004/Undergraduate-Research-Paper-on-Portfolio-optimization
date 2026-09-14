import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.expected_returns import capm_expected_returns, estimate_betas


def main() -> None:
    rng = np.random.default_rng(42)
    n_periods = 120
    dates = pd.bdate_range("2020-01-01", periods=n_periods)
    market = pd.Series(rng.normal(0.0008, 0.01, n_periods), index=dates, name="market")
    asset_a = 0.5 * market + rng.normal(0, 0.008, n_periods)
    asset_b = 1.2 * market + rng.normal(0, 0.012, n_periods)
    assets = pd.DataFrame({"AssetA": asset_a, "AssetB": asset_b}, index=dates)
    risk_free = 0.02 / 252
    betas = estimate_betas(assets, market, risk_free=risk_free)
    annual_rf = 0.02
    annual_market = float(market.mean() * 252)
    mu = capm_expected_returns(annual_rf, betas, annual_market)
    print("Estimated betas (daily excess-return regression):")
    print(betas.round(3))
    print("\nCAPM expected annual returns:")
    print((mu * 100).round(2).astype(str) + "%")


if __name__ == "__main__":
    main()
