import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.covariance import portfolio_volatility, sample_covariance
from portfolio.minimum_variance import minimum_variance_weights


def main() -> None:
    rng = np.random.default_rng(21)
    n_periods = 250
    dates = pd.bdate_range("2018-01-01", periods=n_periods)
    z = rng.normal(0, 0.012, (n_periods, 3))
    returns = pd.DataFrame(
        {
            "LowVol": 0.3 * z[:, 0] + 0.1 * z[:, 1],
            "MidVol": 0.8 * z[:, 0] + 0.4 * z[:, 2],
            "HighVol": 1.2 * z[:, 0] + 0.9 * z[:, 1] + 0.5 * z[:, 2],
        },
        index=dates,
    )
    cov = sample_covariance(returns)
    weights = minimum_variance_weights(cov)
    equal = pd.Series(1 / len(returns.columns), index=returns.columns)
    min_vol = portfolio_volatility(weights, cov, annualize=True)
    eq_vol = portfolio_volatility(equal, cov, annualize=True)
    print("Minimum-variance weights (short selling allowed):")
    print(weights.round(4))
    print(f"\nMin-var portfolio volatility (annualized): {min_vol:.2%}")
    print(f"Equal-weight portfolio volatility (annualized): {eq_vol:.2%}")


if __name__ == "__main__":
    main()
