from __future__ import annotations
import numpy as np
import pandas as pd


def sample_covariance(
    returns: pd.DataFrame, *, annualize: bool = False, periods_per_year: int = 252
) -> pd.DataFrame:
    clean = returns.dropna(how="any")
    if clean.shape[0] < 2:
        raise ValueError("Need at least 2 return observations to estimate covariance.")
    cov = clean.cov()
    if annualize:
        cov = cov * periods_per_year
    return cov


def portfolio_variance(weights: np.ndarray | pd.Series, cov: pd.DataFrame) -> float:
    w = np.asarray(weights, dtype=float)
    sigma = cov.to_numpy()
    return float(w @ sigma @ w)


def portfolio_volatility(
    weights: np.ndarray | pd.Series,
    cov: pd.DataFrame,
    *,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> float:
    var = portfolio_variance(weights, cov)
    vol = float(np.sqrt(var))
    if annualize:
        vol *= np.sqrt(periods_per_year)
    return vol
