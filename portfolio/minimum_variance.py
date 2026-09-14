from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from portfolio.covariance import sample_covariance


def minimum_variance_weights(cov: pd.DataFrame) -> pd.Series:
    asset_names = cov.index
    n_assets = len(asset_names)
    if n_assets < 2:
        raise ValueError("Need at least 2 assets for a minimum-variance portfolio.")
    sigma = cov.to_numpy()

    def full_weights(free_weights: np.ndarray) -> np.ndarray:
        weights = np.empty(n_assets)
        weights[:-1] = free_weights
        weights[-1] = 1.0 - free_weights.sum()
        return weights

    def objective(free_weights: np.ndarray) -> float:
        weights = full_weights(free_weights)
        return float(weights @ sigma @ weights)

    def gradient(free_weights: np.ndarray) -> np.ndarray:
        weights = full_weights(free_weights)
        grad_all = 2.0 * sigma @ weights
        return grad_all[:-1] - grad_all[-1]

    initial_free = np.ones(n_assets - 1) / n_assets
    result = minimize(objective, initial_free, method="L-BFGS-B", jac=gradient)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    optimal_weights = full_weights(result.x)
    return pd.Series(optimal_weights, index=asset_names, name="weight")


def minimum_variance_portfolio(
    returns: pd.DataFrame, *, annualize: bool = False, periods_per_year: int = 252
) -> pd.Series:
    cov = sample_covariance(returns, annualize=annualize, periods_per_year=periods_per_year)
    return minimum_variance_weights(cov)
