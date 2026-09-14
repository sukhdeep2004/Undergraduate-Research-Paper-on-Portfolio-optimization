from __future__ import annotations
import numpy as np
import pandas as pd


def _to_series(x, index: pd.Index) -> pd.Series:
    if np.isscalar(x):
        return pd.Series(float(x), index=index)
    return pd.Series(x, index=index)


def excess_returns(returns: pd.Series, risk_free) -> pd.Series:
    rf = _to_series(risk_free, returns.index)
    return returns - rf


def estimate_beta(asset_returns: pd.Series, market_returns: pd.Series, risk_free=0.0) -> float:
    aligned = pd.concat([asset_returns, market_returns], axis=1, keys=["asset", "market"]).dropna()
    asset_excess = excess_returns(aligned["asset"], risk_free)
    market_excess = excess_returns(aligned["market"], risk_free)
    market_var = market_excess.var()
    if market_var == 0 or np.isnan(market_var):
        raise ValueError("Market excess returns have zero variance; beta is undefined.")
    return float(asset_excess.cov(market_excess) / market_var)


def estimate_betas(
    asset_returns: pd.DataFrame, market_returns: pd.Series, risk_free=0.0
) -> pd.Series:
    betas = {
        col: estimate_beta(asset_returns[col], market_returns, risk_free)
        for col in asset_returns.columns
    }
    return pd.Series(betas, name="beta")


def capm_expected_return(risk_free: float, beta: float, market_expected_return: float) -> float:
    market_premium = market_expected_return - risk_free
    return risk_free + beta * market_premium


def capm_expected_returns(
    risk_free: float, betas: pd.Series, market_expected_return: float
) -> pd.Series:
    market_premium = market_expected_return - risk_free
    mu = risk_free + betas * market_premium
    mu.name = "expected_return"
    return mu
