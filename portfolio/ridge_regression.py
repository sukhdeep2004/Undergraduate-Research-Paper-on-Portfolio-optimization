from __future__ import annotations
import numpy as np
import pandas as pd


def _as_matrix(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be 2-dimensional (n_samples x n_features).")
    return arr


def ols_coefficients(X, y) -> np.ndarray:
    Xm = _as_matrix(X)
    yv = np.asarray(y, dtype=float).ravel()
    if Xm.shape[0] != yv.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    gram = Xm.T @ Xm
    rhs = Xm.T @ yv
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "X^T X is singular; OLS has no unique solution. Use ridge_coefficients with lam > 0."
        ) from exc


def ridge_coefficients(X, y, lam: float) -> np.ndarray:
    if lam < 0:
        raise ValueError("Ridge penalty lam must be non-negative.")
    Xm = _as_matrix(X)
    yv = np.asarray(y, dtype=float).ravel()
    if Xm.shape[0] != yv.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    k = Xm.shape[1]
    gram = Xm.T @ Xm + lam * np.eye(k)
    rhs = Xm.T @ yv
    return np.linalg.solve(gram, rhs)


def ridge_path(X, y, lambdas) -> np.ndarray:
    lams = np.asarray(lambdas, dtype=float).ravel()
    if np.any(lams < 0):
        raise ValueError("All ridge penalties must be non-negative.")
    Xm = _as_matrix(X)
    yv = np.asarray(y, dtype=float).ravel()
    return np.vstack([ridge_coefficients(Xm, yv, float(lam)) for lam in lams])


def _excess_returns(returns: pd.DataFrame, risk_free) -> pd.DataFrame:
    clean = returns.dropna(how="any")
    if clean.shape[0] < 2:
        raise ValueError("Need at least 2 return observations.")
    if clean.shape[1] < 2:
        raise ValueError("Need at least 2 assets.")
    if isinstance(risk_free, pd.Series):
        return clean.sub(risk_free, axis=0).dropna(how="any")
    return clean - float(risk_free)


def _normalize_to_budget(theta: pd.Series) -> pd.Series:
    total = theta.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Coefficients sum to ~0; cannot normalize to a budget.")
    weights = theta / total
    weights.name = "weight"
    return weights


def britten_jones_weights(returns: pd.DataFrame, *, risk_free=0.0) -> pd.Series:
    excess = _excess_returns(returns, risk_free)
    ones = np.ones(excess.shape[0])
    theta = ols_coefficients(excess.to_numpy(), ones)
    return _normalize_to_budget(pd.Series(theta, index=excess.columns))


def ridge_portfolio_weights(returns: pd.DataFrame, lam: float, *, risk_free=0.0) -> pd.Series:
    if lam < 0:
        raise ValueError("Ridge penalty lam must be non-negative.")
    excess = _excess_returns(returns, risk_free)
    ones = np.ones(excess.shape[0])
    theta = ridge_coefficients(excess.to_numpy(), ones, lam)
    return _normalize_to_budget(pd.Series(theta, index=excess.columns))


def ridge_gmv_weights(
    returns: pd.DataFrame, lam: float, *, annualize: bool = False, periods_per_year: int = 252
) -> pd.Series:
    if lam < 0:
        raise ValueError("Ridge penalty lam must be non-negative.")
    clean = returns.dropna(how="any")
    if clean.shape[1] < 2:
        raise ValueError("Need at least 2 assets.")
    if clean.shape[0] < 2:
        raise ValueError("Need at least 2 return observations.")
    cov = np.cov(clean.to_numpy(), rowvar=False)
    if annualize:
        cov = cov * periods_per_year
    n = cov.shape[0]
    weights = np.linalg.solve(cov + lam * np.eye(n), np.ones(n))
    return _normalize_to_budget(pd.Series(weights, index=clean.columns))
