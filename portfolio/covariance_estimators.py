from __future__ import annotations
import numpy as np
import pandas as pd
from portfolio.covariance import sample_covariance


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.dropna(how="any")
    if clean.shape[1] < 2:
        raise ValueError("Need at least 2 assets to estimate a covariance matrix.")
    if clean.shape[0] < 2:
        raise ValueError("Need at least 2 return observations.")
    return clean


def _mle_covariance(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc.T @ Xc / X.shape[0]


def _maybe_annualize(cov: np.ndarray, annualize: bool, periods_per_year: int) -> np.ndarray:
    return cov * periods_per_year if annualize else cov


def _constant_variance_target(S: np.ndarray) -> np.ndarray:
    n = S.shape[0]
    mu = np.trace(S) / n
    return mu * np.eye(n)


def _constant_correlation_target(S: np.ndarray) -> np.ndarray:
    v = np.sqrt(np.diag(S))
    outer_v = np.outer(v, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer_v > 0, S / outer_v, 0.0)
    n = S.shape[0]
    off_diag_sum = corr.sum() - np.trace(corr)
    r_bar = off_diag_sum / (n * (n - 1))
    F = r_bar * outer_v
    np.fill_diagonal(F, np.diag(S))
    return F


def _to_frame(cov: np.ndarray, names: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(cov, index=names, columns=names)


def sample_cov(
    returns: pd.DataFrame, *, annualize: bool = False, periods_per_year: int = 252
) -> pd.DataFrame:
    return sample_covariance(returns, annualize=annualize, periods_per_year=periods_per_year)


def shrunk_cov(
    returns: pd.DataFrame,
    delta: float,
    *,
    target: str = "constant_variance",
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if not 0.0 <= delta <= 1.0:
        raise ValueError("Shrinkage intensity delta must be in [0, 1].")
    clean = _clean_returns(returns)
    S = _mle_covariance(clean.to_numpy(dtype=float))
    if target == "constant_variance":
        F = _constant_variance_target(S)
    elif target == "constant_correlation":
        F = _constant_correlation_target(S)
    else:
        raise ValueError(
            f"Unknown target {target!r}; use 'constant_variance' or 'constant_correlation'."
        )
    cov = (1.0 - delta) * S + delta * F
    cov = _maybe_annualize(cov, annualize, periods_per_year)
    frame = _to_frame(cov, clean.columns)
    frame.attrs.update({"estimator": f"shrunk[{target}]", "shrinkage": float(delta)})
    return frame


def ledoit_wolf_cov(
    returns: pd.DataFrame, *, annualize: bool = False, periods_per_year: int = 252
) -> pd.DataFrame:
    clean = _clean_returns(returns)
    X = clean.to_numpy(dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    T = Xc.shape[0]
    S = Xc.T @ Xc / T
    F = _constant_variance_target(S)
    d2 = float(np.sum((S - F) ** 2))
    norm_sq = np.sum(Xc**2, axis=1)
    pi_hat = float(np.sum(norm_sq**2) - T * np.sum(S**2))
    b_bar2 = pi_hat / T**2
    b2 = min(b_bar2, d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    cov = delta * F + (1.0 - delta) * S
    cov = _maybe_annualize(cov, annualize, periods_per_year)
    frame = _to_frame(cov, clean.columns)
    frame.attrs.update({"estimator": "ledoit_wolf", "shrinkage": float(delta)})
    return frame


def oas_cov(
    returns: pd.DataFrame, *, annualize: bool = False, periods_per_year: int = 252
) -> pd.DataFrame:
    from sklearn.covariance import OAS

    clean = _clean_returns(returns)
    model = OAS().fit(clean.to_numpy(dtype=float))
    cov = _maybe_annualize(model.covariance_, annualize, periods_per_year)
    frame = _to_frame(cov, clean.columns)
    frame.attrs.update({"estimator": "oas", "shrinkage": float(model.shrinkage_)})
    return frame


def ewma_cov(
    returns: pd.DataFrame,
    *,
    lam: float = 0.94,
    demean: bool = True,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    if not 0.0 < lam < 1.0:
        raise ValueError("EWMA decay lam must be in (0, 1).")
    clean = _clean_returns(returns)
    X = clean.to_numpy(dtype=float)
    (T, n) = X.shape
    ages = np.arange(T)[::-1]
    w = lam**ages
    w = w / w.sum()
    if demean:
        mu_w = w @ X
        X = X - mu_w
    Xw = X * np.sqrt(w)[:, None]
    cov = Xw.T @ Xw
    cov = _maybe_annualize(cov, annualize, periods_per_year)
    frame = _to_frame(cov, clean.columns)
    half_life = float(np.log(0.5) / np.log(lam))
    frame.attrs.update({"estimator": "ewma", "lam": float(lam), "half_life": half_life})
    return frame


def cov_diagnostics(cov: pd.DataFrame) -> dict:
    A = np.asarray(cov, dtype=float)
    eig = np.sort(np.linalg.eigvalsh(A))[::-1]
    gamma_max = float(eig[0])
    gamma_min = float(eig[-1])
    condition_number = gamma_max / gamma_min if gamma_min > 0 else float("inf")
    sq = float(np.sum(eig**2))
    effective_rank = float(eig.sum() ** 2 / sq) if sq > 0 else 0.0
    return {
        "eigenvalues": eig,
        "max_eigenvalue": gamma_max,
        "min_eigenvalue": gamma_min,
        "condition_number": condition_number,
        "effective_rank": effective_rank,
    }
