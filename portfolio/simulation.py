from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd

STRUCTURES = ("identity", "constant_correlation", "block", "factor")
RngLike = Union[int, np.random.Generator, None]


def _as_rng(rng: RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _asset_names(n_assets: int) -> list[str]:
    width = max(2, len(str(n_assets - 1)))
    return [f"Asset_{i:0{width}d}" for i in range(n_assets)]


def _resolve_vols(
    vols: Optional[Union[float, Sequence[float], np.ndarray]],
    n_assets: int,
    rng: np.random.Generator,
    *,
    low: float = 0.1,
    high: float = 0.35,
) -> np.ndarray:
    if vols is None:
        return rng.uniform(low, high, size=n_assets)
    arr = np.asarray(vols, dtype=float)
    if arr.ndim == 0:
        return np.full(n_assets, float(arr))
    if arr.shape != (n_assets,):
        raise ValueError(f"Expected {n_assets} volatilities, got shape {arr.shape}.")
    return arr


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _to_frame(cov: np.ndarray, names: Sequence[str], attrs: dict) -> pd.DataFrame:
    frame = pd.DataFrame(cov, index=list(names), columns=list(names))
    frame.attrs.update(attrs)
    return frame


def identity_covariance(
    n_assets: int,
    *,
    vols: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    rng: RngLike = None,
) -> pd.DataFrame:
    gen = _as_rng(rng)
    v = _resolve_vols(vols, n_assets, gen)
    cov = np.diag(v**2)
    return _to_frame(
        cov,
        _asset_names(n_assets),
        {"structure": "identity", "n_assets": n_assets, "avg_vol": float(v.mean())},
    )


def constant_correlation_covariance(
    n_assets: int,
    avg_corr: float = 0.3,
    *,
    vols: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    rng: RngLike = None,
) -> pd.DataFrame:
    lower = -1.0 / (n_assets - 1)
    if not lower < avg_corr < 1.0:
        raise ValueError(
            f"avg_corr={avg_corr} must be in ({lower:.4f}, 1) for a PSD matrix with {n_assets} assets."
        )
    gen = _as_rng(rng)
    v = _resolve_vols(vols, n_assets, gen)
    corr = np.full((n_assets, n_assets), avg_corr)
    np.fill_diagonal(corr, 1.0)
    cov = _symmetrize(np.outer(v, v) * corr)
    return _to_frame(
        cov,
        _asset_names(n_assets),
        {
            "structure": "constant_correlation",
            "n_assets": n_assets,
            "avg_corr": float(avg_corr),
            "avg_vol": float(v.mean()),
        },
    )


def block_covariance(
    block_sizes: Sequence[int],
    intra_corr: float = 0.6,
    inter_corr: float = 0.1,
    *,
    vols: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    rng: RngLike = None,
) -> pd.DataFrame:
    if inter_corr > intra_corr:
        raise ValueError("inter_corr should be <= intra_corr for a block structure.")
    n_assets = int(sum(block_sizes))
    gen = _as_rng(rng)
    v = _resolve_vols(vols, n_assets, gen)
    corr = np.full((n_assets, n_assets), inter_corr)
    start = 0
    for size in block_sizes:
        end = start + size
        corr[start:end, start:end] = intra_corr
        start = end
    np.fill_diagonal(corr, 1.0)
    corr = _nearest_psd_correlation(corr)
    cov = _symmetrize(np.outer(v, v) * corr)
    return _to_frame(
        cov,
        _asset_names(n_assets),
        {
            "structure": "block",
            "n_assets": n_assets,
            "block_sizes": list(block_sizes),
            "intra_corr": float(intra_corr),
            "inter_corr": float(inter_corr),
            "avg_vol": float(v.mean()),
        },
    )


def factor_covariance(
    n_assets: int,
    n_factors: int = 1,
    *,
    factor_vol: float = 0.2,
    idio_vol: Optional[Union[float, Sequence[float], np.ndarray]] = 0.15,
    loading_scale: float = 1.0,
    rng: RngLike = None,
) -> pd.DataFrame:
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1.")
    gen = _as_rng(rng)
    B = gen.normal(0.0, loading_scale, size=(n_assets, n_factors))
    factor_var = np.full(n_factors, float(factor_vol) ** 2)
    idio = _resolve_vols(idio_vol, n_assets, gen, low=0.08, high=0.25)
    cov = _symmetrize(B @ np.diag(factor_var) @ B.T + np.diag(idio**2))
    frame = _to_frame(
        cov,
        _asset_names(n_assets),
        {
            "structure": "factor",
            "n_assets": n_assets,
            "n_factors": int(n_factors),
            "factor_vol": float(factor_vol),
            "avg_idio_vol": float(idio.mean()),
        },
    )
    frame.attrs["loadings"] = B
    frame.attrs["factor_var"] = factor_var
    frame.attrs["idio_var"] = idio**2
    return frame


def build_covariance(structure: str, **kwargs) -> pd.DataFrame:
    if structure == "identity":
        return identity_covariance(**kwargs)
    if structure == "constant_correlation":
        return constant_correlation_covariance(**kwargs)
    if structure == "block":
        return block_covariance(**kwargs)
    if structure == "factor":
        return factor_covariance(**kwargs)
    raise ValueError(f"Unknown structure {structure!r}; use one of {STRUCTURES}.")


def _nearest_psd_correlation(corr: np.ndarray) -> np.ndarray:
    corr = _symmetrize(corr)
    (vals, vecs) = np.linalg.eigh(corr)
    if vals[0] >= 0:
        return corr
    vals = np.clip(vals, 0.0, None)
    psd = _symmetrize(vecs * vals @ vecs.T)
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def simulate_returns(
    cov: pd.DataFrame,
    n_obs: int,
    *,
    mu: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    distribution: str = "normal",
    df: float = 8.0,
    rng: RngLike = None,
) -> pd.DataFrame:
    gen = _as_rng(rng)
    names = list(cov.index)
    n_assets = len(names)
    sigma = _symmetrize(np.asarray(cov, dtype=float))
    mean = _resolve_mean(mu, n_assets)
    L = np.linalg.cholesky(_psd_jitter(sigma))
    z = gen.standard_normal(size=(n_obs, n_assets))
    Lt = np.ascontiguousarray(L.T)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if distribution == "normal":
            draws = z @ Lt
        elif distribution == "t":
            if df <= 2:
                raise ValueError("Student-t df must be > 2 for a finite covariance.")
            g = gen.chisquare(df, size=(n_obs, 1)) / df
            scale = np.sqrt((df - 2.0) / df) / np.sqrt(g)
            draws = np.ascontiguousarray(z * scale) @ Lt
        else:
            raise ValueError(f"Unknown distribution {distribution!r}; use 'normal' or 't'.")
    frame = pd.DataFrame(draws + mean, columns=names)
    frame.attrs["true_cov"] = cov
    frame.attrs["dgp"] = dict(cov.attrs)
    frame.attrs["distribution"] = distribution
    if distribution == "t":
        frame.attrs["df"] = float(df)
    return frame


def simulate_regime_returns(
    covs: Sequence[pd.DataFrame],
    lengths: Sequence[int],
    *,
    mus: Optional[Sequence[Optional[Union[float, Sequence[float], np.ndarray]]]] = None,
    distribution: str = "normal",
    df: float = 8.0,
    rng: RngLike = None,
) -> pd.DataFrame:
    if len(covs) != len(lengths):
        raise ValueError("covs and lengths must have the same number of regimes.")
    if mus is not None and len(mus) != len(covs):
        raise ValueError("mus must match the number of regimes when provided.")
    gen = _as_rng(rng)
    names = list(covs[0].index)
    for c in covs:
        if list(c.index) != names:
            raise ValueError("All regime covariances must share the same asset set.")
    chunks: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    for i, (cov, length) in enumerate(zip(covs, lengths)):
        mu = None if mus is None else mus[i]
        chunk = simulate_returns(cov, length, mu=mu, distribution=distribution, df=df, rng=gen)
        chunk.attrs = {}
        chunks.append(chunk)
        labels.append(np.full(length, i))
    returns = pd.concat(chunks, ignore_index=True)
    returns.attrs["regime"] = np.concatenate(labels)
    returns.attrs["regime_covs"] = list(covs)
    returns.attrs["regime_lengths"] = list(lengths)
    return returns


def _resolve_mean(
    mu: Optional[Union[float, Sequence[float], np.ndarray]], n_assets: int
) -> np.ndarray:
    if mu is None:
        return np.zeros(n_assets)
    arr = np.asarray(mu, dtype=float)
    if arr.ndim == 0:
        return np.full(n_assets, float(arr))
    if arr.shape != (n_assets,):
        raise ValueError(f"Expected {n_assets} means, got shape {arr.shape}.")
    return arr


def _psd_jitter(sigma: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    try:
        np.linalg.cholesky(sigma)
        return sigma
    except np.linalg.LinAlgError:
        scale = float(np.mean(np.diag(sigma)))
        return sigma + np.eye(sigma.shape[0]) * max(eps, eps * scale)


def covariance_error(estimate: pd.DataFrame, true_cov: pd.DataFrame) -> dict:
    A = np.asarray(estimate, dtype=float)
    B = np.asarray(true_cov, dtype=float)
    if A.shape != B.shape:
        raise ValueError("Estimate and true covariance must have the same shape.")
    diff = float(np.linalg.norm(A - B, ord="fro"))
    denom = float(np.linalg.norm(B, ord="fro"))
    true_eig = np.linalg.eigvalsh(_symmetrize(B))
    est_eig = np.linalg.eigvalsh(_symmetrize(A))
    true_kappa = _condition_number(true_eig)
    est_kappa = _condition_number(est_eig)
    return {
        "frobenius_error": diff,
        "relative_error": diff / denom if denom > 0 else float("nan"),
        "true_condition_number": true_kappa,
        "est_condition_number": est_kappa,
        "condition_ratio": est_kappa / true_kappa if true_kappa > 0 else float("nan"),
    }


def _condition_number(eigenvalues: np.ndarray) -> float:
    lo = float(eigenvalues[0])
    hi = float(eigenvalues[-1])
    (lo, hi) = (min(lo, hi), max(lo, hi))
    return hi / lo if lo > 0 else float("inf")


@dataclass
class SimulationScenario:
    name: str
    structure: str
    n_obs: int
    structure_kwargs: dict = field(default_factory=dict)
    mu: Optional[Union[float, Sequence[float], np.ndarray]] = None
    distribution: str = "normal"
    df: float = 8.0
    seed: Optional[int] = 0

    def true_covariance(self) -> pd.DataFrame:
        return build_covariance(self.structure, rng=self.seed, **self.structure_kwargs)

    def generate(self) -> pd.DataFrame:
        cov = self.true_covariance()
        draw_seed = None if self.seed is None else self.seed + 10000
        return simulate_returns(
            cov, self.n_obs, mu=self.mu, distribution=self.distribution, df=self.df, rng=draw_seed
        )
