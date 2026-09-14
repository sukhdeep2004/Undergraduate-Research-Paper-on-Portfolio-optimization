from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import pandas as pd
from portfolio.constrained_optimization import solve_min_variance
from portfolio.covariance_estimators import ledoit_wolf_cov, sample_cov
from portfolio.research_optimizer import research_gmv_weights, risk_budget_weights

WeightFn = Callable[..., pd.Series]
CovEstimator = Callable[..., pd.DataFrame]


@dataclass
class Strategy:
    name: str
    weight_fn: WeightFn
    cov_estimator: CovEstimator = sample_cov
    family: str = "other"

    def __post_init__(self) -> None:
        if not callable(self.weight_fn):
            raise TypeError(f"weight_fn for {self.name!r} must be callable.")
        if not callable(self.cov_estimator):
            raise TypeError(f"cov_estimator for {self.name!r} must be callable.")


def equal_weight(cov: pd.DataFrame) -> pd.Series:
    n = len(cov.index)
    return pd.Series(np.full(n, 1.0 / n), index=cov.index, name="weight")


def markowitz_gmv(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
    return solve_min_variance(cov, x0=x0)


def make_constrained_gmv(*, long_only: bool = True, max_weight: Optional[float] = 0.2) -> WeightFn:

    def weight_fn(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
        cap = max_weight
        if cap is not None:
            cap = max(cap, 1.0 / len(cov.index))
        return solve_min_variance(cov, long_only=long_only, max_weight=cap, x0=x0)

    return weight_fn


def make_ridge_gmv(lambda2: float = 0.1) -> WeightFn:

    def weight_fn(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
        return research_gmv_weights(cov, lambda2=lambda2, x0=x0)

    return weight_fn


def gmv_weights(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
    return solve_min_variance(cov, x0=x0)


def make_proposed(
    *,
    gamma_d: float = 0.05,
    gamma_e: float = 0.05,
    lambda2: float = 0.0,
    long_only: bool = True,
    max_weight: Optional[float] = None,
) -> WeightFn:

    def weight_fn(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
        cap = max_weight
        if cap is not None:
            cap = max(cap, 1.0 / len(cov.index))
        return research_gmv_weights(
            cov,
            lambda2=lambda2,
            gamma_d=gamma_d,
            gamma_e=gamma_e,
            long_only=long_only,
            max_weight=cap,
            x0=x0,
        )

    return weight_fn


def make_risk_parity(budgets: Optional[np.ndarray] = None) -> WeightFn:

    def weight_fn(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
        return risk_budget_weights(cov, budgets=budgets, x0=x0)

    return weight_fn


def make_sparse(gamma_l1: float = 0.02, *, long_only: bool = False) -> WeightFn:

    def weight_fn(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
        return research_gmv_weights(cov, l1=gamma_l1, long_only=long_only, x0=x0)

    return weight_fn


def make_sector_neutral(
    group_targets: "dict[str, tuple[list[int], float]]",
    *,
    gamma_d: float = 0.05,
    gamma_e: float = 0.05,
    long_only: bool = True,
) -> WeightFn:

    def weight_fn(cov: pd.DataFrame, x0: Optional[np.ndarray] = None) -> pd.Series:
        if x0 is None:
            x0 = np.full(len(cov.index), 1.0 / len(cov.index))
            for _, (idx, target) in group_targets.items():
                if idx:
                    x0[list(idx)] = target / len(idx)
        return research_gmv_weights(
            cov,
            gamma_d=gamma_d,
            gamma_e=gamma_e,
            long_only=long_only,
            group_targets=group_targets,
            x0=x0,
        )

    return weight_fn


def default_strategies(
    *, max_weight: float = 0.2, lambda2: float = 0.1, gamma_d: float = 0.05, gamma_e: float = 0.05
) -> "dict[str, Strategy]":
    strategies = [
        Strategy("equal_weight", equal_weight, sample_cov, family="baseline"),
        Strategy("markowitz", markowitz_gmv, sample_cov, family="classical"),
        Strategy(
            "constrained",
            make_constrained_gmv(long_only=True, max_weight=max_weight),
            sample_cov,
            family="classical",
        ),
        Strategy("ridge", make_ridge_gmv(lambda2), sample_cov, family="regularized"),
        Strategy("shrinkage", gmv_weights, ledoit_wolf_cov, family="regularized"),
        Strategy(
            "proposed",
            make_proposed(gamma_d=gamma_d, gamma_e=gamma_e, long_only=True),
            sample_cov,
            family="proposed",
        ),
    ]
    return {s.name: s for s in strategies}
