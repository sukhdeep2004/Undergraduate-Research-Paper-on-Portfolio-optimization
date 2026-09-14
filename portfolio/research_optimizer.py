from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

_WEIGHT_FLOOR = 1e-12


def precision_metric(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    (vals, vecs) = np.linalg.eigh(sigma)
    floor = max(float(vals[-1]), 1e-300) * 1e-12
    vals = np.clip(vals, floor, None)
    inv_sqrt_vals = vals ** (-0.5)
    scaled = np.ascontiguousarray(vecs * inv_sqrt_vals)
    inv_sqrt = scaled @ np.ascontiguousarray(vecs.T)
    mean_diag = float(np.mean(inv_sqrt_vals**2))
    if mean_diag > 0:
        inv_sqrt = inv_sqrt / np.sqrt(mean_diag)
    return 0.5 * (inv_sqrt + inv_sqrt.T)


@dataclass
class OptimizerContext:
    sigma: np.ndarray
    scale: float
    metric: np.ndarray | None = None

    @classmethod
    def from_cov(cls, cov: pd.DataFrame, *, need_metric: bool) -> "OptimizerContext":
        sigma = np.asarray(cov, dtype=float)
        scale = 1.0 / float(np.mean(np.diag(sigma)))
        metric = precision_metric(sigma) if need_metric else None
        return cls(sigma=sigma, scale=scale, metric=metric)


@dataclass
class RiskTerm:
    coef: float = 1.0
    name: str = "risk"
    needs_metric: bool = False

    def value(self, w: np.ndarray, ctx: OptimizerContext) -> float:
        return self.coef * float(w @ ctx.sigma @ w) * ctx.scale

    def grad(self, w: np.ndarray, ctx: OptimizerContext) -> np.ndarray:
        return self.coef * 2.0 * ctx.sigma @ w * ctx.scale


@dataclass
class RidgeTerm:
    coef: float
    name: str = "ridge"
    needs_metric: bool = False

    def value(self, w: np.ndarray, ctx: OptimizerContext) -> float:
        return self.coef * float(w @ w) * ctx.scale

    def grad(self, w: np.ndarray, ctx: OptimizerContext) -> np.ndarray:
        return self.coef * 2.0 * w * ctx.scale


@dataclass
class DiversificationTerm:
    coef: float
    name: str = "diversification"
    needs_metric: bool = False

    def value(self, w: np.ndarray, ctx: OptimizerContext) -> float:
        return -self.coef * float(np.sum(np.log(np.maximum(w, _WEIGHT_FLOOR))))

    def grad(self, w: np.ndarray, ctx: OptimizerContext) -> np.ndarray:
        return -self.coef / np.maximum(w, _WEIGHT_FLOOR)


@dataclass
class PrecisionTerm:
    coef: float
    name: str = "precision"
    needs_metric: bool = True

    def _info_inv(self, w: np.ndarray, ctx: OptimizerContext):
        V = ctx.metric
        wf = np.maximum(w, _WEIGHT_FLOOR)
        scaled = np.ascontiguousarray(V * wf)
        M = scaled @ np.ascontiguousarray(V.T)
        M = M + 1e-12 * np.eye(M.shape[0])
        return (np.linalg.inv(M), V)

    def value(self, w: np.ndarray, ctx: OptimizerContext) -> float:
        (Minv, _) = self._info_inv(w, ctx)
        return self.coef * float(np.trace(Minv))

    def grad(self, w: np.ndarray, ctx: OptimizerContext) -> np.ndarray:
        (Minv, V) = self._info_inv(w, ctx)
        MiV = np.ascontiguousarray(Minv) @ np.ascontiguousarray(V)
        return -self.coef * np.sum(MiV**2, axis=0)


@dataclass
class RobustnessTerm:
    coef: float
    name: str = "robustness"
    needs_metric: bool = True

    def _info_matrix(self, w: np.ndarray, ctx: OptimizerContext) -> np.ndarray:
        V = ctx.metric
        scaled = np.ascontiguousarray(V * w)
        return scaled @ np.ascontiguousarray(V.T)

    def value(self, w: np.ndarray, ctx: OptimizerContext) -> float:
        lam_min = float(np.linalg.eigvalsh(self._info_matrix(w, ctx))[0])
        return -self.coef * lam_min

    def grad(self, w: np.ndarray, ctx: OptimizerContext) -> np.ndarray:
        M = self._info_matrix(w, ctx)
        (eigvals, eigvecs) = np.linalg.eigh(M)
        u = eigvecs[:, 0]
        Vu = ctx.metric.T @ u
        return -self.coef * Vu**2


@dataclass
class L1Term:
    coef: float
    eps: float = 1e-08
    name: str = "l1"
    needs_metric: bool = False

    def value(self, w: np.ndarray, ctx: OptimizerContext) -> float:
        return self.coef * float(np.sum(np.sqrt(w**2 + self.eps))) * ctx.scale

    def grad(self, w: np.ndarray, ctx: OptimizerContext) -> np.ndarray:
        return self.coef * (w / np.sqrt(w**2 + self.eps)) * ctx.scale


ObjectiveTerm = Union[
    RiskTerm, RidgeTerm, PrecisionTerm, DiversificationTerm, RobustnessTerm, L1Term
]


def default_terms(
    *,
    lambda2: float = 0.0,
    gamma_a: float = 0.0,
    gamma_d: float = 0.0,
    gamma_e: float = 0.0,
    l1: float = 0.0,
) -> list[ObjectiveTerm]:
    terms: list[ObjectiveTerm] = [RiskTerm()]
    if lambda2 > 0:
        terms.append(RidgeTerm(lambda2))
    if gamma_a > 0:
        terms.append(PrecisionTerm(gamma_a))
    if gamma_d > 0:
        terms.append(DiversificationTerm(gamma_d))
    if gamma_e > 0:
        terms.append(RobustnessTerm(gamma_e))
    if l1 > 0:
        terms.append(L1Term(l1))
    return terms


def _bounds(
    n: int, *, long_only: bool, min_weight: float | None, max_weight: float | None
) -> list[tuple[float, float]] | None:
    if not long_only and min_weight is None and (max_weight is None):
        return None
    lower = 0.0 if long_only else -np.inf
    if min_weight is not None:
        lower = max(lower, min_weight)
    upper = max_weight if max_weight is not None else np.inf
    if max_weight is not None and max_weight < 1.0 / n:
        raise ValueError(
            f"max_weight={max_weight} is infeasible for {n} assets; it must be at least 1/N = {1.0 / n:.4f}."
        )
    return [(lower, upper)] * n


def _group_constraints(groups: dict[str, tuple[list[int], float]] | None, n: int) -> list[dict]:
    if not groups:
        return []
    cons = []
    for name, (idx, cap) in groups.items():
        indicator = np.zeros(n)
        indicator[list(idx)] = 1.0
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w, a=indicator, c=cap: float(c - a @ w),
                "jac": lambda w, a=indicator: -a,
            }
        )
    return cons


def _group_target_constraints(
    group_targets: dict[str, tuple[list[int], float]] | None, n: int
) -> list[dict]:
    if not group_targets:
        return []
    cons = []
    for name, (idx, target) in group_targets.items():
        indicator = np.zeros(n)
        indicator[list(idx)] = 1.0
        cons.append(
            {
                "type": "eq",
                "fun": lambda w, a=indicator, t=target: float(a @ w - t),
                "jac": lambda w, a=indicator: a,
            }
        )
    return cons


@dataclass
class ResearchPortfolioResult:
    weights: pd.Series
    variance: float
    objective: float
    term_values: dict[str, float]
    iterations: int
    converged: bool

    @property
    def volatility(self) -> float:
        return float(np.sqrt(self.variance))


def solve_research_portfolio(
    cov: pd.DataFrame,
    *,
    terms: list[ObjectiveTerm] | None = None,
    lambda2: float = 0.0,
    gamma_a: float = 0.0,
    gamma_d: float = 0.0,
    gamma_e: float = 0.0,
    l1: float = 0.0,
    long_only: bool = False,
    min_weight: float | None = None,
    max_weight: float | None = None,
    groups: dict[str, tuple[list[int], float]] | None = None,
    group_targets: dict[str, tuple[list[int], float]] | None = None,
    x0: np.ndarray | pd.Series | None = None,
    max_iter: int = 500,
    ftol: float = 1e-12,
) -> ResearchPortfolioResult:
    names = cov.index
    n = len(names)
    if n < 2:
        raise ValueError("Need at least 2 assets for a portfolio.")
    if terms is None:
        terms = default_terms(
            lambda2=lambda2, gamma_a=gamma_a, gamma_d=gamma_d, gamma_e=gamma_e, l1=l1
        )
    need_metric = any((getattr(t, "needs_metric", False) for t in terms))
    ctx = OptimizerContext.from_cov(cov, need_metric=need_metric)

    def objective(w: np.ndarray) -> float:
        return float(sum((t.value(w, ctx) for t in terms)))

    def gradient(w: np.ndarray) -> np.ndarray:
        g = np.zeros(n)
        for t in terms:
            g = g + t.grad(w, ctx)
        return g

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)}
    ]
    constraints.extend(_group_constraints(groups, n))
    constraints.extend(_group_target_constraints(group_targets, n))
    bounds = _bounds(n, long_only=long_only, min_weight=min_weight, max_weight=max_weight)
    if x0 is None:
        x0_arr = np.ones(n) / n
    else:
        x0_arr = np.asarray(x0, dtype=float)
    result = minimize(
        objective,
        x0_arr,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": ftol, "maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    w = result.x
    weights = pd.Series(w, index=names, name="weight")
    term_values = {t.name: float(t.value(w, ctx)) for t in terms}
    return ResearchPortfolioResult(
        weights=weights,
        variance=float(w @ ctx.sigma @ w),
        objective=float(result.fun),
        term_values=term_values,
        iterations=int(result.nit),
        converged=bool(result.success),
    )


def research_gmv_weights(
    cov: pd.DataFrame,
    *,
    lambda2: float = 0.0,
    gamma_a: float = 0.0,
    gamma_d: float = 0.0,
    gamma_e: float = 0.0,
    l1: float = 0.0,
    long_only: bool = False,
    max_weight: float | None = None,
    groups: dict[str, tuple[list[int], float]] | None = None,
    group_targets: dict[str, tuple[list[int], float]] | None = None,
    x0: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    return solve_research_portfolio(
        cov,
        lambda2=lambda2,
        gamma_a=gamma_a,
        gamma_d=gamma_d,
        gamma_e=gamma_e,
        l1=l1,
        long_only=long_only,
        max_weight=max_weight,
        groups=groups,
        group_targets=group_targets,
        x0=x0,
    ).weights


def risk_budget_weights(
    cov: pd.DataFrame,
    *,
    budgets: np.ndarray | pd.Series | None = None,
    x0: np.ndarray | pd.Series | None = None,
    max_iter: int = 500,
) -> pd.Series:
    names = cov.index
    n = len(names)
    if n < 2:
        raise ValueError("Need at least 2 assets for a portfolio.")
    sigma = np.asarray(cov, dtype=float)
    scale = 1.0 / float(np.mean(np.diag(sigma)))
    sigma_s = sigma * scale
    if budgets is None:
        b = np.full(n, 1.0 / n)
    else:
        b = np.asarray(budgets, dtype=float)
        if np.any(b <= 0):
            raise ValueError("Risk budgets must be strictly positive.")
        b = b / b.sum()
    if x0 is None:
        y0 = 1.0 / np.sqrt(np.diag(sigma_s))
    else:
        y0 = np.abs(np.asarray(x0, dtype=float)) + 1e-06

    def objective(y: np.ndarray) -> float:
        return 0.5 * float(y @ sigma_s @ y) - float(b @ np.log(y))

    def gradient(y: np.ndarray) -> np.ndarray:
        return sigma_s @ y - b / y

    result = minimize(
        objective,
        y0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=[(1e-10, None)] * n,
        options={"maxiter": max_iter, "ftol": 1e-14},
    )
    if not result.success:
        raise RuntimeError(f"Risk-budgeting solve failed: {result.message}")
    w = result.x / result.x.sum()
    return pd.Series(w, index=names, name="weight")


def risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    sigma = np.asarray(cov, dtype=float)
    mrc = sigma @ w
    total = float(w @ mrc)
    rc = w * mrc / total if total > 0 else np.zeros_like(w)
    return pd.Series(rc, index=cov.index, name="risk_contribution")
