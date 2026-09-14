from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

CRITERIA = ("D", "A", "E")


def polynomial_design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    powers = np.arange(degree + 1)
    return (x[:, None] ** powers[None, :]).T


def information_matrix(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    return V * p @ V.T


def variance_function(V: np.ndarray, p: np.ndarray) -> np.ndarray:
    M = information_matrix(V, p)
    Minv = np.linalg.inv(M)
    return np.einsum("ij,ik,kj->j", V, Minv, V)


def _criterion_derivative(V: np.ndarray, p: np.ndarray, criterion: str) -> np.ndarray:
    M = information_matrix(V, p)
    if criterion == "D":
        Minv = np.linalg.inv(M)
        return np.einsum("ij,ik,kj->j", V, Minv, V)
    if criterion == "A":
        Minv = np.linalg.inv(M)
        Minv2 = Minv @ Minv
        return np.einsum("ij,ik,kj->j", V, Minv2, V)
    if criterion == "E":
        (eigvals, eigvecs) = np.linalg.eigh(M)
        u = eigvecs[:, 0]
        return (u @ V) ** 2
    raise ValueError(f"Unknown criterion {criterion!r}; use one of {CRITERIA}.")


def criterion_value(M: np.ndarray, criterion: str) -> float:
    if criterion == "D":
        (sign, logabsdet) = np.linalg.slogdet(M)
        return float(logabsdet) if sign > 0 else float("-inf")
    if criterion == "A":
        return float(-np.trace(np.linalg.inv(M)))
    if criterion == "E":
        return float(np.linalg.eigvalsh(M)[0])
    raise ValueError(f"Unknown criterion {criterion!r}; use one of {CRITERIA}.")


@dataclass
class DesignResult:
    weights: np.ndarray
    support_x: np.ndarray
    support_p: np.ndarray
    criterion: str
    iterations: int
    converged: bool
    max_directional_derivative: float
    criterion_trace: list[float] = field(default_factory=list)
    n_params: int = 0

    def summary(self) -> str:
        pts = ", ".join(
            (f"x={x:+.3f}: p={p:.3f}" for (x, p) in zip(self.support_x, self.support_p))
        )
        return f"[{self.criterion}-optimal] {self.iterations} iters, converged={self.converged}, max F_j={self.max_directional_derivative:.2e}\n  support ({len(self.support_x)} pts): {pts}"


def optimal_design(
    V: np.ndarray,
    criterion: str = "D",
    *,
    step: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-08,
    support_x: np.ndarray | None = None,
    support_tol: float = 0.0001,
) -> DesignResult:
    if criterion not in CRITERIA:
        raise ValueError(f"Unknown criterion {criterion!r}; use one of {CRITERIA}.")
    (k, J) = V.shape
    if support_x is None:
        support_x = np.arange(J, dtype=float)
    support_x = np.asarray(support_x, dtype=float).ravel()
    p = np.full(J, 1.0 / J)
    trace: list[float] = []
    converged = False
    max_F = np.inf
    for it in range(1, max_iter + 1):
        d = np.maximum(_criterion_derivative(V, p, criterion), 0.0)
        weighted_mean = float(p @ d)
        F = d - weighted_mean
        max_F = float(np.max(F))
        trace.append(criterion_value(information_matrix(V, p), criterion))
        if max_F <= tol * max(1.0, abs(weighted_mean)):
            converged = True
            break
        if weighted_mean > 0:
            p = p * (d / weighted_mean) ** step
            total = p.sum()
            if total > 0:
                p /= total
    on = p >= support_tol
    return DesignResult(
        weights=p,
        support_x=support_x[on],
        support_p=p[on],
        criterion=criterion,
        iterations=it,
        converged=converged,
        max_directional_derivative=max_F,
        criterion_trace=trace,
        n_params=k,
    )


@dataclass
class GMVResult:
    weights: np.ndarray
    variance: float
    iterations: int
    converged: bool

    @property
    def volatility(self) -> float:
        return float(np.sqrt(self.variance))


def min_variance_simplex(
    Sigma: np.ndarray, *, step: float = 1.0, max_iter: int = 20000, tol: float = 1e-12
) -> GMVResult:
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    scale = float(np.max(np.abs(np.diag(Sigma)))) or 1.0
    p = np.full(n, 1.0 / n)
    converged = False
    for it in range(1, max_iter + 1):
        grad = Sigma @ p
        p_new = p * np.exp(-(step / scale) * grad)
        p_new /= p_new.sum()
        if np.max(np.abs(p_new - p)) <= tol:
            p = p_new
            converged = True
            break
        p = p_new
    variance = float(p @ Sigma @ p)
    return GMVResult(weights=p, variance=variance, iterations=it, converged=converged)


def gmv_closed_form(Sigma: np.ndarray) -> np.ndarray:
    Sigma = np.asarray(Sigma, dtype=float)
    ones = np.ones(Sigma.shape[0])
    z = np.linalg.solve(Sigma, ones)
    return z / z.sum()
