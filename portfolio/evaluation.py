from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from portfolio.constrained_optimization import weight_turnover
from portfolio.covariance_estimators import cov_diagnostics, sample_cov

CovEstimator = Callable[[pd.DataFrame], pd.DataFrame]
WeightFn = Callable[[pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    portfolio_returns: pd.Series
    metrics: dict
    diagnostics: pd.DataFrame

    def summary_row(self) -> pd.Series:
        return pd.Series(self.metrics)


def effective_n(weights: pd.Series) -> float:
    w = weights.to_numpy(dtype=float)
    hhi = float(np.sum(w**2))
    return 1.0 / hhi if hhi > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    curve = (1.0 + returns).cumprod()
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    return float(drawdown.min())


def annualized_stats(returns: pd.Series, periods_per_year: int = 252) -> dict:
    if returns.empty or returns.std(ddof=1) == 0:
        return {"ann_return": float("nan"), "ann_volatility": float("nan"), "sharpe": float("nan")}
    mu = float(returns.mean()) * periods_per_year
    vol = float(returns.std(ddof=1)) * np.sqrt(periods_per_year)
    return {
        "ann_return": mu,
        "ann_volatility": vol,
        "sharpe": mu / vol if vol > 0 else float("nan"),
    }


def rolling_backtest(
    returns: pd.DataFrame,
    weight_fn: WeightFn,
    *,
    cov_estimator: CovEstimator | None = None,
    window: int = 252,
    step: int = 21,
    annualize_cov: bool = True,
    periods_per_year: int = 252,
    warm_start: bool = True,
) -> BacktestResult:
    if cov_estimator is None:
        cov_estimator = sample_cov
    n_obs = len(returns)
    if window >= n_obs:
        raise ValueError(f"window={window} needs to be < {n_obs} observations.")
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    diag_rows: dict[pd.Timestamp, dict] = {}
    oos_chunks: list[pd.Series] = []
    warm: np.ndarray | None = None
    for end in range(window, n_obs, step):
        win = returns.iloc[end - window : end]
        cov = (
            cov_estimator(win, annualize=annualize_cov, periods_per_year=periods_per_year)
            if _accepts_annualize(cov_estimator)
            else cov_estimator(win)
        )
        rebalance_date = returns.index[end - 1]
        try:
            weights = _call_weight_fn(weight_fn, cov, warm if warm_start else None)
        except (RuntimeError, ValueError):
            continue
        weights = weights.reindex(returns.columns).fillna(0.0)
        weight_rows[rebalance_date] = weights
        warm = weights.to_numpy()
        diag = cov_diagnostics(cov)
        diag_rows[rebalance_date] = {
            "condition_number": diag["condition_number"],
            "effective_rank": diag["effective_rank"],
            "effective_n": effective_n(weights),
        }
        oos = returns.iloc[end : min(end + step, n_obs)]
        if not oos.empty:
            oos_chunks.append(oos @ weights)
    weights_df = pd.DataFrame(weight_rows).T
    diagnostics = pd.DataFrame(diag_rows).T
    port_returns = pd.concat(oos_chunks).sort_index() if oos_chunks else pd.Series(dtype=float)
    turnover = weight_turnover(weights_df) if len(weights_df) > 1 else pd.Series(dtype=float)
    stats = annualized_stats(port_returns, periods_per_year)
    metrics = {
        "sharpe": stats["sharpe"],
        "ann_return": stats["ann_return"],
        "ann_volatility": stats["ann_volatility"],
        "avg_turnover": float(turnover.mean()) if not turnover.empty else float("nan"),
        "avg_effective_n": (
            float(diagnostics["effective_n"].mean()) if not diagnostics.empty else float("nan")
        ),
        "avg_condition_number": (
            float(diagnostics["condition_number"].mean()) if not diagnostics.empty else float("nan")
        ),
        "max_drawdown": max_drawdown(port_returns),
        "n_rebalances": int(len(weights_df)),
    }
    return BacktestResult(
        weights=weights_df, portfolio_returns=port_returns, metrics=metrics, diagnostics=diagnostics
    )


def compare_backtests(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({label: r.metrics for (label, r) in results.items()}).T


def apply_transaction_costs(
    result: BacktestResult, cost_bps: float = 10.0
) -> tuple[pd.Series, dict]:
    net = result.portfolio_returns.copy()
    w = result.weights
    if net.empty or w.empty:
        return (net, dict(result.metrics))
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = float(w.iloc[0].abs().sum())
    cost = cost_bps * 0.0001 * turnover
    idx = net.index
    for rebalance_date, c in cost.items():
        pos = idx.searchsorted(rebalance_date, side="right")
        if pos < len(idx):
            net.iloc[pos] -= float(c)
    stats = annualized_stats(net)
    metrics = {
        "sharpe": stats["sharpe"],
        "ann_return": stats["ann_return"],
        "ann_volatility": stats["ann_volatility"],
        "avg_turnover": result.metrics.get("avg_turnover", float("nan")),
        "avg_effective_n": result.metrics.get("avg_effective_n", float("nan")),
        "max_drawdown": max_drawdown(net),
        "cost_bps": float(cost_bps),
        "total_cost_drag": float(cost.sum()),
        "n_rebalances": result.metrics.get("n_rebalances", len(w)),
    }
    return (net, metrics)


def _accepts_annualize(fn: Callable) -> bool:
    import inspect

    try:
        return "annualize" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _call_weight_fn(weight_fn: WeightFn, cov: pd.DataFrame, warm: np.ndarray | None) -> pd.Series:
    import inspect

    if warm is not None:
        try:
            if "x0" in inspect.signature(weight_fn).parameters:
                return weight_fn(cov, x0=warm)
        except (TypeError, ValueError):
            pass
    return weight_fn(cov)
