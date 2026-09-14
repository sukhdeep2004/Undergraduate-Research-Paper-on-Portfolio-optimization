from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from portfolio.covariance import sample_covariance


def _bounds(
    n_assets: int, *, long_only: bool, max_weight: float | None
) -> list[tuple[float, float]] | None:
    if not long_only and max_weight is None:
        return None
    lower = 0.0 if long_only else -np.inf
    upper = max_weight if max_weight is not None else np.inf
    if max_weight is not None and max_weight < 1.0 / n_assets:
        raise ValueError(
            f"max_weight={max_weight} is infeasible for {n_assets} assets; it must be at least 1/N = {1.0 / n_assets:.4f}."
        )
    return [(lower, upper)] * n_assets


def solve_min_variance(
    cov: pd.DataFrame,
    *,
    mu: pd.Series | None = None,
    target_return: float | None = None,
    long_only: bool = False,
    max_weight: float | None = None,
    x0: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    asset_names = cov.index
    n_assets = len(asset_names)
    if n_assets < 2:
        raise ValueError("Need at least 2 assets for a portfolio.")
    sigma = cov.to_numpy()
    if target_return is not None and mu is None:
        raise ValueError("mu is required when target_return is set.")
    scale = 1.0 / float(np.mean(np.diag(sigma)))

    def objective(weights: np.ndarray) -> float:
        return float(weights @ sigma @ weights) * scale

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * sigma @ weights * scale

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)}
    ]
    if target_return is not None:
        mu_vec = mu.reindex(asset_names).to_numpy()
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w, m=mu_vec: float(w @ m - target_return),
                "jac": lambda w, m=mu_vec: m,
            }
        )
    bounds = _bounds(n_assets, long_only=long_only, max_weight=max_weight)
    if x0 is None:
        x0_arr = np.ones(n_assets) / n_assets
    else:
        x0_arr = np.asarray(x0, dtype=float)
    result = minimize(
        objective,
        x0_arr,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    return pd.Series(result.x, index=asset_names, name="weight")


def constrained_min_variance_portfolio(
    returns: pd.DataFrame,
    *,
    long_only: bool = False,
    max_weight: float | None = None,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    cov = sample_covariance(returns, annualize=annualize, periods_per_year=periods_per_year)
    return solve_min_variance(cov, long_only=long_only, max_weight=max_weight)


def long_only_min_variance(
    cov: pd.DataFrame, *, max_weight: float | None = None, x0: np.ndarray | pd.Series | None = None
) -> pd.Series:
    return solve_min_variance(cov, long_only=True, max_weight=max_weight, x0=x0)


def max_weight_min_variance(
    cov: pd.DataFrame,
    max_weight: float,
    *,
    long_only: bool = True,
    x0: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    return solve_min_variance(cov, long_only=long_only, max_weight=max_weight, x0=x0)


def max_return_target_volatility(
    cov: pd.DataFrame,
    mu: pd.Series,
    sigma_target: float,
    *,
    long_only: bool = False,
    max_weight: float | None = None,
    x0: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    asset_names = cov.index
    n_assets = len(asset_names)
    if n_assets < 2:
        raise ValueError("Need at least 2 assets for a portfolio.")
    if sigma_target <= 0:
        raise ValueError("sigma_target must be positive.")
    sigma = cov.to_numpy()
    mu_vec = mu.reindex(asset_names).to_numpy()
    target_var = float(sigma_target) ** 2
    mean_abs_mu = float(np.mean(np.abs(mu_vec)))
    ret_scale = 1.0 / mean_abs_mu if mean_abs_mu > 0 else 1.0
    var_scale = 1.0 / float(np.mean(np.diag(sigma)))

    def neg_return(weights: np.ndarray) -> float:
        return -float(weights @ mu_vec) * ret_scale

    def neg_return_grad(weights: np.ndarray) -> np.ndarray:
        return -mu_vec * ret_scale

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones_like(w)},
        {
            "type": "ineq",
            "fun": lambda w: (target_var - float(w @ sigma @ w)) * var_scale,
            "jac": lambda w: -2.0 * sigma @ w * var_scale,
        },
    ]
    bounds = _bounds(n_assets, long_only=long_only, max_weight=max_weight)
    if x0 is None:
        x0_arr = np.ones(n_assets) / n_assets
    else:
        x0_arr = np.asarray(x0, dtype=float)
    result = minimize(
        neg_return,
        x0_arr,
        method="SLSQP",
        jac=neg_return_grad,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    weights = result.x
    achieved_var = float(weights @ sigma @ weights)
    if achieved_var > target_var * (1.0 + 0.0001) + 1e-12:
        raise RuntimeError(
            f"sigma_target={sigma_target:.4f} is below the minimum achievable volatility ({achieved_var ** 0.5:.4f}); no feasible portfolio."
        )
    return pd.Series(weights, index=asset_names, name="weight")


def _max_constrained_return(
    cov: pd.DataFrame, mu: pd.Series, *, long_only: bool, max_weight: float | None
) -> float:
    asset_names = cov.index
    mu_vec = mu.reindex(asset_names).to_numpy()
    if max_weight is None:
        return float(mu_vec.max())
    n_assets = len(asset_names)
    bounds = _bounds(n_assets, long_only=long_only, max_weight=max_weight)
    result = minimize(
        lambda w: -float(w @ mu_vec),
        np.ones(n_assets) / n_assets,
        method="SLSQP",
        jac=lambda w: -mu_vec,
        bounds=bounds,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        ],
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success:
        return float(mu_vec.max())
    return float(result.x @ mu_vec)


def constrained_efficient_frontier(
    cov: pd.DataFrame,
    mu: pd.Series,
    *,
    long_only: bool = False,
    max_weight: float | None = None,
    n_points: int = 40,
) -> pd.DataFrame:
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    asset_names = cov.index
    mu_vec = mu.reindex(asset_names).to_numpy()
    sigma = cov.to_numpy()
    w_gmv = solve_min_variance(cov, long_only=long_only, max_weight=max_weight)
    ret_min = float(w_gmv.to_numpy() @ mu_vec)
    ret_max = _max_constrained_return(cov, mu, long_only=long_only, max_weight=max_weight)
    if ret_max <= ret_min:
        ret_max = ret_min + abs(ret_min) * 0.001 + 1e-06
    targets = np.linspace(ret_min, ret_max, n_points)
    warm = w_gmv.to_numpy()
    rows = []
    for target in targets:
        try:
            weights = solve_min_variance(
                cov,
                mu=mu,
                target_return=float(target),
                long_only=long_only,
                max_weight=max_weight,
                x0=warm,
            )
        except RuntimeError:
            continue
        warm = weights.to_numpy()
        rows.append(
            {
                "target_return": float(target),
                "expected_return": float(weights.to_numpy() @ mu_vec),
                "volatility": float(np.sqrt(weights.to_numpy() @ sigma @ weights.to_numpy())),
            }
        )
    return pd.DataFrame(rows)


def plot_frontier_comparison(
    frontiers: dict[str, pd.DataFrame],
    *,
    expected_returns: pd.Series | None = None,
    cov: pd.DataFrame | None = None,
    title: str = "Efficient frontier: unconstrained vs constrained",
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    (fig, ax) = plt.subplots(figsize=(9, 6))
    for label, frontier in frontiers.items():
        if frontier.empty:
            continue
        line = ax.plot(
            frontier["volatility"],
            frontier["expected_return"],
            linewidth=2,
            marker=".",
            label=label,
        )[0]
        idx_min = frontier["volatility"].idxmin()
        ax.scatter(
            frontier.loc[idx_min, "volatility"],
            frontier.loc[idx_min, "expected_return"],
            color=line.get_color(),
            s=90,
            zorder=4,
        )
    if expected_returns is not None and cov is not None:
        sigma = cov.to_numpy()
        for ticker in expected_returns.index:
            w = np.zeros(len(expected_returns))
            w[expected_returns.index.get_loc(ticker)] = 1.0
            ret = float(w @ expected_returns.to_numpy())
            vol = float(np.sqrt(w @ sigma @ w))
            ax.scatter(vol, ret, c="gray", s=40, zorder=3)
            ax.annotate(ticker, (vol, ret), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected return (annualized)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def rolling_constrained_weights(
    returns: pd.DataFrame,
    *,
    long_only: bool = False,
    max_weight: float | None = None,
    window: int = 252,
    step: int = 21,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    n_obs = len(returns)
    if window < 2:
        raise ValueError("window must be at least 2.")
    if window > n_obs:
        raise ValueError(f"window={window} exceeds available {n_obs} observations.")
    if step < 1:
        raise ValueError("step must be at least 1.")
    rows: dict[pd.Timestamp, pd.Series] = {}
    warm: np.ndarray | None = None
    for end in range(window, n_obs + 1, step):
        win = returns.iloc[end - window : end]
        cov = sample_covariance(win, annualize=annualize, periods_per_year=periods_per_year)
        try:
            weights = solve_min_variance(cov, long_only=long_only, max_weight=max_weight, x0=warm)
        except RuntimeError:
            continue
        warm = weights.to_numpy()
        rows[returns.index[end - 1]] = weights
    return pd.DataFrame(rows).T


def weight_turnover(weights_over_time: pd.DataFrame) -> pd.Series:
    return weights_over_time.diff().abs().sum(axis=1).iloc[1:]


def plot_weights_comparison(
    weights: dict[str, pd.Series],
    *,
    title: str = "Portfolio weights: unconstrained vs constrained",
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(weights.keys())
    assets = list(next(iter(weights.values())).index)
    x = np.arange(len(assets))
    bar_width = 0.8 / max(len(labels), 1)
    (fig, ax) = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        values = weights[label].reindex(assets).to_numpy()
        ax.bar(x + i * bar_width, values, bar_width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x + bar_width * (len(labels) - 1) / 2)
    ax.set_xticklabels(assets)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
