from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from portfolio.covariance import portfolio_variance
from portfolio.minimum_variance import minimum_variance_weights


def historical_expected_returns(returns: pd.DataFrame, *, periods_per_year: int = 252) -> pd.Series:
    mu = returns.mean() * periods_per_year
    mu.name = "expected_return"
    return mu


def portfolio_return(weights: np.ndarray | pd.Series, expected_returns: pd.Series) -> float:
    w = np.asarray(weights, dtype=float)
    mu = expected_returns.to_numpy()
    return float(w @ mu)


def weights_for_target_return(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    target_return: float,
    initial_weights: np.ndarray | pd.Series | None = None,
) -> pd.Series:
    sigma = cov.to_numpy()
    mu = expected_returns.reindex(cov.index).to_numpy()
    n_assets = len(mu)
    if initial_weights is None:
        x0 = np.ones(n_assets) / n_assets
    else:
        x0 = np.asarray(initial_weights, dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ sigma @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * sigma @ weights

    constraints = [
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        {"type": "eq", "fun": lambda weights: float(weights @ mu - target_return)},
    ]
    result = minimize(objective, x0, method="SLSQP", jac=gradient, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"No solution for target return {target_return:.4f}: {result.message}")
    return pd.Series(result.x, index=cov.index, name="weight")


def efficient_frontier(
    cov: pd.DataFrame, expected_returns: pd.Series, *, n_points: int = 40
) -> pd.DataFrame:
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    w_gmv = minimum_variance_weights(cov)
    ret_min = portfolio_return(w_gmv, expected_returns)
    ret_max = float(expected_returns.max())
    targets = np.linspace(ret_min, ret_max, n_points)
    rows = []
    warm_start = minimum_variance_weights(cov).to_numpy()
    for target in targets:
        weights = weights_for_target_return(
            cov, expected_returns, target, initial_weights=warm_start
        )
        warm_start = weights.to_numpy()
        ret = portfolio_return(weights, expected_returns)
        vol = np.sqrt(portfolio_variance(weights, cov))
        rows.append({"target_return": target, "expected_return": ret, "volatility": vol})
    return pd.DataFrame(rows)


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    *,
    expected_returns: pd.Series | None = None,
    cov: pd.DataFrame | None = None,
    gmv_weights: pd.Series | None = None,
    title: str = "Efficient frontier",
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    (fig, ax) = plt.subplots(figsize=(9, 6))
    ax.plot(
        frontier["volatility"],
        frontier["expected_return"],
        "b-",
        linewidth=2,
        label="Efficient frontier",
    )
    if expected_returns is not None and cov is not None:
        for ticker in expected_returns.index:
            w = pd.Series(0.0, index=expected_returns.index)
            w[ticker] = 1.0
            ret = portfolio_return(w, expected_returns)
            vol = np.sqrt(portfolio_variance(w, cov))
            ax.scatter(vol, ret, s=60, zorder=3)
            ax.annotate(ticker, (vol, ret), textcoords="offset points", xytext=(4, 4), fontsize=9)
    if gmv_weights is not None and expected_returns is not None and (cov is not None):
        ret = portfolio_return(gmv_weights, expected_returns)
        vol = np.sqrt(portfolio_variance(gmv_weights, cov))
        ax.scatter(vol, ret, c="red", s=100, zorder=4, label="Min-variance portfolio")
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
