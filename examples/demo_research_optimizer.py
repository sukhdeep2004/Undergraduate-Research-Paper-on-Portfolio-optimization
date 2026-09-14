from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.constrained_optimization import solve_min_variance
from portfolio.covariance_estimators import cov_diagnostics, sample_cov
from portfolio.evaluation import compare_backtests, effective_n, rolling_backtest
from portfolio.research_optimizer import research_gmv_weights, solve_research_portfolio
from portfolio.ridge_regression import ridge_gmv_weights


def synthetic_returns(
    n_assets: int = 8, n_days: int = 1000, n_factors: int = 2, seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    B = rng.normal(0.0, 1.0, size=(n_assets, n_factors)) * np.array([0.9, 0.5])
    factor_vol = np.array([0.011, 0.007])
    factors = rng.normal(0.0, 1.0, size=(n_days, n_factors)) * factor_vol
    idio = rng.normal(0.0, 1.0, size=(n_days, n_assets)) * rng.uniform(0.004, 0.01, n_assets)
    drift = rng.uniform(0.0002, 0.0007, n_assets)
    R = factors @ B.T + idio + drift
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    names = [f"A{i + 1}" for i in range(n_assets)]
    return pd.DataFrame(R, index=dates, columns=names)


def run_reductions(cov: pd.DataFrame) -> None:
    print("=" * 70)
    print("Part A -- reductions (must match Weeks 2-3 to tolerance)")
    print("=" * 70)
    w_research = research_gmv_weights(cov, lambda2=0.0, gamma_d=0.0, gamma_e=0.0)
    w_week2 = solve_min_variance(cov)
    gmv_diff = float(np.max(np.abs(w_research.to_numpy() - w_week2.to_numpy())))
    print(
        f"\n  GMV      : max|research - Week2 GMV|      = {gmv_diff:.2e}  ({('MATCH' if gmv_diff < 1e-05 else 'DIFFER')})"
    )
    lam = 5.0 * float(np.mean(np.diag(cov.to_numpy())))
    w_research_ridge = research_gmv_weights(cov, lambda2=lam)
    Sigma = cov.to_numpy()
    n = Sigma.shape[0]
    w_closed = np.linalg.solve(Sigma + lam * np.eye(n), np.ones(n))
    w_closed = pd.Series(w_closed / w_closed.sum(), index=cov.index)
    ridge_diff = float(np.max(np.abs(w_research_ridge.to_numpy() - w_closed.to_numpy())))
    print(
        f"  Ridge    : max|research - (Sigma+lam I)^-1 1| = {ridge_diff:.2e}  ({('MATCH' if ridge_diff < 1e-05 else 'DIFFER')})  [lambda={lam:.2e}]"
    )


def _info_matrix(Sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    from portfolio.research_optimizer import precision_metric

    V = precision_metric(Sigma)
    return V * w @ V.T


def run_sweeps(cov: pd.DataFrame, figures_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("Part B -- A/D/E design-penalty sweeps (long-only)")
    print("=" * 70)
    Sigma = cov.to_numpy()
    gamma_a_grid = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    gamma_d_grid = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0])
    gamma_e_grid = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    (a_trace, a_vol) = ([], [])
    for g in gamma_a_grid:
        w = research_gmv_weights(cov, gamma_a=float(g), long_only=True).to_numpy()
        a_trace.append(float(np.trace(np.linalg.inv(_info_matrix(Sigma, w)))))
        a_vol.append(float(np.sqrt(w @ Sigma @ w)))
    (d_effn, d_vol) = ([], [])
    for g in gamma_d_grid:
        w = research_gmv_weights(cov, gamma_d=float(g), long_only=True)
        d_effn.append(effective_n(w))
        d_vol.append(float(np.sqrt(w.to_numpy() @ Sigma @ w.to_numpy())))
    (e_kappa, e_vol) = ([], [])
    for g in gamma_e_grid:
        w = research_gmv_weights(cov, gamma_e=float(g), long_only=True).to_numpy()
        e_vol.append(float(np.sqrt(w @ Sigma @ w)))
        eig = np.linalg.eigvalsh(_info_matrix(Sigma, w))
        e_kappa.append(float(eig[-1] / max(eig[0], 1e-12)))
    print("\n  A-sweep (average variance): tr(M^-1) should fall with gamma_a")
    for g, t, v in zip(gamma_a_grid, a_trace, a_vol):
        print(f"    gamma_a={g:>4}: tr(M^-1)={t:8.3f}, vol={v:.4f}")
    print("\n  D-sweep (diversification): effective N should rise with gamma_d")
    for g, en, v in zip(gamma_d_grid, d_effn, d_vol):
        print(f"    gamma_d={g:>4}: effective_N={en:5.2f}, vol={v:.4f}")
    print("\n  E-sweep (robustness): kappa(M) should fall with gamma_e")
    for g, k, v in zip(gamma_e_grid, e_kappa, e_vol):
        print(f"    gamma_e={g:>4}: kappa(M)={k:8.2f}, vol={v:.4f}")
    (fig, (axa, axd, axe)) = plt.subplots(1, 3, figsize=(16, 4.8))
    axa.plot(gamma_a_grid, a_trace, marker="o", color="C4")
    axa.set_xlabel("A-penalty $\\gamma_A$")
    axa.set_ylabel("average variance $\\mathrm{tr}\\,M(w)^{-1}$")
    axa.set_title("A-penalty lowers average variance")
    axa.grid(True, alpha=0.3)
    axd.plot(gamma_d_grid, d_effn, marker="o", color="C0")
    axd.set_xlabel("D-penalty $\\gamma_D$")
    axd.set_ylabel("effective N (1 / HHI)")
    axd.set_title("D-penalty raises diversification (H2)")
    axd.grid(True, alpha=0.3)
    axe.plot(gamma_e_grid, e_kappa, marker="o", color="C2")
    axe.set_yscale("log")
    axe.set_xlabel("E-penalty $\\gamma_E$")
    axe.set_ylabel("condition number $\\kappa(M(w))$")
    axe.set_title("E-penalty improves conditioning (H1)")
    axe.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week6_design_penalty_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {path.name}")


def run_scoreboard(returns: pd.DataFrame, figures_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("Part C -- unified backtest scoreboard (rolling, long-only)")
    print("=" * 70)
    base = float(np.mean(np.diag(np.cov(returns.to_numpy(), rowvar=False)))) * 252
    lam = 5.0 * base

    def equal_weight(cov, x0=None):
        n = cov.shape[0]
        return pd.Series(np.ones(n) / n, index=cov.index)

    strategies = {
        "Equal weight": equal_weight,
        "GMV (long-only)": lambda cov, x0=None: solve_min_variance(cov, long_only=True, x0=x0),
        "Ridge GMV": lambda cov, x0=None: research_gmv_weights(
            cov, lambda2=lam, long_only=True, x0=x0
        ),
        "A-penalized": lambda cov, x0=None: research_gmv_weights(
            cov, gamma_a=0.25, long_only=True, x0=x0
        ),
        "D-penalized": lambda cov, x0=None: research_gmv_weights(
            cov, gamma_d=0.1, long_only=True, x0=x0
        ),
        "E-penalized": lambda cov, x0=None: research_gmv_weights(
            cov, gamma_e=0.25, long_only=True, x0=x0
        ),
    }
    results = {
        name: rolling_backtest(returns, fn, cov_estimator=sample_cov, window=252, step=21)
        for (name, fn) in strategies.items()
    }
    board = compare_backtests(results)
    cols = ["sharpe", "ann_volatility", "avg_turnover", "avg_effective_n", "max_drawdown"]
    print("\n" + board[cols].round(4).to_string())
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 4.5))
    board["avg_effective_n"].plot.bar(ax=axes[0], color="C0")
    axes[0].set_title("Diversification (avg effective N)")
    axes[0].set_ylabel("effective N")
    board["avg_turnover"].plot.bar(ax=axes[1], color="C1")
    axes[1].set_title("Stability (avg turnover, lower=better)")
    board["ann_volatility"].plot.bar(ax=axes[2], color="C3")
    axes[2].set_title("OOS volatility (annualized)")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = figures_dir / "week6_backtest_scoreboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 6 research optimizer demo.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional: real returns instead of synthetic factor data.",
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    if args.tickers:
        from portfolio.data import download_prices, prices_to_returns

        print(f"Downloading {len(args.tickers)} tickers from Yahoo ...")
        prices = download_prices(args.tickers, start=args.start, end=args.end)
        returns = prices_to_returns(prices)
    else:
        returns = synthetic_returns()
        print(f"Synthetic factor returns: {returns.shape[0]} days, {returns.shape[1]} assets")
    cov = sample_cov(returns, annualize=True)
    diag = cov_diagnostics(cov)
    print(
        f"Covariance: kappa={diag['condition_number']:.2f}, effective_rank={diag['effective_rank']:.2f}\n"
    )
    run_reductions(cov)
    run_sweeps(cov, figures_dir)
    run_scoreboard(returns, figures_dir)
    print(f"\nFigures written to {figures_dir}")


if __name__ == "__main__":
    main()
