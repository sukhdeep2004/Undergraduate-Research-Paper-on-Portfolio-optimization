import argparse
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.constrained_optimization import plot_weights_comparison, weight_turnover
from portfolio.data import download_prices, prices_to_returns
from portfolio.ridge_regression import (
    britten_jones_weights,
    ridge_gmv_weights,
    ridge_path,
    ridge_portfolio_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge-regression portfolio experiments (Week 3).")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0],
        help="Ridge penalties for the tangency weight comparison (R^T R scale).",
    )
    parser.add_argument(
        "--window", type=int, default=252, help="Rolling window length (trading days)."
    )
    parser.add_argument(
        "--rebalance", type=int, default=21, help="Rebalance step (trading days; 21 ~ monthly)."
    )
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "JPM", "XOM"])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def fig_coefficient_paths(returns, lambdas_marked, figures_dir):
    R = returns.to_numpy()
    ones = np.ones(R.shape[0])
    grid = np.logspace(-4, 2, 60)
    paths = ridge_path(R, ones, grid)
    (fig, ax) = plt.subplots(figsize=(10, 6))
    for j, name in enumerate(returns.columns):
        ax.plot(grid, paths[:, j], label=name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    for lam in lambdas_marked:
        ax.axvline(lam, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("ridge penalty $\\lambda$ (log scale)")
    ax.set_ylabel("coefficient $\\theta_i$")
    ax.set_title("Ridge coefficient paths: regress 1 on excess returns")
    ax.legend(title="asset")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week3_coefficient_paths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return (path, paths, grid)


def fig_weights_comparison(returns, lambdas, figures_dir):
    weights = {"OLS (Britten-Jones)": britten_jones_weights(returns)}
    for lam in lambdas:
        weights[f"Ridge lambda={lam:g}"] = ridge_portfolio_weights(returns, lam)
    path = figures_dir / "week3_weights_comparison.png"
    plot_weights_comparison(
        weights, title="Tangency weights: OLS vs ridge shrinkage (full sample)", save_path=str(path)
    )
    return (path, weights)


def _rolling_weights(returns, weight_fn, window, step):
    (dates, rows) = ([], [])
    for end in range(window, len(returns) + 1, step):
        rows.append(weight_fn(returns.iloc[end - window : end]))
        dates.append(returns.index[end - 1])
    return pd.DataFrame(rows, index=dates)


def fig_weight_stability(returns, window, rebalance, figures_dir):
    base = float(np.mean(np.diag(np.cov(returns.to_numpy(), rowvar=False))))
    lam_grid = [0.0, 0.1 * base, 1.0 * base, 10.0 * base]
    weights_by_lam = {
        lam: _rolling_weights(returns, lambda r, l=lam: ridge_gmv_weights(r, l), window, rebalance)
        for lam in lam_grid
    }
    turnover_by_lam = {lam: float(weight_turnover(w).mean()) for (lam, w) in weights_by_lam.items()}
    (fig, (ax_time, ax_curve)) = plt.subplots(1, 2, figsize=(13, 5))
    to_plain = weight_turnover(weights_by_lam[lam_grid[0]])
    to_ridge = weight_turnover(weights_by_lam[lam_grid[2]])
    ax_time.plot(
        to_plain.index, to_plain.to_numpy(), label=f"GMV, no shrinkage (mean {to_plain.mean():.3f})"
    )
    ax_time.plot(
        to_ridge.index,
        to_ridge.to_numpy(),
        label=f"GMV ridge $\\lambda$={lam_grid[2]:.1e} (mean {to_ridge.mean():.3f})",
    )
    ax_time.set_ylabel("L1 turnover between rebalances")
    ax_time.set_title(f"Turnover over time (window={window}d, rebalance={rebalance}d)")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    ax_curve.plot(list(turnover_by_lam), list(turnover_by_lam.values()), marker="o")
    ax_curve.set_xscale("symlog", linthresh=max(lam_grid[1], 1e-12))
    ax_curve.set_xlabel("ridge penalty $\\lambda$ (covariance scale, symlog)")
    ax_curve.set_ylabel("average L1 turnover")
    ax_curve.set_title("Shrinkage reduces turnover")
    ax_curve.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week3_weight_stability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return (path, turnover_by_lam)


def fig_eigenvalue_spectrum(returns, figures_dir):
    cov = np.cov(returns.to_numpy(), rowvar=False)
    base = float(np.mean(np.diag(cov)))
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    idx = np.arange(1, len(eig) + 1)
    (fig, (ax_spec, ax_cond)) = plt.subplots(1, 2, figsize=(13, 5))
    ax_spec.plot(idx, eig, marker="o", label="$\\Sigma$ (sample)")
    for lam in (0.1 * base, 1.0 * base):
        ax_spec.plot(
            idx, eig + lam, marker="s", label=f"$\\Sigma+\\lambda I,\\ \\lambda$={lam:.1e}"
        )
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("eigenvalue index (largest to smallest)")
    ax_spec.set_ylabel("eigenvalue (log scale)")
    ax_spec.set_title("Ridge lifts the small eigenvalues")
    ax_spec.legend()
    ax_spec.grid(True, alpha=0.3)
    lam_cont = np.concatenate([[0.0], np.logspace(-7, -1, 80)])
    cond = (eig.max() + lam_cont) / (eig.min() + lam_cont)
    ax_cond.plot(lam_cont, cond)
    ax_cond.set_xscale("symlog", linthresh=1e-07)
    ax_cond.set_yscale("log")
    ax_cond.set_xlabel("ridge penalty $\\lambda$ (symlog)")
    ax_cond.set_ylabel("condition number $\\kappa(\\Sigma+\\lambda I)$")
    ax_cond.set_title(f"Conditioning improves with lambda (kappa0={cond[0]:.0f})")
    ax_cond.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week3_eigenvalue_spectrum.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return (path, eig)


def main() -> None:
    args = parse_args()
    print(f"Downloading {args.tickers} from Yahoo Finance (start={args.start}, end={args.end})...")
    prices = download_prices(args.tickers, start=args.start, end=args.end)
    returns = prices_to_returns(prices)
    print(
        f"{len(returns)} daily returns, {len(returns.columns)} assets ({returns.index.min().date()} to {returns.index.max().date()})"
    )
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    print("\n[1/4] Ridge coefficient paths ...")
    (p1, paths, grid) = fig_coefficient_paths(returns, args.lambdas, figures_dir)
    print(
        f"  ||theta|| {np.linalg.norm(paths[0]):.3f} -> {np.linalg.norm(paths[-1]):.3f} as lambda {grid[0]:.0e} -> {grid[-1]:.0f}   (saved {p1.name})"
    )
    print("\n[2/4] Tangency weights: OLS vs ridge ...")
    (p2, weights) = fig_weights_comparison(returns, args.lambdas, figures_dir)
    print(pd.DataFrame(weights).round(4).to_string())
    print(f"  saved {p2.name}")
    print("\n[3/4] Rolling GMV stability ...")
    (p3, turnover_by_lam) = fig_weight_stability(returns, args.window, args.rebalance, figures_dir)
    print("  average turnover by lambda (covariance scale):")
    for lam, turn in turnover_by_lam.items():
        print(f"    lambda={lam:.2e}: {turn:.4f}")
    print(f"  saved {p3.name}")
    print("\n[4/4] Covariance spectrum and conditioning ...")
    (p4, eig) = fig_eigenvalue_spectrum(returns, figures_dir)
    print(f"  kappa(Sigma) = {eig.max() / eig.min():.0f}   (saved {p4.name})")
    print(f"\nAll figures written to {figures_dir}")


if __name__ == "__main__":
    main()
