from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.constrained_optimization import solve_min_variance
from portfolio.covariance_estimators import cov_diagnostics
from portfolio.design_optimality import (
    gmv_closed_form,
    min_variance_simplex,
    optimal_design,
    polynomial_design_matrix,
    variance_function,
)


def run_design_part(figures_dir: Path) -> None:
    grid = np.round(np.arange(-1.0, 1.0001, 0.1), 2)
    V = polynomial_design_matrix(grid, degree=2)
    k = V.shape[0]
    print("=" * 70)
    print("Part A -- classical optimal design (quadratic regression on [-1, 1])")
    print("=" * 70)
    results = {}
    for crit, step in (("D", 1.0), ("A", 0.5), ("E", 0.5)):
        res = optimal_design(V, criterion=crit, step=step, support_x=grid)
        results[crit] = res
        print("\n" + res.summary())
    d_opt = results["D"]
    d_var = variance_function(V, d_opt.weights)
    print("\nKiefer-Wolfowitz check (D-optimal):")
    print(f"  max_x d(x, p*) = {d_var.max():.4f}   (should equal k = {k})")
    print(f"  number of parameters k = {k}")
    ref_support = {-1.0: 1 / 3, 0.0: 1 / 3, 1.0: 1 / 3}
    print("\n  D-optimal weights at textbook support points:")
    for x, target in ref_support.items():
        idx = int(np.argmin(np.abs(grid - x)))
        print(f"    x={x:+.0f}: p={d_opt.weights[idx]:.4f}  (theory {target:.4f})")
    (fig, (ax_w, ax_d)) = plt.subplots(1, 2, figsize=(13, 5))
    for crit in ("D", "A", "E"):
        ax_w.plot(grid, results[crit].weights, marker="o", markersize=4, label=f"{crit}-optimal")
    ax_w.set_xlabel("design point x")
    ax_w.set_ylabel("design weight p(x)")
    ax_w.set_title("Optimal design weights (quadratic regression)")
    ax_w.legend()
    ax_w.grid(True, alpha=0.3)
    ax_d.plot(grid, d_var, marker="o", markersize=4, color="C0", label="d(x, p*)")
    ax_d.axhline(k, color="k", ls="--", label=f"k = {k} (KW bound)")
    ax_d.set_xlabel("design point x")
    ax_d.set_ylabel("standardised variance d(x, p*)")
    ax_d.set_title("Equivalence theorem: $\\max_x d(x,p^*) = k$ at the D-optimum")
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week5_optimal_design.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {path.name}")


def _synthetic_covariance(seed: int = 0) -> pd.DataFrame:
    names = ["A", "B", "C", "D", "E"]
    vols = np.array([0.18, 0.22, 0.25, 0.2, 0.3])
    rng = np.random.default_rng(seed)
    base = 0.35
    corr = np.full((5, 5), base) + rng.uniform(-0.1, 0.1, size=(5, 5))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    cov = (cov + cov.T) / 2
    return pd.DataFrame(cov, index=names, columns=names)


def run_bridge_part(figures_dir: Path, returns: pd.DataFrame | None) -> None:
    print("\n" + "=" * 70)
    print("Part B -- the bridge: long-only GMV via the same simplex update")
    print("=" * 70)
    if returns is not None:
        cov = returns.cov()
        print(f"  covariance from {returns.shape[0]} daily returns, {returns.shape[1]} assets")
    else:
        cov = _synthetic_covariance()
        print("  covariance: synthetic 5-asset (positive correlations)")
    diag = cov_diagnostics(cov)
    print(
        f"  cov_diagnostics: kappa={diag['condition_number']:.2f}, effective_rank={diag['effective_rank']:.2f}"
    )
    Sigma = cov.to_numpy()
    names = list(cov.index)
    gmv = min_variance_simplex(Sigma)
    w_simplex = pd.Series(gmv.weights, index=names, name="simplex (design update)")
    w_week2 = solve_min_variance(cov, long_only=True)
    w_week2.name = "Week 2 long-only GMV"
    w_uncon = pd.Series(gmv_closed_form(Sigma), index=names, name="unconstrained GMV")
    table = pd.concat([w_simplex, w_week2, w_uncon], axis=1).round(4)
    print(
        f"\n  simplex solver: {gmv.iterations} iters, converged={gmv.converged}, vol={gmv.volatility:.4f}"
    )
    print("\n  Weights:")
    print(table.to_string())
    max_diff = float(np.max(np.abs(w_simplex.to_numpy() - w_week2.to_numpy())))
    print(
        f"\n  max |simplex - Week2 long-only| = {max_diff:.2e}  ({('MATCH' if max_diff < 0.001 else 'DIFFER')})"
    )
    if (w_uncon < -1e-06).any():
        print(
            "  note: unconstrained GMV uses shorts (negative weights) that the budget simplex cannot represent -- the linear/quadratic caveat."
        )
    (fig, ax) = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.27
    ax.bar(x - width, w_simplex, width, label="simplex (design update)")
    ax.bar(x, w_week2, width, label="Week 2 long-only GMV")
    ax.bar(x + width, w_uncon, width, label="unconstrained GMV")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("weight")
    ax.set_title("GMV bridge: the design simplex update reproduces long-only GMV")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week5_gmv_bridge.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  saved {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 5 optimal-design <-> portfolio bridge demo.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional: pull real returns instead of the synthetic cov.",
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    returns = None
    if args.tickers:
        from portfolio.data import download_prices, prices_to_returns

        print(f"Downloading {len(args.tickers)} tickers from Yahoo ...")
        prices = download_prices(args.tickers, start=args.start, end=args.end)
        returns = prices_to_returns(prices)
    run_design_part(figures_dir)
    run_bridge_part(figures_dir, returns)
    print(f"\nFigures written to {figures_dir}")


if __name__ == "__main__":
    main()
