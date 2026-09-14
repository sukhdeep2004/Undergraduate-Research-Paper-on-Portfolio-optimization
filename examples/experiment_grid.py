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
from portfolio.covariance_estimators import ledoit_wolf_cov, sample_cov
from portfolio.evaluation import compare_backtests, rolling_backtest
from portfolio.simulation import covariance_error, factor_covariance, simulate_returns
from portfolio.strategies import default_strategies

TRADING_DAYS = 252


def _rewrap(cov: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(values, index=cov.index, columns=cov.columns)
    out.attrs.update(cov.attrs)
    return out


def make_true_cov(n_assets: int, n_factors: int, seed: int) -> pd.DataFrame:
    ann = factor_covariance(n_assets, n_factors=n_factors, factor_vol=0.18, idio_vol=None, rng=seed)
    daily = _rewrap(ann, ann.to_numpy() / TRADING_DAYS)
    daily.attrs["scale"] = "daily"
    return daily


def make_mean(cov: pd.DataFrame, seed: int, avg_sharpe: float = 0.4) -> np.ndarray:
    rng = np.random.default_rng(seed + 777)
    daily_vol = np.sqrt(np.diag(cov.to_numpy()))
    sharpe = rng.normal(avg_sharpe, 0.15, size=len(daily_vol))
    return sharpe * daily_vol / np.sqrt(TRADING_DAYS)


def score_panel(returns: pd.DataFrame, window: int, step: int, knobs: dict) -> pd.DataFrame:
    strategies = default_strategies(**knobs)
    results = {}
    for name, strat in strategies.items():
        try:
            results[name] = rolling_backtest(
                returns,
                strat.weight_fn,
                cov_estimator=strat.cov_estimator,
                window=window,
                step=step,
            )
        except (RuntimeError, ValueError):
            continue
    return compare_backtests(results)


def sweep_nt(args, knobs) -> "tuple[pd.DataFrame, pd.DataFrame]":
    (window, step) = (args.window, args.step)
    n_obs = window + step * args.rebalances
    (metric_rows, error_rows) = ([], [])
    for n_assets in args.n_assets:
        q = n_assets / window
        (boards, errs) = ([], {"sample": [], "ledoit_wolf": []})
        for seed in range(args.seeds):
            cov = make_true_cov(n_assets, args.n_factors, seed)
            mu = make_mean(cov, seed)
            rets = simulate_returns(cov, n_obs, mu=mu, distribution="normal", rng=seed + 1)
            boards.append(score_panel(rets, window, step, knobs))
            win = rets.iloc[:window]
            errs["sample"].append(covariance_error(sample_cov(win), cov)["relative_error"])
            errs["ledoit_wolf"].append(
                covariance_error(ledoit_wolf_cov(win), cov)["relative_error"]
            )
        avg_board = pd.concat(boards).groupby(level=0).mean(numeric_only=True)
        for strat, row in avg_board.iterrows():
            metric_rows.append(
                {
                    "sweep": "n_over_t",
                    "n_assets": n_assets,
                    "q": q,
                    "strategy": strat,
                    **row.to_dict(),
                }
            )
        for est, vals in errs.items():
            error_rows.append(
                {
                    "n_assets": n_assets,
                    "q": q,
                    "estimator": est,
                    "relative_error": float(np.mean(vals)),
                }
            )
    return (pd.DataFrame(metric_rows), pd.DataFrame(error_rows))


def sweep_noise(args, knobs) -> pd.DataFrame:
    (window, step) = (args.window, args.step)
    n_obs = window + step * args.rebalances
    rows = []
    for df in args.dfs:
        dist = "normal" if df is None else "t"
        boards = []
        for seed in range(args.seeds):
            cov = make_true_cov(args.noise_n_assets, args.n_factors, seed)
            mu = make_mean(cov, seed)
            rets = simulate_returns(
                cov, n_obs, mu=mu, distribution=dist, df=df or 8.0, rng=seed + 1
            )
            boards.append(score_panel(rets, window, step, knobs))
        avg_board = pd.concat(boards).groupby(level=0).mean(numeric_only=True)
        label = "normal" if df is None else f"t(df={df})"
        for strat, row in avg_board.iterrows():
            rows.append(
                {
                    "sweep": "noise",
                    "tail": label,
                    "df": df or np.inf,
                    "strategy": strat,
                    **row.to_dict(),
                }
            )
    return pd.DataFrame(rows)


_ORDER = ["equal_weight", "markowitz", "constrained", "ridge", "shrinkage", "proposed"]


def _strategy_style(name: str):
    palette = {
        "equal_weight": ("#7f7f7f", "--"),
        "markowitz": ("#d62728", "-"),
        "constrained": ("#ff7f0e", "-"),
        "ridge": ("#1f77b4", "-"),
        "shrinkage": ("#2ca02c", "-"),
        "proposed": ("#9467bd", "-"),
    }
    return palette.get(name, ("#333333", "-"))


def plot_estimator_error(err: pd.DataFrame, path: Path) -> None:
    (fig, ax) = plt.subplots(figsize=(8, 5))
    for est, color in (("sample", "#d62728"), ("ledoit_wolf", "#2ca02c")):
        sub = err[err["estimator"] == est].sort_values("q")
        ax.plot(
            sub["q"],
            sub["relative_error"],
            marker="o",
            color=color,
            label="sample covariance" if est == "sample" else "Ledoit-Wolf",
        )
    ax.set_xlabel("concentration ratio  q = N / T")
    ax.set_ylabel("relative error  $\\|\\hat\\Sigma-\\Sigma\\|_F / \\|\\Sigma\\|_F$")
    ax.set_title("Covariance estimation error vs. N/T (factor DGP, known truth)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_method_vs_nt(metrics: pd.DataFrame, path: Path) -> None:
    df = metrics[metrics["sweep"] == "n_over_t"]
    (fig, (ax_v, ax_t)) = plt.subplots(1, 2, figsize=(13, 5))
    for name in _ORDER:
        sub = df[df["strategy"] == name].sort_values("q")
        if sub.empty:
            continue
        (color, ls) = _strategy_style(name)
        ax_v.plot(sub["q"], sub["ann_volatility"], marker="o", color=color, ls=ls, label=name)
        ax_t.plot(sub["q"], sub["avg_turnover"], marker="o", color=color, ls=ls, label=name)
    ax_v.set_xlabel("q = N / T")
    ax_v.set_ylabel("OOS annualized volatility")
    ax_v.set_title("Out-of-sample risk vs. N/T (H4)")
    ax_t.set_xlabel("q = N / T")
    ax_t.set_ylabel("avg turnover")
    ax_t.set_title("Turnover vs. N/T")
    for ax in (ax_v, ax_t):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise_scoreboard(noise: pd.DataFrame, path: Path) -> None:
    tails = list(dict.fromkeys(noise["tail"]))
    strategies = [s for s in _ORDER if s in set(noise["strategy"])]
    x = np.arange(len(tails))
    width = 0.8 / max(len(strategies), 1)
    (fig, (ax_v, ax_t)) = plt.subplots(1, 2, figsize=(13, 5))
    for i, name in enumerate(strategies):
        (color, _) = _strategy_style(name)
        vols = [
            float(
                noise[(noise["tail"] == t) & (noise["strategy"] == name)]["ann_volatility"].mean()
            )
            for t in tails
        ]
        turn = [
            float(noise[(noise["tail"] == t) & (noise["strategy"] == name)]["avg_turnover"].mean())
            for t in tails
        ]
        ax_v.bar(x + i * width, vols, width, color=color, label=name)
        ax_t.bar(x + i * width, turn, width, color=color, label=name)
    for ax, title, ylab in (
        (ax_v, "OOS volatility vs. tail heaviness", "OOS annualized volatility"),
        (ax_t, "Turnover vs. tail heaviness", "avg turnover"),
    ):
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels(tails)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 7 simulation experiment grid.")
    p.add_argument(
        "--n-assets",
        nargs="+",
        type=int,
        default=[15, 40, 80, 110],
        help="universe sizes for the N/T sweep (T = window).",
    )
    p.add_argument("--n-factors", type=int, default=3)
    p.add_argument("--window", type=int, default=120, help="estimation window T (days).")
    p.add_argument("--step", type=int, default=21, help="rebalance stride (~monthly).")
    p.add_argument("--rebalances", type=int, default=10, help="out-of-sample rebalances.")
    p.add_argument("--seeds", type=int, default=3, help="Monte Carlo repetitions.")
    p.add_argument(
        "--dfs",
        nargs="+",
        type=lambda s: None if s.lower() in ("inf", "normal") else int(s),
        default=[3, 5, 10, None],
        help="Student-t dfs for the noise sweep (use 'normal' for Gaussian).",
    )
    p.add_argument(
        "--noise-n-assets", type=int, default=30, help="universe size for the noise sweep."
    )
    p.add_argument("--max-weight", type=float, default=0.2)
    p.add_argument("--lambda2", type=float, default=0.1)
    p.add_argument("--gamma-d", type=float, default=0.05)
    p.add_argument("--gamma-e", type=float, default=0.05)
    p.add_argument("--quick", action="store_true", help="smaller grid for a fast smoke run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_assets = [8, 20]
        args.dfs = [5, None]
        args.seeds = 2
        args.rebalances = 6
    knobs = dict(
        max_weight=args.max_weight, lambda2=args.lambda2, gamma_d=args.gamma_d, gamma_e=args.gamma_e
    )
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    print("=" * 72)
    print(
        f"Week 7 experiment grid  |  T={args.window}, step={args.step}, rebalances={args.rebalances}, seeds={args.seeds}"
    )
    print("=" * 72)
    print("\n[1/2] N/T sweep ...")
    (nt_metrics, est_err) = sweep_nt(args, knobs)
    print("\n[2/2] noise sweep ...")
    noise_metrics = sweep_noise(args, knobs)
    all_metrics = pd.concat([nt_metrics, noise_metrics], ignore_index=True)
    all_metrics.to_csv(figures_dir / "week7_grid_results.csv", index=False)
    est_err.to_csv(figures_dir / "week7_estimator_error.csv", index=False)
    plot_estimator_error(est_err, figures_dir / "week7_estimator_error.png")
    plot_method_vs_nt(nt_metrics, figures_dir / "week7_method_vs_nt.png")
    plot_noise_scoreboard(noise_metrics, figures_dir / "week7_noise_scoreboard.png")
    print("\nEstimator relative error vs. N/T:")
    pivot = est_err.pivot(index="q", columns="estimator", values="relative_error").round(4)
    print(pivot.to_string())
    print("\nOOS annualized volatility by strategy vs. q = N/T:")
    vol_pivot = (
        nt_metrics.pivot_table(index="q", columns="strategy", values="ann_volatility")
        .reindex(columns=[c for c in _ORDER if c in nt_metrics["strategy"].values])
        .round(4)
    )
    print(vol_pivot.to_string())
    print("\nAvg turnover by strategy vs. q = N/T:")
    to_pivot = (
        nt_metrics.pivot_table(index="q", columns="strategy", values="avg_turnover")
        .reindex(columns=[c for c in _ORDER if c in nt_metrics["strategy"].values])
        .round(4)
    )
    print(to_pivot.to_string())
    print(f"\nSaved tables + 3 figures to {figures_dir}")


if __name__ == "__main__":
    main()
