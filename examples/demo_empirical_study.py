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
from portfolio.data import DATA_CACHE_DIR, SECTOR_ETFS, load_prices, prices_to_returns
from portfolio.evaluation import apply_transaction_costs, compare_backtests, rolling_backtest
from portfolio.strategies import default_strategies

_ORDER = ["equal_weight", "markowitz", "constrained", "ridge", "shrinkage", "proposed"]
_STYLE = {
    "equal_weight": ("#7f7f7f", "--"),
    "markowitz": ("#d62728", "-"),
    "constrained": ("#ff7f0e", "-"),
    "ridge": ("#1f77b4", "-"),
    "shrinkage": ("#2ca02c", "-"),
    "proposed": ("#9467bd", "-"),
}


def load_universe_returns(start: str, end: "str | None", refresh: bool) -> pd.DataFrame:
    cached = DATA_CACHE_DIR / "universe_returns.csv"
    if cached.exists() and (not refresh):
        return pd.read_csv(cached, index_col=0, parse_dates=True)
    prices = load_prices(SECTOR_ETFS, start, end, refresh=refresh).dropna(how="any")
    return prices_to_returns(prices)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 8 empirical sector-ETF study.")
    p.add_argument("--window", type=int, default=252, help="estimation window (days).")
    p.add_argument("--step", type=int, default=21, help="rebalance stride (~monthly).")
    p.add_argument("--start", default="2018-07-01")
    p.add_argument("--end", default=None)
    p.add_argument("--max-weight", type=float, default=0.3)
    p.add_argument("--lambda2", type=float, default=0.1)
    p.add_argument("--gamma-d", type=float, default=0.05)
    p.add_argument("--gamma-e", type=float, default=0.05)
    p.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="one-way transaction cost per unit turnover (bps).",
    )
    p.add_argument(
        "--cost-grid",
        nargs="+",
        type=float,
        default=[0, 5, 10, 20, 50],
        help="cost levels (bps) for the net-Sharpe sweep.",
    )
    p.add_argument("--refresh", action="store_true", help="re-download instead of cache.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    returns = load_universe_returns(args.start, args.end, args.refresh)
    print("=" * 70)
    print(
        f"Empirical study | {returns.shape[1]} sector ETFs, {returns.shape[0]} days ({returns.index.min().date()} -> {returns.index.max().date()})"
    )
    print(f"window={args.window}, step={args.step}")
    print("=" * 70)
    knobs = dict(
        max_weight=args.max_weight, lambda2=args.lambda2, gamma_d=args.gamma_d, gamma_e=args.gamma_e
    )
    strategies = default_strategies(**knobs)
    results = {}
    for name, strat in strategies.items():
        try:
            results[name] = rolling_backtest(
                returns,
                strat.weight_fn,
                cov_estimator=strat.cov_estimator,
                window=args.window,
                step=args.step,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"  {name}: skipped ({exc})")
    board = compare_backtests(results)
    board = board.reindex([s for s in _ORDER if s in board.index])
    board.to_csv(figures_dir / "week8_scoreboard.csv")
    pd.set_option("display.width", 170, "display.max_columns", 20)
    print("\nScoreboard:")
    print(
        board[
            [
                "sharpe",
                "ann_return",
                "ann_volatility",
                "avg_turnover",
                "avg_effective_n",
                "avg_condition_number",
                "max_drawdown",
                "n_rebalances",
            ]
        ]
        .round(4)
        .to_string()
    )
    _plot_equity_curves(results, figures_dir / "week8_equity_curves.png")
    _plot_metrics(board, figures_dir / "week8_metrics.png")
    _run_cost_analysis(results, board, args, figures_dir)
    print(f"\nSaved scoreboard + net table + 3 figures to {figures_dir}")


def _run_cost_analysis(results, board, args, figures_dir) -> None:
    names = [s for s in _ORDER if s in results]
    rows = {}
    for name in names:
        (_, net_metrics) = apply_transaction_costs(results[name], cost_bps=args.cost_bps)
        rows[name] = {
            "gross_sharpe": board.loc[name, "sharpe"],
            "net_sharpe": net_metrics["sharpe"],
            "sharpe_drag": board.loc[name, "sharpe"] - net_metrics["sharpe"],
            "net_ann_return": net_metrics["ann_return"],
            "avg_turnover": net_metrics["avg_turnover"],
            "total_cost_drag": net_metrics["total_cost_drag"],
        }
    net_board = pd.DataFrame(rows).T.reindex(names)
    net_board.to_csv(figures_dir / "week8_net_scoreboard.csv")
    print(f"\nGross vs net Sharpe @ {args.cost_bps:.0f} bps:")
    print(net_board.round(4).to_string())
    sweep = {name: [] for name in names}
    for c in args.cost_grid:
        for name in names:
            (_, m) = apply_transaction_costs(results[name], cost_bps=c)
            sweep[name].append(m["sharpe"])
    (fig, ax) = plt.subplots(figsize=(9, 6))
    for name in names:
        (color, ls) = _STYLE[name]
        ax.plot(args.cost_grid, sweep[name], marker="o", color=color, ls=ls, label=name)
    ax.set_xlabel("transaction cost (bps per unit turnover)")
    ax.set_ylabel("net Sharpe (annualized)")
    ax.set_title("Week 8 — net Sharpe vs. transaction cost (sector ETFs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "week8_net_of_costs.png", dpi=150)
    plt.close(fig)


def _plot_equity_curves(results: dict, path: Path) -> None:
    (fig, ax) = plt.subplots(figsize=(11, 6))
    for name in _ORDER:
        if name not in results:
            continue
        r = results[name].portfolio_returns
        if r.empty:
            continue
        curve = (1.0 + r).cumprod()
        (color, ls) = _STYLE[name]
        ax.plot(curve.index, curve.values, color=color, ls=ls, lw=1.8, label=name)
    ax.set_ylabel("growth of $1 (out-of-sample)")
    ax.set_xlabel("date")
    ax.set_title("Week 8 — out-of-sample cumulative performance (sector ETFs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metrics(board: pd.DataFrame, path: Path) -> None:
    panels = [
        ("sharpe", "Sharpe (annualized)"),
        ("ann_volatility", "OOS volatility"),
        ("avg_turnover", "avg turnover"),
        ("avg_effective_n", "avg effective N"),
    ]
    (fig, axes) = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.ravel(), panels):
        names = list(board.index)
        colors = [_STYLE.get(n, ("#333", "-"))[0] for n in names]
        ax.bar(np.arange(len(names)), board[col].values, color=colors)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Week 8 — empirical metric panel (sector ETFs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
