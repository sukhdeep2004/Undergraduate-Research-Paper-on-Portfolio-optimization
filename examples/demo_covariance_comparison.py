import argparse
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.constrained_optimization import solve_min_variance, weight_turnover
from portfolio.covariance_estimators import ewma_cov, ledoit_wolf_cov, oas_cov, sample_cov
from portfolio.data import download_prices, prices_to_returns

DEFAULT_SMALL = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
DEFAULT_WIDE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "ADBE",
    "JPM",
    "BAC",
    "GS",
    "WFC",
    "JNJ",
    "PFE",
    "UNH",
    "MRK",
    "AMZN",
    "HD",
    "MCD",
    "KO",
    "PEP",
    "PG",
    "WMT",
    "XOM",
    "CVX",
    "CAT",
    "BA",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling covariance-estimator comparison (Week 4, Steps 6-7)."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_SMALL)
    parser.add_argument("--wide-tickers", nargs="+", default=DEFAULT_WIDE)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--window", type=int, default=252, help="Trailing estimation window in trading days."
    )
    parser.add_argument(
        "--rebalance", type=int, default=21, help="Rebalance step in trading days (21 ~ monthly)."
    )
    parser.add_argument("--lam", type=float, default=0.94, help="EWMA decay factor.")
    return parser.parse_args()


def estimator_fns(lam: float) -> dict:
    return {
        "Sample": sample_cov,
        "Ledoit-Wolf": ledoit_wolf_cov,
        "OAS": oas_cov,
        f"EWMA(lam={lam:g})": lambda r: ewma_cov(r, lam=lam),
    }


def rolling_gmv_backtest(
    returns: pd.DataFrame, estimator_fn, *, window: int, step: int
) -> tuple[pd.DataFrame, pd.Series]:
    n_obs = len(returns)
    rows: dict[pd.Timestamp, pd.Series] = {}
    oos_pieces: list[pd.Series] = []
    warm: np.ndarray | None = None
    for end in range(window, n_obs + 1, step):
        train = returns.iloc[end - window : end]
        cov = estimator_fn(train)
        try:
            w = solve_min_variance(cov, x0=warm)
        except RuntimeError:
            continue
        warm = w.to_numpy()
        rows[returns.index[end - 1]] = w
        test = returns.iloc[end : end + step]
        if len(test) > 0:
            oos_pieces.append(test.reindex(columns=w.index) @ w)
    weights_over_time = pd.DataFrame(rows).T
    oos = pd.concat(oos_pieces) if oos_pieces else pd.Series(dtype=float)
    return (weights_over_time, oos)


def backtest_metrics(weights: pd.DataFrame, oos: pd.Series, periods_per_year: int = 252) -> dict:
    turnover = weight_turnover(weights)
    hhi = (weights**2).sum(axis=1)
    eff_n = 1.0 / hhi
    gross = weights.abs().sum(axis=1)
    oos_std = float(oos.std())
    return {
        "rebalances": int(len(weights)),
        "avg_turnover": float(turnover.mean()),
        "avg_gross": float(gross.mean()),
        "avg_eff_N": float(eff_n.mean()),
        "oos_vol_ann_%": 100.0 * oos_std * np.sqrt(periods_per_year),
        "oos_sharpe": (
            float(oos.mean() / oos_std * np.sqrt(periods_per_year)) if oos_std > 0 else float("nan")
        ),
    }


def run_universe(returns, lam, window, step):
    (table_rows, weights_by, oos_by) = ({}, {}, {})
    for name, fn in estimator_fns(lam).items():
        (w, oos) = rolling_gmv_backtest(returns, fn, window=window, step=step)
        weights_by[name] = w
        oos_by[name] = oos
        table_rows[name] = backtest_metrics(w, oos)
    table = pd.DataFrame(table_rows).T
    return (table, weights_by)


def fig_weight_stability(weights_by, window, step, figures_dir) -> Path:
    (fig, (ax_to, ax_en)) = plt.subplots(1, 2, figsize=(13, 5))
    for name, w in weights_by.items():
        to = weight_turnover(w)
        ax_to.plot(to.index, to.to_numpy(), marker=".", markersize=4, label=name)
        eff_n = 1.0 / (w**2).sum(axis=1)
        ax_en.plot(eff_n.index, eff_n.to_numpy(), marker=".", markersize=4, label=name)
    ax_to.set_ylabel("L1 turnover between rebalances")
    ax_to.set_xlabel("rebalance date")
    ax_to.set_title(f"Turnover over time (window={window}d, rebalance={step}d)")
    ax_to.legend(fontsize=8)
    ax_to.grid(True, alpha=0.3)
    ax_en.set_ylabel("effective N = 1 / sum(w^2)")
    ax_en.set_xlabel("rebalance date")
    ax_en.set_title("Diversification over time")
    ax_en.legend(fontsize=8)
    ax_en.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week4_weight_stability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_turnover_concentration(table: pd.DataFrame, figures_dir: Path) -> Path:
    names = list(table.index)
    x = range(len(names))
    colors = ["C0", "C1", "C2", "C3"]
    (fig, axes) = plt.subplots(1, 3, figsize=(14, 4.5))
    specs = [
        ("avg_turnover", "average L1 turnover", "Stability (lower = steadier)"),
        ("avg_eff_N", "average effective N", "Diversification (higher = better)"),
        ("oos_vol_ann_%", "OOS volatility (annual %)", "Out-of-sample risk (lower = better)"),
    ]
    for ax, (col, ylabel, title) in zip(axes, specs):
        ax.bar(x, table[col].to_numpy(), color=colors[: len(names)])
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for i, v in enumerate(table[col].to_numpy()):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week4_turnover_concentration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    print("=" * 64)
    print(f"Week 4 rolling comparison (window={args.window}d, rebalance={args.rebalance}d)")
    print("=" * 64)
    print(f"\nDownloading WIDE universe ({len(args.wide_tickers)} tickers) ...")
    wide_returns = prices_to_returns(
        download_prices(args.wide_tickers, start=args.start, end=args.end)
    )
    print(f"  {len(wide_returns)} daily returns, {wide_returns.shape[1]} assets")
    (table_wide, weights_by) = run_universe(wide_returns, args.lam, args.window, args.rebalance)
    print("\n[Steps 6-7] WIDE universe performance:")
    print(table_wide.round(4).to_string())
    table_wide.round(6).to_csv(figures_dir / "week4_performance_wide.csv")
    p_stab = fig_weight_stability(weights_by, args.window, args.rebalance, figures_dir)
    p_tc = fig_turnover_concentration(table_wide, figures_dir)
    print(f"  saved {p_stab.name}, {p_tc.name}")
    print(f"\nDownloading SMALL universe ({len(args.tickers)} tickers) ...")
    small_returns = prices_to_returns(download_prices(args.tickers, start=args.start, end=args.end))
    print(f"  {len(small_returns)} daily returns, {small_returns.shape[1]} assets")
    (table_small, _) = run_universe(small_returns, args.lam, args.window, args.rebalance)
    print("\n[Steps 6-7] SMALL universe performance:")
    print(table_small.round(4).to_string())
    table_small.round(6).to_csv(figures_dir / "week4_performance_small.csv")
    print(f"\nAll outputs written to {figures_dir}")


if __name__ == "__main__":
    main()
