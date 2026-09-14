import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.constrained_optimization import (
    plot_weights_comparison,
    rolling_constrained_weights,
    solve_min_variance,
    weight_turnover,
)
from portfolio.covariance import sample_covariance
from portfolio.data import download_prices, prices_to_returns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Constrained vs unconstrained allocation experiments."
    )
    parser.add_argument(
        "--window", type=int, default=252, help="Trailing window length in trading days."
    )
    parser.add_argument(
        "--rebalance", type=int, default=21, help="Rebalance step in trading days (21 ~ monthly)."
    )
    parser.add_argument(
        "--max-weight", type=float, default=0.25, help="Per-asset cap for the capped method."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL", "JPM", "XOM"],
        help="Ticker universe.",
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_weight = args.max_weight
    print(f"Downloading {args.tickers} from Yahoo Finance ({args.start} to {args.end})...")
    prices = download_prices(args.tickers, start=args.start, end=args.end)
    returns = prices_to_returns(prices)
    print(f"{len(returns)} daily returns, {len(returns.columns)} assets")
    methods = {
        "Unconstrained": dict(long_only=False, max_weight=None),
        "Long-only": dict(long_only=True, max_weight=None),
        f"Long-only + cap {max_weight}": dict(long_only=True, max_weight=max_weight),
    }
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print("Experiment 1 - full-sample minimum-variance weights")
    print("=" * 60)
    cov_full = sample_covariance(returns, annualize=True)
    full_weights = {label: solve_min_variance(cov_full, **cfg) for (label, cfg) in methods.items()}
    table = pd.DataFrame(full_weights).round(4)
    print(table.to_string())
    weights_path = figures_dir / "week2_weights_comparison.png"
    plot_weights_comparison(
        full_weights,
        title=f"Full-sample minimum-variance weights ({args.start} to {args.end})",
        save_path=str(weights_path),
    )
    print(f"\nSaved weight comparison plot to {weights_path}")
    print("\n" + "=" * 60)
    print(f"Experiment 2 - rolling stability (window={args.window}d, rebalance={args.rebalance}d)")
    print("=" * 60)
    rolling = {
        label: rolling_constrained_weights(returns, window=args.window, step=args.rebalance, **cfg)
        for (label, cfg) in methods.items()
    }
    print(f"\n{'Method':<26}{'rebal':>7}{'avg turnover':>14}{'avg max wt':>12}{'avg shorts':>12}")
    for label, w in rolling.items():
        turnover = weight_turnover(w)
        avg_turnover = float(turnover.mean())
        avg_max_w = float(w.max(axis=1).mean())
        avg_shorts = float((w < -1e-06).sum(axis=1).mean())
        print(f"{label:<26}{len(w):>7}{avg_turnover:>14.4f}{avg_max_w:>12.4f}{avg_shorts:>12.2f}")
    _plot_stability(rolling, args, figures_dir)


def _plot_stability(
    rolling: dict[str, pd.DataFrame], args: argparse.Namespace, figures_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    (fig, (ax_turnover, ax_maxw)) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label, w in rolling.items():
        turnover = weight_turnover(w)
        ax_turnover.plot(turnover.index, turnover.to_numpy(), marker=".", label=label)
        ax_maxw.plot(w.index, w.max(axis=1).to_numpy(), marker=".", label=label)
    ax_turnover.set_ylabel("Turnover (L1 weight change)")
    ax_turnover.set_title(
        f"Rolling allocation stability (window={args.window}d, rebalance={args.rebalance}d)"
    )
    ax_turnover.legend()
    ax_turnover.grid(True, alpha=0.3)
    ax_maxw.set_ylabel("Max single weight")
    ax_maxw.set_xlabel("Rebalance date")
    ax_maxw.legend()
    ax_maxw.grid(True, alpha=0.3)
    fig.tight_layout()
    stability_path = figures_dir / "week2_weight_stability.png"
    fig.savefig(stability_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved stability plot to {stability_path}")


if __name__ == "__main__":
    main()
