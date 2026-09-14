import argparse
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portfolio.constrained_optimization import plot_weights_comparison, solve_min_variance
from portfolio.covariance_estimators import (
    cov_diagnostics,
    ewma_cov,
    ledoit_wolf_cov,
    oas_cov,
    sample_cov,
)
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
        description="Covariance estimator comparison (Week 4, Steps 4-5)."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_SMALL,
        help="Small universe (readable GMV weight bars).",
    )
    parser.add_argument(
        "--wide-tickers",
        nargs="+",
        default=DEFAULT_WIDE,
        help="Wide universe (high-dimensional spectrum).",
    )
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--window",
        type=int,
        default=252,
        help="Recent days used for the eigenvalue spectrum (q = N/T).",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.94,
        help="EWMA decay factor (RiskMetrics daily default 0.94).",
    )
    return parser.parse_args()


def build_estimators(returns: pd.DataFrame, lam: float) -> dict[str, pd.DataFrame]:
    return {
        "Sample": sample_cov(returns),
        "Ledoit-Wolf": ledoit_wolf_cov(returns),
        "OAS": oas_cov(returns),
        f"EWMA(lam={lam:g})": ewma_cov(returns, lam=lam),
    }


def print_estimator_summary(estimators: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, cov in estimators.items():
        diag = cov_diagnostics(cov)
        rows.append(
            {
                "estimator": name,
                "shrinkage/half-life": _intensity_label(cov),
                "condition_number": round(diag["condition_number"], 1),
                "effective_rank": round(diag["effective_rank"], 2),
                "min_eig": f"{diag['min_eigenvalue']:.2e}",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def _intensity_label(cov: pd.DataFrame) -> str:
    if "shrinkage" in cov.attrs:
        return f"delta={cov.attrs['shrinkage']:.4f}"
    if "half_life" in cov.attrs:
        return f"half-life={cov.attrs['half_life']:.1f}d"
    return "-"


def _to_correlation(cov_array: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov_array))
    outer = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(outer > 0, cov_array / outer, 0.0)
    return corr


def fig_eigenvalue_spectrum(
    returns_window: pd.DataFrame, lam: float, figures_dir: Path
) -> tuple[Path, int]:
    estimators = build_estimators(returns_window, lam)
    n_assets = returns_window.shape[1]
    n_obs = returns_window.shape[0]
    q = n_assets / n_obs
    lam_minus = (1.0 - np.sqrt(q)) ** 2
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    sample_corr = _to_correlation(sample_cov(returns_window).to_numpy())
    sample_eig = np.sort(np.linalg.eigvalsh(sample_corr))[::-1]
    n_in_band = int(np.sum((sample_eig >= lam_minus) & (sample_eig <= lam_plus)))
    (fig, (ax_spec, ax_cond)) = plt.subplots(1, 2, figsize=(13, 5))
    idx = np.arange(1, n_assets + 1)
    for name, cov in estimators.items():
        corr = _to_correlation(cov.to_numpy())
        eig = np.sort(np.linalg.eigvalsh(corr))[::-1]
        eig = np.clip(eig, 1e-14, None)
        ax_spec.plot(idx, eig, marker="o", markersize=4, label=name)
    ax_spec.axhspan(
        lam_minus,
        lam_plus,
        color="gray",
        alpha=0.25,
        label=f"MP band [{lam_minus:.2f}, {lam_plus:.2f}]",
    )
    ax_spec.set_yscale("log")
    ax_spec.set_xlabel("eigenvalue index (largest to smallest)")
    ax_spec.set_ylabel("correlation eigenvalue (log scale)")
    ax_spec.set_title(
        f"Correlation spectrum vs Marchenko-Pastur band\nN={n_assets}, T={n_obs}, q=N/T={q:.2f}  ({n_in_band}/{n_assets} sample eigenvalues in band)"
    )
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3)
    names = list(estimators)
    kappas = [cov_diagnostics(cov)["condition_number"] for cov in estimators.values()]
    ax_cond.bar(range(len(names)), kappas, color=["C0", "C1", "C2", "C3"])
    ax_cond.set_yscale("log")
    ax_cond.set_xticks(range(len(names)))
    ax_cond.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax_cond.set_ylabel("condition number $\\kappa(\\hat\\Sigma)$ (log scale)")
    ax_cond.set_title("Conditioning by estimator")
    for i, k in enumerate(kappas):
        ax_cond.text(i, k, f"{k:.0f}", ha="center", va="bottom", fontsize=8)
    ax_cond.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "week4_eigenvalue_spectrum.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return (path, n_in_band)


def gmv_weights(estimators: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    weights = {}
    for name, cov in estimators.items():
        try:
            weights[name] = solve_min_variance(cov)
        except RuntimeError as exc:
            print(f"  [warn] GMV solve failed for {name}: {exc}")
    return weights


def _weight_stats(w: pd.Series) -> dict:
    arr = w.to_numpy()
    hhi = float(np.sum(arr**2))
    return {
        "sum": round(float(arr.sum()), 4),
        "gross_|w|": round(float(np.abs(arr).sum()), 3),
        "max": round(float(arr.max()), 4),
        "min": round(float(arr.min()), 4),
        "HHI": round(hhi, 4),
        "eff_N": round(1.0 / hhi, 2) if hhi > 0 else float("nan"),
    }


def print_weight_stats(weights: dict[str, pd.Series], label: str) -> None:
    print(f"\nGMV concentration / leverage ({label}):")
    rows = [{"estimator": name, **_weight_stats(w)} for (name, w) in weights.items()]
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    print(f"Downloading WIDE universe ({len(args.wide_tickers)} tickers) from Yahoo ...")
    wide_prices = download_prices(args.wide_tickers, start=args.start, end=args.end)
    wide_returns = prices_to_returns(wide_prices)
    print(
        f"  {len(wide_returns)} daily returns, {wide_returns.shape[1]} assets ({wide_returns.index.min().date()} to {wide_returns.index.max().date()})"
    )
    window = min(args.window, len(wide_returns))
    wide_window = wide_returns.iloc[-window:]
    print(f"\n[Step 4] Spectrum diagnostics on last {window} days of the wide universe:")
    print_estimator_summary(build_estimators(wide_window, args.lam))
    (p_spec, n_in_band) = fig_eigenvalue_spectrum(wide_window, args.lam, figures_dir)
    print(
        f"  {n_in_band}/{wide_window.shape[1]} sample eigenvalues fall in the Marchenko-Pastur noise band   (saved {p_spec.name})"
    )
    print(f"\nDownloading SMALL universe ({len(args.tickers)} tickers) from Yahoo ...")
    small_prices = download_prices(args.tickers, start=args.start, end=args.end)
    small_returns = prices_to_returns(small_prices)
    print(f"  {len(small_returns)} daily returns, {small_returns.shape[1]} assets")
    print("\n[Step 5] GMV weights per estimator (small universe, full history):")
    small_estimators = build_estimators(small_returns, args.lam)
    small_weights = gmv_weights(small_estimators)
    print(pd.DataFrame(small_weights).round(4).to_string())
    print_weight_stats(small_weights, "small universe")
    p_w = figures_dir / "week4_gmv_weights.png"
    plot_weights_comparison(
        small_weights,
        title="GMV weights by covariance estimator (small universe)",
        save_path=str(p_w),
    )
    print(f"  saved {p_w.name}")
    print("\n[Step 5] GMV concentration on the wide universe (full history):")
    wide_estimators = build_estimators(wide_returns, args.lam)
    wide_weights = gmv_weights(wide_estimators)
    print_weight_stats(wide_weights, "wide universe")
    print(f"\nFigures written to {figures_dir}")


if __name__ == "__main__":
    main()
