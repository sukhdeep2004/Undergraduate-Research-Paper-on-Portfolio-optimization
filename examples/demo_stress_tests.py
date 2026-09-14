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
from portfolio.data import DATA_CACHE_DIR, SECTOR_ETFS, load_prices, prices_to_returns
from portfolio.evaluation import (
    annualized_stats,
    compare_backtests,
    effective_n,
    max_drawdown,
    rolling_backtest,
)
from portfolio.screening import build_sector_groups, screen_universe
from portfolio.simulation import constant_correlation_covariance, simulate_regime_returns
from portfolio.strategies import (
    Strategy,
    equal_weight,
    gmv_weights,
    make_proposed,
    make_risk_parity,
    make_sector_neutral,
    make_sparse,
    markowitz_gmv,
)

SUPER_SECTORS = {
    "XLB": "Cyclical",
    "XLF": "Cyclical",
    "XLRE": "Cyclical",
    "XLY": "Cyclical",
    "XLP": "Defensive",
    "XLV": "Defensive",
    "XLU": "Defensive",
    "XLC": "Sensitive",
    "XLE": "Sensitive",
    "XLI": "Sensitive",
    "XLK": "Sensitive",
}
CRISES = {
    "COVID-19 crash": ("2020-02-19", "2020-04-30"),
    "2022 rate shock": ("2022-01-01", "2022-10-31"),
}
_STYLE = {
    "equal_weight": ("#7f7f7f", "--"),
    "markowitz": ("#d62728", "-"),
    "shrinkage": ("#2ca02c", "-"),
    "proposed": ("#9467bd", "-"),
    "robust_E": ("#1f77b4", "-"),
    "risk_parity": ("#17becf", "-"),
    "sparse": ("#bcbd22", "-"),
    "sector_neutral": ("#e377c2", "-"),
    "screened_exE": ("#8c564b", "-"),
}
_CORE = ["equal_weight", "markowitz", "shrinkage", "robust_E", "proposed", "risk_parity"]


def build_strategies(args, columns) -> "dict[str, Strategy]":
    groups = build_sector_groups(columns, SUPER_SECTORS)
    g = len(groups)
    super_targets = {s: (idx, 1.0 / g) for (s, (idx, _)) in groups.items()}
    strategies = [
        Strategy("equal_weight", equal_weight, sample_cov, family="baseline"),
        Strategy("markowitz", markowitz_gmv, sample_cov, family="classical"),
        Strategy("shrinkage", gmv_weights, ledoit_wolf_cov, family="regularized"),
        Strategy(
            "robust_E",
            make_proposed(gamma_d=0.0, gamma_e=args.gamma_e, long_only=True),
            sample_cov,
            family="proposed",
        ),
        Strategy(
            "proposed",
            make_proposed(gamma_d=args.gamma_d, gamma_e=args.gamma_e, long_only=True),
            sample_cov,
            family="proposed",
        ),
        Strategy("risk_parity", make_risk_parity(), sample_cov, family="extension"),
        Strategy("sparse", make_sparse(args.gamma_l1), sample_cov, family="extension"),
        Strategy(
            "sector_neutral",
            make_sector_neutral(super_targets, gamma_d=args.gamma_d, gamma_e=args.gamma_e),
            sample_cov,
            family="extension",
        ),
    ]
    return {s.name: s for s in strategies}


def run_backtests(returns, strategies, args) -> "dict":
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
    return results


def crisis_mask(index: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(False, index=index)
    for _, (lo, hi) in CRISES.items():
        mask |= (index >= pd.Timestamp(lo)) & (index <= pd.Timestamp(hi))
    return mask


def calm_vs_crisis(results: dict) -> pd.DataFrame:
    rows = {}
    for name in _CORE:
        if name not in results:
            continue
        r = results[name].portfolio_returns
        if r.empty:
            continue
        in_crisis = crisis_mask(r.index)
        (crisis_r, calm_r) = (r[in_crisis], r[~in_crisis])
        (c_stats, k_stats) = (annualized_stats(crisis_r), annualized_stats(calm_r))
        rows[name] = {
            "calm_sharpe": k_stats["sharpe"],
            "calm_ann_return": k_stats["ann_return"],
            "crisis_ann_vol": c_stats["ann_volatility"],
            "crisis_max_drawdown": max_drawdown(crisis_r),
            "full_sharpe": annualized_stats(r)["sharpe"],
            "full_max_drawdown": max_drawdown(r),
        }
    return pd.DataFrame(rows).T.reindex([n for n in _CORE if n in rows])


def regime_experiment(args) -> "tuple[dict, np.ndarray]":
    n = args.sim_assets
    rng = np.random.default_rng(args.seed)
    calm = constant_correlation_covariance(n, avg_corr=0.25, vols=0.011, rng=rng)
    crisis = constant_correlation_covariance(n, avg_corr=0.75, vols=0.028, rng=rng)
    lengths = [args.sim_calm, args.sim_crisis, args.sim_calm]
    mus = [0.0004, -0.003, 0.0004]
    returns = simulate_regime_returns([calm, crisis, calm], lengths, mus=mus, rng=rng)
    regime = returns.attrs["regime"]
    strategies = {
        "markowitz": Strategy("markowitz", markowitz_gmv, sample_cov),
        "shrinkage": Strategy("shrinkage", gmv_weights, ledoit_wolf_cov),
        "robust_E": Strategy(
            "robust_E", make_proposed(gamma_d=0.0, gamma_e=args.gamma_e, long_only=True), sample_cov
        ),
        "proposed": Strategy(
            "proposed",
            make_proposed(gamma_d=args.gamma_d, gamma_e=args.gamma_e, long_only=True),
            sample_cov,
        ),
        "risk_parity": Strategy("risk_parity", make_risk_parity(), sample_cov),
    }
    results = run_backtests(returns, strategies, args)
    return (results, regime)


def regime_crisis_metrics(results: dict, regime: np.ndarray) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        r = res.portfolio_returns
        if r.empty:
            continue
        is_crisis = pd.Series(regime, name="regime").reindex(r.index).eq(1).fillna(False)
        (crisis_r, calm_r) = (r[is_crisis.values], r[~is_crisis.values])
        rows[name] = {
            "crisis_ann_vol": annualized_stats(crisis_r)["ann_volatility"],
            "crisis_max_drawdown": max_drawdown(crisis_r),
            "calm_ann_return": annualized_stats(calm_r)["ann_return"],
        }
    order = [
        n for n in ["markowitz", "shrinkage", "robust_E", "proposed", "risk_parity"] if n in rows
    ]
    return pd.DataFrame(rows).T.reindex(order)


def hypothesis_verdicts(board, cv, regime_cv) -> pd.DataFrame:

    def g(df, row, col):
        try:
            return float(df.loc[row, col])
        except Exception:
            return float("nan")

    (turn_mk, turn_pr) = (
        g(board, "markowitz", "avg_turnover"),
        g(board, "proposed", "avg_turnover"),
    )
    (en_mk, en_pr) = (
        g(board, "markowitz", "avg_effective_n"),
        g(board, "proposed", "avg_effective_n"),
    )
    (mdd_mk, mdd_rb) = (
        g(cv, "markowitz", "crisis_max_drawdown"),
        g(cv, "robust_E", "crisis_max_drawdown"),
    )
    (vol_mk, vol_rb) = (g(cv, "markowitz", "crisis_ann_vol"), g(cv, "robust_E", "crisis_ann_vol"))
    (sh_mk, sh_rb) = (g(cv, "markowitz", "calm_sharpe"), g(cv, "robust_E", "calm_sharpe"))
    rmdd_mk = g(regime_cv, "markowitz", "crisis_max_drawdown")
    rmdd_rb = g(regime_cv, "robust_E", "crisis_max_drawdown")
    h1 = "SUPPORTED" if turn_pr < turn_mk else "not supported"
    h2 = "SUPPORTED" if en_pr > en_mk else "not supported"
    h5_real = vol_rb < vol_mk and mdd_rb > mdd_mk
    h5 = "SUPPORTED" if h5_real else "mixed"
    rows = [
        {
            "hypothesis": "H1  E-reg lowers turnover at matched risk",
            "verdict": h1,
            "evidence": f"avg turnover proposed {turn_pr:.3f} vs markowitz {turn_mk:.3f}",
        },
        {
            "hypothesis": "H2  D-penalty raises effective N",
            "verdict": h2,
            "evidence": f"effective N proposed {en_pr:.1f} vs markowitz {en_mk:.1f}",
        },
        {
            "hypothesis": "H3  shrinkage approx info-optimal ridge",
            "verdict": "SUPPORTED (qualitative)",
            "evidence": "spectral/weight agreement shown in the method note (Week 6)",
        },
        {
            "hypothesis": "H4  design penalty beats ridge as N/T->1",
            "verdict": "PARTIAL",
            "evidence": "wins on turnover/effective N, not raw OOS vol vs ridge (Week 7)",
        },
        {
            "hypothesis": "H5  E-optimal more crisis-robust than GMV",
            "verdict": h5,
            "evidence": f"real crisis vol robust_E {vol_rb:.3f} vs GMV {vol_mk:.3f}; crisis MDD {mdd_rb:.3f} vs {mdd_mk:.3f}; calm Sharpe {sh_rb:.2f} vs {sh_mk:.2f}; sim crisis MDD {rmdd_rb:.3f} vs {rmdd_mk:.3f}",
        },
    ]
    return pd.DataFrame(rows).set_index("hypothesis")


def plot_stress_bars(cv: pd.DataFrame, path: Path) -> None:
    panels = [
        ("crisis_max_drawdown", "crisis max drawdown (worse = lower)"),
        ("crisis_ann_vol", "crisis annualized volatility"),
        ("calm_sharpe", "calm-period Sharpe (the give-up)"),
    ]
    (fig, axes) = plt.subplots(1, 3, figsize=(15, 5))
    names = list(cv.index)
    colors = [_STYLE.get(n, ("#333", "-"))[0] for n in names]
    for ax, (col, title) in zip(axes, panels):
        ax.bar(np.arange(len(names)), cv[col].values, color=colors)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Week 9 — crisis robustness vs calm give-up (real sector ETFs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regime_equity(results: dict, regime: np.ndarray, path: Path) -> None:
    (fig, ax) = plt.subplots(figsize=(11, 6))
    crisis_idx = np.where(regime == 1)[0]
    if crisis_idx.size:
        ax.axvspan(
            crisis_idx.min(), crisis_idx.max(), color="red", alpha=0.08, label="crisis regime"
        )
    for name, res in results.items():
        r = res.portfolio_returns
        if r.empty:
            continue
        curve = (1.0 + r).cumprod()
        (color, ls) = _STYLE.get(name, ("#333", "-"))
        ax.plot(curve.index, curve.values, color=color, ls=ls, lw=1.8, label=name)
    ax.set_xlabel("simulated trading day")
    ax.set_ylabel("growth of $1 (out-of-sample)")
    ax.set_title("Week 9 — traversing a simulated calm → crisis → calm regime shift")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_extensions(board: pd.DataFrame, path: Path) -> None:
    panels = [
        ("avg_effective_n", "avg effective N (diversification)"),
        ("avg_turnover", "avg turnover (stability)"),
    ]
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 5))
    names = list(board.index)
    colors = [_STYLE.get(n, ("#333", "-"))[0] for n in names]
    for ax, (col, title) in zip(axes, panels):
        ax.bar(np.arange(len(names)), board[col].values, color=colors)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Week 9 — extensions on the full sample (sector ETFs)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_universe_returns(start: str, end: "str | None", refresh: bool) -> pd.DataFrame:
    cached = DATA_CACHE_DIR / "universe_returns.csv"
    if cached.exists() and (not refresh):
        return pd.read_csv(cached, index_col=0, parse_dates=True)
    prices = load_prices(SECTOR_ETFS, start, end, refresh=refresh).dropna(how="any")
    return prices_to_returns(prices)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 9 robustness, stress tests, extensions.")
    p.add_argument("--window", type=int, default=252)
    p.add_argument("--step", type=int, default=21)
    p.add_argument("--start", default="2018-07-01")
    p.add_argument("--end", default=None)
    p.add_argument("--gamma-d", type=float, default=0.05)
    p.add_argument("--gamma-e", type=float, default=0.1)
    p.add_argument("--gamma-l1", type=float, default=0.05)
    p.add_argument("--sim-assets", type=int, default=15)
    p.add_argument("--sim-calm", type=int, default=500)
    p.add_argument("--sim-crisis", type=int, default=120)
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(exist_ok=True)
    returns = load_universe_returns(args.start, args.end, args.refresh)
    print("=" * 74)
    print(
        f"Week 9 | {returns.shape[1]} sector ETFs, {returns.shape[0]} days ({returns.index.min().date()} -> {returns.index.max().date()})"
    )
    print(f"window={args.window}, step={args.step}, gamma_e={args.gamma_e}")
    print("=" * 74)
    strategies = build_strategies(args, returns.columns)
    results = run_backtests(returns, strategies, args)
    screened = screen_universe(returns, exclude_sectors=["Energy"], sector_map=SECTOR_ETFS)
    try:
        results["screened_exE"] = rolling_backtest(
            screened,
            make_proposed(gamma_d=args.gamma_d, gamma_e=args.gamma_e, long_only=True),
            cov_estimator=sample_cov,
            window=args.window,
            step=args.step,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"  screened_exE: skipped ({exc})")
    board = compare_backtests(results)
    ext_order = [
        "equal_weight",
        "markowitz",
        "shrinkage",
        "robust_E",
        "proposed",
        "risk_parity",
        "sparse",
        "sector_neutral",
        "screened_exE",
    ]
    board = board.reindex([s for s in ext_order if s in board.index])
    board.to_csv(figures_dir / "week9_extensions_scoreboard.csv")
    pd.set_option("display.width", 180, "display.max_columns", 20)
    print("\nExtensions scoreboard (full sample):")
    print(
        board[
            [
                "sharpe",
                "ann_volatility",
                "avg_turnover",
                "avg_effective_n",
                "max_drawdown",
                "n_rebalances",
            ]
        ]
        .round(4)
        .to_string()
    )
    cv = calm_vs_crisis(results)
    cv.to_csv(figures_dir / "week9_crisis_vs_calm.csv")
    print("\nStudy 1 — real-data calm vs crisis (H5):")
    print(cv.round(4).to_string())
    plot_stress_bars(cv, figures_dir / "week9_stress_bars.png")
    (regime_results, regime) = regime_experiment(args)
    regime_cv = regime_crisis_metrics(regime_results, regime)
    print("\nStudy 2 — simulated calm -> crisis -> calm (H5):")
    print(regime_cv.round(4).to_string())
    plot_regime_equity(regime_results, regime, figures_dir / "week9_regime_equity.png")
    plot_extensions(board, figures_dir / "week9_extensions.png")
    verdicts = hypothesis_verdicts(board, cv, regime_cv)
    verdicts.to_csv(figures_dir / "week9_hypotheses.csv")
    print("\nH1-H5 verdicts:")
    print(verdicts.to_string())
    print(f"\nSaved 3 CSVs + 3 figures to {figures_dir}")


if __name__ == "__main__":
    main()
