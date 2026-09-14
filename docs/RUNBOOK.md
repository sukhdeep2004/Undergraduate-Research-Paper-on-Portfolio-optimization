# Runbook

How to install, run, and navigate this repository. Narrative that used to live as inline code comments is collected here so the Python stays clean.

## Environment

```bash
cd /path/to/URA
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: NumPy, Pandas, SciPy, Matplotlib, yfinance, scikit-learn, Jupyter.

Always run scripts from the **repo root** so `import portfolio` resolves. Example scripts already prepend the repo root to `sys.path`.

## Data cache

- Default cache directory: `data_cache/` (gitignored).
- `portfolio.data.load_prices` downloads via Yahoo Finance once, then reuses CSV keyed by tickers + date range.
- Default empirical universe: the 11 SPDR Select Sector ETFs (`SECTOR_ETFS` in `portfolio.data`).

```bash
python examples/fetch_universe.py
```

## Demo catalog

| Script | What it shows | Network? |
|--------|----------------|----------|
| `examples/demo_expected_returns.py` | CAPM / historical means | Yes |
| `examples/demo_covariance.py` | Sample covariance | Yes |
| `examples/demo_minimum_variance.py` | Classical GMV | Yes |
| `examples/demo_efficient_frontier.py` | Frontier (+ optional `--capm`) | Yes |
| `examples/demo_constrained_optimization.py` | Long-only, max-weight, target-vol cases | Yes |
| `examples/demo_constrained_comparison.py` | Unconstrained vs constrained stability | Yes |
| `examples/demo_ridge_portfolios.py` | OLS / ridge / Britten–Jones | Yes |
| `examples/demo_covariance_estimators.py` | Shrinkage spectra + GMV plug-in | Yes |
| `examples/demo_covariance_comparison.py` | Rolling estimator backtests | Yes |
| `examples/demo_design_optimality.py` | A/D/E design + GMV bridge | No (default) |
| `examples/demo_research_optimizer.py` | Multi-term research objective | Synthetic / optional |
| `examples/experiment_grid.py` | Simulation grid over N/T, noise, strategies | No |
| `examples/demo_empirical_study.py` | Sector-ETF rolling study + costs | Yes (or cache) |
| `examples/demo_stress_tests.py` | Crisis regimes + extensions | Synthetic |

Figures and CSV scoreboards write to `figures/`.

### Useful flags (examples)

```bash
python examples/demo_constrained_comparison.py --window 504 --rebalance 63
python examples/demo_covariance_estimators.py --window 252 --lam 0.94
python examples/demo_efficient_frontier.py --capm
python examples/demo_design_optimality.py --tickers AAPL MSFT GOOGL JPM XOM
```

## Library map

### `portfolio.data`

Market ingest with on-disk caching. `download_prices` / `load_prices` / `prices_to_returns`, plus `SECTOR_ETFS`.

### `portfolio.covariance` / `covariance_estimators`

Sample covariance helpers and shrinkage estimators (Ledoit–Wolf, OAS, EWMA) with conditioning diagnostics. Shrinkage pulls extreme sample eigenvalues toward the mean and improves GMV out-of-sample risk when \(N/T\) is large.

### `portfolio.minimum_variance` / `efficient_frontier` / `expected_returns`

Classical Markowitz building blocks: GMV weights, historical or CAPM means, frontier construction and plots.

### `portfolio.constrained_optimization`

SLSQP quadratic programs with budget, optional long-only, max-weight, and target-return / target-volatility constraints. Objectives are scaled to \(O(1)\) because daily-return variances are tiny relative to default solver tolerances. Helpers for rolling weights, turnover, and frontier overlays.

### `portfolio.ridge_regression`

OLS and ridge coefficient paths, Britten–Jones regression interpretation of portfolio weights, and ridge GMV.

### `portfolio.design_optimality`

NumPy port of the multiplicative exchange algorithm used in optimal design (A/D/E criteria). The same simplex update applied to a covariance matrix recovers long-only global minimum variance — the design ↔ portfolio bridge.

### `portfolio.research_optimizer`

Composable objective terms (risk, ridge, precision / design penalties, diversification, robustness, L1 sparsity) solved as a research portfolio. Includes risk-budget / risk-contribution helpers.

### `portfolio.strategies`

Named strategy factories (`equal_weight`, constrained GMV, ridge GMV, proposed research portfolio, risk parity, sparse, sector-neutral) used by grids and backtests.

### `portfolio.simulation`

Covariance DGPs (identity, constant correlation, block, factor), Gaussian / heavy-tailed sampling, regime returns, and estimator-error scoring against known truth.

### `portfolio.evaluation`

Rolling train/test backtests, annualized stats, max drawdown, effective \(N\), scoreboard tables, and proportional transaction-cost overlays (`cost_bps` × turnover).

### `portfolio.screening`

Universe filters (e.g. ticker/sector exclusions) and sector group targets for constrained extensions.

## Reproducing paper-style results

1. Install deps and activate the venv.
2. Warm the cache: `python examples/fetch_universe.py`.
3. Simulation / method comparison: `python examples/experiment_grid.py`.
4. Empirical study: `python examples/demo_empirical_study.py`.
5. Stress / extensions: `python examples/demo_stress_tests.py`.
6. Design bridge: `python examples/demo_design_optimality.py`.

Outputs land in `figures/` (PNG + CSV). The LaTeX paper under `paper/` includes selected figures from that workflow.

## Optimal design R notes

See [optimal-design/r-code-explained.md](optimal-design/r-code-explained.md). Place companion R scripts under `docs/r-scripts/` if you distribute them (see that folder’s README).

## Notebooks

```bash
jupyter notebook notebooks/week3_regression_portfolios.ipynb
```

## Design notes worth keeping

- **Why constrain?** Long-only and weight caps reduce extreme shorts and turnover when covariance is noisy.
- **Why shrink?** Sample covariance is ill-conditioned for wide universes; Ledoit–Wolf / OAS typically cut out-of-sample volatility of GMV.
- **Why design criteria?** A/D/E penalties on the information matrix \(M\) encourage portfolios that are stable / diversified in the same geometric sense as optimal experimental designs.
- **Fair evaluation:** Prefer out-of-sample volatility, Sharpe net of costs, turnover, and effective \(N\) over in-sample fit.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: portfolio` | Run from repo root, or ensure `sys.path` includes it |
| yfinance / empty prices | Check tickers, date range, and network; delete bad cache files under `data_cache/` and rerun with `refresh=True` |
| SLSQP barely moves | Solver scaling is handled in `constrained_optimization`; ensure covariance is annualized consistently with the demo you copy |
| Figures missing | Create `figures/` (demos mkdir as needed) and rerun the relevant script |
