# Optimal Portfolio Design

Research codebase linking **Markowitz / constrained portfolio optimization** with ideas from **optimal experimental design** (A/D/E-optimality). Built as a structured undergraduate research project: classical methods → shrinkage & ridge portfolios → design-inspired objectives → simulate real-market backtests.

## Highlights

- Constrained global minimum-variance and efficient-frontier solvers (SLSQP)
- Covariance estimators: sample, Ledoit–Wolf, OAS, EWMA
- Ridge / Britten–Jones regression portfolios
- Design-optimality bridge (multiplicative simplex updates ↔ long-only GMV)
- Extensible research optimizer (risk, ridge, precision, diversification, robustness, sparsity)
- Rolling backtests with turnover, drawdown, Sharpe, and transaction-cost overlays
- Synthetic stress tests and an empirical study on SPDR sector ETFs

## Repository layout

```text
URA/
├── portfolio/          # Core Python library
├── examples/           # Runnable demos and experiment scripts
├── notebooks/          # Jupyter notebooks
├── figures/            # Generated plots and result tables
├── paper/              # LaTeX research paper sources
├── docs/               # Runbook and reference notes
├── requirements.txt
└── README.md
```

Private notes, drafts, and tooling stay local (see `.gitignore`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run demos from the **repository root** (so `portfolio` is importable):

```bash
python examples/demo_minimum_variance.py
python examples/demo_empirical_study.py
python examples/experiment_grid.py
```

Market-data demos need network access once; prices cache under `data_cache/` (gitignored). Refresh the default sector-ETF universe with:

```bash
python examples/fetch_universe.py
```

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Install, run demos, reproduce experiments, module map |
| [docs/optimal-design/r-code-explained.md](docs/optimal-design/r-code-explained.md) | Walkthrough of A/D/E-optimality R scripts |
| `paper/` | Paper source (`main.tex` + sections) |

## Quick module map

| Area | Package module |
|------|----------------|
| Data ingest | `portfolio.data` |
| Covariance | `portfolio.covariance`, `portfolio.covariance_estimators` |
| Classical MV / frontier | `portfolio.minimum_variance`, `portfolio.efficient_frontier` |
| Constraints | `portfolio.constrained_optimization` |
| Ridge portfolios | `portfolio.ridge_regression` |
| Design optimality | `portfolio.design_optimality` |
| Research objective | `portfolio.research_optimizer`, `portfolio.strategies` |
| Simulation / backtest | `portfolio.simulation`, `portfolio.evaluation` |
| Screening | `portfolio.screening` |

## Paper

LaTeX sources live in `paper/`. Build with your usual TeX toolchain from that directory (e.g. `pdflatex` / `latexmk`). Compiled PDFs are gitignored.

## Acknowledgments

This work was supervised by **Professor Saumen Mandal** (University of Manitoba). Thank you for your guidance and support throughout the project.

Contact: [Saumen.Mandal@umanitoba.ca](mailto:Saumen.Mandal@umanitoba.ca)


