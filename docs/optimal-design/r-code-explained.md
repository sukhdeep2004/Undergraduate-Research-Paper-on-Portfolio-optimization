# Optimal Design R Code — Detailed Explanation

This document explains the R scripts for **A-optimality**, **c-optimality**, and **D-optimality** in regression experimental design. The code implements a **multiplicative exchange / weight-update algorithm** on a **discretized design space**. The same update idea is ported to Python in `portfolio/design_optimality.py`.

**Source files** (place in `docs/r-scripts/`):

| File | Contents |
|------|----------|
| `docs/r-scripts/R-code-A-optimality.txt` | A-optimal designs: simple linear regression (SLR) and quadratic regression (QR) |
| `docs/r-scripts/R-code-c-optimality.txt` | c-optimal design for estimating the quadratic coefficient $\theta_2$ |
| `docs/r-scripts/R-code-D-optimality.txt` | D-optimal design for quadratic regression |

---

## Table of contents

1. [Statistical background](#1-statistical-background)
2. [Shared code structure](#2-shared-code-structure)
3. [Variable and matrix dictionary](#3-variable-and-matrix-dictionary)
4. [The iteration loop (all three scripts)](#4-the-iteration-loop-all-three-scripts)
5. [A-optimality (`R-code-A-optimality.txt`)](#5-a-optimality-r-code-a-optimalitytxt)
6. [c-optimality (`R-code-c-optimality.txt`)](#6-c-optimality-r-code-c-optimalitytxt)
7. [D-optimality (`R-code-D-optimality.txt`)](#7-d-optimality-r-code-d-optimalitytxt)
8. [Post-processing and plots](#8-post-processing-and-plots)
9. [How to run and tune](#9-how-to-run-and-tune)
10. [Expected theoretical answers](#10-expected-theoretical-answers)
11. [Relationship to the portfolio project](#11-relationship-to-the-portfolio-project)

---

## 1. Statistical background

### 1.1 Linear model

For a design point $x$ on $[-1, 1]$, the mean response is

$$
E(y \mid x) = f(x)^T \theta
$$

where $f(x)$ is the **regressor vector** and $\theta$ is the parameter vector.

| Model | Regressor $f(x)$ | $k$ (number of parameters) |
|-------|------------------|----------------------------|
| Simple linear regression (SLR) | $(1, x)^T$ | 2 |
| Quadratic regression (QR) | $(1, x, x^2)^T$ | 3 |

Errors are assumed i.i.d. with constant variance $\sigma^2$.

### 1.2 Approximate (continuous) design

An **approximate design** is a probability distribution on support points:

$$
p = \{(x_1, p_1), (x_2, p_2), \ldots, (x_J, p_J)\}, \quad p_j \geq 0,\ \sum_{j=1}^J p_j = 1
$$

- $x_j$ = support points (fixed after discretization)
- $p_j$ = proportion of total sample size at $x_j$

In the code, `c2` holds the $x_j$ and `c1` holds the $p_j$.

### 1.3 Information matrix

For regressor vectors $v_j = f(x_j)$,

$$
M(p) = \sum_{j=1}^J p_j\, v_j v_j^T
$$

The covariance matrix of the least-squares estimator is proportional to $M(p)^{-1}$. **Optimal designs** choose $p$ to make $M(p)$ “large” in a criterion-specific sense (equivalently, make $M(p)^{-1}$ “small”).

In R, if `m2` has rows $v_j^T$ stacked (so `m2` is $k \times J$), then:

```r
m1 <- diag(c1)           # weights
m5 <- m2 %*% m1 %*% t(m2)  # M(p)
m6 <- solve(m5)            # M(p)^{-1}
```

### 1.4 Optimality criteria (what each script maximizes)

The algorithms **maximize** $\phi(p) = \psi(M(p))$. Equivalently, they minimize a variance measure.

**Criterion functions**

| Criterion | Criterion function $\phi(p)$ | Goal |
|-----------|------------------------------|------|
| **D** | $\log \det M(p) = -\log \det M(p)^{-1}$ | Minimize generalized variance of $\hat{\theta}$ |
| **A** | $-\mathrm{tr}(M(p)^{-1})$ | Minimize **sum** of parameter variances |
| **c** | $-c^T M(p)^{-1} c$ | Minimize variance of **one** linear combination $c^T\theta$ |

**Partial derivatives** (needed for the update; at iteration $r$):

| Criterion | $\dfrac{\partial \phi}{\partial p_j}$ |
|-----------|--------------------------------------|
| **D** | $v_j^T M^{-1} v_j$ |
| **A** | $v_j^T M^{-2} v_j$ |
| **c** | $\bigl[c^T M^{-1} v_j\bigr]^2$ |

These appear in the code as `c5` (possibly squared for c-optimality).

### 1.5 Equivalence / vertex directional derivative

For improving design $p$, the **vertex directional derivative** toward support point $j$ is

$$
F_j = \frac{\partial \phi}{\partial p_j} - \sum_{i=1}^J p_i \frac{\partial \phi}{\partial p_i}
$$

In the code:

```r
c22 <- c5 - sum(c1 * c5)   # F_j for each j; max(c22) tracked in k23
```

At a **D-optimal** design, $F_j \leq 0$ for all $j$, with equality on the support (General Equivalence Theorem). The loop tracks `max(c22)` in `k23`; when this is near zero, the design is close to optimal.

### 1.6 Multiplicative algorithm

From `OptimalDesign_Note4.tex`, weights are updated by

$$
p_j^{(r+1)} = \frac{p_j^{(r)}\, f\!\left(d_j^{(r)}\right)}{\sum_{i=1}^J p_i^{(r)}\, f\!\left(d_i^{(r)}\right)}
$$

where $d_j = \partial \phi / \partial p_j$ (or a monotone transform of it), and $f$ is positive and strictly increasing. Common choices:

- $f(x) = e^{\delta x}$ → `c11 <- exp(k1 * ...)`
- Logistic: $f(x) = e^{\delta x}/(1 + e^{\delta x})$ → `c11 <- (exp(k1*c222))/(1+exp(k1*c222))`

`k1` in the code is the step-size / learning-rate parameter $\delta$. Larger values can speed convergence but may overshoot; smaller values are more stable.

---

## 2. Shared code structure

All three files follow the same pipeline:

```
Setup design space (c2, c1, m2)
    ↓
for i in 1:n:                    # n iterations
    Build M(p), M^{-1}
    Compute criterion-specific d_j  → c5
    Compute F_j                     → c22
    Renormalize c1
    Apply f(·) multiplicative update
    Record max(F_j)               → k23
    ↓
Summarize convergence (k23, c100)
Plot weights and derivatives
```

**Differences** between files:

| Aspect | A-opt | c-opt | D-opt |
|--------|-------|-------|-------|
| Model in first block | SLR then QR | QR only | QR only |
| `m2` rows | 2 (SLR) or 3 (QR) | 3 | 3 |
| Formula for `c5` | $v_j^T M^{-2} v_j$ | $[c^T M^{-1} v_j]^2$ | $v_j^T M^{-1} v_j$ |
| Update function `c11` | logistic on `c222` or `c5` | logistic on `c22` | `exp(k1*c5)` |
| Default `k1` | 0.9 | 0.3 | 0.5 |
| Default `n` | 200 | 1 (see note below) | 200 |

**Note on c-optimality `n <- 1`:** The file is labeled “correct program” and is meant to be run with a **small** $\delta$ (`k1 = 0.3`). With `n = 1` you only get one update from the uniform start; for convergence, set `n` to a large value (e.g. 200) like the other scripts.

---

## 3. Variable and matrix dictionary

| Symbol in code | Typical size | Meaning |
|----------------|--------------|---------|
| `k1` | scalar | $\delta$ — step size in $f(x)=e^{\delta x}$ or logistic |
| `c2` | length `J` | Support points $x_j \in [-1,1]$ |
| `J` | scalar | Number of support points ($J=21$ for `seq(-1,1,by=0.1)`) |
| `c1` | length `J` | Design weights $p_j$ |
| `c3` | length `J` | Constant 1 (intercept in $f(x)$) |
| `c4` | length `J` | $x_j^2$ |
| `c44` | length `J` | $x_j^3$ (defined but unused in shown loops) |
| `m2` | $k \times J$ | Regressor matrix; row $r$ is the $r$-th basis function across all $x_j$ |
| `m1` | $J \times J$ | `diag(c1)` |
| `m3` | $J \times k$ | `t(m2)` |
| `m4` | $k \times J$ | `m2 %*% m1` |
| `m5` | $k \times k$ | Information matrix $M(p)$ |
| `m6` | $k \times k$ | $M(p)^{-1}$ |
| `m7` | $J \times k$ | $V^T M^{-1}$ where $V = [v_1,\ldots,v_J]$ |
| `m77` | $J \times k$ | $V^T M^{-2}$ (A-opt only) |
| `m8` | $J \times J$ | Used to extract per-vertex derivatives on the diagonal |
| `c5` | length `J` | $d_j = \partial\phi/\partial p_j$ (per support point) |
| `c6` | length `k` | `diag(m5)` — diagonal of information matrix (A, c) |
| `k5` | scalar | `length(c6)` = $k$ |
| `c55` | length `J` | `c5/k5` — scaled derivative (used in some update variants) |
| `c22` | length `J` | Vertex directional derivative $F_j$ |
| `c222` | length `J` | `c55 - 1` — shifted scaled derivative for logistic update |
| `c11` | length `J` | Multiplicative factor $f(\cdot)$ |
| `c12` | length `J` | `c1 * c11` before renormalization |
| `k23` | length `n` | `max(c22)` at each iteration (convergence trace) |
| `Cmat` | $k \times J$ | c-opt only: each column is the vector $c$ |
| `n` | scalar | Number of algorithm iterations |
| `xp`, `xpF` | matrices | Rounded $(x, p)$ and $(x, p, F_j)$ for inspection |
| `In1`…`In6` | scalars | First iteration where `max(c22)` ≤ $10^{-r}$ |
| `c100` | vector | Collection of convergence iteration indices |

### Building `m2` (quadratic example)

```r
c2 <- seq(-1, 1, by = 0.1)   # 21 points
c3 <- rep(1, J)               # 1
c4 <- c2^2                    # x^2
m2 <- matrix(c(c3, c2, c4), nrow = 3, ncol = J, byrow = TRUE)
```

Row 1 = intercept, row 2 = $x$, row 3 = $x^2$ at every support point.

**SLR block in A-optimality** uses only two rows:

```r
m2 <- matrix(c(c3, c2), nrow = 2, ncol = J, byrow = TRUE)
```

---

## 4. The iteration loop (all three scripts)

Below is the logical flow inside `for(i in 1:n)`.

### Step 1 — Information matrix and inverse

```r
m1 <- diag(c1)
m3 <- t(m2)
m4 <- m2 %*% m1
m5 <- m4 %*% m3      # M(p)
m6 <- solve(m5)      # M(p)^{-1}
```

### Step 2 — Criterion-specific partial derivatives `c5`

**D-optimality:**

```r
m7 <- m3 %*% m6              # V^T M^{-1}
m8 <- m7 %*% m2              # J×J matrix; (i,j) entry = v_i^T M^{-1} v_j
c5 <- diag(m8)               # d_j = v_j^T M^{-1} v_j
```

**A-optimality:**

```r
m7 <- m3 %*% m6
m77 <- m7 %*% m6             # V^T M^{-2}
m8 <- m77 %*% m2
c5 <- diag(m8)               # d_j = v_j^T M^{-2} v_j
```

**c-optimality** (interest vector $c = (0,0,1)^T$ for $\theta_2$ in QR):

```r
c33 <- rep(0, J)
Cmat <- matrix(c(c33, c33, c3), nrow = 3, ncol = J, byrow = TRUE)  # each column c
m7 <- m3 %*% m6
m8 <- m7 %*% Cmat
c5 <- diag(m8)^2             # d_j = [c^T M^{-1} v_j]^2
```

Here `diag(m8)` equals $c^T M^{-1} v_j$ because each column of `Cmat` is $c$.

### Step 3 — Directional derivatives and normalization

```r
c6 <- diag(m5)               # optional; used for c55 in A/c
k5 <- length(c6)
c55 <- c5 / k5
c22 <- c5 - sum(c1 * c5)     # F_j
c222 <- c55 - 1              # used in logistic updates
c1 <- c1 / sum(c1)           # renormalize weights before update
```

### Step 4 — Multiplicative weight update

Active line varies by file (others left commented):

```r
c11 <- f(...)                # see per-criterion section
c12 <- c1 * c11
c1 <- c12 / sum(c12)         # new weights, sum to 1
k23 <- c(k23, max(c22))      # convergence monitor
```

### Step 5 — After the loop

Outputs final weights, convergence index `c100`, and diagnostic plots.

---

## 5. A-optimality (`R-code-A-optimality.txt`)

The file contains **two complete programs**, separated by a line of dots.

### 5.1 Block 1 — Simple linear regression (lines 1–86)

**Model:** $E(y \mid x) = \theta_0 + \theta_1 x$

**Criterion:** A-optimality minimizes $\mathrm{tr}(M^{-1}) = \sum_{i=1}^k \mathrm{Var}(\hat{\theta}_i)$.

**Parameters:**

- `k1 <- 0.9`
- `n <- 200`

**Active update** (SLR):

```r
c11 <- (exp(k1 * c222)) / (1 + exp(k1 * c222))
```

Uses **logistic** $f$ on `c222 = c5/k - 1`, not on raw `c5`. Commented alternatives:

```r
# c11 <- exp(k1 * c5)
# c11 <- exp(k1 * c222)
# c11 <- (exp(k1 * c55)) / (1 + exp(k1 * c55))
# c11 <- (exp(k1 * c222)) / (1 + exp(k1 * c222))  # active in SLR
```

**Comment `#400`:** Likely a note that 400 iterations were used in experiments (current code uses `n <- 200`).

### 5.2 Block 2 — Quadratic regression (lines 89–176)

**Model:** $E(y \mid x) = \theta_0 + \theta_1 x + \theta_2 x^2$

**Same structure** as Block 1 but `m2` has three rows and the **active update** changes to:

```r
c11 <- (exp(k1 * c5)) / (1 + exp(k1 * c5))
```

So QR uses logistic on **raw** A-derivatives `c5`, while SLR used logistic on **shifted** `c222`.

**Comment `#2000`:** Suggests longer runs (2000 iterations) were tried for QR.

### 5.3 Interpreting A-optimal results

- **Not scale-invariant:** Unlike D-optimality, A-optimal designs depend on how regressors are scaled (e.g. inches vs cm).
- **Plots:** `plot(c2, c1)` shows optimal weights over the grid; mass should concentrate on a **small** optimal support (often 3 points for QR on $[-1,1]$).
- **`plot(c2, c22)`:** Shows which grid points still want more weight (positive $F_j$).

---

## 6. c-optimality (`R-code-c-optimality.txt`)

### 6.1 Objective

Header comment:

```r
# c-optimality : min var(theta2^hat)
```

For quadratic model $(1, x, x^2)$, **`c = (0, 0, 1)`** estimates the **quadratic coefficient** $\theta_2$ with minimum variance:

$$
\mathrm{Var}(c^T \hat{\theta}) \propto c^T M(p)^{-1} c
$$

### 6.2 Implementation details

- `k1 <- 0.3` — smaller step than A/D scripts (stability for c-opt).
- `n <- 1` — **increase to 200+** for actual convergence from uniform start.
- Update:

```r
c11 <- (exp(k1 * c22)) / (1 + exp(k1 * c22))
```

Uses logistic on **directional derivative** `c22`, not on `c5`.

### 6.3 Documented answer (in-file comment)

```r
# Answer: c-optimal design for quadratic model with c=(0,0,1)
# (that is, for theta2 x^2) is x*=(-1,0,1), p*=(1/4,1/2,1/4).
# works with small values of delta, e.g., 0.3 etc.
```

So the **theoretical** three-point design on $\{-1, 0, 1\}$ with weights $(\tfrac{1}{4}, \tfrac{1}{2}, \tfrac{1}{4})$ should emerge when:

1. The grid includes $-1, 0, 1$ (it does with `by=0.1`),
2. `n` is large enough,
3. `k1` is small enough (e.g. 0.3).

### 6.4 Why `c5` is squared

Theory: $\partial \phi_c / \partial p_j = [c^T M^{-1} v_j]^2$. The code computes `c5 <- diag(m8)^2` after `m8 <- m7 %*% Cmat`.

---

## 7. D-optimality (`R-code-D-optimality.txt`)

### 7.1 Objective

**D-optimality** maximizes $\det M(p)$, equivalently minimizes $\det M(p)^{-1}$ (generalized variance of $\hat{\theta}$).

For QR on $[-1,1]$, the **analytic** D-optimal design uses support $\{-1, 0, 1\}$ with equal weights $1/3$ each (three parameters → three support points).

### 7.2 Implementation

```r
k1 <- 0.5
n <- 200
# ...
c5 <- diag(m7 %*% m2)   # v_j^T M^{-1} v_j
c22 <- c5 - sum(c1 * c5)
c11 <- exp(k1 * c5)      # exponential multiplicative update
```

**No** `c55` / `c222` in the active path (those lines are commented).

### 7.3 Extra convergence levels

D-optimality tracks six tolerances:

```r
c100 <- c(In1, In2, In3, In4, In5, In6)  # 10^{-1} … 10^{-6}
```

### 7.4 Connection to G-optimality

For D-optimal designs, the standardized prediction variance $d(x,p) = f(x)^T M^{-1} f(x)$ satisfies $\max_x d(x,p^*) = k$ (here $k=3$). The diagonal entries `c5` at support points relate to the equivalence check.

### 7.5 Handwritten PDF (missing from repo)

`Program_D-optimality_Details_Handwritten.pdf` likely walks through:

- Derivation of $d_j = v_j^T M^{-1} v_j$
- The Kiefer–Wolfowitz equivalence $D \Leftrightarrow G$
- Step-by-step hand calculation for quadratic/cubic models

If you obtain the PDF, place it in the repo root and cross-check numeric examples against `k23` and final `c1`.

**Analytic reference** (cubic, four points) from `OptimalDesign_Note4.tex`:

- Support: $\pm 1$, $\pm 1/\sqrt{5} \approx \pm 0.447$
- Weights: $0.25$ each

For **quadratic** ($k=3$), homework in the same note asks you to derive support $\{-1, 0, 1\}$, weights $1/3$ — what the D-code should approximate.

---

## 8. Post-processing and plots

Common tail in all scripts:

### 8.1 Design table

```r
c2c1 <- t(matrix(c(c2, c1), nrow = 2, ncol = J, byrow = TRUE))
xp <- round(c2c1, 3)
```

Rows = $(x_j, p_j)$ after optimization.

### 8.2 Convergence iteration

```r
In1 <- which(k23 <= 1e-1)[1]
# ...
c100 <- c(In1, In2, In3, In4)   # D adds In5, In6
```

`c100[r]` = first iteration where $\max_j F_j \leq 10^{-r}$. `NA` if threshold never met.

### 8.3 Plots

| Plot | Purpose |
|------|---------|
| `plot(1:n, k23)` | $\max F_j$ vs iteration — should decrease |
| `plot(c2, c1, type='l'/'b')` | Optimal weight function on grid |
| `plot(c2, c5)` | Partial derivatives across support |
| `plot(c2, c22)` | Directional derivatives — should be ≤ 0, ≈ 0 on support |

**Practical check:** After convergence, only a few `p_j` should be noticeably positive; others ≈ 0.

---

## 9. How to run and tune

### 9.1 Running in R

```r
# Example: D-optimality
source("docs/r-scripts/R-code-D-optimality.txt")   # or copy-paste into RStudio

# Inspect
xp      # support and weights
k23     # convergence trace
c100    # iteration counts to reach tolerances
```

### 9.2 Parameters to adjust

| Parameter | Effect |
|-----------|--------|
| `c2 <- seq(-1, 1, by=0.1)` | Finer grid → more support candidates, slower |
| `c1 <- rep(1/J, J)` | Starting design (uniform) |
| `k1` | Step size; too large → oscillation; too small → slow |
| `n` | Iterations; increase if `k23` still decreasing at end |
| Choice of `c11` line | Different $f$ — uncomment alternatives to experiment |

### 9.3 Troubleshooting

| Issue | Likely fix |
|-------|------------|
| Weights stay uniform | Increase `n`; adjust `k1`; check active `c11` line |
| `solve(m5)` fails | Support does not span parameter space — widen `c2` range or add points |
| c-opt does not match $(1/4, 1/2, 1/4)$ | Set `n <- 200`, keep `k1 <- 0.3` |
| A-opt differs between SLR and QR blocks | Expected — different models and different `c11` formulas |

---

## 10. Expected theoretical answers

| Problem | Criterion | Model | Known optimal support | Known weights |
|---------|-----------|-------|----------------------|---------------|
| SLR | A | $1, x$ | Problem-dependent on scaling | — |
| QR | A | $1, x, x^2$ | Depends on scaling (not D-invariant) | — |
| QR | c | $c=(0,0,1)$, estimate $\theta_2$ | $\{-1, 0, 1\}$ | $(1/4, 1/2, 1/4)$ |
| QR | D | all $\theta$ | $\{-1, 0, 1\}$ | $(1/3, 1/3, 1/3)$ |
| Cubic | D | $1, x, x^2, x^3$ | $\pm 1$, $\pm 1/\sqrt{5}$ | $1/4$ each |

Use these to validate numeric output from `xp` after sufficient iterations.

---

## 11. Relationship to the portfolio project

| Portfolio | Design theory |
|-----------|----------------|
| Asset weights | Design weights $p_j$ |
| Covariance / precision | Related to $M(p)$ and $M(p)^{-1}$ |
| Risk minimization | Variance / criterion minimization |

These R scripts are the computational core behind the design ↔ portfolio bridge: an optimal allocation of “effort” (`c1`) over locations (`c2`) is found by iterative improvement using derivatives of a criterion — the same structural idea as reallocating portfolio weights under constraints. See `portfolio/design_optimality.py`.


---

## Quick reference: which `c11` line is active?

| File | Active update |
|------|----------------|
| A-opt SLR | `c11 <- (exp(k1*c222))/(1+exp(k1*c222))` |
| A-opt QR | `c11 <- (exp(k1*c5))/(1+exp(k1*c5))` |
| c-opt | `c11 <- (exp(k1*c22))/(1+exp(k1*c22))` |
| D-opt | `c11 <- exp(k1*c5)` |

---

## Further reading in this repo

- `portfolio/design_optimality.py` — Python port of the multiplicative algorithm
- `examples/demo_design_optimality.py` — A/D/E verification + GMV bridge demo
- [RUNBOOK.md](../RUNBOOK.md) — how to run the demos

---

*Part of the URA optimal portfolio / optimal design research repository.*
