# PyBME Performance Guide

## Overview

PyBME fuses hard observations with soft probabilistic data (e.g., physics-model output) using Bayesian Maximum Entropy on network graphs. The computational bottleneck is the **Laplace integration** step, which must solve a Newton optimization problem at every point on a z-grid, for every prediction location.

This guide explains the three optimizations that reduced a real 1,400-point SWMM sewer-network prediction from **119 seconds to 1.1 seconds** (108x speedup), the equations behind them, and when each applies.

---

## The Problem: Why Soft Data Is Expensive

For each prediction point, BME evaluates a posterior integral over *ns* soft-data dimensions at each of *M* z-grid values. With the Laplace method, this means:

- **M = 100** z-grid points per prediction
- **5–20 Newton iterations** per z-grid point
- Each Newton step requires a **gradient** (ns PDF evaluations) and **Hessian** (3 × ns PDF evaluations via finite differences)
- Per prediction point: ~35,000 PDF evaluations
- For 1,400 predictions: **~49 million PDF evaluations**

Before optimization, each PDF evaluation went through piecewise-linear interpolation and finite-difference derivatives — all in Python scalar loops.

---

## Optimization 1: Analytic Derivatives for Gaussian and Truncated-Normal Soft PDFs

### What Changed

For a truncated normal on $[a,b]$ with mean $\mu$ and standard deviation $\sigma$,

$$
f(x) = \frac{\phi\!\left(\frac{x-\mu}{\sigma}\right)}{\sigma\,Z},
\qquad
Z = \Phi(\beta) - \Phi(\alpha),
\qquad
\alpha = \frac{a-\mu}{\sigma},
\quad
\beta = \frac{b-\mu}{\sigma},
$$

for $x \in [a,b]$, and $f(x)=0$ outside the support. Because $Z$ is constant with respect to $x$, the derivatives of $\log f(x)$ inside the support are the same as for a Gaussian kernel:

| Quantity | Formula | Cost |
|----------|---------|------|
| $\log f(x)$ | $-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2 - \log\!\bigl(\sigma\sqrt{2\pi}\bigr) - \log Z$ | constant-time |
| $\frac{d}{dx}\log f(x)$ | $-\frac{x - \mu}{\sigma^2}$ | 1 subtract + 1 divide |
| $\frac{d^2}{dx^2}\log f(x)$ | $-\frac{1}{\sigma^2}$ | **Constant** |

In the implementation, `SoftPDF.log_pdf(x)` returns `-inf` outside the stored support, and the Newton solver clamps iterates to that support. Inside the support, the derivatives above are exact for the implemented density.

The constant Hessian is the key insight. For the Laplace target

$$
\log g(x) = -\frac{1}{2}(x-\mu)^T Q (x-\mu) + \sum_{j=1}^{n_s} \log f_j(x_j),
$$

the gradient and Hessian are

$$
\nabla \log g(x) = -Q(x-\mu) +
\begin{bmatrix}
\frac{d}{dx_1}\log f_1(x_1) \\
\vdots \\
\frac{d}{dx_{n_s}}\log f_{n_s}(x_{n_s})
\end{bmatrix},
$$

$$
\nabla^2 \log g(x) = -Q + \operatorname{diag}\!\left(
\frac{d^2}{dx_1^2}\log f_1(x_1), \ldots,
\frac{d^2}{dx_{n_s}^2}\log f_{n_s}(x_{n_s})
\right).
$$

For Gaussian and truncated-normal soft factors,

$$
\nabla^2 \log g(x) = -Q - \operatorname{diag}\!\left(\frac{1}{\sigma_1^2}, \ldots, \frac{1}{\sigma_{n_s}^2}\right),
$$

so Newton's negative Hessian

$$
-H = Q + \operatorname{diag}\!\left(\frac{1}{\sigma_1^2}, \ldots, \frac{1}{\sigma_{n_s}^2}\right)
$$

never changes across z-grid points or Newton iterations. This means:

- **One matrix inverse** instead of 100 × 15 = 1,500
- **One log-determinant** instead of 1,500
- **Zero finite-difference evaluations** — the 49 million PDF calls are eliminated entirely

### When It Applies

The analytic path activates automatically when all soft PDFs in a prediction point's neighborhood were created with:

```python
SoftPDF.from_gaussian(mean, var)
SoftPDF.from_truncnorm(mu, sigma, a=0)      # one-sided truncation
SoftPDF.from_truncnorm(mu, sigma, a=0, b=10) # two-sided truncation
```

**It does not apply to:** lognormal, triangular, uniform, histogram, interval, mixture, or callable PDFs. Those fall back to finite-difference derivatives automatically.

The code path was numerically checked against SciPy and finite differences. For a representative truncated normal, the implementation matched SciPy's `truncnorm.logpdf` exactly, the first derivative matched central finite differences to about $2 \times 10^{-12}$, and the second derivative matched to finite-difference error tolerance.

### Practical Impact

| Metric | Before | After |
|--------|--------|-------|
| PDF evaluations per prediction | ~35,000 | 0 |
| Hessian inversions per prediction | ~1,500 | 1 |
| Integration inner loop | 231 ms | 0.12 ms |

---

## Optimization 2: Vectorized Laplace Batch

### What Changed

Instead of looping over *M* = 100 z-grid points in Python, the entire batch is processed simultaneously using NumPy broadcasting:

```
x:       (M, ns)     — all 100 mode vectors stacked
mu_grid: (M, ns)     — conditional means at each z-value
Q:       (ns, ns)    — precision matrix (same for all z)
neg_H:   (ns, ns)    — constant Hessian (same for all z and all iterations)
```

One Newton step for all 100 z-grid points:

```python
grad = -(d @ Q) - (x - sp_mu) * sp_inv_var   # (M, ns)
step = grad @ neg_H_inv                        # (M, ns)
x = np.clip(x + step, lo, hi)                 # (M, ns)
```

This replaces 100 separate Python-level Newton solves with a single batch of matrix operations handled by BLAS.

The Laplace approximation itself is

$$
I(\mu) = \mathbb{E}_{x \sim N(\mu,\Sigma)}\!\left[\prod_{j=1}^{n_s} f_j(x_j)\right]
\approx |Q|^{1/2} |{-H(x^*)}|^{-1/2} \exp\!\bigl(\log g(x^*)\bigr),
$$

where $Q = \Sigma^{-1}$ and $x^*$ is the mode of $\log g(x)$. In the vectorized path, every z-grid point has its own conditional mean row in `mu_grid`, but all rows share the same covariance, so the same $Q$, $-H$, inverse, and log-determinant are reused across the whole batch.

### When It Applies

The vectorized path is used when **all** soft PDFs in the neighborhood have analytic derivatives (`has_analytic_deriv == True`). This is the same condition as Optimization 1 — they work in tandem.

When any PDF lacks analytic derivatives, the solver falls back to the scalar warm-started Newton loop (still faster than the original due to warm-starting and cho_solve reuse).

### Practical Impact

Combined with analytic derivatives, the vectorized batch delivers a **1,919× speedup** on the integration inner loop (0.12 ms vs 231 ms).

---

## Optimization 3: Parallel Outer Loop

### What Changed

`bme_predict_network_st` accepts an `n_jobs` parameter for parallel dispatch over prediction points:

```python
results = bme_predict_network_st(
    ck_nodes, tk, ch_nodes, th, zh,
    cs_nodes, ts, soft_pdfs,
    net_cov_st,
    nhmax=30, nsmax=6,
    method="laplace",
    n_jobs=4,          # use 4 cores
)
```

Each prediction point is fully independent — its neighborhood selection, covariance assembly, kriging, and integration are self-contained. This makes the outer loop embarrassingly parallel.

### When to Use It

| Scenario | Recommendation |
|----------|---------------|
| < 50 prediction points | `n_jobs=1` — overhead exceeds benefit |
| 50–500 points, fast integration (hard-only or analytic soft) | `n_jobs=1` — serial is already fast |
| 500+ points with non-analytic soft PDFs | `n_jobs=N` where N = number of cores |
| Very large runs (10,000+ points) | `n_jobs=-1` (all cores) |

With the analytic+vectorized optimizations active, integration is so fast that per-point overhead is dominated by covariance assembly and kriging — parallelism gives marginal benefit for typical problem sizes. It becomes valuable when using non-analytic soft PDFs or very large prediction grids.

### Dependencies

For `bme_predict_network_st`, the current implementation uses `concurrent.futures.ProcessPoolExecutor` from the Python standard library and falls back to serial if parallel dispatch fails. No extra package is required for this path.

Other prediction APIs in PyBME still use `joblib` when `n_jobs > 1`, so the dependency story is API-specific.

---

## When to Use Each Integration Method

PyBME supports multiple integration methods via the `method` parameter:

| Method | Best For | Accuracy | Speed |
|--------|----------|----------|-------|
| `"laplace"` | Truncated-normal or Gaussian soft data | High; exact for ideal Gaussian factors, very accurate here | **Fast** with analytic path (1.1s / 1400 pts) |
| `"ep"` | Any soft PDF type; near-Gaussian posteriors | Good (approximate) | Fast (avoids Newton entirely) |
| `"gauss_hermite"` | Low-dimensional soft (ns ≤ 3) | High (quadrature) | Slow for ns > 3 |
| `"qmc"` | Moderate-dimensional soft data | Approximate | Moderate |
| `"lis"` | Difficult non-Gaussian soft integrals | Approximate | Moderate |
| `"auto"` | General use | Varies | Uses `"laplace"` when `ns >= 6`, else `"gauss_hermite"` |

**Recommendation for SWMM-type workflows** (truncated-normal soft data with physical non-negativity):

```python
method="laplace"  # uses the fast vectorized path when all local soft PDFs are analytic
```

---

## End-to-End Benchmark

**Problem:** 421-node sewer network, 17 flow meters, 1,400 truncated-normal soft PDFs, 1,400 prediction points (56 nodes × 25 hourly steps).

| Configuration | Wall Time | Relative |
|---------------|-----------|----------|
| Original (no neighborhood limits) | > 10 min | — |
| + nhmax=30, nsmax=6 | 119 s | 1× |
| + Analytic derivatives | — | — |
| + Vectorized batch | — | — |
| + All three optimizations | **1.1 s** | **108×** |
| Hard-data only (no soft) | 0.4 s | — |

The soft-data overhead is now only **0.7 seconds** for 1,400 predictions, down from 119 seconds.

---

## Quick-Start: Getting Maximum Performance

```python
from pybme.soft_data import SoftPDF
from pybme.predict import bme_predict_network_st

# 1. Use from_truncnorm or from_gaussian to get analytic derivatives
soft_pdfs = [
    SoftPDF.from_truncnorm(mu=q_swmm, sigma=sigma, a=0)
    for q_swmm, sigma in zip(swmm_values, uncertainties)
]

# 2. Keep neighborhoods tight
nhmax = 30   # hard-data neighbors
nsmax = 6    # soft-data neighbors

# 3. Use Laplace method
results = bme_predict_network_st(
    ck_nodes, tk, ch_nodes, th, zh,
    cs_nodes, ts, soft_pdfs,
    net_cov_st,
    nhmax=nhmax,
    nsmax=nsmax,
    n_grid=100,
    method="laplace",
)

# Optional: parallelize over prediction points
# results = bme_predict_network_st(..., method="laplace", n_jobs=4)
```

### Checklist

- [x] Soft PDFs created with `from_truncnorm()` or `from_gaussian()` → analytic path active
- [x] `nhmax` ≤ 30, `nsmax` ≤ 8 → small linear algebra per point
- [x] `n_grid` = 100 → sufficient resolution, vectorized batch handles it
- [x] `method="laplace"` and analytic local PDFs → vectorized Laplace fast path

### Verifying the Fast Path

Check that your soft PDFs have analytic support:

```python
assert all(sp.has_analytic_deriv for sp in soft_pdfs), \
    "Some PDFs lack analytic derivatives — will use slower FD fallback"
```

---

## Architecture Summary

```
bme_predict_network_st(n_jobs=...)
│
├── Per prediction point (parallel if n_jobs > 1):
│   ├── Stage A: Neighbourhood selection (covariance-ranked top-k)
│   ├── Stage B: Covariance block assembly
│   ├── Stage C: Kriging (hard data → prior mean + variance)
│   ├── Stage D: Conditional soft distribution (Schur complement)
│   ├── Stage E: Laplace integration <- where most of the speedup lives
│   │   ├── All soft PDFs analytic? -> _laplace_batch_vectorized()
│   │   │   ├── Constant Hessian: neg_H = Q + diag(1/sigma^2) [computed once]
│   │   │   ├── Vectorized Newton: (M, ns) arrays [all z-grid points at once]
│   │   │   └── Vectorized log-target evaluation
│   │   └── Mixed PDF types? -> scalar Newton with FD fallback (warm-started)
│   └── Stage F: Posterior PDF assembly (trapezoidal integration)
│
└── Collect BMEResult list
```

---

## Limitations & Future Work

- **Non-analytic PDFs** (lognormal, mixture, custom) still use the finite-difference fallback. For these, `method="ep"` may be faster.
- **Group-by-node caching** — prediction points sharing the same spatial node recompute identical spatial covariance blocks. Caching these would reduce Stages A–D cost by ~25× for structured prediction grids.
- **Precision-based batch kriging** — for hard-data-only portions, a single sparse solve could replace per-point Cholesky decompositions.
