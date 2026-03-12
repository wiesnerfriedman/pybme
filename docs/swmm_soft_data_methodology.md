# SWMM-as-Soft-Data: Network BME Methodology

## 1. Problem Statement

We have a sewer network with *N* = 421 nodes. At any (node *i*, time *t*):

- **Hard data** — real flow meter readings *z_h(i,t)*. Known exactly. Available at 17 meter locations.
- **Soft data** — SWMM hydraulic model output *Q_swmm(i,t)*. Known *approximately*. Available at all 421 nodes.

**Goal**: fuse the physically-based SWMM model with real observations using Bayesian Maximum Entropy (BME) on the network graph, producing a posterior estimate and uncertainty at every node and time step.

---

## 2. Mathematical Framework

### 2.1 BME Posterior

The BME posterior PDF at estimation point *(k, t_k)* is:

```
f(z_k | hard, soft)  proportional to  f_prior(z_k | hard) * E[ prod_j f_j(s_j) ]
```

where:
- `f_prior` is the kriging posterior given hard data (Gaussian)
- `f_j(s_j)` is the soft PDF at the *j*-th soft data point
- The expectation is over the conditional distribution of soft variables given hard data and *z_k*

### 2.2 Covariance Model

Separable space-time on the network graph:

```
C((i,t), (j,t')) = sigma^2 * rho_s(i,j) * rho_t(|t - t'|)
```

| Component          | Model                                                | Parameters                    |
|--------------------|------------------------------------------------------|-------------------------------|
| Spatial `rho_s`    | Graph-Laplacian: `(kappa^2 I + L)^{-1}`, normalized | kappa = 0.1, unit edge weights |
| Temporal `rho_t`   | Exponential: `exp(-h/a)`                             | range *a* = 6 hours            |
| Sill `sigma^2`     | Scaled to observed data variance                     | `Var(z_h) / mean(diag(C_net))` |

### 2.3 Soft PDF Construction

At each (node *i*, time *t*) where SWMM output exists:

```
f_i^soft(z) = TruncatedNormal(mu = Q_swmm(i,t),  sigma = sigma_i,  a = 0)
```

**Variance assignment** (`sigma_i^2`):
- At metered nodes: computed from time-series residuals `r(t) = Q_swmm(i,t) - z_obs(i,t)` -> `sigma_i^2 = Var(r)`
- At unmetered nodes: `sigma_i^2 = median({sigma_j^2} for j in meters)`
- Floor: `sigma_i^2 >= 0.01 MGD^2`

Truncation at `a = 0` enforces non-negative flow (physical constraint).

---

## 3. Algorithm: Per-Prediction-Point Pipeline

Each prediction point *(k, t_k)* goes through 7 stages. Below we score each on computational cost for the current problem dimensions:

```
nk = 1400 prediction points (56 nodes x 25 hourly)
nh_total = 297 hard observations
ns_total = 1400 soft PDFs
nhmax = 30 (hard neighbors per point)
nsmax = 6 (soft neighbors per point)
n_grid = 100 (z-grid resolution)
```

---

### Stage A: Neighbourhood Selection

**What**: For each prediction point, rank all hard/soft data by |Cov(k, data_i)| and select the top nhmax / nsmax.

**Operations per point**:
- Compute `net_cov_st(k, all_hard)` -> (1, 297) vector = 297 spatial lookups + temporal eval
- Compute `net_cov_st(k, all_soft)` -> (1, 1400) vector = 1400 spatial lookups + temporal eval
- Two partial argsorts

**Complexity**: `O(nh_total + ns_total)` per point; spatial lookup = O(1) array indexing into cached `C_dense`.

**Cost Score: 3/10** (~1697 index lookups + element-wise multiply + sort per point)

| Parallelizable? | Yes — each prediction point is independent. Embarrassingly parallel. |
|-----------------|----------------------------------------------------------------------|
| Approximation?  | Could use graph-distance cutoff instead of covariance ranking (avoid computing covariance to distant nodes). Spatial locality on graphs: nodes beyond ~5 hops have negligible covariance with kappa=0.1. |

---

### Stage B: Covariance Block Construction

**What**: Build the covariance sub-matrices needed for kriging and BME:
- `Ckh` (1 x nh) — estimation-to-hard
- `Chh` (nh x nh) — hard-to-hard
- `Cks` (1 x ns) — estimation-to-soft
- `Css` (ns x ns) — soft-to-soft
- `Chs` (nh x ns) — hard-to-soft

**Operations per point**:
- Each block: index `C_dense[idx1, idx2]`, multiply by `rho_t` element-wise
- Largest block: `Chh` at 30 x 30 = 900 entries
- Total entries: 30 + 900 + 6 + 36 + 180 = 1152

**Complexity**: `O(nh^2 + nh*ns + ns^2)` per point — dominated by the (30 x 30) Chh block.

**Cost Score: 2/10** (just array indexing into cached dense matrix + element-wise temporal eval)

| Parallelizable? | Yes — each point's blocks are independent. Could also pre-compute and cache `Chh` if the same hard neighbors recur across prediction points. |
|-----------------|----------------------------------------------------------------------|
| Approximation?  | **Tensor structure**: For prediction points sharing the same spatial node but different times, the spatial part of all blocks is identical — only temporal factors change. Could factor `C = sigma^2 * rho_s * rho_t` and cache `rho_s` blocks per unique spatial node (56 unique), then only multiply by temporal factors (25 time values). This would reduce covariance block construction by ~25x. |

---

### Stage C: Kriging (Hard Data Only)

**What**: Compute the kriging mean and variance from hard data:
```
k_mu = Ckh * Chh^{-1} * (zh - m)
k_var = sigma^2 - Ckh * Chh^{-1} * Ckh^T
```

**Operations per point**:
- Cholesky of Chh: `O(nh^3)` = `O(30^3)` = 27,000 FLOPS
- Two triangular solves: `O(nh^2)` each

**Complexity**: `O(nh^3)` per point.

**Cost Score: 2/10** (30^3 = 27K FLOPS is fast in BLAS)

| Parallelizable? | Yes — independent per point. |
|-----------------|----------------------------------------------------------------------|
| Approximation?  | If same hard neighbors recur, cache the Cholesky factor and reuse. For network BME, could use a **precision-based approach** (already implemented as `network_kriging_precision`) that solves one sparse linear system for ALL prediction nodes simultaneously — `O(N)` total instead of `O(nk * nh^3)`. |

---

### Stage D: Conditional Soft Distribution

**What**: Compute the conditional mean and covariance of soft data given (k, hard):
```
mu_s(z_k) = b_k * z_k + b_c
K_s|kh = Css - C_s,kh * C_kh,kh^{-1} * C_kh,s
```

**Operations per point**:
- Build `C_kh_kh` (31 x 31), Cholesky, cho_solve for B matrix
- Schur complement for `K_s|kh` (6 x 6)
- Soft product screening: `integrate_soft_product` — Gauss-Hermite with `n_quad=15`

**Complexity**: `O((1+nh)^3 + ns * (1+nh)^2)` per point.

**Cost Score: 3/10** (31^3 ~ 30K FLOPS + 6x31^2 matrix ops)

| Parallelizable? | Yes — independent per point. |
|-----------------|----------------------------------------------------------------------|
| Approximation?  | `K_s|kh` only depends on the *indices* of the selected hard+soft neighbors, not on the hard data values. If multiple prediction points share the same neighbor set (likely for same-node-different-time predictions), `K_s|kh` can be cached and reused. |

---

### Stage E: Laplace Integration (DOMINANT BOTTLENECK)

**What**: For each of M = `n_grid` = 100 z-values, evaluate:
```
I(z_k) = E_{s ~ N(mu_s(z_k), K_s|kh)}[ prod_j f_j(s_j) ]
```

using the Laplace approximation:

1. **Find mode** via Newton's method on `log g(x) = -0.5(x-mu)^T Q (x-mu) + sum_j log f_j(x_j)`
2. **Compute Hessian** at mode
3. **Laplace formula**: `I ~ |Q|^{1/2} |negH|^{-1/2} exp(log_g(x*))`

**Operations per z-grid point** (ns = 6):
- Newton iterations: ~5-20 (warm-started from previous z-point)
- Per Newton iteration:
  - Gradient: `O(ns)` = 6 PDF evaluations + `O(ns^2)` for Q*(x-mu)
  - Hessian: `O(ns)` = 18 PDF evaluations (3 per dimension for central FD) + copy of -Q
  - Linear solve: `O(ns^3)` = 216 FLOPS
  - Line search: 1-3 `_log_target` evaluations = 6-18 PDF evaluations each
- After convergence:
  - Hessian at mode: 18 PDF evaluations
  - Eigenvalue check: `O(ns^3)` = 216 FLOPS
  - slogdet: `O(ns^3)` = 216 FLOPS

**Per z-grid point total**: ~10 Newton iters * (~30 PDF evals + 432 FLOPS) + overhead ~ 350 PDF evals
**Per prediction point total**: 100 z-grid points * 350 = **35,000 PDF evaluations**
**All prediction points**: 1400 * 35,000 = **49 million PDF evaluations**

**Complexity**: `O(n_grid * n_newton * ns)` PDF evaluations per prediction point.

**Cost Score: 9/10** — **THIS IS THE BOTTLENECK.** Even with the separable optimization (diagonal Hessian), 49M PDF evaluations dominate the runtime.

| Parallelizable? | **YES — at multiple levels**: (1) The 100 z-grid points per prediction are independent (trivially parallelizable with `multiprocessing` or vectorization). (2) The 1400 prediction points are fully independent (embarrassingly parallel across cores/workers). |
|-----------------|----------------------------------------------------------------------|
| Approximation?  | **Multiple opportunities**: |
|                 | 1. **Analytic Laplace for truncated normals**: Since `f_j` is TruncNormal, `log f_j(x_j)` has a closed-form Gaussian + log-erf expression. Both gradient and Hessian can be computed *analytically* — no finite differencing needed. This eliminates ALL 49M PDF evaluations and replaces them with fast `erfc` calls. **Estimated speedup: 10-50x.** |
|                 | 2. **Taylor expansion of I(z_k)**: Since `mu_s(z_k) = b_k * z_k + b_c` is *linear* in z_k, and the Laplace integral is smooth in the conditioning mean, one could compute I(z) at a few z-values and interpolate with a polynomial/spline. Instead of 100 Laplace solves, ~5-10 may suffice. **Estimated speedup: 10-20x.** |
|                 | 3. **Moment-matching shortcut**: For near-Gaussian soft data (which truncated normals with mu >> 0 are), the soft data integral can be approximated by matching moments — reducing the problem to a multivariate Gaussian integral with known closed form. No Newton needed. **Estimated speedup: 100x+.** |
|                 | 4. **EP (Expectation Propagation)**: Already implemented as `method="ep"`. Avoids Newton entirely by iteratively fitting Gaussian approximations to each soft factor. Often 5-10x faster than Laplace for smooth soft PDFs. |

---

### Stage F: Posterior PDF Assembly

**What**: Combine prior with Laplace integral values:
```
f(z_k) = prior(z_k) * I(z_k) / integral
```
Extract mean, variance, CI via trapezoidal integration over z-grid.

**Operations per point**:
- Element-wise multiply: `O(n_grid)` = 100
- Trapezoidal integration: `O(n_grid)` = 100
- CDF + searchsorted for CI: `O(n_grid)` = 100

**Complexity**: `O(n_grid)` per point.

**Cost Score: 1/10** (trivial)

| Parallelizable? | Yes, trivially vectorizable. |
|-----------------|----------------------------------------------------------------------|
| Approximation?  | If the posterior is nearly Gaussian (which it often is for flow data), could skip the z-grid entirely and compute mean/variance from the Laplace mode + curvature analytically. |

---

### Stage G: Outer Loop Over Prediction Points

**What**: Sequential Python loop over 1400 prediction points.

**Cost Score: 5/10** — not for per-point compute, but for **Python loop overhead** and **redundant covariance evaluation** across points that share the same spatial node.

| Parallelizable? | **Yes — embarrassingly parallel.** All 1400 points are independent. `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel` would give near-linear speedup with N_cores. |
|-----------------|----------------------------------------------------------------------|
| Approximation?  | **Group-by-node strategy**: The 1400 points are 56 nodes x 25 times. For each unique node, the spatial covariance to all data points is the same — only temporal factors differ. Restructuring the loop to iterate over 56 nodes (outer) x 25 times (inner) and caching spatial covariance blocks would eliminate ~24/25 = 96% of redundant spatial covariance lookups. |

---

## 4. Cost Summary

| Stage | Description                    | Score | Calls/Point | Total Calls (1400 pts) | Parallelizable? |
|-------|-------------------------------|-------|-------------|------------------------|-----------------|
| A     | Neighbourhood selection       | 3/10  | 1           | 1,400                  | Yes             |
| B     | Covariance block construction | 2/10  | 5 blocks    | 7,000                  | Yes (cacheable) |
| C     | Kriging (hard only)           | 2/10  | 1 Cholesky  | 1,400                  | Yes (cacheable) |
| D     | Conditional soft distribution | 3/10  | 1           | 1,400                  | Yes (cacheable) |
| **E** | **Laplace integration**       | **9/10** | **100 Newton solves** | **140,000** | **Yes (multi-level)** |
| F     | Posterior PDF assembly        | 1/10  | 1           | 1,400                  | Yes             |
| G     | Outer Python loop             | 5/10  | —           | 1,400 iterations       | Yes             |

**Stage E (Laplace integration) accounts for an estimated 70-85% of total wall-clock time.**

---

## 5. Optimization Roadmap

### Already Implemented (v0.5.0+)
- [x] Neighbourhood selection (nhmax/nsmax enforcement)
- [x] Separable Hessian (diagonal soft-PDF contribution, no cross-terms)
- [x] Warm-start Newton (previous z-grid solution as initial guess)
- [x] cho_solve reuse (avoids redundant matrix inversion)
- [x] Reduced n_grid (200 -> 100)

### High-Impact Opportunities (Not Yet Implemented)

| Priority | Optimization | Expected Speedup | Complexity |
|----------|-------------|-------------------|------------|
| **1**    | Analytic grad/Hessian for TruncNormal soft PDFs | 10-50x on Stage E | Medium — need closed-form d/dx log(TruncNormal) |
| **2**    | Parallel outer loop (multiprocessing/joblib) | Nx for N cores | Low — embarrassingly parallel |
| **3**    | Taylor/spline interpolation of I(z_k) over z-grid | 10-20x on Stage E | Medium — need smoothness analysis |
| **4**    | Group-by-node caching (spatial cov blocks) | ~25x on Stages A-D | Medium — restructure outer loop |
| **5**    | Moment-matching for near-Gaussian soft data | 100x+ on Stage E | High — new integration path |
| **6**    | Precision-based batch kriging (sparse solve) | Replace Stage C entirely | Medium — already have `network_kriging_precision` |
| **7**    | EP integration method (already implemented) | 5-10x vs Laplace | Low — just change `method="ep"` |

---

## 6. Current Problem Dimensions

| Quantity                          | Count |
|-----------------------------------|-------|
| Network nodes                     | 421   |
| Network edges                     | 491   |
| Flow meter locations              | 17    |
| Hard data points (meters x times) | 297   |
| Soft data points (nodes x times)  | 1400  |
| Prediction points                 | 1400  |
| Hard neighbors per point (nhmax)  | 30    |
| Soft neighbors per point (nsmax)  | 6     |
| z-grid resolution                 | 100   |
| Newton iterations per z-point     | 5-20  |
| Temporal range                    | 6 hours |
| Analysis window                   | 24 hours (hourly steps) |

---

## 7. File References

| File | Role |
|------|------|
| `examples/swmm_soft_data_bme.py` | Main example script |
| `src/pybme/predict.py` | `bme_predict_network_st`, `_bme_network_st_point` |
| `src/pybme/integration.py` | `integrate_soft_laplace_batch`, `_find_mode`, gradient/Hessian |
| `src/pybme/network.py` | `NetworkCovariance`, `NetworkCovarianceST` |
| `src/pybme/soft_data.py` | `SoftPDF`, `from_truncnorm` |
| `src/pybme/network_plots.py` | Visualization functions |
