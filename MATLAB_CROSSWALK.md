# PyBME ↔ MATLAB BMElib 2.0 — Crosswalk & Getting Started

**Prepared by:** Corinne Wiesner-Friedman
**For:** Marc Serre — creator of MATLAB BMElib

PyBME is a pure-Python port of the core algorithms in MATLAB BMElib 2.0.
This document maps MATLAB functions to their Python equivalents, shows
side-by-side code examples, and explains how to install and run it.

---

## Table of Contents

1. [Quick Start — Install & First Prediction](#1-quick-start)
2. [Function Crosswalk — MATLAB → Python](#2-function-crosswalk)
3. [Side-by-Side Code Examples](#3-side-by-side-code-examples)
4. [What's New in Python (Not in MATLAB BMElib)](#4-whats-new-in-python)
5. [What's Not Ported (MATLAB Only)](#5-whats-not-ported)
6. [Tutorials](#6-tutorials)

---

## 1. Quick Start

### Installation (one command)

```bash
pip install -e ".[dev]"
```

That's it — no MEX compilation, no MATLAB license, no Fortran compiler.
Requires Python ≥ 3.10, NumPy, and SciPy (installed automatically).

### First Prediction (5 lines)

```python
import numpy as np
from pybme import bme_predict, SoftPDF

# Hard data: 5 points with exact measurements
ch = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]])
zh = np.array([3.2, 2.8, 3.5, 2.1, 3.0])

# Soft data: 2 points with uncertain (Gaussian) measurements
cs = np.array([[0.3, 0.7], [0.8, 0.2]])
soft_pdfs = [SoftPDF.from_gaussian(mean=3.1, var=0.5),
             SoftPDF.from_gaussian(mean=2.5, var=0.8)]

# Predict at a new location — returns mode, mean, variance,
# skewness, full posterior PDF, and confidence interval
results = bme_predict(
    ck=np.array([[0.5, 0.5]]),    # estimation point
    ch=ch, zh=zh,                  # hard data
    cs=cs, soft_pdfs=soft_pdfs,    # soft data
    model="exponential",           # covariance model
    params=[1.0, 0.5],             # [sill, range]
)

r = results[0]
print(f"BME mean  = {r.mean:.3f}")
print(f"BME mode  = {r.mode:.3f}")
print(f"Variance  = {r.variance:.3f}")
print(f"95% CI    = [{r.ci_lower:.3f}, {r.ci_upper:.3f}]")
```

---

## 2. Function Crosswalk

### Covariance Models (MATLAB `modelslib` → Python `pybme.covariance`)

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `exponentialC(D, [c a])` | `exponential_cov(h, [sill, range])` | Same formula: C(h) = c·exp(−3h/a) |
| `gaussianC(D, [c a])` | `gaussian_cov(h, [sill, range])` | Same formula |
| `sphericalC(D, [c a])` | `spherical_cov(h, [sill, range])` | Same formula |
| `nuggetC(D, [c])` | `nugget_cov(h, [sill])` | Same formula |
| `holecosC(D, [c a])` | `hole_cos_cov(h, [sill, range])` | Same formula |
| — | `matern_cov(h, [sill, range, nu])` | **New** — Matérn family |
| `exponentialV`, `gaussianV`, etc. | — | Variograms not explicitly ported; use γ(h) = C(0) − C(h) |
| `coord2K(ck, ch, model, param)` | `build_cov_matrix(c1, c2, model, params)` | Same purpose — builds covariance matrix |
| `coord2dist(c1, c2)` | `coord2dist(c1, c2)` | **Same name** — Euclidean distance matrix |
| `modelsyntax` | — | Not needed — Python has docstrings & help() |

**Nested/additive models** work the same way:

```matlab
% MATLAB:  nested exponential + nugget
covmodel = 'exponentialC/nuggetC';
covparam = [0.8, 5.0, 0.2];
```

```python
# Python:  nested exponential + nugget
model  = ["exponential", "nugget"]
params = [[0.8, 5.0], [0.2]]
```

### Space-Time Covariance

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `gaussianCST(Ds, Dt, [c as at])` | `gaussian_cov_st(hs, ht, [sill, range_s, range_t])` | Direct port |
| `nuggetCST(Ds, Dt, [c])` | `nugget_cov_st(hs, ht, [sill])` | Direct port |
| `'exponentialC/exponentialC'` (separable) | `build_cov_matrix_st(..., model_s, params_s, model_t, params_t, sigma2)` | Separable kernel: σ²·Cs(h)·Ct(τ) |

---

### BME Prediction (MATLAB `bmeprobalib` → Python `pybme.predict`)

This is the biggest simplification — **4 MATLAB functions become 1 Python function**:

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `BMEprobaPdf(z, ck, ch, cs, zh, softpdftype, nl, limi, probdens, covmodel, covparam, nhmax, nsmax, dmax, order, options)` | `bme_predict(ck, ch, zh, cs, soft_pdfs, model, params, ...)` | Returns full PDF in `result.z_grid`, `result.pdf` |
| `BMEprobaMode(...)` | (same call) | Returns `result.mode` |
| `BMEprobaMoments(...)` | (same call) | Returns `result.mean`, `result.variance`, `result.skewness` |
| `BMEprobaCI(...)` | (same call) | Returns `result.ci_lower`, `result.ci_upper` |
| `BMEoptions` | keyword arguments | No separate options struct needed |
| `BMEprobaMomentsXvalidation(...)` | `cross_validate(ch, zh, cs, soft_pdfs, ...)` | Leave-one-out cross-validation |

**Key simplification:** In MATLAB you call 4 separate functions and manage
`softpdftype`, `nl`, `limi`, `probdens` arrays.  In Python you call
`bme_predict()` once and get everything in a `BMEResult` object.

---

### Soft Data (MATLAB `bmeprobalib` → Python `pybme.SoftPDF`)

In MATLAB BMElib, soft data is described by a numeric `softpdftype` code
(1–4) and separate arrays `nl`, `limi`, `probdens`.  In Python, each soft
datum is a `SoftPDF` object with a named constructor:

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `softpdftype=1` (histogram) | `SoftPDF.from_histogram(breaks, densities)` | Same representation |
| `softpdftype=2` (linear interp) | `SoftPDF.from_linear(z_grid, pdf_values)` | Same representation |
| `softpdftype=3` (grid + table) | `SoftPDF.from_linear(z_grid, pdf_values)` | Merged into type 2 |
| `softpdftype=4` (grid + table) | `SoftPDF.from_linear(z_grid, pdf_values)` | Merged into type 2 |
| `probaGaussian(...)` | `SoftPDF.from_gaussian(mean, var)` | Same distribution |
| `probaUniform(...)` | `SoftPDF.from_uniform(a, b)` | Same distribution |
| `probaTriangular(...)` | `SoftPDF.from_triangular(a, mode, b)` | Same distribution |
| `probaStudentT(...)` | — | Not ported |
| — | `SoftPDF.from_truncnorm(mu, sigma, a, b)` | **New** — truncated Gaussian |
| — | `SoftPDF.from_lognormal(mu_log, sigma_log)` | **New** — lognormal |
| — | `SoftPDF.from_callable(func, a, b)` | **New** — any Python function |
| — | `SoftPDF.from_mixture(components, weights)` | **New** — mixture of PDFs |
| `proba2stat(softpdftype, nl, limi, probdens)` | `sp.moments()` → `(mean, var)` | Method on the SoftPDF object |
| `proba2val(z, softpdftype, nl, limi, probdens)` | `sp.evaluate(z)` | Method on the SoftPDF object |
| `probaplot(...)` | `plt.plot(sp.z_grid, sp.pdf_values)` | Use matplotlib directly |

---

### BME with Interval Data (MATLAB `bmeintlib` → Python `pybme`)

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `BMEintervalMode(...)` | `bme_predict(cs=cs, soft_pdfs=[SoftPDF.from_interval(a,b), ...])` | Intervals are just a SoftPDF type |
| `BMEintervalPdf(...)` | (same call — `.pdf` attribute) | |

In Python, intervals aren't a separate library — they're just another
`SoftPDF` constructor.  Use `SoftPDF.from_interval(lower, upper)` and pass
it to the same `bme_predict()`.

---

### Hard-Data Kriging (MATLAB `bmehrlib` → Python `pybme`)

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `kriging(ck, ch, zh, covmodel, covparam, nhmax, dmax, order)` | `bme_predict(ck, ch, zh, model=..., params=...)` | Just omit `cs` and `soft_pdfs` — falls back to kriging |
| `krigingME(...)` | — | Measurement errors not separately modeled |
| `krigingMP(...)` | — | Multi-point kriging not ported |
| `krigingT(...)` | — | Kriging with transforms not ported |

---

### Neighbourhood, Trend, Fitting

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `neighbours(ck, c_data, n, dmax)` | `select_neighbors(ck, c_data, n, nmax, dmax)` | Same purpose |
| `probaneighbours(...)` | `select_neighbors(...)` (for soft data too) | Unified function |
| — | `SpatialIndex(coords)` | **New** — KD-tree O(log N) lookup |
| — | `SpatialTemporalIndex(coords, times)` | **New** — KD-tree S/T lookup |
| `designmatrix(c, order)` | `design_matrix(coords, order)` | Same purpose |
| `modelfit(...)` | `fit_covariance(ch, zh, model, ...)` | MATLAB: weighted LS; Python: REML |

---

### Integration Engine (MATLAB `mvnlib` → Python `pybme.integration`)

| MATLAB BMElib | PyBME | Notes |
|---|---|---|
| `mvPro` / `mvProAG2` (Fortran MEX) | `integrate_soft_product(...)` | GH quadrature (≤ 8 dims) + MC fallback |
| — | `integrate_soft_laplace(...)` | **New** — O(ns³) Laplace approximation |
| — | `integrate_soft_ep(...)` | **New** — Expectation Propagation |
| — | `integrate_soft_qmc(...)` | **New** — Quasi-Monte Carlo (Sobol) |
| — | `integrate_soft_lis(...)` | **New** — Laplace Importance Sampling |

These are controlled by the `method` parameter in `bme_predict()`:

| `method=` | Algorithm | When to use |
|---|---|---|
| `"auto"` *(default)* | Laplace if ns ≥ 6, else GH | General purpose |
| `"gauss_hermite"` | Tensor-product Gauss-Hermite | ns ≤ 5, exact posterior |
| `"laplace"` | Laplace approximation | Many soft neighbours (ns ≫ 8) |
| `"ep"` | Expectation Propagation | Non-log-concave soft PDFs |
| `"qmc"` | Quasi-Monte Carlo (Sobol) | Moderate ns, better convergence than MC |
| `"lis"` | Laplace Importance Sampling | Unbiased with low variance |
| `"mc"` | Plain Monte Carlo | Very high dimensions, diagnostics |

---

## 3. Side-by-Side Code Examples

### Example 1: BME Prediction with Soft Data

**MATLAB BMElib:**

```matlab
% Load data
ch = [0 0; 1 0; 0 1; 1 1; 0.5 0.5];
zh = [3.2; 2.8; 3.5; 2.1; 3.0];
cs = [0.3 0.7; 0.8 0.2];

% Set up soft data (softpdftype=2, linear interpolation)
softpdftype = 2;
nl  = [3; 3];
limi   = [1.5 3.1 4.5;
          1.0 2.5 4.0];
probdens = [0 0.5 0;
            0 0.4 0];

% Covariance model
covmodel = 'exponentialC';
covparam = [1.0, 0.5];

% Parameters
nhmax = 10;  nsmax = 3;  dmax = 100;  order = 0;
options = BMEoptions;

% Get moments (mean, variance) at estimation grid
ck = [0.5 0.5];
[moments, info] = BMEprobaMoments(ck, ch, cs, zh, ...
    softpdftype, nl, limi, probdens, ...
    covmodel, covparam, nhmax, nsmax, dmax, order, options);
zk = moments(:,1);  % mean
vk = moments(:,2);  % variance

% Get full posterior PDF
[z, pdf, info] = BMEprobaPdf([], ck, ch, cs, zh, ...
    softpdftype, nl, limi, probdens, ...
    covmodel, covparam, nhmax, nsmax, dmax, order, options);

% Get confidence interval
options(20) = 0.95;
[zlCI, zuCI] = BMEprobaCI(ck, ch, cs, zh, ...
    softpdftype, nl, limi, probdens, ...
    covmodel, covparam, nhmax, nsmax, dmax, order, options);

fprintf('Mean=%.3f  Var=%.3f  CI=[%.3f, %.3f]\n', zk, vk, zlCI, zuCI);
```

**PyBME (Python):**

```python
import numpy as np
from pybme import bme_predict, SoftPDF

# Hard data
ch = np.array([[0,0],[1,0],[0,1],[1,1],[0.5,0.5]])
zh = np.array([3.2, 2.8, 3.5, 2.1, 3.0])

# Soft data — each point is a SoftPDF object
cs = np.array([[0.3, 0.7], [0.8, 0.2]])
soft_pdfs = [
    SoftPDF.from_linear([1.5, 3.1, 4.5], [0, 0.5, 0]),  # same as softpdftype=2
    SoftPDF.from_linear([1.0, 2.5, 4.0], [0, 0.4, 0]),
]

# Single call returns everything: mean, mode, variance, PDF, CI
r = bme_predict(
    ck=np.array([[0.5, 0.5]]),
    ch=ch, zh=zh,
    cs=cs, soft_pdfs=soft_pdfs,
    model="exponential", params=[1.0, 0.5],
    nhmax=10, nsmax=3,
)[0]

print(f"Mean={r.mean:.3f}  Var={r.variance:.3f}  "
      f"CI=[{r.ci_lower:.3f}, {r.ci_upper:.3f}]")
```

**Key differences:**
- **1 function** instead of 3–4 separate calls
- No `softpdftype`/`nl`/`limi`/`probdens` arrays — each soft datum is a named object
- No `options` vector with magic index numbers — keyword arguments instead
- Result is a named object (`r.mean`, `r.variance`) instead of positional arrays

---

### Example 2: Covariance Models

**MATLAB BMElib:**

```matlab
h = linspace(0, 1.5, 300);
C_exp = exponentialC(h, [1.0, 1.0]);
C_sph = sphericalC(h, [1.0, 1.0]);
C_gau = gaussianC(h, [1.0, 1.0]);
plot(h, C_exp, h, C_sph, h, C_gau);
legend('Exponential', 'Spherical', 'Gaussian');
```

**PyBME (Python):**

```python
import numpy as np
from pybme import exponential_cov, spherical_cov, gaussian_cov

h = np.linspace(0, 1.5, 300)
C_exp = exponential_cov(h, [1.0, 1.0])
C_sph = spherical_cov(h, [1.0, 1.0])
C_gau = gaussian_cov(h, [1.0, 1.0])

import matplotlib.pyplot as plt
plt.plot(h, C_exp, h, C_sph, h, C_gau)
plt.legend(['Exponential', 'Spherical', 'Gaussian'])
plt.show()
```

**Nearly identical** call signatures — just different naming convention
(`exponentialC` → `exponential_cov`).

---

### Example 3: Space-Time BME

**MATLAB BMElib:**

```matlab
covmodel = 'exponentialC/exponentialC';
covparam = [1.0, 5.0, 10.0];
[moments, info] = BMEprobaMoments(ck, ch, cs, zh, ...
    softpdftype, nl, limi, probdens, ...
    covmodel, covparam, nhmax, nsmax, dmax, order, options);
```

**PyBME (Python):**

```python
results = bme_predict_st(
    ck=ck, tk=tk,         # estimation coords + times
    ch=ch, th=th, zh=zh,  # hard data coords + times + values
    cs=cs, ts=ts,         # soft data coords + times
    soft_pdfs=soft_pdfs,
    model_s="exponential", params_s=[1.0, 5.0],   # spatial model
    model_t="exponential", params_t=[1.0, 10.0],  # temporal model
    sigma2=1.0,                                     # overall variance
)
```

---

## 4. What's New in Python (Not in MATLAB BMElib)

These are original contributions that extend BMElib's capabilities:

| Feature | What it does | Why it matters |
|---|---|---|
| **Matérn covariance** | Flexible smoothness parameter ν | Industry standard; generalises exponential (ν=0.5) and Gaussian (ν→∞) |
| **Laplace approximation** | O(ns³) integration for soft data | Handles 10–100+ soft neighbours without exponential cost blowup |
| **Expectation Propagation** | Iterative Gaussian-site approximation | Better than Laplace for hard-edged PDFs (uniforms, intervals, bimodals) |
| **Quasi-Monte Carlo** | Sobol sequences for soft-data integral | O(1/N) convergence vs O(1/√N) for plain MC — much more efficient |
| **Laplace Importance Sampling** | Importance sampling with Laplace proposal | Unbiased correction to Laplace with low variance |
| **KD-tree neighbourhood** | O(log N) neighbour lookup | Fast neighbourhood for large datasets (10,000+ points) |
| **REML covariance fitting** | `fit_covariance()` — auto-fit sill, range, nugget | Replaces manual variogram fitting |
| **SPDE / GMRF kriging** | Sparse-precision Matérn kriging on FEM mesh | O(n^{3/2}) vs O(n³) for large 2-D fields (hard data only) |
| **Truncated-normal, lognormal, callable, mixture soft PDFs** | Additional SoftPDF constructors | More flexible soft data representation |
| **Cross-validation** | `cross_validate()` — built-in LOO | One function call vs manual scripting |

---

## 5. What's Not Ported (MATLAB Only)

| MATLAB Library | Functions | Status |
|---|---|---|
| `simulib` | `simuchol`, `simuseq`, conditional simulation | Not ported — use `scipy` or `gstools` for simulation |
| `graphlib` | `colorplot`, `valplot`, `probaplot`, etc. | Not ported — use `matplotlib` directly |
| `iolib` | `readGeoEAS`, `writeGeoEAS`, etc. | Not ported — use `pandas.read_csv()`, etc. |
| `mvnlib` | Fortran MEX routines | Replaced by NumPy/SciPy equivalents |
| `bmecatlib` | Categorical BME | Not ported |
| `statlib` (partial) | Variogram estimation (`vario`, `crossvario`), Gaussian transforms (`other2gauss`), distribution PDFs | Not ported — use `scipy.stats`, `scikit-gstat` |
| `bmehrlib` (partial) | `krigingME`, `krigingMP`, `krigingT` | Not as separate functions — basic kriging is embedded in `bme_predict` |
| `bmeprobalib` (partial) | `probaStudentT`, transform variants (`BMEprobaTMode`, etc.) | Not ported |

---

## 6. Tutorials

Each MATLAB tutorial has a corresponding Python tutorial:

| MATLAB Tutorial | Python Command | Topic |
|---|---|---|
| `MODELSLIBtutorial.m` | `python -m pybme.tutorials.tutorial_models` | Covariance models |
| `BMEPROBALIBtutorial.m` | `python -m pybme.tutorials.tutorial_bme_proba` | BME with soft PDFs |
| `BMEINTLIBtutorial.m` | `python -m pybme.tutorials.tutorial_bme_interval` | BME with intervals |
| `BMEHRLIBtutorial.m` | `python -m pybme.tutorials.tutorial_kriging` | Kriging (hard only) |
| `STATLIBtutorial.m` | `python -m pybme.tutorials.tutorial_statistics` | Statistics & variograms |
| `GENLIBtutorial.m` | `python -m pybme.tutorials.tutorial_genlib` | Grid, NN, smoothing |

Run any tutorial from the command line — no IDE needed.  Each prints
results to the console and saves plots (if matplotlib is installed).

---

## Architecture Summary

```
MATLAB BMElib                          PyBME
─────────────                          ─────
modelslib/                       →     covariance.py
  exponentialC, gaussianC, ...           exponential_cov, gaussian_cov, ...
  coord2K                                build_cov_matrix
                                  
bmeprobalib/                     →     predict.py  +  soft_data.py
  BMEprobaPdf                            bme_predict() → result.pdf
  BMEprobaMode                           bme_predict() → result.mode
  BMEprobaMoments                        bme_predict() → result.mean, .variance
  BMEprobaCI                             bme_predict() → result.ci_lower/upper
  probaGaussian, probaUniform            SoftPDF.from_gaussian(), .from_uniform()
  softpdftype/nl/limi/probdens           SoftPDF objects
  BMEoptions                             keyword arguments

bmeintlib/                       →     predict.py  +  soft_data.py
  BMEintervalMode/Pdf                    bme_predict(soft_pdfs=[SoftPDF.from_interval(...)])

bmehrlib/                        →     predict.py
  kriging                                bme_predict() with no soft data
                                  
genlib/                          →     distance.py  +  neighborhood.py  +  trend.py
  coord2dist                             coord2dist
  neighbours                             select_neighbors + SpatialIndex
  designmatrix                           design_matrix

statlib/ (partial)               →     fitting.py  +  validation.py
  modelfit                               fit_covariance (REML)
  BMEprobaMomentsXvalidation             cross_validate

mvnlib/ (Fortran MEX)            →     integration.py
  mvPro, mvProAG2                        integrate_soft_product (+ 5 new methods)

— (none) —                       →     spde.py          (NEW — GMRF kriging)
```

---

## Test Suite

PyBME includes 105 automated tests covering all modules.  Run them with:

```bash
cd pybme
pytest
```

All tests pass and verify numerical agreement between the Python and
MATLAB implementations for standard BME workflows.
