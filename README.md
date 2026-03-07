# PyBME — Bayesian Maximum Entropy Geostatistical Library

**Author:** Corinne Wiesner-Friedman

A pure-Python implementation of the Bayesian Maximum Entropy (BME) framework for spatial and space-time geostatistical estimation.  PyBME is a modern port of the core algorithms in the [MATLAB BMElib 2.0](https://mserre.sph.unc.edu/BMElab_web/) library and includes:

* **True non-Gaussian soft-data integration** via Gauss-Hermite tensor-product quadrature (replaces moment-matching).
* **10+ soft-data types** — Gaussian, uniform / interval, triangular, truncated-normal, lognormal, histogram, callable, mixture.
* **6 covariance models** — exponential, Gaussian, spherical, Matérn, nugget, hole-cosine — with nesting support.
* **Neighbourhood selection**, polynomial trend (order 0/1/2 / simple kriging), REML covariance fitting, leave-one-out cross-validation.
* **Separable space-time BME** with independent spatial and temporal covariance models.
* **SPDE / GMRF sparse-precision kriging** — Matérn fields on FEM meshes with O(n^{3/2}) Cholesky; hard-data kriging only (see [limitations](#when-to-use-which-approach)). *(v0.3.0, original contribution)*
* **Laplace approximation** for soft-data integration within `bme_predict` — O(ns³) per point, replacing exponential-cost GH quadrature when ns is large. *(v0.3.0, original contribution)*
* **Expectation Propagation (EP)**, **Quasi-Monte Carlo (QMC)**, and **Laplace Importance Sampling (LIS)** — three additional integration methods for diverse soft-data scenarios. *(v0.4.0, original contribution)*

---

## Installation

```bash
# From the pybme/ directory:
pip install -e ".[dev]"     # editable install with pytest + matplotlib
```

Requirements: Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.  Matplotlib is optional (only for plotting).

---

## Attribution

PyBME is a Python port of **BMElib 2.0**, the MATLAB library for
**Bayesian Maximum Entropy** (BME) — an information-theoretic geostatistical
framework that maximises Shannon entropy subject to physical knowledge
constraints.  BMElib was developed by **Marc L. Serre**, **Patrick Bogaert**, and
**George Christakos** at the University of North Carolina at Chapel Hill.

If you use PyBME in published work please cite the original BMElib and the
foundational BME references:

> Serre M.L. & Christakos G. (1999).  Modern geostatistics for environmental
> and health sciences: BMElib.  *Stochastic Environmental Research and Risk
> Assessment*, **13**, 1–26.  <https://doi.org/10.1007/s004770050030>

> Christakos G. (1990).  A Bayesian/maximum-entropy view to the spatial
> estimation problem.  *Mathematical Geology*, **22**(7), 763–777.
> <https://doi.org/10.1007/BF00890661>

> Christakos G. (2000).  *Modern Spatiotemporal Geostatistics*.  Oxford
> University Press, New York.

> Christakos G., Bogaert P. & Serre M.L. (2002).  *Temporal GIS: Advanced
> Functions for Field-Based Applications*.  Springer-Verlag, Berlin.
> <https://doi.org/10.1007/978-3-642-56540-3>

BMElib homepage: <http://www.unc.edu/depts/case/BMElib/>

### INLA-SPDE extensions (v0.3.0)

The SPDE/GMRF sparse-precision module and Laplace approximation for
soft-data integration are **original contributions by Corinne Wiesner-Friedman**
and are not part of the original MATLAB BMElib.  These features are inspired by
the INLA-SPDE methodology.  If you use them please also cite:

> Lindgren F., Rue H. & Lindström J. (2011).  An explicit link between Gaussian
> fields and Gaussian Markov random fields: the stochastic partial differential
> equation approach.  *Journal of the Royal Statistical Society: Series B*,
> **73**(4), 423–498.  <https://doi.org/10.1111/j.1467-9868.2011.00777.x>

> Rue H., Martino S. & Chopin N. (2009).  Approximate Bayesian inference for
> latent Gaussian models by using integrated nested Laplace approximations.
> *Journal of the Royal Statistical Society: Series B*, **71**(2), 319–392.
> <https://doi.org/10.1111/j.1467-9868.2008.00700.x>

---

## Quick start

```python
import numpy as np
from pybme import bme_predict, SoftPDF, fit_covariance

# Hard data
ch = np.array([[0], [5], [10], [20]])
zh = np.array([1.2, 0.8, 1.5, 0.3])

# Soft data — one Gaussian, one interval
cs = np.array([[7], [15]])
soft = [
    SoftPDF.from_gaussian(mean=1.0, var=0.3),
    SoftPDF.from_interval(0.5, 2.0),
]

# Fit covariance
fit = fit_covariance(ch, zh, model="exponential", order=0)
print(f"sill={fit['sill']:.2f}  range={fit['range']:.2f}  nugget={fit['nugget']:.4f}")

# Predict
ck = np.linspace(0, 20, 50).reshape(-1, 1)
results = bme_predict(
    ck, ch, zh, cs, soft,
    model="exponential",
    params=[fit["sill"], fit["range"]],
    order=0,
)

for r in results[:3]:
    print(f"  mean={r.mean:.3f}  var={r.variance:.3f}  CI=[{r.ci_lower:.2f}, {r.ci_upper:.2f}]")
```

---

## Package structure

```
pybme/
├── pyproject.toml
├── README.md
├── src/
│   └── pybme/
│       ├── __init__.py          # public API re-exports
│       ├── distance.py          # coord2dist — Euclidean distance matrix
│       ├── covariance.py        # 6 covariance models + nested evaluation
│       ├── soft_data.py         # SoftPDF class with 10+ constructors
│       ├── neighborhood.py      # spatial & space-time neighbour selection
│       ├── trend.py             # polynomial design matrix + trend estimation
│       ├── integration.py       # GH / Laplace / EP / QMC / LIS / MC integration
│       ├── predict.py           # bme_predict, bme_predict_st, BMEResult
│       ├── spde.py              # SPDE/GMRF sparse-precision Matérn fields (v0.3)
│       ├── fitting.py           # REML covariance fitting
│       └── validation.py        # leave-one-out cross-validation
├── tests/
│   ├── test_covariance.py       # ≈ MATLAB MODELSLIBtest
│   ├── test_soft_pdf.py         # ≈ MATLAB probaGenerationTest
│   ├── test_bme_proba.py        # ≈ MATLAB BMEPROBALIBtest (8 cases)
│   ├── test_integration.py      # ≈ MATLAB MVNLIBtest
│   └── test_bme_interval.py     # ≈ MATLAB BMEINTLIBtest
│       ├── bme_core.py          # complete monolithic script (standalone)
│       └── tutorials/
│           ├── __init__.py
│           ├── tutorial_models.py         # ≈ MODELSLIBtutorial
│           ├── tutorial_bme_proba.py      # ≈ BMEPROBALIBtutorial
│           ├── tutorial_bme_interval.py   # ≈ BMEINTLIBtutorial
│           ├── tutorial_kriging.py        # ≈ BMEHRLIBtutorial
│           ├── tutorial_statistics.py     # ≈ STATLIBtutorial
│           └── tutorial_genlib.py         # ≈ GENLIBtutorial
├── examples/
│   └── example01_bme_vs_kriging.py   # 1-D demo with 5 soft-data types
```

---

## API reference

### Soft data — `SoftPDF`

| Constructor | Description | MATLAB equivalent |
|---|---|---|
| `SoftPDF.from_gaussian(mean, var)` | Discretised Gaussian | `probaGaussian` |
| `SoftPDF.from_uniform(a, b)` | Uniform on [a, b] | `probaUniform` |
| `SoftPDF.from_interval(a, b)` | Interval-censored (= uniform) | `BMEinterval*` |
| `SoftPDF.from_triangular(a, mode, b)` | Triangular | `probaTriangular` |
| `SoftPDF.from_truncnorm(mu, sigma, a, b)` | Truncated normal | custom |
| `SoftPDF.from_lognormal(mu_log, sigma_log)` | Lognormal | custom |
| `SoftPDF.from_histogram(breaks, densities)` | Piecewise-constant | `softpdftype=1` |
| `SoftPDF.from_linear(z_grid, pdf_values)` | Piecewise-linear | `softpdftype=2` |
| `SoftPDF.from_callable(func, a, b)` | Arbitrary Python callable | — |
| `SoftPDF.from_mixture(components, weights)` | Weighted mixture | — |

### Covariance models

| Name | Function | Parameters |
|---|---|---|
| `exponential` | `sill·exp(−3h/range)` | `[sill, range]` |
| `gaussian` | `sill·exp(−3(h/range)²)` | `[sill, range]` |
| `spherical` | `sill·(1 − 1.5t + 0.5t³)` | `[sill, range]` |
| `matern` | Matérn(ν) | `[sill, range, nu]` |
| `nugget` | `sill·δ(h≈0)` | `[sill]` |
| `hole_cos` | `sill·cos(πh/range)` | `[sill, range]` |

**Nested models**: pass a list of names and a list-of-lists for params:

```python
model = ["nugget", "exponential"]
params = [[0.1], [0.9, 10.0]]
```

### Prediction

```python
# Spatial BME
results = bme_predict(ck, ch, zh, cs, soft_pdfs,
                      model, params,
                      nhmax=20, nsmax=8, dmax=np.inf,
                      order=0, n_grid=200, ci_prob=0.95)

# Space-time BME (separable kernel)
results = bme_predict_st(ck, tk, ch, th, zh,
                         cs, ts, soft_pdfs,
                         model_s, params_s, model_t, params_t,
                         sigma2=1.0)
```

**Integration method** (v0.3.0, expanded v0.4.0): both `bme_predict` and `bme_predict_st` accept
a `method` parameter:

| Value | Algorithm | When to use |
|---|---|---|
| `"auto"` *(default)* | Laplace if ns ≥ 6, else GH | General purpose — good default for any problem |
| `"gauss_hermite"` | Tensor-product Gauss-Hermite | ns ≤ 5–8, exact posterior shape |
| `"laplace"` | Laplace approximation | Many soft neighbours (ns ≫ 8), near-Gaussian soft PDFs |
| `"ep"` | Expectation Propagation | Non-log-concave soft PDFs (uniforms, intervals, bimodal) |
| `"qmc"` | Quasi-Monte Carlo (Sobol) | Moderate ns (3–20), O(1/N) convergence — better than MC |
| `"lis"` | Laplace Importance Sampling | Unbiased correction to Laplace, low-variance alternative to MC |
| `"mc"` | Monte Carlo sampling | Very high dimensions or diagnostics (highest cost) |

```python
# Force Laplace approximation for large soft-data problems
results = bme_predict(ck, ch, zh, cs, soft_pdfs,
                      model, params, method="laplace")
```

Each `BMEResult` contains:

| Attribute | Description |
|---|---|
| `mean`, `mode` | Central estimates |
| `variance` | Posterior variance |
| `skewness` | Third-moment skewness (non-zero when soft data is non-Gaussian) |
| `z_grid`, `pdf` | Full posterior PDF (arrays) |
| `ci_lower`, `ci_upper` | Confidence interval bounds |
| `kriging_mean`, `kriging_var` | Hard-data-only kriging results |
| `n_hard`, `n_soft` | Number of neighbours used |

### Fitting & validation

```python
fit = fit_covariance(ch, zh, model="exponential", order=0)
# → {'sill': ..., 'range': ..., 'nugget': ..., 'nll': ..., 'success': True}

cv = cross_validate(ch, zh, model="exponential", params=[fit['sill'], fit['range']])
# → {'rmse': ..., 'mae': ..., 'predicted': ..., 'errors': ...}
```

### SPDE / GMRF kriging *(v0.3.0 — original contribution)*

Sparse-precision kriging on a FEM triangular mesh, bypassing the dense
covariance matrix entirely.  **This is hard-data (kriging) only** — it does
not integrate soft probabilistic data.  For full BME with soft data, use
`bme_predict()` with `method="laplace"` (see above).

```python
from pybme import SPDEMesh, build_precision_matrix, spde_kriging, snap_to_mesh

# Build a Delaunay mesh from observation coordinates
mesh = SPDEMesh.from_points(coords_2d, margin=0.1)

# Convert Matérn parameters to SPDE form
from pybme.spde import matern_to_spde_params
kappa, tau = matern_to_spde_params(sigma2=1.0, range_param=10.0, nu=1.0)

# Sparse precision matrix Q (n × n, O(n) non-zeros)
Q = build_precision_matrix(mesh, kappa, tau, alpha=2)

# Map observation coordinates to nearest mesh nodes
obs_idx = snap_to_mesh(obs_coords, mesh)

# Kriging via sparse Cholesky — O(n^{3/2}) in 2-D
mu, var = spde_kriging(mesh, Q, obs_idx, z_obs, nugget=0.01)
```

**Limitations of `spde_kriging`:**

| Constraint | Detail |
|---|---|
| Hard data only | No soft-data (PDF) integration — use `bme_predict()` for that |
| Simple kriging only | Assumes zero mean; no trend estimation (order 0/1/2) |
| 2-D spatial only | Mesh is Delaunay triangulation in ℝ²; no 1-D, 3-D, or space-time |
| Matérn covariance only | The SPDE link is specific to the Matérn family (ν = α − d/2) |
| Mesh-node predictions | Predictions are returned at mesh nodes — use `snap_to_mesh()` to map |

---

### When to use which approach

| Scenario | Recommended approach | Why |
|---|---|---|
| **Hard + soft data, ns ≤ 5** | `bme_predict()` with default `method="auto"` | GH quadrature is exact and fast for few soft points |
| **Hard + soft data, ns ≥ 6** | `bme_predict(..., method="laplace")` | Laplace scales as O(ns³) vs exponential for GH |
| **Non-log-concave soft PDFs** | `bme_predict(..., method="ep")` | EP handles uniforms, intervals, and bimodal soft PDFs better than Laplace |
| **Moderate ns (3-20), balanced cost/accuracy** | `bme_predict(..., method="qmc")` | Sobol QMC gives O(1/N) convergence vs O(1/√N) for MC |
| **Unbiased estimate with Laplace speed** | `bme_predict(..., method="lis")` | Laplace Importance Sampling: unbiased correction to Laplace |
| **Hard data only, any covariance** | `bme_predict()` with no `cs`/`soft_pdfs` | Falls back to standard kriging internally |
| **Hard data only, Matérn, large 2-D field** | `spde_kriging()` | Sparse Cholesky is O(n^{3/2}) vs O(n³) dense |
| **Very high ns or diagnostics** | `bme_predict(..., method="mc")` | Monte Carlo — unbiased but highest variance |
| **Space-time with soft data** | `bme_predict_st()` | Separable S/T kernel; supports all `method` options |

**In short:**

* Use **`bme_predict()`** (the standard BME pipeline) whenever you have
  soft probabilistic data — it supports all covariance models, trend
  orders, and the full posterior PDF.  The `method` parameter controls
  only *how* the soft-data integral is computed (GH / Laplace / MC).
* Use **`spde_kriging()`** when you have a large 2-D spatial field with
  **hard data only**, Matérn covariance, and need the computational
  savings of sparse linear algebra.  It is *not* a replacement for full
  BME — it is an alternative kriging back-end for the hard-data-only case.

---

## Tutorials

Python translations of the MATLAB BMElib `tutorlib` tutorials.  Each is a standalone runnable script that prints results and saves plots:

| Tutorial | MATLAB equivalent | Run command |
|---|---|---|
| Covariance models | `MODELSLIBtutorial` | `python -m pybme.tutorials.tutorial_models` |
| BME with soft PDFs | `BMEPROBALIBtutorial` | `python -m pybme.tutorials.tutorial_bme_proba` |
| BME with intervals | `BMEINTLIBtutorial` | `python -m pybme.tutorials.tutorial_bme_interval` |
| Kriging (hard only) | `BMEHRLIBtutorial` | `python -m pybme.tutorials.tutorial_kriging` |
| Statistics & variograms | `STATLIBtutorial` | `python -m pybme.tutorials.tutorial_statistics` |
| Grid, NN, smoothing | `GENLIBtutorial` | `python -m pybme.tutorials.tutorial_genlib` |

### Monolithic script

The complete BME framework is also available as a single-file script at `src/pybme/bme_core.py` — useful for quick prototyping or standalone use without installing the package.

---

## Running tests

```bash
cd pybme
pytest
```

Expected output: ~100+ tests across 8 test files matching the MATLAB BMElib test suite structure
plus SPDE, Laplace, EP, QMC, and LIS integration tests.

---

## How it works

BME (Bayesian Maximum Entropy) is a two-stage geostatistical estimation
framework developed by Christakos (1990, 2000) and Serre (1999):

**Stage A — General Knowledge:**  The mean and covariance structure of the
random field define global constraints.  Maximising Shannon entropy subject
to these constraints yields the *prior* PDF $f_G$, which for a field with
known mean and covariance is the multivariate Gaussian.

**Stage B — Specificatory Knowledge:**  Site-specific data is integrated
into the prior.  **Hard data** (exact measurements) condition $f_G$ exactly;
**soft data** (PDFs representing uncertain knowledge at nearby locations)
are integrated out, yielding the BME posterior $f_K(z_k)$.

For each estimation point $\mathbf{x}_k$, the BME posterior PDF is:

$$
f_K(z_k) = A^{-1} \underbrace{p(z_k \mid z_{\text{hard}})}_{\text{hard-data conditional (kriging)}} \times \underbrace{\int \prod_{i=1}^{n_s} f_{S_i}(s_i) \; p(s_1, \dots, s_{n_s} \mid z_k, z_{\text{hard}}) \, d\mathbf{s}}_{\text{soft-data integration}}
$$

where $p(z_k \mid z_{\text{hard}})$ is the Gaussian conditional from kriging,
$f_{S_i}$ is the soft PDF at the $i$-th soft-data location,
$p(\mathbf{s} \mid z_k, z_{\text{hard}})$ is the Gaussian conditional of the
soft-data values given $z_k$ and the hard data, and $A$ is a normalising
constant.

The soft-data integral is evaluated numerically using one of six strategies:

1. **Gauss-Hermite tensor-product quadrature** (default for ns ≤ 5) — exact up to polynomial degree, preserves the full non-Gaussian shape of the posterior.
2. **Laplace approximation** (default for ns ≥ 6, v0.3.0) — finds the posterior mode and uses a second-order Taylor expansion of the log-posterior, giving O(ns³) per point instead of exponential cost.  Based on the INLA methodology of Rue et al. (2009).
3. **Expectation Propagation (EP)** (v0.4.0) — iteratively approximates each soft factor with a Gaussian site via moment matching.  O(ns² × iters) per point.  More accurate than Laplace for non-log-concave soft PDFs (uniforms, intervals, bimodal mixtures).  Based on Minka (2001).
4. **Quasi-Monte Carlo (QMC)** (v0.4.0) — Sobol low-discrepancy sequences mapped through the inverse-normal CDF.  O(1/N) convergence vs O(1/√N) for plain MC.  Drop-in upgrade to MC when ns > 8.
5. **Laplace Importance Sampling (LIS)** (v0.4.0) — uses the Laplace mode and Hessian as a Gaussian proposal for importance sampling.  Unbiased (unlike Laplace alone) with greatly reduced variance compared to plain MC.
6. **Monte Carlo sampling** — fallback for very high dimensions or when requested explicitly.

Unlike moment-matching approaches, all six methods can capture the non-Gaussian shape of the posterior when soft data is non-Gaussian.

---

## Comparison to MATLAB BMElib

| Feature | MATLAB BMElib 2.0 | PyBME |
|---|---|---|
| Language | MATLAB + Fortran MEX | Pure Python (NumPy/SciPy) |
| Soft-data types | 4 softpdftype codes | 10+ named constructors |
| Integration engine | Fortran mvPro / mvProAG2 | GH + Laplace + EP + QMC + LIS + MC |
| SPDE / GMRF kriging | — | `spde_kriging()` on FEM mesh (hard data only) |
| Laplace approximation | — | `method="laplace"` in `bme_predict` (full BME) |
| Expectation Propagation | — | `method="ep"` in `bme_predict` (v0.4.0) |
| Quasi-Monte Carlo | — | `method="qmc"` in `bme_predict` (v0.4.0) |
| Laplace Importance Sampling | — | `method="lis"` in `bme_predict` (v0.4.0) |
| Space-time | Full S/T framework | Separable S/T |
| Covariance fitting | Manual | REML auto-fit |
| Cross-validation | Manual scripting | Built-in `cross_validate()` |
| Neighbourhood | `neighbours()` | `select_neighbors()` / `_st()` |
| Installation | MATLAB path setup | `pip install -e .` |

---

## License

MIT
