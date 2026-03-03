# PyBME — Bayesian Maximum Entropy Geostatistical Library

**Author:** Corinne Wiesner-Friedman

A pure-Python implementation of the Bayesian Maximum Entropy (BME) framework for spatial and space-time geostatistical estimation.  PyBME is a modern port of the core algorithms in the [MATLAB BMElib 2.0](http://www.unc.edu/depts/case/BMElib/](https://mserre.sph.unc.edu/BMElib_web/BMELIB.htm ) library and includes:

* **True non-Gaussian soft-data integration** via Gauss-Hermite tensor-product quadrature (replaces moment-matching).
* **10+ soft-data types** — Gaussian, uniform / interval, triangular, truncated-normal, lognormal, histogram, callable, mixture.
* **6 covariance models** — exponential, Gaussian, spherical, Matérn, nugget, hole-cosine — with nesting support.
* **Neighbourhood selection**, polynomial trend (order 0/1/2 / simple kriging), REML covariance fitting, leave-one-out cross-validation.
* **Separable space-time BME** with independent spatial and temporal covariance models.

---

## Installation

```bash
# From the pybme/ directory:
pip install -e ".[dev]"     # editable install with pytest + matplotlib
```

Requirements: Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.  Matplotlib is optional (only for plotting).

---

## Attribution

PyBME is a Python port of **BMElib 2.0**, the MATLAB Bayesian Maximum Entropy
library developed by **Marc L. Serre** and **George Christakos** at the
University of North Carolina at Chapel Hill.

If you use PyBME in published work please cite the original BMElib:

> Serre M.L. & Christakos G. (1999).  Modern geostatistics for environmental
> and health sciences: BMElib.  *Stochastic Environmental Research and Risk
> Assessment*, **13**, 1–26.  <https://doi.org/10.1007/s004770050030>

BMElib homepage: <http://www.unc.edu/depts/case/BMElib/>

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
│       ├── integration.py       # Gauss-Hermite / Monte Carlo integration
│       ├── predict.py           # bme_predict, bme_predict_st, BMEResult
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

Expected output: ~50 tests across 5 test files matching the MATLAB BMElib test suite structure.

---

## How it works

BME prediction integrates **hard data** (exact measurements) and **soft probabilistic data** (PDFs representing uncertain knowledge) using Bayesian conditioning on a Gaussian random field prior.

For each estimation point $\mathbf{x}_k$:

$$
f(z_k | \text{data}) \propto \underbrace{p(z_k | z_{\text{hard}})}_{\text{kriging prior}} \times \underbrace{\int \prod_i f_{S_i}(s_i) \, p(s_1, \dots, s_{n_s} | z_k, z_{\text{hard}}) \, ds}_{\text{soft-data likelihood}}
$$

The integral is evaluated numerically using **Gauss-Hermite tensor-product quadrature** (≤ 8 soft data dimensions) with **Monte Carlo fallback** for higher dimensions.  This preserves the full non-Gaussian shape of the posterior — unlike moment-matching approaches that force a Gaussian approximation.

---

## Comparison to MATLAB BMElib

| Feature | MATLAB BMElib 2.0 | PyBME |
|---|---|---|
| Language | MATLAB + Fortran MEX | Pure Python (NumPy/SciPy) |
| Soft-data types | 4 softpdftype codes | 10+ named constructors |
| Integration engine | Fortran mvPro / mvProAG2 | Gauss-Hermite quadrature + MC |
| Space-time | Full S/T framework | Separable S/T |
| Covariance fitting | Manual | REML auto-fit |
| Cross-validation | Manual scripting | Built-in `cross_validate()` |
| Neighbourhood | `neighbours()` | `select_neighbors()` / `_st()` |
| Installation | MATLAB path setup | `pip install -e .` |

---

## License

MIT
