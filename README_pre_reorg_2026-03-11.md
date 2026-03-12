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
* **Network-domain BME** via graph-Laplacian covariance — rivers, sewers, pipe networks, road systems.  Supports hard + soft data, all integration methods, and separable space-time.  No Euclidean distance assumption required. *(v0.5.0, original contribution)*

---

## Installation

```bash
# From the pybme/ directory:
pip install -e ".[dev]"     # editable install with pytest + matplotlib
```

Requirements: Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.  Matplotlib is optional (only for plotting).

---

## Attribution

### How To Cite

This repository now includes a standard citation file, [CITATION.cff](CITATION.cff).
GitHub and other tooling can use that file to generate software citations automatically.

If you use PyBME in published work, the recommended setup is:

1. Cite **PyBME** as the software implementation you used.
2. Also cite the original **BMElib / BME methodology** references below.
3. If you use the SPDE, Laplace, EP, QMC, LIS, or network-domain extensions, cite the methodological papers relevant to those features as well.

For **network-domain BME**, PyBME's baseline implementation is the
graph-Laplacian covariance framework documented in this repository: a
topology-based construction on arbitrary graphs rather than a
distance-along-network covariance fit. That makes it methodologically
distinct from the earlier network BME formulations associated with **Eric
Money and Marc Serre**, and also distinct from the more explicitly
flow-informed or **gradual-flow covariance** BME formulations developed by
**Prahlad Jat and Marc Serre**. If your work relies on those formulations,
cite those methodological papers separately in addition to the software. In
particular, the most relevant references are:

> Money E.S., Carter G.P. & Serre M.L. (2009). Modern space/time
> geostatistics using river distances: data integration of turbidity and E.
> coli measurements to assess fecal contamination along the Raritan River in
> New Jersey. *Environmental Science & Technology*, **43**(10), 3736-3742.
> <https://doi.org/10.1021/es803236j>

> Money E., Carter G.P. & Serre M.L. (2009). Using river distances in the
> space/time estimation of dissolved oxygen along two impaired river networks
> in New Jersey. *Water Research*, **43**(7), 1948-1958.
> <https://doi.org/10.1016/j.watres.2009.01.034>

> Jat P. & Serre M.L. (2016). Bayesian maximum entropy space/time estimation
> of surface water chloride in Maryland using river distances.
> *Environmental Pollution*, **219**, 1148-1155.
> <https://doi.org/10.1016/j.envpol.2016.09.020>

> Jat P. & Serre M.L. (2018). A novel geostatistical approach combining
> Euclidean and gradual-flow covariance models to estimate fecal coliform
> along the Haw and Deep rivers in North Carolina. *Stochastic Environmental
> Research and Risk Assessment*, **32**(9), 2537-2549.
> <https://doi.org/10.1007/s00477-018-1512-6>

For comparisons with the stream-network modelling ecosystem built around
**STARS / SSN / SSN2**, see Peterson & Ver Hoef (2014), Ver Hoef et al.
(2006), Ver Hoef & Peterson (2010), and Dumelle et al. (2024).

More specifically, the distinctions are:

| Comparison target | What those approaches emphasize | How PyBME differs |
|---|---|---|
| Money-Serre river-distance BME | BME on river networks using distance-along-river covariance structure tied to hydrography | PyBME's baseline network prior does not fit a river-distance covariance; it uses graph-Laplacian, diffusion, or precision operators on the network graph |
| Jat-Serre gradual-flow covariance BME | Hybrid covariance design that combines Euclidean and gradual-flow behaviour for river water-quality applications | PyBME does not implement the gradual-flow covariance family; flow effects enter separately through topology, space-time structure, or the optional mass-balance precision penalty |
| STARS / SSN / SSN2 | Stream-network covariance modelling and prediction with tail-up, tail-down, and Euclidean components on dendritic hydrography | PyBME's network module is an arbitrary-graph BME framework that also supports soft probabilistic data and physics-informed priors beyond the standard SSN covariance family |

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
├── notebooks/
│   ├── 01_covariance_models.ipynb
│   ├── 02_kriging.ipynb
│   ├── 03_statistics.ipynb
│   ├── 04_general_utilities.ipynb
│   ├── 05_bme_interval.ipynb
│   ├── 06_bme_soft_proba.ipynb
│   ├── 07_public_swmm_network_bme.ipynb   # public SWMM network soft-data tutorial
│   ├── 08_synthetic_routed_network_bme.ipynb  # synthetic sewer network BME tutorial
│   └── 09_physics_informed_prior.ipynb    # mass-balance prior vs standard Laplacian
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
│       ├── network.py           # graph-Laplacian covariance + physics-informed prior
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

### Network-domain BME *(v0.5.0 — original contribution)*

BME prediction on **network-constrained domains** (rivers, sewers, pipe
systems, road networks) where Euclidean distance is inappropriate.
Covariance is derived from the **graph Laplacian** of the network topology,
guaranteeing a valid (symmetric positive-definite) covariance structure on
arbitrary graph topologies — trees, cycles, and general networks.

```python
from pybme import (
    NetworkCovariance, NetworkCovarianceST,
    adjacency_from_edges,
    bme_predict_network, bme_predict_network_st,
    network_kriging_precision,
    SoftPDF,
)
import numpy as np

# Define a river network: 0—1—2—3—4 with a tributary 2—5—6
edges = np.array([[0,1],[1,2],[2,3],[3,4],[2,5],[5,6]])
W = adjacency_from_edges(7, edges)

# Build graph-Laplacian covariance (regularised inverse — Matérn-like on graph)
net_cov = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)

# Hard data at nodes 0, 4, 6
ch_nodes = np.array([0, 4, 6])
zh = np.array([2.1, 0.8, 1.5])

# Soft (uncertain) data at node 5
soft = [SoftPDF.from_gaussian(1.2, 0.3)]

# Full BME at nodes 1, 2, 3
results = bme_predict_network(
    ck_nodes=np.array([1, 2, 3]),
    ch_nodes=ch_nodes, zh=zh,
    cs_nodes=np.array([5]), soft_pdfs=soft,
    net_cov=net_cov,
)
for i, r in enumerate(results):
    print(f"Node {[1,2,3][i]}: mean={r.mean:.3f}, var={r.variance:.4f}")
```

**Separable space-time on a network:**

```python
# Pair the spatial network covariance with a temporal model
net_cov_st = NetworkCovarianceST(
    net_cov, model_t="exponential", params_t=[1.0, 5.0], sigma2=1.0,
)

# Space-time BME
results = bme_predict_network_st(
    ck_nodes=np.array([2]), tk=np.array([3.0]),
    ch_nodes=np.array([0, 4]), th=np.array([0.0, 1.0]),
    zh=np.array([2.0, 1.0]),
    net_cov_st=net_cov_st,
)
```

**Precision-based kriging** (hard data only, sparse Cholesky):

```python
mu, var = network_kriging_precision(net_cov, obs_nodes, z_obs, nugget=0.01)
```

**Three covariance constructions:**

| Method | Formula | Character |
|---|---|---|
| `"regularised"` *(default)* | $C = \sigma^2(\kappa^2 I + L)^{-1}$ | Matérn(ν=1) analog on graph; exponential-like decay |
| `"diffusion"` | $C = \sigma^2 \exp(-\kappa L)$ | Smoother; Gaussian-like decay |
| `"precision"` | $Q = \sigma^{-2}(\kappa^2 I + L)$ | Sparse precision matrix for large networks |

**Physics-informed prior** (mass-balance penalty):

```python
from pybme import PhysicsInformedNetworkCovariance, build_mass_balance_operator

# directed_edges: (E, 2) array of upstream → downstream edges
directed_edges = np.array([[0,1],[1,2],[2,3],[3,4],[2,5],[5,6]])

# Build covariance with mass-balance penalty in the precision
net_cov = PhysicsInformedNetworkCovariance(
    W, directed_edges,
    kappa=1.0, sigma2=1.0,
    alpha=1.0,   # weight on Laplacian smoothness (0 = off)
    lam=2.0,     # weight on mass-balance penalty H^T H
    from_adjacency=True,
)

# Precision: Q = (1/σ²)(κ²I + α·L + λ·H^T H)
# H encodes flow conservation: x_j ≈ Σ x_parents at each junction
# Drop-in compatible with NetworkCovarianceST and bme_predict_network_st
```

| Term | Role |
|------|------|
| $\kappa^2 I$ | Regularisation (keeps Q strictly positive-definite) |
| $\alpha L$ | Undirected-Laplacian smoothness (optional, set α=0 to disable) |
| $\lambda H^\top H$ | Mass-balance penalty — penalises flow-conservation violations |

See [notebook 09](notebooks/09_physics_informed_prior.ipynb) for a full comparison.

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
| **Network domain (river, sewer, pipe)** | `bme_predict_network()` | Graph-Laplacian covariance; no Euclidean distance needed |
| **Network with flow-conservation prior** | `PhysicsInformedNetworkCovariance` | Mass-balance penalty in precision; parent-child correlations increase |
| **Network domain, space-time** | `bme_predict_network_st()` | Separable graph-Laplacian × temporal covariance |
| **Network, hard data only, large graph** | `network_kriging_precision()` | Sparse precision Cholesky on graph Laplacian |
| **Very high ns or diagnostics** | `bme_predict(..., method="mc")` | Monte Carlo — unbiased but highest variance |
| **Space-time with soft data** | `bme_predict_st()` | Separable S/T kernel; supports all `method` options |

**In short:**

* Use **`bme_predict()`** (the standard BME pipeline) whenever you have
  soft probabilistic data — it supports all covariance models, trend
  orders, and the full posterior PDF.  The `method` parameter controls
  only *how* the soft-data integral is computed (GH / Laplace / MC).
* Use **`bme_predict_network()`** / **`bme_predict_network_st()`** for
  domains where distances are defined by a graph topology (rivers,
  sewers, roads).  Supports hard + soft data, all integration methods.
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

### Notebook tutorials

Interactive notebooks live in `notebooks/`. The main shareable network-domain soft-data example is:

| Notebook | Focus | Notes |
|---|---|---|
| `notebooks/07_public_swmm_network_bme.ipynb` | Public SWMM network BME with simulated hard data and SWMM-derived soft data | Uses the public KBWI model. Set `PYBME_PUBLIC_SWMM_DIR` if the data is not in the default local path. |
| `notebooks/08_synthetic_routed_network_bme.ipynb` | Shareable routed sewer-like network with dry-weather inflow, one RTK triangle, and space-time network BME | Uses the internal synthetic generator in `pybme.synthetic_network`, with a nominal routed run as soft data and sparse simulated observations as hard data. |
| `notebooks/09_physics_informed_prior.ipynb` | Mass-balance prior vs standard Laplacian — how to encode flow conservation in the BME prior | Compares precision/correlation structure, BME predictions, and sensitivity to the mass-balance weight λ. |

### Example scripts

Standalone scripts in `examples/` include a fully shareable routed-network example that does not depend on SWMM input files:

| Script | Focus | Notes |
|---|---|---|
| `examples/synthetic_routed_network_bme.py` | Synthetic routed sewer-like network with dry-weather inflows, one RTK triangle, and network space-time BME | Uses a nominal routed run as soft data and sparse simulated observations as hard data. Routing is travel-time based with velocity fixed at 0.5 ft/s for the soft run. |

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


### PyBME vs SSN/STARS

PyBME and the **STARS / SSN / SSN2** ecosystem address related problems, but
they are not implementing the same model family.

| Aspect | PyBME network implementation | SSN/STARS family |
|---|---|---|
| Primary domain | Arbitrary graphs: rivers, sewers, pipes, roads | Dendritic stream networks with directed flow |
| Core covariance idea | Graph-Laplacian / diffusion / precision construction on the network graph | Tail-up, tail-down, and Euclidean moving-average covariance models based on stream distance and flow connectivity |
| Data model | Full BME: hard data plus probabilistic soft data | Spatial stream-network regression/kriging with observed responses |
| Space-time support | Separable graph-space × time in `bme_predict_network_st()` | Core STARS/SSN workflow is primarily spatial; newer Bayesian stream-network space-time models live in adjacent tools such as `SSNbayes` |
| Physics-informed prior | Supported via `PhysicsInformedNetworkCovariance` and the mass-balance term $H^\top H$ | Not the standard SSN/STARS covariance construction |
| Preprocessing | Native Python graph objects / adjacency matrices | STARS supplies the spatial information needed to fit SSN models on stream networks |

### PyBME vs Money/Jat network BME

The distinctions from the **Money-Serre** and **Jat-Serre** papers are also
more specific than just "different citations":

| Method family | Network dependence encoded through | What the papers are doing | What PyBME implements |
|---|---|---|---|
| Money-Serre river-distance BME | River distance along a hydrographic network | Space-time BME on river networks where covariance is built from river distances between monitoring sites | A graph-operator prior on nodes or graph coordinates; dependence comes from the Laplacian or related operator rather than a fitted river-distance covariance kernel |
| Jat-Serre gradual-flow covariance BME | Euclidean plus gradual-flow covariance mixture | River-water-quality estimation where flow behaviour is embedded directly in the covariance construction | Optional flow-informed structure is introduced through separate modelling choices such as routed soft data or the mass-balance penalty, not through the Jat gradual-flow covariance itself |
| PyBME network-domain BME | Graph topology, optional temporal kernel, optional precision penalty | General network-domain BME on arbitrary graphs, with hard and soft data and multiple integration back-ends | Implemented in `NetworkCovariance`, `NetworkCovarianceST`, `PhysicsInformedNetworkCovariance`, `bme_predict_network()`, and `bme_predict_network_st()` |

So the closest comparison is:

* If you want **stream-distance covariance models** with tail-up / tail-down structure on hydrographic networks, the SSN/STARS literature is the appropriate baseline.
* If you want **river-distance BME** in the style of Money and collaborators, or **gradual-flow covariance BME** in the style of Jat and Serre, those are distinct model families that should be cited on their own terms.
* If you want **BME with soft probabilistic data on an arbitrary network graph**, PyBME is implementing a different graph-based prior construction.
* If you want **flow-informed penalties or conservation constraints** on sewer or routed infrastructure graphs, PyBME's physics-informed precision formulation goes beyond the standard STARS/SSN setup and is separate from the Jat gradual-flow covariance formulation.

Representative SSN/STARS references:

> Ver Hoef J.M., Peterson E. & Theobald D. (2006). Spatial statistical models
> that use flow and stream distance. *Environmental and Ecological
> Statistics*, **13**(4), 449-464.
> <https://doi.org/10.1007/s10651-006-0022-8>

> Ver Hoef J.M. & Peterson E.E. (2010). A moving average approach for spatial
> statistical models of stream networks. *Journal of the American Statistical
> Association*, **105**(489), 6-18.
> <https://doi.org/10.1198/jasa.2009.ap08248>

> Peterson E.E. & Ver Hoef J.M. (2014). *STARS*: An *ArcGIS* toolset used to
> calculate the spatial information needed to fit spatial statistical models
> to stream network data. *Journal of Statistical Software*, **56**(2), 1-17.
> <https://doi.org/10.18637/jss.v056.i02>

> Dumelle M., Peterson E.E., Ver Hoef J.M., Pearse A. & Isaak D.J. (2024).
> SSN2: The next generation of spatial stream network modeling in R.
> *Journal of Open Source Software*, **9**(99), 6389.
> <https://doi.org/10.21105/joss.06389>
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
| Network-domain BME | — | `bme_predict_network()` / `_st()` via graph Laplacian (v0.5.0) |
| Space-time | Full S/T framework | Separable S/T (Euclidean and network) |
| Covariance fitting | Manual | REML auto-fit |
| Cross-validation | Manual scripting | Built-in `cross_validate()` |
| Neighbourhood | `neighbours()` | `select_neighbors()` / `_st()` |
| Installation | MATLAB path setup | `pip install -e .` |

---

## License

MIT
