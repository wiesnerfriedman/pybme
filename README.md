<p align="center">
    <img src="docs/pybmelogo-cropped.png" alt="PyBME logo" width="760">
</p>

# PyBME, Bayesian Maximum Entropy Geostatistical Library

**Author:** Corinne Wiesner-Friedman

PyBME is a pure-Python implementation of **Bayesian Maximum Entropy (BME)** for spatial and space-time geostatistical estimation. It is a modern port of the core workflow in [MATLAB BMElib 2.0](https://mserre.sph.unc.edu/BMElab_web/), with additional original work for sparse-precision kriging, scalable soft-data integration, and network-domain BME on arbitrary graphs.

At a high level, the repository contains:

* **BMElib-derived core BME workflow**: hard + soft data, multiple covariance models, polynomial trend handling, REML fitting, cross-validation, and separable space-time BME.
* **Original extensions in PyBME**: Laplace, EP, QMC, LIS, SPDE / GMRF kriging, KD-tree-backed neighborhood acceleration, and graph-based network-domain BME.
* **Tutorial material**: MATLAB tutorlib translations, notebooks, standalone examples, and a monolithic script version.

## Contents

* [Installation](#installation)
* [What BME adds beyond kriging](#what-bme-adds-beyond-kriging)
* [Quick start](#quick-start)
* [BMElib-derived core workflow](#bmelib-derived-core-workflow)
* [Original contributions in PyBME](#original-contributions-in-pybme)
* [Network-domain BME](#network-domain-bme)
* [When to use which approach](#when-to-use-which-approach)
* [Tutorials, examples, tests, and support material](#tutorials-examples-tests-and-support-material)
* [How to cite and attribution](#how-to-cite-and-attribution)
* [Comparison to MATLAB BMElib](#comparison-to-matlab-bmelib)
* [AI-assisted development](#ai-assisted-development)
* [License](#license)

---

## Installation

```bash
# From the pybme/ directory:
pip install -e ".[dev]"
```

Requirements: Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10. Matplotlib is optional and only needed for plotting.

---

## What BME adds beyond kriging

BME (Bayesian Maximum Entropy) is a two-stage geostatistical estimation framework developed by Christakos (1990, 2000) and Serre (1999).

**Stage A, General Knowledge:** the mean and covariance structure define global constraints. Maximising Shannon entropy under those constraints yields the prior PDF $f_G$, which is multivariate Gaussian when the mean and covariance are known.

**Stage B, Specificatory Knowledge:** site-specific information is folded into the prior. Hard data condition $f_G$ exactly, while soft data are represented as PDFs and integrated out rather than collapsed to point values.

For each estimation point $\mathbf{x}_k$, the BME posterior is

$$
f_K(z_k) = A^{-1} \underbrace{p(z_k \mid z_{\text{hard}})}_{\text{hard-data conditional (kriging)}} \times \underbrace{\int \prod_{i=1}^{n_s} f_{S_i}(s_i) \; p(s_1, \dots, s_{n_s} \mid z_k, z_{\text{hard}}) \, d\mathbf{s}}_{\text{soft-data integration}}
$$

where $p(z_k \mid z_{\text{hard}})$ is the Gaussian conditional from kriging, $f_{S_i}$ is the soft PDF at the $i$-th soft-data location, $p(\mathbf{s} \mid z_k, z_{\text{hard}})$ is the Gaussian conditional of the soft-data values given $z_k$ and the hard data, and $A$ is a normalising constant.

The practical difference from ordinary kriging is that PyBME can propagate **probabilistic soft information** directly into the posterior instead of forcing uncertain data into means, intervals, or ad hoc weights.

The soft-data integral is evaluated numerically using one of six strategies:

1. **Gauss-Hermite tensor-product quadrature** (default for ns ≤ 5): exact up to polynomial degree and preserves the full non-Gaussian posterior shape.
2. **Laplace approximation** (default for ns ≥ 6, v0.3.0): mode-finding plus second-order expansion, reducing cost to $O(n_s^3)$ per point instead of exponential growth.
3. **Expectation Propagation (EP)** (v0.4.0): Gaussian site approximation via moment matching, often better than Laplace for non-log-concave soft PDFs.
4. **Quasi-Monte Carlo (QMC)** (v0.4.0): Sobol low-discrepancy integration with better convergence than plain Monte Carlo in moderate dimensions.
5. **Laplace Importance Sampling (LIS)** (v0.4.0): unbiased correction to Laplace using the Laplace Gaussian as the proposal.
6. **Monte Carlo sampling**: fallback for very high dimensions or explicit diagnostics.

Unlike moment-matching approximations, all six routes preserve genuinely non-Gaussian soft-data effects in the posterior.

---

## Quick start

```python
import numpy as np
from pybme import bme_predict, SoftPDF, fit_covariance

# Hard data
ch = np.array([[0], [5], [10], [20]])
zh = np.array([1.2, 0.8, 1.5, 0.3])

# Soft data, one Gaussian, one interval
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

## BMElib-derived core workflow

PyBME ports the main BMElib-style BME workflow into a pure-Python package. The core translated capabilities are:

* **True non-Gaussian soft-data integration** via Gauss-Hermite tensor-product quadrature.
* **10+ soft-data types**: Gaussian, uniform / interval, triangular, truncated-normal, lognormal, histogram, callable, and mixture forms.
* **6 covariance models**: exponential, Gaussian, spherical, Matérn, nugget, and hole-cosine, with nesting support.
* **Neighbourhood selection**, polynomial trend handling (order 0/1/2 or simple kriging), REML covariance fitting, and leave-one-out cross-validation.
* **Separable space-time BME** with independent spatial and temporal covariance models.

### Package structure

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
│   ├── 07_public_swmm_network_bme.ipynb
│   ├── 08_synthetic_routed_network_bme.ipynb
│   └── 09_physics_informed_prior.ipynb
├── src/
│   └── pybme/
│       ├── __init__.py
│       ├── distance.py
│       ├── covariance.py
│       ├── soft_data.py
│       ├── neighborhood.py
│       ├── trend.py
│       ├── integration.py
│       ├── predict.py
│       ├── network.py
│       ├── spde.py
│       ├── fitting.py
│       └── validation.py
├── tests/
│   ├── test_covariance.py
│   ├── test_soft_pdf.py
│   ├── test_bme_proba.py
│   ├── test_integration.py
│   └── test_bme_interval.py
├── examples/
│   └── example01_bme_vs_kriging.py
```

### Soft data, `SoftPDF`

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
| `SoftPDF.from_callable(func, a, b)` | Arbitrary Python callable | not in BMElib |
| `SoftPDF.from_mixture(components, weights)` | Weighted mixture | not in BMElib |

### Covariance models

| Name | Function | Parameters |
|---|---|---|
| `exponential` | `sill·exp(−3h/range)` | `[sill, range]` |
| `gaussian` | `sill·exp(−3(h/range)²)` | `[sill, range]` |
| `spherical` | `sill·(1 − 1.5t + 0.5t³)` | `[sill, range]` |
| `matern` | Matérn(ν) | `[sill, range, nu]` |
| `nugget` | `sill·δ(h≈0)` | `[sill]` |
| `hole_cos` | `sill·cos(πh/range)` | `[sill, range]` |

Nested models are passed as a list of names and a list-of-lists for parameters:

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
| `skewness` | Third-moment skewness when soft data is non-Gaussian |
| `z_grid`, `pdf` | Full posterior PDF |
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

### Tutorlib translations

PyBME includes Python translations of the MATLAB BMElib `tutorlib` material.

| Tutorial | MATLAB equivalent | Run command |
|---|---|---|
| Covariance models | `MODELSLIBtutorial` | `python -m pybme.tutorials.tutorial_models` |
| BME with soft PDFs | `BMEPROBALIBtutorial` | `python -m pybme.tutorials.tutorial_bme_proba` |
| BME with intervals | `BMEINTLIBtutorial` | `python -m pybme.tutorials.tutorial_bme_interval` |
| Kriging (hard only) | `BMEHRLIBtutorial` | `python -m pybme.tutorials.tutorial_kriging` |
| Statistics & variograms | `STATLIBtutorial` | `python -m pybme.tutorials.tutorial_statistics` |
| Grid, NN, smoothing | `GENLIBtutorial` | `python -m pybme.tutorials.tutorial_genlib` |

---

## Original contributions in PyBME

The following sections describe features that are not part of the original MATLAB BMElib baseline.

These contributions are still under review and should be considered experimental until peer review is complete.

### Advanced integration methods

Both `bme_predict` and `bme_predict_st` accept a `method` parameter.

| Value | Algorithm | When to use |
|---|---|---|
| `"auto"` *(default)* | Laplace if ns ≥ 6, else GH | General-purpose default |
| `"gauss_hermite"` | Tensor-product Gauss-Hermite | ns ≤ 5–8, best exactness |
| `"laplace"` | Laplace approximation | Many soft neighbours, near-Gaussian soft PDFs |
| `"ep"` | Expectation Propagation | Uniforms, intervals, or other non-log-concave soft PDFs |
| `"qmc"` | Quasi-Monte Carlo (Sobol) | Moderate ns with better convergence than MC |
| `"lis"` | Laplace Importance Sampling | Unbiased correction to Laplace |
| `"mc"` | Monte Carlo sampling | Very high dimensions or diagnostics |

```python
# Force Laplace approximation for larger soft-data problems
results = bme_predict(ck, ch, zh, cs, soft_pdfs,
                      model, params, method="laplace")
```

### SPDE / GMRF kriging *(v0.3.0)*

Sparse-precision kriging on a FEM triangular mesh bypasses the dense covariance matrix entirely. This is a **hard-data kriging** back-end, not a replacement for full BME with soft PDFs.

```python
from pybme import SPDEMesh, build_precision_matrix, spde_kriging, snap_to_mesh
from pybme.spde import matern_to_spde_params

# Build a Delaunay mesh from observation coordinates
mesh = SPDEMesh.from_points(coords_2d, margin=0.1)
kappa, tau = matern_to_spde_params(sigma2=1.0, range_param=10.0, nu=1.0)
Q = build_precision_matrix(mesh, kappa, tau, alpha=2)
obs_idx = snap_to_mesh(obs_coords, mesh)
mu, var = spde_kriging(mesh, Q, obs_idx, z_obs, nugget=0.01)
```

Limitations of `spde_kriging`:

| Constraint | Detail |
|---|---|
| Hard data only | No soft-data integration |
| Simple kriging only | Assumes zero mean |
| 2-D spatial only | Delaunay mesh in $\mathbb{R}^2$ |
| Matérn covariance only | SPDE construction is Matérn-specific |
| Mesh-node predictions | Use `snap_to_mesh()` to map observations and outputs |

### Performance and neighborhood extensions

PyBME also adds reusable spatial and space-time neighborhood indices beyond the original ad hoc neighbor lookup style. In particular, `SpatialIndex` and `SpatialTemporalIndex` provide **KD-tree-backed** repeated neighbor search, which is useful when many prediction locations are queried against the same observation set.

---

## Network-domain BME

PyBME includes a graph-based BME framework for **network-constrained domains** such as rivers, sewers, pipes, and road systems where Euclidean distance is not the right geometry.

The baseline prior is built from the **graph Laplacian** or related graph operators, which gives a valid covariance or precision structure on arbitrary graph topologies, including trees, cycles, and more general infrastructure networks.

### Core network API

```python
from pybme import (
    NetworkCovariance, NetworkCovarianceST,
    adjacency_from_edges,
    bme_predict_network, bme_predict_network_st,
    network_kriging_precision,
    SoftPDF,
)
import numpy as np

# Define a river or sewer network, 0-1-2-3-4 with a tributary 2-5-6
edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6]])
W = adjacency_from_edges(7, edges)

# Regularised graph-Laplacian covariance
net_cov = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)

# Hard data at nodes 0, 4, 6
ch_nodes = np.array([0, 4, 6])
zh = np.array([2.1, 0.8, 1.5])

# Soft data at node 5
soft = [SoftPDF.from_gaussian(1.2, 0.3)]

# Full BME prediction at nodes 1, 2, 3
results = bme_predict_network(
    ck_nodes=np.array([1, 2, 3]),
    ch_nodes=ch_nodes, zh=zh,
    cs_nodes=np.array([5]), soft_pdfs=soft,
    net_cov=net_cov,
)
for i, r in enumerate(results):
    print(f"Node {[1, 2, 3][i]}: mean={r.mean:.3f}, var={r.variance:.4f}")
```

### Space-time network BME

```python
net_cov_st = NetworkCovarianceST(
    net_cov, model_t="exponential", params_t=[1.0, 5.0], sigma2=1.0,
)

results = bme_predict_network_st(
    ck_nodes=np.array([2]), tk=np.array([3.0]),
    ch_nodes=np.array([0, 4]), th=np.array([0.0, 1.0]),
    zh=np.array([2.0, 1.0]),
    net_cov_st=net_cov_st,
)
```

### Precision-based network kriging

```python
mu, var = network_kriging_precision(net_cov, obs_nodes, z_obs, nugget=0.01)
```

### Covariance constructions

| Method | Formula | Character |
|---|---|---|
| `"regularised"` *(default)* | $C = \sigma^2(\kappa^2 I + L)^{-1}$ | Matérn-like graph covariance with exponential-style decay |
| `"diffusion"` | $C = \sigma^2 \exp(-\kappa L)$ | Smoother diffusion covariance |
| `"precision"` | $Q = \sigma^{-2}(\kappa^2 I + L)$ | Sparse precision form for large graphs |

### Physics-informed prior

```python
from pybme import PhysicsInformedNetworkCovariance, build_mass_balance_operator

directed_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [2, 5], [5, 6]])

net_cov = PhysicsInformedNetworkCovariance(
    W, directed_edges,
    kappa=1.0, sigma2=1.0,
    alpha=1.0,
    lam=2.0,
    from_adjacency=True,
)

# Precision: Q = (1/σ²)(κ²I + α·L + λ·H^T H)
# H encodes flow conservation: x_j ≈ Σ x_parents at each junction
```

| Term | Role |
|---|---|
| $\kappa^2 I$ | Regularisation to keep $Q$ strictly positive-definite |
| $\alpha L$ | Optional undirected-Laplacian smoothness |
| $\lambda H^\top H$ | Mass-balance penalty for flow-conservation violations |

See [notebook 09](notebooks/09_physics_informed_prior.ipynb) for a full comparison.

### How PyBME differs from Money and Jat

PyBME's network module is a graph-operator BME framework. It is not a direct implementation of the Money-Serre river-distance BME family or the Jat-Serre gradual-flow covariance family.

| Method family | Network dependence encoded through | What those papers do | What PyBME implements |
|---|---|---|---|
| Money-Serre river-distance BME | River distance along hydrographic networks | Space-time BME with covariance built from river distances between monitoring sites | A graph-operator prior on nodes or graph coordinates, with dependence induced by Laplacian-style operators rather than fitted river-distance kernels |
| Jat-Serre gradual-flow covariance BME | Euclidean plus gradual-flow covariance mixture | River-water-quality estimation with flow behavior embedded directly in the covariance family | Flow-informed structure enters separately through topology, routed soft data, or the optional mass-balance penalty, not through the Jat gradual-flow covariance |
| PyBME network-domain BME | Graph topology, optional temporal kernel, optional precision penalty | General network-domain BME on arbitrary graphs with hard and soft data | Implemented in `NetworkCovariance`, `NetworkCovarianceST`, `PhysicsInformedNetworkCovariance`, `bme_predict_network()`, and `bme_predict_network_st()` |

### PyBME vs SSN/STARS

PyBME and the **STARS / SSN / SSN2** ecosystem address overlapping network-estimation problems, but they are built around different covariance and data-model assumptions.

| Aspect | PyBME network implementation | SSN/STARS family |
|---|---|---|
| Primary domain | Arbitrary graphs: rivers, sewers, pipes, roads | Dendritic stream networks with directed flow |
| Core covariance idea | Graph-Laplacian / diffusion / precision construction on the network graph | Tail-up, tail-down, and Euclidean moving-average covariance models built from stream distance and flow connectivity |
| Data model | Full BME: hard data plus probabilistic soft data | Stream-network regression / kriging with observed responses |
| Space-time support | Separable graph-space × time in `bme_predict_network_st()` | Primarily spatial in the core SSN/STARS workflow |
| Physics-informed prior | Supported via `PhysicsInformedNetworkCovariance` and $H^\top H$ | Not the standard SSN/STARS covariance construction |
| Preprocessing | Native Python graph objects and adjacency matrices | STARS computes the stream-network spatial information used to fit SSN models |

The practical comparison is:

* If you want **tail-up / tail-down stream-distance covariance models** on hydrographic networks, the SSN/STARS literature is the right baseline.
* If you want **river-distance BME** or **gradual-flow covariance BME** in the Money or Jat sense, those are distinct methodological families and should be cited on their own terms.
* If you want **soft-data BME on an arbitrary graph** with optional physics-informed precision penalties, PyBME is implementing a different graph-based prior construction.

### Network-domain examples

The main shareable network-domain materials are:

| Resource | Focus | Notes |
|---|---|---|
| `notebooks/07_public_swmm_network_bme.ipynb` | Public SWMM network BME with simulated hard data and SWMM-derived soft data | Uses the public KBWI model. Set `PYBME_PUBLIC_SWMM_DIR` if needed. |
| `notebooks/08_synthetic_routed_network_bme.ipynb` | Shareable routed sewer-like network with dry-weather inflow, one RTK triangle, and space-time network BME | Uses the internal synthetic generator with a nominal routed run as soft data. |
| `notebooks/09_physics_informed_prior.ipynb` | Mass-balance prior versus standard Laplacian | Compares precision structure, BME predictions, and sensitivity to $\lambda$. |
| `examples/synthetic_routed_network_bme.py` | Standalone routed sewer-like network example | Uses a nominal routed run as soft data and sparse simulated observations as hard data. |

---

## When to use which approach

| Scenario | Recommended approach | Why |
|---|---|---|
| Hard + soft data, ns ≤ 5 | `bme_predict()` with `method="auto"` | GH quadrature is exact and fast |
| Hard + soft data, ns ≥ 6 | `bme_predict(..., method="laplace")` | Laplace scales as $O(n_s^3)$ |
| Non-log-concave soft PDFs | `bme_predict(..., method="ep")` | EP often handles intervals and mixtures better |
| Moderate ns, balanced cost/accuracy | `bme_predict(..., method="qmc")` | Better convergence than MC |
| Unbiased estimate with Laplace speed | `bme_predict(..., method="lis")` | Low-variance unbiased correction |
| Hard data only, any covariance | `bme_predict()` with no `cs` / `soft_pdfs` | Falls back to kriging internally |
| Hard data only, Matérn, large 2-D field | `spde_kriging()` | Sparse Cholesky is cheaper than dense covariance algebra |
| Network domain with soft data | `bme_predict_network()` | Graph-based BME on arbitrary networks |
| Network with flow-conservation prior | `PhysicsInformedNetworkCovariance` | Adds mass-balance structure in the precision |
| Network domain, space-time | `bme_predict_network_st()` | Separable graph-space × time covariance |
| Network, hard data only, large graph | `network_kriging_precision()` | Sparse precision Cholesky |
| Very high ns or diagnostics | `bme_predict(..., method="mc")` | Most general but highest variance |
| Space-time with soft data | `bme_predict_st()` | Separable space-time BME with all integration methods |

In short:

* Use **`bme_predict()`** whenever you have soft probabilistic data in Euclidean space.
* Use **`bme_predict_network()`** or **`bme_predict_network_st()`** when the domain geometry is a graph.
* Use **`spde_kriging()`** only for large hard-data-only Matérn kriging problems where sparse precision methods are the point.

---

## Tutorials, examples, tests, and support material

### Notebook tutorials

Interactive notebooks live in `notebooks/`.

| Notebook | Focus | Notes |
|---|---|---|
| `notebooks/01_covariance_models.ipynb` | Covariance models | Core covariance behavior |
| `notebooks/02_kriging.ipynb` | Kriging | Hard-data baseline |
| `notebooks/03_statistics.ipynb` | Statistics | Variogram and supporting utilities |
| `notebooks/04_general_utilities.ipynb` | General utilities | Helper workflows |
| `notebooks/05_bme_interval.ipynb` | Interval BME | Soft interval example |
| `notebooks/06_bme_soft_proba.ipynb` | Soft-probability BME | General soft PDF example |
| `notebooks/07_public_swmm_network_bme.ipynb` | Public SWMM network BME | Public KBWI-based soft-data workflow |
| `notebooks/08_synthetic_routed_network_bme.ipynb` | Synthetic routed network BME | Shareable sewer-like example |
| `notebooks/09_physics_informed_prior.ipynb` | Physics-informed prior | Mass-balance penalty study |

### Example scripts

| Script | Focus | Notes |
|---|---|---|
| `examples/example01_bme_vs_kriging.py` | 1-D BME vs kriging demo | Includes multiple soft-data types |
| `examples/synthetic_routed_network_bme.py` | Synthetic routed sewer-like network | Standalone shareable network BME example |

### Monolithic script

The full framework is also available as a single-file script at `src/pybme/bme_core.py` for quick prototyping or standalone use.

### Running tests

```bash
cd pybme
pytest
```

Expected output is roughly 100+ tests across the BMElib-style core suites plus SPDE, Laplace, EP, QMC, LIS, and network-related tests.

---

## How to cite and attribution

This repository includes [CITATION.cff](CITATION.cff), so GitHub and other tools can generate software citations automatically.

Recommended citation practice:

1. Cite **PyBME** as the software implementation.
2. Cite the original **BME / BMElib** references for the foundational methodology.
3. If you use SPDE, Laplace, EP, QMC, LIS, or network-domain features, also cite the corresponding methodological papers.

For network applications in particular, cite PyBME for the software, then add the specific methodological family you are actually using or comparing against: BMElib/BME foundations, Money-Serre river-distance BME, Jat-Serre gradual-flow covariance BME, or SSN/STARS-style stream-network covariance modelling.

PyBME is a Python port of **BMElib 2.0**, the MATLAB library for **Bayesian Maximum Entropy** developed by **Marc L. Serre**, **Patrick Bogaert**, and **George Christakos** at UNC Chapel Hill.

Foundational BMElib / BME references:

> Serre M.L. & Christakos G. (1999). Modern geostatistics for environmental and health sciences: BMElib. *Stochastic Environmental Research and Risk Assessment*, **13**, 1–26. <https://doi.org/10.1007/s004770050030>

> Christakos G. (1990). A Bayesian/maximum-entropy view to the spatial estimation problem. *Mathematical Geology*, **22**(7), 763–777. <https://doi.org/10.1007/BF00890661>

> Christakos G. (2000). *Modern Spatiotemporal Geostatistics*. Oxford University Press, New York.

> Christakos G., Bogaert P. & Serre M.L. (2002). *Temporal GIS: Advanced Functions for Field-Based Applications*. Springer-Verlag, Berlin. <https://doi.org/10.1007/978-3-642-56540-3>

BMElib homepage: <http://www.unc.edu/depts/case/BMElib/>

### Original-contribution references

The SPDE / GMRF sparse-precision module and Laplace-style soft-data integration features are original contributions in PyBME and are not part of the original MATLAB BMElib baseline. Relevant references include:

> Lindgren F., Rue H. & Lindström J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach. *Journal of the Royal Statistical Society: Series B*, **73**(4), 423–498. <https://doi.org/10.1111/j.1467-9868.2011.00777.x>

> Rue H., Martino S. & Chopin N. (2009). Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. *Journal of the Royal Statistical Society: Series B*, **71**(2), 319–392. <https://doi.org/10.1111/j.1467-9868.2008.00700.x>

### Network-method references

If your work uses or compares against river-distance BME, gradual-flow covariance BME, or the SSN/STARS family, the main references are:

> Money E.S., Carter G.P. & Serre M.L. (2009). Modern space/time geostatistics using river distances: data integration of turbidity and E. coli measurements to assess fecal contamination along the Raritan River in New Jersey. *Environmental Science & Technology*, **43**(10), 3736-3742. <https://doi.org/10.1021/es803236j>

> Money E., Carter G.P. & Serre M.L. (2009). Using river distances in the space/time estimation of dissolved oxygen along two impaired river networks in New Jersey. *Water Research*, **43**(7), 1948-1958. <https://doi.org/10.1016/j.watres.2009.01.034>

> Jat P. & Serre M.L. (2016). Bayesian maximum entropy space/time estimation of surface water chloride in Maryland using river distances. *Environmental Pollution*, **219**, 1148-1155. <https://doi.org/10.1016/j.envpol.2016.09.020>

> Jat P. & Serre M.L. (2018). A novel geostatistical approach combining Euclidean and gradual-flow covariance models to estimate fecal coliform along the Haw and Deep rivers in North Carolina. *Stochastic Environmental Research and Risk Assessment*, **32**(9), 2537-2549. <https://doi.org/10.1007/s00477-018-1512-6>

> Ver Hoef J.M., Peterson E. & Theobald D. (2006). Spatial statistical models that use flow and stream distance. *Environmental and Ecological Statistics*, **13**(4), 449-464. <https://doi.org/10.1007/s10651-006-0022-8>

> Ver Hoef J.M. & Peterson E.E. (2010). A moving average approach for spatial statistical models of stream networks. *Journal of the American Statistical Association*, **105**(489), 6-18. <https://doi.org/10.1198/jasa.2009.ap08248>

> Peterson E.E. & Ver Hoef J.M. (2014). *STARS*: An *ArcGIS* toolset used to calculate the spatial information needed to fit spatial statistical models to stream network data. *Journal of Statistical Software*, **56**(2), 1-17. <https://doi.org/10.18637/jss.v056.i02>

> Dumelle M., Peterson E.E., Ver Hoef J.M., Pearse A. & Isaak D.J. (2024). SSN2: The next generation of spatial stream network modeling in R. *Journal of Open Source Software*, **9**(99), 6389. <https://doi.org/10.21105/joss.06389>

---

## Comparison to MATLAB BMElib

| Feature | MATLAB BMElib 2.0 | PyBME |
|---|---|---|
| Language | MATLAB + Fortran MEX | Pure Python (NumPy/SciPy) |
| Soft-data types | 4 softpdftype codes | 10+ named constructors |
| Integration engine | Fortran mvPro / mvProAG2 | GH + Laplace + EP + QMC + LIS + MC |
| SPDE / GMRF kriging | not in BMElib | `spde_kriging()` on FEM mesh (hard data only) |
| Laplace approximation | not in BMElib | `method="laplace"` in `bme_predict` |
| Expectation Propagation | not in BMElib | `method="ep"` in `bme_predict` |
| Quasi-Monte Carlo | not in BMElib | `method="qmc"` in `bme_predict` |
| Laplace Importance Sampling | not in BMElib | `method="lis"` in `bme_predict` |
| Network-domain BME | not in BMElib | `bme_predict_network()` / `_st()` via graph Laplacian |
| Space-time | Full S/T framework | Separable S/T in Euclidean and network settings |
| Covariance fitting | Manual | REML auto-fit |
| Cross-validation | Manual scripting | Built-in `cross_validate()` |
| Neighbourhood | `neighbours()` | `select_neighbors()` / `_st()` plus KD-tree-backed indices |
| Installation | MATLAB path setup | `pip install -e .` |

---

## AI-assisted development

Code implementation, documentation, and test scaffolding were developed with the assistance of GitHub Copilot (LLM-based coding assistant). All mathematical formulations, methodological choices, research direction, and validation were performed by the author.

---

## License

MIT
