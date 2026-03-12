"""PyBME — Bayesian Maximum Entropy geostatistical library.

Author
------
Corinne Wiesner-Friedman

Based on BMElib
---------------
PyBME is a Python port of the MATLAB BMElib 2.0 library, originally
developed by Marc L. Serre and George Christakos at the University of
North Carolina at Chapel Hill.  If you use PyBME in published work,
please cite the original BMElib:

    Serre M.L. & Christakos G. (1999). Modern geostatistics for
    environmental and health sciences: BMElib. Stochastic Environmental
    Research and Risk Assessment, 13, 1–26.
    https://doi.org/10.1007/s004770050030

BMElib homepage: http://www.unc.edu/depts/case/BMElib/

INLA-SPDE extensions (v0.3.0)
-----------------------------
The SPDE/GMRF module (``spde.py``) and Laplace approximation for
soft-data integration (in ``integration.py``) are original contributions
by Corinne Wiesner-Friedman and are not part of the original MATLAB
BMElib.  They are inspired by the INLA-SPDE methodology.

* ``spde_kriging()`` provides **hard-data kriging only** via sparse
  precision matrices (Matérn, 2-D, simple kriging).
* The Laplace approximation (``method='laplace'`` in ``bme_predict``)
  works within the **full BME pipeline** with soft probabilistic data.

If you use these features please also cite:

    Lindgren F., Rue H. & Lindström J. (2011).  An explicit link between
    Gaussian fields and Gaussian Markov random fields: the stochastic
    partial differential equation approach.  JRSS-B, 73(4), 423–498.
    https://doi.org/10.1111/j.1467-9868.2011.00777.x

    Rue H., Martino S. & Chopin N. (2009).  Approximate Bayesian
    inference for latent Gaussian models by using integrated nested
    Laplace approximations.  JRSS-B, 71(2), 319–392.
    https://doi.org/10.1111/j.1467-9868.2008.00700.x

Quick start::

    from pybme import bme_predict, SoftPDF, fit_covariance

    soft = [SoftPDF.from_lognormal(mu_log=1.0, sigma_log=0.4)]
    results = bme_predict(ck, ch, zh, cs, soft, model='exponential', params=[1.0, 10.0])
"""

from __future__ import annotations

# core types
from .soft_data import SoftPDF
from .predict import BMEResult

# distance / covariance
from .distance import coord2dist
from .covariance import (
    exponential_cov, gaussian_cov, spherical_cov, matern_cov,
    nugget_cov, hole_cos_cov, COV_MODELS, eval_cov,
    build_cov_matrix, build_cov_matrix_st,
    # space-time covariance
    gaussian_cov_st, exponential_cov_st, nugget_cov_st,
    COV_MODELS_ST, eval_cov_st,
)

# prediction
from .predict import (
    bme_predict, bme_predict_st,
    bme_predict_network, bme_predict_network_st,
)
from .neighborhood import (
    select_neighbors, select_neighbors_st,
    SpatialIndex, SpatialTemporalIndex,
)
from .trend import design_matrix, estimate_trend
from .integration import (
    integrate_soft_product, integrate_soft_product_batch,
    integrate_soft_laplace, integrate_soft_laplace_batch,
    integrate_soft_ep, integrate_soft_ep_batch,
    integrate_soft_qmc, integrate_soft_qmc_batch,
    integrate_soft_lis, integrate_soft_lis_batch,
)

# SPDE / GMRF (INLA-inspired scalability)
from .spde import (
    SPDEMesh,
    matern_to_spde_params,
    build_precision_matrix,
    spde_kriging,
    snap_to_mesh,
)

# Network-domain covariance (graph Laplacian)
from .network import (
    NetworkCovariance,
    NetworkCovarianceST,
    PhysicsInformedNetworkCovariance,
    build_graph_laplacian,
    build_mass_balance_operator,
    adjacency_from_edges,
    network_kriging_precision,
)

# Network-domain plotting
from .network_plots import (
    plot_network_observations,
    plot_network_field,
    plot_network_correlation,
    plot_operator,
)
from .swmm import (
    SwmmNetwork,
    ObservationTable,
    parse_swmm_inp,
    build_edge_array,
    read_meter_node_map,
    read_observation_csv,
    nearest_timeseries_value,
)

# fitting & validation
from .fitting import fit_covariance
from .validation import cross_validate

# The original monolithic script is available as pybme.bme_core
# Tutorials are available as pybme.tutorials.*

__version__ = "0.5.0"
__author__  = "Corinne Wiesner-Friedman"
__credits__ = [
    "Corinne Wiesner-Friedman",
    "Marc L. Serre (original MATLAB BMElib, UNC Chapel Hill)",
    "George Christakos (BME theory and original MATLAB BMElib)",
]

__all__ = [
    # types
    "SoftPDF", "BMEResult",
    # covariance
    "coord2dist", "eval_cov", "build_cov_matrix", "build_cov_matrix_st",
    "exponential_cov", "gaussian_cov", "spherical_cov",
    "matern_cov", "nugget_cov", "hole_cos_cov", "COV_MODELS",
    # prediction
    "bme_predict", "bme_predict_st",
    "bme_predict_network", "bme_predict_network_st",
    # neighborhood / trend / integration
    "select_neighbors", "select_neighbors_st",
    "SpatialIndex", "SpatialTemporalIndex",
    "design_matrix", "estimate_trend",
    "integrate_soft_product", "integrate_soft_product_batch",
    "integrate_soft_laplace", "integrate_soft_laplace_batch",
    "integrate_soft_ep", "integrate_soft_ep_batch",
    "integrate_soft_qmc", "integrate_soft_qmc_batch",
    "integrate_soft_lis", "integrate_soft_lis_batch",
    # SPDE / GMRF
    "SPDEMesh", "matern_to_spde_params", "build_precision_matrix",
    "spde_kriging", "snap_to_mesh",
    # Network-domain
    "NetworkCovariance", "NetworkCovarianceST",
    "PhysicsInformedNetworkCovariance",
    "build_graph_laplacian", "build_mass_balance_operator",
    "adjacency_from_edges",
    "network_kriging_precision",
    "bme_predict_network", "bme_predict_network_st",
    # Network-domain plotting
    "plot_network_observations", "plot_network_field",
    "plot_network_correlation", "plot_operator",
    # SWMM utilities
    "SwmmNetwork", "ObservationTable",
    "parse_swmm_inp", "build_edge_array",
    "read_meter_node_map", "read_observation_csv",
    "nearest_timeseries_value",
    # fitting / validation
    "fit_covariance", "cross_validate",
]
