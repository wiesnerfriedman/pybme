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
from .predict import bme_predict, bme_predict_st
from .neighborhood import select_neighbors, select_neighbors_st
from .trend import design_matrix, estimate_trend
from .integration import integrate_soft_product

# fitting & validation
from .fitting import fit_covariance
from .validation import cross_validate

# The original monolithic script is available as pybme.bme_core
# Tutorials are available as pybme.tutorials.*

__version__ = "0.1.0"
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
    # neighborhood / trend / integration
    "select_neighbors", "select_neighbors_st",
    "design_matrix", "estimate_trend",
    "integrate_soft_product",
    # fitting / validation
    "fit_covariance", "cross_validate",
]
