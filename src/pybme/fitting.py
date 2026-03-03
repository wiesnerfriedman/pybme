"""Covariance parameter fitting via restricted maximum likelihood (REML)."""

from __future__ import annotations
import math

import numpy as np
from scipy import linalg
from scipy.optimize import minimize

from .covariance import build_cov_matrix
from .distance import coord2dist
from .trend import design_matrix


def fit_covariance(ch, zh, model="exponential", order=0,
                   range_bounds=(0.01, None), sill_bounds=(0.01, None),
                   nugget_bounds=(0.0, None)):
    """Fit covariance parameters by restricted maximum likelihood.

    Fits *sill*, *range*, and *nugget* for a single isotropic model.

    Parameters
    ----------
    ch     : (n, d) coordinates of hard data
    zh     : (n,) observed values
    model  : covariance model name (e.g. ``'exponential'``)
    order  : polynomial trend order (NaN/0/1/2)

    Returns
    -------
    dict with ``'sill'``, ``'range'``, ``'nugget'``, ``'nll'``, ``'success'``
    """
    ch = np.atleast_2d(ch)
    zh = np.asarray(zh, dtype=np.float64)
    n = len(zh)

    def neg_reml(log_theta):
        sill = np.exp(log_theta[0])
        rng = np.exp(log_theta[1])
        nug = np.exp(log_theta[2])
        K = build_cov_matrix(ch, ch, model, [sill, rng]) + np.eye(n) * (nug + 1e-10)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        X = design_matrix(ch, order)
        alpha = linalg.cho_solve((L, True), zh)
        if X.shape[1] > 0:
            XtKi = linalg.cho_solve((L, True), X)
            beta = np.linalg.lstsq(X.T @ XtKi, X.T @ alpha, rcond=None)[0]
            resid = zh - X @ beta
        else:
            resid = zh
        alpha_r = linalg.cho_solve((L, True), resid)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        return 0.5 * (float(resid @ alpha_r) + logdet + n * math.log(2 * math.pi))

    dists = coord2dist(ch, ch)
    dmax_data = np.max(dists) if len(dists) > 0 else 1.0
    z_var = max(np.var(zh), 1e-6)
    x0 = np.log([z_var, dmax_data / 3.0, z_var * 0.05 + 1e-6])
    res = minimize(neg_reml, x0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
    sill, rng, nug = np.exp(res.x)
    return {"sill": sill, "range": rng, "nugget": nug,
            "nll": res.fun, "success": res.success}
