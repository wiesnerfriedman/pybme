"""Covariance models matching MATLAB BMElib modelslib.

Supports: exponential, gaussian, spherical, matern, nugget, hole_cos.
All models accept ``(h, params)`` where *h* is a distance array (or scalar).

Nested (additive) models are supported via lists:
    model=['nugget', 'exponential'], params=[[0.1], [0.9, 10.0]]
"""

from __future__ import annotations
import numpy as np
from scipy import special
from .distance import coord2dist


# ── individual models ────────────────────────────────────────────

def exponential_cov(h, params):
    """C(h) = sill·exp(−3h/range).  params=[sill, range]"""
    s, r = params[0], params[1]
    return s * np.exp(-3.0 * np.asarray(h, dtype=np.float64) / r)


def gaussian_cov(h, params):
    """C(h) = sill·exp(−3(h/range)²).  params=[sill, range]"""
    s, r = params[0], params[1]
    return s * np.exp(-3.0 * (np.asarray(h, dtype=np.float64) / r) ** 2)


def spherical_cov(h, params):
    """C(h) = sill·(1 − 1.5t + 0.5t³)  for t = min(h/range, 1).  params=[sill, range]"""
    s, r = params[0], params[1]
    t = np.minimum(np.asarray(h, dtype=np.float64) / r, 1.0)
    return s * (1.0 - 1.5 * t + 0.5 * t ** 3)


def matern_cov(h, params):
    """Matérn covariance.  params=[sill, range, nu].

    Special cases: nu=0.5 → exponential,  nu→∞ → Gaussian.
    """
    s, r, nu = params[0], params[1], params[2]
    h = np.asarray(h, dtype=np.float64)
    out = np.full_like(h, s, dtype=np.float64)
    mask = h > 1e-15
    if np.any(mask):
        sc = np.sqrt(2.0 * nu) * h[mask] / r
        out[mask] = s * (2.0 ** (1 - nu) / special.gamma(nu)) * sc ** nu * special.kv(nu, sc)
    return out


def nugget_cov(h, params):
    """Pure nugget: C(h) = sill · δ(h ≈ 0).  params=[sill]"""
    return params[0] * (np.asarray(h) < 1e-10).astype(np.float64)


def hole_cos_cov(h, params):
    """Hole-effect cosine: C(h) = sill·cos(π h/range).  params=[sill, range]"""
    return params[0] * np.cos(np.pi * np.asarray(h, dtype=np.float64) / params[1])


# ── registry ─────────────────────────────────────────────────────

COV_MODELS: dict = {
    "exponential": exponential_cov,
    "gaussian": gaussian_cov,
    "spherical": spherical_cov,
    "matern": matern_cov,
    "nugget": nugget_cov,
    "hole_cos": hole_cos_cov,
}


# ── evaluation helpers ───────────────────────────────────────────

def eval_cov(h, model, params):
    """Evaluate a single or *nested* covariance model.

    Parameters
    ----------
    h      : distance array (any shape)
    model  : str or list[str]
    params : list (single) or list[list] (nested)
    """
    h = np.asarray(h, dtype=np.float64)
    if isinstance(model, str):
        return COV_MODELS[model](h, params)
    return sum(COV_MODELS[m](h, p) for m, p in zip(model, params))


def build_cov_matrix(c1, c2, model, params):
    """Covariance matrix between two coordinate sets.

    Parameters
    ----------
    c1, c2 : (n, d) arrays
    model  : str or list[str]
    params : corresponding parameter(s)

    Returns
    -------
    (n1, n2) covariance matrix
    """
    return eval_cov(coord2dist(np.atleast_2d(c1), np.atleast_2d(c2)), model, params)


def build_cov_matrix_st(c1, t1, c2, t2,
                        model_s, params_s, model_t, params_t, sigma2):
    """Separable space-time covariance matrix.

    C = σ² · Cs(‖Δx‖) · Ct(|Δt|)

    Spatial/temporal models should have sill = 1;  overall sill in *sigma2*.
    """
    hs = coord2dist(np.atleast_2d(c1), np.atleast_2d(c2))
    ht = np.abs(np.atleast_1d(t1)[:, None] - np.atleast_1d(t2)[None, :])
    return sigma2 * eval_cov(hs, model_s, params_s) * eval_cov(ht, model_t, params_t)


# ── space-time covariance models ────────────────────────────────
#
# Follows the MATLAB BMElib convention:
#   - Separable:     'modelS/modelT'   →  C(r,t) = sill · Cs(r) · Ct(t)
#   - Non-separable: 'modelST'         →  custom function of (Ds, Dt)
#
# In MATLAB the eval syntax is  coord2K → Ds, Dt → covmodelST(Ds, Dt, param).
# Here we provide an eval_cov_st() helper that works on lag arrays directly.

def gaussian_cov_st(hs, ht, params):
    """Non-separable Gaussian S/T covariance (≈ MATLAB ``gaussianCST``).

    Uses a space-time metric: d_st = hs + k·ht, then applies a
    Gaussian covariance to that combined distance.

    Parameters
    ----------
    hs     : spatial lag(s)  (≥ 0)
    ht     : temporal lag(s) (≥ 0), broadcast-compatible with *hs*
    params : [sill, range_st, k]
             sill     – variance at (0, 0)
             range_st – S/T range (distance to reach 5 % of sill)
             k        – S/T metric  (d_st = hs + k·ht)
    """
    sill, r_st, k = params[0], params[1], params[2]
    d = np.asarray(hs, dtype=np.float64) + k * np.asarray(ht, dtype=np.float64)
    return sill * np.exp(-3.0 * (d / r_st) ** 2)


def exponential_cov_st(hs, ht, params):
    """Non-separable Exponential S/T covariance.

    Parameters
    ----------
    hs, ht : spatial / temporal lags
    params : [sill, range_st, k]   (d_st = hs + k·ht)
    """
    sill, r_st, k = params[0], params[1], params[2]
    d = np.asarray(hs, dtype=np.float64) + k * np.asarray(ht, dtype=np.float64)
    return sill * np.exp(-3.0 * d / r_st)


def nugget_cov_st(hs, ht, params):
    """S/T nugget: C = sill when *both* hs ≈ 0 and ht ≈ 0 (≈ MATLAB ``nuggetCST``).

    params : [sill]
    """
    return params[0] * ((np.asarray(hs) < 1e-10) & (np.asarray(ht) < 1e-10)).astype(np.float64)


COV_MODELS_ST: dict = {
    "gaussian_st": gaussian_cov_st,
    "exponential_st": exponential_cov_st,
    "nugget_st": nugget_cov_st,
}


def eval_cov_st(hs, ht, model_s, params_s, model_t=None, params_t=None, sill=None):
    """Evaluate a space-time covariance model.

    Supports two calling conventions:

    **Separable** (like MATLAB ``'modelS/modelT'``)::

        eval_cov_st(hs, ht, 'exponential', [1,5], 'exponential', [1,2], sill=1.0)
        # C(r,t) = sill · Cs(r) · Ct(t)

    **Non-separable** (like MATLAB ``'modelST'``)::

        eval_cov_st(hs, ht, 'gaussian_st', [1.0, 5.0, 0.5])
        # C(r,t) = gaussian_cov_st(hs, ht, params)

    Parameters
    ----------
    hs        : spatial lag(s)
    ht        : temporal lag(s), broadcast-compatible with *hs*
    model_s   : spatial model name (separable) **or** ST model name (non-separable)
    params_s  : parameters for the spatial (or ST) model
    model_t   : temporal model name (separable only, None for non-separable)
    params_t  : temporal parameters  (separable only)
    sill      : overall sill for separable model (default 1.0)

    Returns
    -------
    Covariance value(s), same shape as broadcast(hs, ht).
    """
    hs = np.asarray(hs, dtype=np.float64)
    ht = np.asarray(ht, dtype=np.float64)

    # ── non-separable ──
    if model_t is None:
        if model_s in COV_MODELS_ST:
            return COV_MODELS_ST[model_s](hs, ht, params_s)
        raise KeyError(f"Unknown ST model '{model_s}'. "
                       f"Available: {list(COV_MODELS_ST.keys())}")

    # ── separable ──
    if sill is None:
        sill = 1.0
    return sill * eval_cov(hs, model_s, params_s) * eval_cov(ht, model_t, params_t)
