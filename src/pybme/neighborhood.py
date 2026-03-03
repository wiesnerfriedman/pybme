"""Neighbourhood selection for BME estimation points."""

from __future__ import annotations
import numpy as np
from .distance import coord2dist


def select_neighbors(ck: np.ndarray, c_data: np.ndarray,
                     n_valid: int, nmax: int, dmax: float) -> np.ndarray:
    """Return indices of the ≤ *nmax* nearest data points within *dmax* of *ck*.

    Parameters
    ----------
    ck      : (d,) estimation-point coordinates
    c_data  : (N, d) data coordinates
    n_valid : number of valid rows in *c_data*
    nmax    : max neighbours to return
    dmax    : max distance

    Returns
    -------
    1-D integer index array (sorted by ascending distance)
    """
    if n_valid == 0:
        return np.array([], dtype=int)
    c_data = np.atleast_2d(c_data[:n_valid])
    d = coord2dist(np.atleast_1d(ck).reshape(1, -1), c_data).ravel()
    within = np.where(d <= dmax)[0]
    if len(within) == 0:
        return np.array([], dtype=int)
    return within[np.argsort(d[within])][:nmax]


def select_neighbors_st(ck, tk, c_data, t_data,
                        nmax, dmax_s, dmax_t, st_ratio=1.0):
    """Space-time neighbourhood with separate spatial and temporal distance limits.

    Combined ranking distance: ``ds + st_ratio * dt``.
    """
    if len(c_data) == 0:
        return np.array([], dtype=int)
    ds = coord2dist(np.asarray(ck).reshape(1, -1), np.atleast_2d(c_data)).ravel()
    dt = np.abs(float(tk) - np.atleast_1d(t_data))
    within = np.where((ds <= dmax_s) & (dt <= dmax_t))[0]
    if len(within) == 0:
        return np.array([], dtype=int)
    d_comb = ds[within] + st_ratio * dt[within]
    return within[np.argsort(d_comb)][:nmax]
