"""Neighbourhood selection for BME estimation points.

Two modes of operation
----------------------
1. **Ad-hoc** (original API): ``select_neighbors`` / ``select_neighbors_st``
   compute brute-force distances per query.  Fine for ≤ ~1 000 data points.

2. **Index-accelerated** (new): build a ``SpatialIndex`` or
   ``SpatialTemporalIndex`` once from the data, then call ``.query()``
   for each estimation point in O(log N).  Essential for radar-scale data.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .distance import coord2dist


# ════════════════════════════════════════════════════════════════
# Legacy brute-force API (preserved for backward compatibility)
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# KD-tree accelerated spatial index  (build once, query many)
# ════════════════════════════════════════════════════════════════

class SpatialIndex:
    """KD-tree backed spatial neighbour index.

    Build once from all data coordinates, then call ``query()`` for each
    estimation point in O(log N) instead of O(N).

    Parameters
    ----------
    coords : (N, d) data coordinates
    leafsize : cKDTree leaf size (tune for performance; 16–32 is typical)

    Example
    -------
    >>> idx = SpatialIndex(ch)
    >>> neighbours = idx.query(ck_i, nmax=20, dmax=50.0)
    """

    def __init__(self, coords: np.ndarray, leafsize: int = 16):
        self.coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        self.n = self.coords.shape[0]
        self._tree = cKDTree(self.coords, leafsize=leafsize) if self.n > 0 else None

    def query(self, ck: np.ndarray, nmax: int = 20,
              dmax: float = np.inf) -> np.ndarray:
        """Return indices of ≤ *nmax* nearest neighbours within *dmax*.

        Parameters
        ----------
        ck   : (d,) query point
        nmax : max neighbours
        dmax : max Euclidean distance

        Returns
        -------
        1-D integer index array sorted by ascending distance
        """
        if self._tree is None or self.n == 0:
            return np.array([], dtype=int)
        k = min(nmax, self.n)
        dd, ii = self._tree.query(np.atleast_1d(ck), k=k,
                                  distance_upper_bound=dmax)
        if k == 1:
            dd, ii = np.atleast_1d(dd), np.atleast_1d(ii)
        mask = ii < self.n  # cKDTree returns self.n for "not found"
        return ii[mask]

    def query_batch(self, ck: np.ndarray, nmax: int = 20,
                    dmax: float = np.inf) -> list:
        """Batch query for multiple estimation points.

        Parameters
        ----------
        ck : (K, d) estimation-point coordinates

        Returns
        -------
        list of 1-D integer index arrays, one per estimation point
        """
        ck = np.atleast_2d(ck)
        if self._tree is None or self.n == 0:
            return [np.array([], dtype=int) for _ in range(ck.shape[0])]
        k = min(nmax, self.n)
        dd, ii = self._tree.query(ck, k=k, distance_upper_bound=dmax)
        if k == 1:
            dd = dd.reshape(-1, 1)
            ii = ii.reshape(-1, 1)
        out = []
        for row in range(ii.shape[0]):
            mask = ii[row] < self.n
            out.append(ii[row][mask])
        return out


class SpatialTemporalIndex:
    """Spatial KD-tree + temporal filtering for space-time data.

    The spatial component uses a KD-tree.  Temporal filtering is applied
    as a post-filter on the spatial candidates (since temporal "distance"
    is usually 1-D and cheap to check linearly on a small candidate set).

    Parameters
    ----------
    coords : (N, d_spatial) spatial coordinates
    times  : (N,) temporal coordinates
    leafsize : cKDTree leaf size
    """

    def __init__(self, coords: np.ndarray, times: np.ndarray,
                 leafsize: int = 16):
        self.coords = np.ascontiguousarray(np.atleast_2d(coords), dtype=np.float64)
        self.times = np.atleast_1d(times).astype(np.float64)
        self.n = self.coords.shape[0]
        self._tree = cKDTree(self.coords, leafsize=leafsize) if self.n > 0 else None

    def query(self, ck: np.ndarray, tk: float,
              nmax: int = 20, dmax_s: float = np.inf,
              dmax_t: float = np.inf, st_ratio: float = 1.0) -> np.ndarray:
        """Return ≤ *nmax* neighbours within spatial and temporal distance limits.

        Ranking: ``ds + st_ratio * dt``.

        Parameters
        ----------
        ck       : (d,) spatial query point
        tk       : temporal query value
        nmax     : max neighbours
        dmax_s   : spatial distance limit
        dmax_t   : temporal distance limit
        st_ratio : weighting for temporal distance in combined ranking

        Returns
        -------
        1-D integer index array
        """
        if self._tree is None or self.n == 0:
            return np.array([], dtype=int)
        # spatial candidates: retrieve more than nmax to allow temporal filtering
        k_spatial = min(max(nmax * 4, 50), self.n)
        dd_s, ii_s = self._tree.query(np.atleast_1d(ck), k=k_spatial,
                                      distance_upper_bound=dmax_s)
        if k_spatial == 1:
            dd_s, ii_s = np.atleast_1d(dd_s), np.atleast_1d(ii_s)
        valid = ii_s < self.n
        ii_s, dd_s = ii_s[valid], dd_s[valid]
        if len(ii_s) == 0:
            return np.array([], dtype=int)
        # temporal filter
        dt = np.abs(tk - self.times[ii_s])
        tmask = dt <= dmax_t
        ii_s, dd_s, dt = ii_s[tmask], dd_s[tmask], dt[tmask]
        if len(ii_s) == 0:
            return np.array([], dtype=int)
        d_comb = dd_s + st_ratio * dt
        order = np.argsort(d_comb)[:nmax]
        return ii_s[order]
