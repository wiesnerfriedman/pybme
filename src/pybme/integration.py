"""Core numerical integration engine for BME.

Implements the expectation  E_{x ~ N(μ, Σ)}[ ∏ fᵢ(xᵢ) ]  that is the
heart of BME — the integral that distinguishes it from kriging.

Methods:
  * Gauss-Hermite tensor-product quadrature  (up to ~8 soft dimensions)
  * Monte Carlo fallback  (> 8 dimensions)
"""

from __future__ import annotations
import math
from typing import List

import numpy as np
from .soft_data import SoftPDF

# ── Gauss-Hermite node cache ─────────────────────────────────

_GH_CACHE: dict = {}


def _gh_nodes(n: int):
    """Cached physicist-Hermite quadrature nodes and weights."""
    if n not in _GH_CACHE:
        _GH_CACHE[n] = np.polynomial.hermite.hermgauss(n)
    return _GH_CACHE[n]


def _adaptive_nquad(ns: int, base: int = 15) -> int:
    """Select quadrature points per dimension to keep total cost manageable."""
    if ns <= 1:
        return max(base, 20)
    if ns == 2:
        return min(base, 12)
    if ns == 3:
        return min(base, 8)
    if ns == 4:
        return min(base, 6)
    if ns <= 6:
        return min(base, 5)
    if ns <= 8:
        return min(base, 4)
    return 0  # → Monte Carlo


def integrate_soft_product(soft_pdfs: List[SoftPDF],
                           mu: np.ndarray, cov: np.ndarray,
                           n_quad: int = 15) -> float:
    """Compute  E_{x ~ N(μ, Σ)}[ ∏ᵢ fᵢ(xᵢ) ]  via Gauss-Hermite quadrature.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF, one per soft datum
    mu        : conditional mean vector  (ns,)
    cov       : conditional covariance matrix  (ns, ns)
    n_quad    : base quadrature points per dimension

    Returns
    -------
    float  (strictly positive; clamped to 1e-300 minimum)
    """
    ns = len(soft_pdfs)
    if ns == 0:
        return 1.0
    mu = np.asarray(mu, dtype=np.float64).ravel()
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T) + np.eye(ns) * 1e-10

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        ev, Q = np.linalg.eigh(cov)
        L = Q @ np.diag(np.sqrt(np.maximum(ev, 1e-10)))

    nq = _adaptive_nquad(ns, n_quad)
    if nq == 0 or ns > 8:
        return _mc_integrate(soft_pdfs, mu, L, n_samples=30000)

    nodes, weights = _gh_nodes(nq)

    if ns == 1:
        x = mu[0] + math.sqrt(2.0) * L[0, 0] * nodes
        fv = soft_pdfs[0].evaluate(x)
        return max(float(np.dot(weights, fv)) / math.sqrt(math.pi), 1e-300)

    # Tensor-product Gauss-Hermite  (cost = nq^ns, feasible for ns ≤ 8)
    grids = np.meshgrid(*([nodes] * ns), indexing="ij")
    u = np.column_stack([g.ravel() for g in grids])
    wg = np.meshgrid(*([weights] * ns), indexing="ij")
    w = np.ones(u.shape[0])
    for wgi in wg:
        w *= wgi.ravel()

    x = mu[None, :] + math.sqrt(2.0) * (u @ L.T)

    prod_f = np.ones(x.shape[0])
    for i, sp in enumerate(soft_pdfs):
        prod_f *= sp.evaluate(x[:, i])

    return max(float(np.dot(w, prod_f)) / (math.pi ** (ns / 2.0)), 1e-300)


def _mc_integrate(soft_pdfs, mu, L, n_samples=30000):
    """Monte Carlo fallback for high-dimensional integrals."""
    ns = len(soft_pdfs)
    u = np.random.randn(n_samples, ns)
    x = mu[None, :] + u @ L.T
    prod_f = np.ones(n_samples)
    for i, sp in enumerate(soft_pdfs):
        prod_f *= sp.evaluate(x[:, i])
    return max(float(np.mean(prod_f)), 1e-300)
