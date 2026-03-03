"""Core numerical integration engine for BME.

Implements the expectation  E_{x ~ N(μ, Σ)}[ ∏ fᵢ(xᵢ) ]  that is the
heart of BME — the integral that distinguishes it from kriging.

Methods:
  * Gauss-Hermite tensor-product quadrature  (up to ~8 soft dimensions)
  * Monte Carlo fallback  (> 8 dimensions)
  * **Laplace approximation** — O(ns³) per evaluation, accurate
    for near-Gaussian soft PDFs, replaces exponential-cost GH for ns ≥ 6.
    Original contribution by Corinne Wiesner-Friedman (not part of
    MATLAB BMElib), inspired by the INLA methodology of Rue et al. (2009).

    Rue H., Martino S. & Chopin N. (2009).  Approximate Bayesian inference
    for latent Gaussian models by using integrated nested Laplace
    approximations.  JRSS-B, 71(2), 319–392.
    https://doi.org/10.1111/j.1467-9868.2008.00700.x
"""

from __future__ import annotations
import math
from typing import List, Optional

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


# ════════════════════════════════════════════════════════════════
# BATCH integration — evaluate over a grid of conditional means
# ════════════════════════════════════════════════════════════════

def integrate_soft_product_batch(soft_pdfs: List[SoftPDF],
                                 mu_grid: np.ndarray,
                                 cov: np.ndarray,
                                 n_quad: int = 15) -> np.ndarray:
    """Vectorised batch:  E_{x ~ N(μ_j, Σ)}[ ∏ fᵢ(xᵢ) ]  for j = 1 … M.

    This is the same integral as ``integrate_soft_product`` but evaluated
    for many different mean vectors at once.  The covariance matrix (and
    therefore the Cholesky factor and quadrature nodes) are shared across
    all M evaluations, which amortises the expensive setup.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF  (length ns)
    mu_grid   : (M, ns) array — one mean vector per row
    cov       : (ns, ns) conditional covariance (same for all rows)
    n_quad    : base quadrature points per dimension

    Returns
    -------
    (M,) array of integral values  (clamped ≥ 1e-300)
    """
    ns = len(soft_pdfs)
    M = mu_grid.shape[0]
    if ns == 0:
        return np.ones(M)

    mu_grid = np.asarray(mu_grid, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T) + np.eye(ns) * 1e-10

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        ev, Q = np.linalg.eigh(cov)
        L = Q @ np.diag(np.sqrt(np.maximum(ev, 1e-10)))

    nq = _adaptive_nquad(ns, n_quad)
    if nq == 0 or ns > 8:
        return _mc_integrate_batch(soft_pdfs, mu_grid, L)

    nodes, weights = _gh_nodes(nq)

    if ns == 1:
        # u: (nq,),  L[0,0]: scalar
        # x_j = mu_j + sqrt(2) * L[0,0] * nodes   →  (M, nq) via broadcast
        s2L = math.sqrt(2.0) * L[0, 0]
        x = mu_grid[:, 0:1] + s2L * nodes[None, :]  # (M, nq)
        fv = soft_pdfs[0].evaluate(x.ravel()).reshape(M, -1)  # (M, nq)
        return np.maximum(fv @ weights / math.sqrt(math.pi), 1e-300)

    # Tensor-product nodes  (cost = nq^ns per mean-vector)
    grids = np.meshgrid(*([nodes] * ns), indexing="ij")
    u = np.column_stack([g.ravel() for g in grids])     # (Q, ns)  Q = nq^ns
    wg = np.meshgrid(*([weights] * ns), indexing="ij")
    w = np.ones(u.shape[0])
    for wgi in wg:
        w *= wgi.ravel()                                 # (Q,)

    # Transformed nodes:  x_j = mu_j + sqrt(2) * u @ L.T  → (M, Q, ns)
    #   but we avoid allocating (M, Q, ns) by broadcasting cleverly.
    uL = math.sqrt(2.0) * (u @ L.T)                      # (Q, ns)

    # For each evaluation we need  prod_i f_i(x_ji)
    # Evaluate all soft PDFs at all (M * Q) points:
    #   x[:, :, i] = mu_grid[:, i:i+1] + uL[:, i]
    # Then prod over i, dot with w.
    result = np.empty(M)
    # Process in chunks to limit memory  (chunk × Q × ns ≤ ~50 M floats)
    chunk = max(1, min(M, int(50_000_000 / max(u.shape[0] * ns, 1))))
    invpi = math.pi ** (ns / 2.0)
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        Mc = end - start
        # (Mc, Q, ns)
        x_chunk = mu_grid[start:end, None, :] + uL[None, :, :]
        prod_f = np.ones((Mc, u.shape[0]))
        for i, sp in enumerate(soft_pdfs):
            prod_f *= sp.evaluate(x_chunk[:, :, i].ravel()).reshape(Mc, -1)
        result[start:end] = (prod_f @ w) / invpi

    return np.maximum(result, 1e-300)


def _mc_integrate_batch(soft_pdfs, mu_grid, L, n_samples=30000):
    """Monte Carlo fallback for batch integration."""
    ns = len(soft_pdfs)
    M = mu_grid.shape[0]
    u = np.random.randn(n_samples, ns)
    uL = u @ L.T                                          # (n_samples, ns)
    result = np.empty(M)
    chunk = max(1, min(M, int(50_000_000 / max(n_samples * ns, 1))))
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        Mc = end - start
        x = mu_grid[start:end, None, :] + uL[None, :, :]  # (Mc, n_samples, ns)
        prod_f = np.ones((Mc, n_samples))
        for i, sp in enumerate(soft_pdfs):
            prod_f *= sp.evaluate(x[:, :, i].ravel()).reshape(Mc, -1)
        result[start:end] = np.mean(prod_f, axis=1)
    return np.maximum(result, 1e-300)


# ════════════════════════════════════════════════════════════════
# LAPLACE APPROXIMATION  (INLA-style, for soft data)
# ════════════════════════════════════════════════════════════════
#
# Instead of tensor-product Gauss-Hermite (cost O(nq^ns)) or Monte
# Carlo, approximate the integral via a second-order Taylor expansion
# of the log-integrand around its mode.  Cost: O(ns³) per evaluation
# (find mode via Newton + Hessian).
#
# References:
#   Rue, Martino & Chopin (2009) — INLA, JRSS-B 71(2), 319–392.
#   Tierney & Kadane (1986) — J. Amer. Statist. Assoc. 81, 82–86.

def _soft_log_pdf(soft_pdfs: List[SoftPDF], x: np.ndarray) -> float:
    """Sum of log soft-PDF values:  Σ log fᵢ(xᵢ).

    Returns -inf when any fᵢ(xᵢ) ≤ 0.
    """
    s = 0.0
    for i, sp in enumerate(soft_pdfs):
        fi = sp.evaluate(float(x[i]))
        if fi <= 0.0:
            return -np.inf
        s += math.log(fi)
    return s


def _log_target(x: np.ndarray, soft_pdfs: List[SoftPDF],
                mu: np.ndarray, Q: np.ndarray) -> float:
    """log g(x) = -0.5 (x-μ)^T Q (x-μ) + Σ log fᵢ(xᵢ).

    Q is the precision matrix (inverse covariance).
    """
    d = x - mu
    return -0.5 * float(d @ Q @ d) + _soft_log_pdf(soft_pdfs, x)


def _adaptive_eps(soft_pdfs: List[SoftPDF]) -> np.ndarray:
    """Choose per-dimension finite-difference step sizes.

    SoftPDF objects use piecewise-linear interpolation, so the step
    must span several grid segments to capture curvature.  We use
    ~3× the grid spacing of each PDF, clamped to [0.05, 2.0].
    """
    eps_arr = np.empty(len(soft_pdfs))
    for i, sp in enumerate(soft_pdfs):
        zg = sp.z_grid
        if len(zg) > 1:
            grid_dx = (zg[-1] - zg[0]) / (len(zg) - 1)
            eps_arr[i] = np.clip(3.0 * grid_dx, 0.05, 2.0)
        else:
            lo, hi = sp.support
            eps_arr[i] = np.clip((hi - lo) / 30.0, 0.05, 2.0)
    return eps_arr


def _log_target_grad(x: np.ndarray, soft_pdfs: List[SoftPDF],
                     mu: np.ndarray, Q: np.ndarray,
                     eps_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Gradient of log g(x) w.r.t. x.

    The Gaussian part is computed analytically (−Q(x−μ)); the soft-PDF
    part uses central finite differences with adaptive step sizes.
    """
    ns = len(x)
    if eps_arr is None:
        eps_arr = _adaptive_eps(soft_pdfs)
    # Analytic Gaussian gradient
    grad = -Q @ (x - mu)
    # Add numerical soft-PDF gradient via central differences
    for i in range(ns):
        ei = eps_arr[i]
        x_p, x_m = x.copy(), x.copy()
        x_p[i] += ei
        x_m[i] -= ei
        lp = _soft_log_pdf(soft_pdfs, x_p)
        lm = _soft_log_pdf(soft_pdfs, x_m)
        grad[i] += (lp - lm) / (2.0 * ei)
    return grad


def _log_target_hessian(x: np.ndarray, soft_pdfs: List[SoftPDF],
                        mu: np.ndarray, Q: np.ndarray,
                        eps_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Hessian of log g(x).

    The Gaussian part contributes −Q exactly; the soft-PDF curvature
    is approximated via central finite differences with adaptive step
    sizes (must span multiple interpolation grid segments).
    """
    ns = len(x)
    if eps_arr is None:
        eps_arr = _adaptive_eps(soft_pdfs)
    H = -Q.copy()  # Exact Gaussian contribution
    # Add soft-PDF Hessian numerically
    l0 = _soft_log_pdf(soft_pdfs, x)
    for i in range(ns):
        ei = eps_arr[i]
        x_p, x_m = x.copy(), x.copy()
        x_p[i] += ei
        x_m[i] -= ei
        lp = _soft_log_pdf(soft_pdfs, x_p)
        lm = _soft_log_pdf(soft_pdfs, x_m)
        H[i, i] += (lp - 2.0 * l0 + lm) / (ei ** 2)
        for j in range(i + 1, ns):
            ej = eps_arr[j]
            xpp = x.copy(); xpp[i] += ei; xpp[j] += ej
            xpm = x.copy(); xpm[i] += ei; xpm[j] -= ej
            xmp = x.copy(); xmp[i] -= ei; xmp[j] += ej
            xmm = x.copy(); xmm[i] -= ei; xmm[j] -= ej
            cross = (_soft_log_pdf(soft_pdfs, xpp) - _soft_log_pdf(soft_pdfs, xpm)
                     - _soft_log_pdf(soft_pdfs, xmp) + _soft_log_pdf(soft_pdfs, xmm)
                     ) / (4.0 * ei * ej)
            H[i, j] += cross
            H[j, i] = H[i, j]
    return H


def _find_mode(soft_pdfs: List[SoftPDF], mu: np.ndarray,
               Q: np.ndarray, max_iter: int = 30,
               tol: float = 1e-6) -> np.ndarray:
    """Newton's method to find the mode of log g(x).

    Uses a modified Newton step with backtracking line search.
    Clamps iterates to stay within soft-PDF supports.
    """
    ns = len(mu)
    # Compute support bounds for clamping
    lo = np.array([sp.support[0] for sp in soft_pdfs])
    hi = np.array([sp.support[1] for sp in soft_pdfs])
    # Start from mu clamped to within supports
    x = np.clip(mu.copy(), lo, hi)

    eps_arr = _adaptive_eps(soft_pdfs)
    for _ in range(max_iter):
        g = _log_target_grad(x, soft_pdfs, mu, Q, eps_arr)
        H = _log_target_hessian(x, soft_pdfs, mu, Q, eps_arr)
        try:
            step = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            ev = np.linalg.eigvalsh(H)
            shift = max(-ev.min() + 1e-4, 1e-4)
            step = np.linalg.solve(-(H - shift * np.eye(ns)), g)

        # Backtracking line search with clamping
        f0 = _log_target(x, soft_pdfs, mu, Q)
        alpha = 1.0
        for _ in range(20):
            x_new = np.clip(x + alpha * step, lo, hi)
            if _log_target(x_new, soft_pdfs, mu, Q) > f0 - 1e-4:
                break
            alpha *= 0.5
        x = np.clip(x + alpha * step, lo, hi)
        if np.linalg.norm(alpha * step) < tol:
            break
    return x


def integrate_soft_laplace(soft_pdfs: List[SoftPDF],
                           mu: np.ndarray, cov: np.ndarray) -> float:
    """Laplace approximation to  E_{x ~ N(μ, Σ)}[ ∏ fᵢ(xᵢ) ].

    Cost: O(ns³) per call — independent of quadrature order.
    Accurate when soft PDFs are close to log-concave (Gaussian, interval,
    truncated-normal, moderately skewed lognormal).

    Parameters
    ----------
    soft_pdfs : list of SoftPDF (length ns)
    mu        : (ns,) conditional mean
    cov       : (ns, ns) conditional covariance

    Returns
    -------
    float ≥ 1e-300
    """
    ns = len(soft_pdfs)
    if ns == 0:
        return 1.0

    mu = np.asarray(mu, dtype=np.float64).ravel()
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T) + np.eye(ns) * 1e-10
    Q = np.linalg.inv(cov)

    try:
        x_star = _find_mode(soft_pdfs, mu, Q)
        H = _log_target_hessian(x_star, soft_pdfs, mu, Q)
        neg_H = -H
        ev = np.linalg.eigvalsh(neg_H)
        if ev.min() <= 0:
            neg_H += (abs(ev.min()) + 1e-6) * np.eye(ns)

        _, logdet_post = np.linalg.slogdet(neg_H)
        _, logdet_prior = np.linalg.slogdet(Q)
        log_g = _log_target(x_star, soft_pdfs, mu, Q)
        if not np.isfinite(log_g) or not np.isfinite(logdet_post):
            return 1e-300
    except Exception:
        return 1e-300

    # E[∏fᵢ] ≈ |Q|^{1/2} |neg_H|^{-1/2} exp(log_g(x*))
    log_I = log_g + 0.5 * (logdet_prior - logdet_post)
    return max(math.exp(min(log_I, 500.0)), 1e-300)


def integrate_soft_laplace_batch(soft_pdfs: List[SoftPDF],
                                 mu_grid: np.ndarray,
                                 cov: np.ndarray) -> np.ndarray:
    """Batch Laplace approximation for multiple conditional means.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF (length ns)
    mu_grid   : (M, ns) — one mean per row
    cov       : (ns, ns) shared covariance

    Returns
    -------
    (M,) array of integral values
    """
    ns = len(soft_pdfs)
    M = mu_grid.shape[0]
    if ns == 0:
        return np.ones(M)

    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T) + np.eye(ns) * 1e-10
    Q = np.linalg.inv(cov)
    _, logdet_prior = np.linalg.slogdet(Q)

    result = np.empty(M)
    x_prev = None
    for j in range(M):
        mu_j = mu_grid[j]
        try:
            x_star = _find_mode(soft_pdfs, mu_j, Q, max_iter=20)
            x_prev = x_star

            H = _log_target_hessian(x_star, soft_pdfs, mu_j, Q)
            neg_H = -H
            ev = np.linalg.eigvalsh(neg_H)
            if ev.min() <= 0:
                neg_H += (abs(ev.min()) + 1e-6) * np.eye(ns)

            _, logdet_post = np.linalg.slogdet(neg_H)
            log_g = _log_target(x_star, soft_pdfs, mu_j, Q)
            if not np.isfinite(log_g) or not np.isfinite(logdet_post):
                result[j] = 1e-300
                continue
            log_I = log_g + 0.5 * (logdet_prior - logdet_post)
            result[j] = max(math.exp(min(log_I, 500.0)), 1e-300)
        except Exception:
            result[j] = 1e-300

    return result
