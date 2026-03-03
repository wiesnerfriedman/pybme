"""Core numerical integration engine for BME.

Implements the expectation  E_{x ~ N(μ, Σ)}[ ∏ fᵢ(xᵢ) ]  that is the
heart of BME — the integral that distinguishes it from kriging.

Methods:
  * Gauss-Hermite tensor-product quadrature  (up to ~8 soft dimensions)
  * Monte Carlo fallback  (> 8 dimensions)
  * **Laplace approximation** — O(ns³) per evaluation, accurate
    for near-Gaussian soft PDFs, replaces exponential-cost GH for ns ≥ 6.
  * **Expectation Propagation (EP)** — O(ns² · iters) per evaluation,
    iteratively approximates each soft factor with a Gaussian site.
    More accurate than Laplace for non-log-concave PDFs (uniforms,
    intervals, bimodal).
  * **Quasi-Monte Carlo (QMC)** — O(n_samples × ns) with O(1/N)
    convergence via Sobol sequences instead of O(1/√N) for plain MC.
  * **Laplace Importance Sampling (LIS)** — uses the Laplace mode and
    Hessian as a Gaussian proposal for importance sampling, giving
    unbiased results with much lower variance than raw MC.

Original contributions by Corinne Wiesner-Friedman (not part of MATLAB
BMElib), inspired by:

    Rue H., Martino S. & Chopin N. (2009).  Approximate Bayesian inference
    for latent Gaussian models by using integrated nested Laplace
    approximations.  JRSS-B, 71(2), 319–392.
    https://doi.org/10.1111/j.1467-9868.2008.00700.x

    Minka T. (2001).  Expectation Propagation for approximate Bayesian
    inference.  UAI 2001, 362–369.

    Rasmussen C.E. & Williams C.K.I. (2006).  Gaussian Processes for
    Machine Learning.  MIT Press, Ch. 3.
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


# ════════════════════════════════════════════════════════════════
# QUASI-MONTE CARLO  (QMC — Sobol sequences)
# ════════════════════════════════════════════════════════════════
#
# Same idea as plain MC, but uses low-discrepancy Sobol points
# instead of pseudo-random draws.  Convergence: O(1/N) instead of
# O(1/√N).  Useful when ns is moderate (3-20) and GH is too
# expensive.
#
# References:
#   Joe S. & Kuo F.Y. (2008).  Constructing Sobol sequences with
#   better two-dimensional projections.
#   Scipy: scipy.stats.qmc.Sobol

def integrate_soft_qmc(soft_pdfs: List[SoftPDF],
                       mu: np.ndarray, cov: np.ndarray,
                       n_samples: int = 4096) -> float:
    """Quasi-Monte Carlo integration for  E_{x ~ N(μ, Σ)}[ ∏ fᵢ(xᵢ) ].

    Uses a scrambled Sobol sequence mapped through the inverse-normal
    CDF so that points are distributed as N(μ, Σ).

    Parameters
    ----------
    soft_pdfs : list of SoftPDF  (length ns)
    mu        : (ns,) conditional mean
    cov       : (ns, ns) conditional covariance
    n_samples : number of Sobol points (rounded up to next power of 2)

    Returns
    -------
    float ≥ 1e-300
    """
    from scipy.stats import qmc, norm as sp_norm

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

    # Round n_samples up to next power of 2 (Sobol requirement)
    m = max(int(np.ceil(np.log2(max(n_samples, 2)))), 1)
    sampler = qmc.Sobol(d=ns, scramble=True)
    u_unit = sampler.random_base2(m)   # (2^m, ns) in [0, 1)
    u_norm = sp_norm.ppf(np.clip(u_unit, 1e-10, 1 - 1e-10))  # → N(0,1)
    x = mu[None, :] + u_norm @ L.T    # (N, ns)

    prod_f = np.ones(x.shape[0])
    for i, sp in enumerate(soft_pdfs):
        prod_f *= sp.evaluate(x[:, i])
    return max(float(np.mean(prod_f)), 1e-300)


def integrate_soft_qmc_batch(soft_pdfs: List[SoftPDF],
                             mu_grid: np.ndarray,
                             cov: np.ndarray,
                             n_samples: int = 4096) -> np.ndarray:
    """Batch QMC integration for multiple conditional means.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF  (length ns)
    mu_grid   : (M, ns) array — one mean per row
    cov       : (ns, ns) conditional covariance
    n_samples : Sobol sample count (rounded up to power of 2)

    Returns
    -------
    (M,) array of integral values (clamped ≥ 1e-300)
    """
    from scipy.stats import qmc, norm as sp_norm

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
        ev, Qm = np.linalg.eigh(cov)
        L = Qm @ np.diag(np.sqrt(np.maximum(ev, 1e-10)))

    m = max(int(np.ceil(np.log2(max(n_samples, 2)))), 1)
    sampler = qmc.Sobol(d=ns, scramble=True)
    u_unit = sampler.random_base2(m)
    u_norm = sp_norm.ppf(np.clip(u_unit, 1e-10, 1 - 1e-10))
    uL = u_norm @ L.T   # (N, ns)
    N = uL.shape[0]

    result = np.empty(M)
    chunk = max(1, min(M, int(50_000_000 / max(N * ns, 1))))
    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        Mc = end - start
        x = mu_grid[start:end, None, :] + uL[None, :, :]  # (Mc, N, ns)
        prod_f = np.ones((Mc, N))
        for i, sp in enumerate(soft_pdfs):
            prod_f *= sp.evaluate(x[:, :, i].ravel()).reshape(Mc, -1)
        result[start:end] = np.mean(prod_f, axis=1)
    return np.maximum(result, 1e-300)


# ════════════════════════════════════════════════════════════════
# EXPECTATION PROPAGATION  (EP)
# ════════════════════════════════════════════════════════════════
#
# Approximate each non-Gaussian soft factor fᵢ(xᵢ) with a
# *Gaussian site*  t̃_i(x_i) = s_i exp(τ_i x_i - ½ λ_i x_i²),
# so that the full posterior is a product of Gaussians — exact in
# closed form.  Sites are refined iteratively by moment matching.
#
# Cost: O(ns² × iters)  — no quadrature, no sampling.
#
# References:
#   Minka T. (2001)  Expectation Propagation for approximate
#     Bayesian inference.  UAI 2001.
#   Rasmussen & Williams (2006), Ch. 3.6.

# NumPy compat: use trapezoid if available (NumPy >= 2.0)
_trapz_compat = getattr(np, "trapezoid", np.trapz)


def _ep_1d_moments(sp: SoftPDF, cavity_mean: float,
                   cavity_var: float) -> tuple:
    """Compute zeroth, first, and second moments of  f(x) * N(x|m,v).

    Uses the SoftPDF z_grid for numerical quadrature.

    Returns (Z, mean, var) where Z = ∫ f(x) N(x|m,v) dx.
    """
    z = sp.z_grid
    if len(z) < 2:
        return 1.0, cavity_mean, cavity_var

    fv = sp.evaluate(z)
    # Gaussian cavity density at the grid points
    if cavity_var <= 0:
        cavity_var = 1e-10
    sd = math.sqrt(cavity_var)
    g = np.exp(-0.5 * ((z - cavity_mean) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
    h = fv * g  # unnormalised integrand

    # Trapezoidal integration
    Z = float(_trapz_compat(h, z))
    if Z <= 0:
        return 1e-300, cavity_mean, cavity_var
    m = float(_trapz_compat(z * h, z)) / Z
    v = float(_trapz_compat((z - m) ** 2 * h, z)) / Z
    v = max(v, 1e-12)
    return Z, m, v


def integrate_soft_ep(soft_pdfs: List[SoftPDF],
                      mu: np.ndarray, cov: np.ndarray,
                      max_iter: int = 50, damp: float = 0.8,
                      tol: float = 1e-6) -> float:
    """Expectation Propagation for  E_{x ~ N(μ, Σ)}[ ∏ fᵢ(xᵢ) ].

    Iteratively approximates each soft factor with a Gaussian site
    and returns the normalising constant of the resulting Gaussian
    approximation multiplied by the accumulated site scales.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF  (length ns)
    mu        : (ns,) prior conditional mean
    cov       : (ns, ns) prior conditional covariance
    max_iter  : EP sweeps
    damp      : damping factor in (0, 1] — lower = more stable
    tol       : convergence threshold on site parameter change

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

    # Natural parameters of prior:  Λ = Σ⁻¹,  η = Λ μ
    Lambda = np.linalg.inv(cov)
    eta = Lambda @ mu

    # Site natural parameters (initially zero → non-informative)
    tau = np.zeros(ns)    # precision ≡ λ_i
    nu = np.zeros(ns)     # precision-weighted mean ≡ τ_i × m_i

    for _sweep in range(max_iter):
        max_delta = 0.0
        for i in range(ns):
            # --- Cavity distribution for dimension i ---
            Lambda_post = Lambda.copy()
            Lambda_post[np.diag_indices(ns)] += tau
            eta_post = eta + nu

            try:
                Sigma_post = np.linalg.inv(Lambda_post)
            except np.linalg.LinAlgError:
                continue
            mu_post = Sigma_post @ eta_post

            # Remove site i to get cavity
            cavity_var = Sigma_post[i, i]
            if cavity_var <= 0:
                continue
            tau_cav = 1.0 / cavity_var - tau[i]
            nu_cav = mu_post[i] / cavity_var - nu[i]
            if tau_cav <= 0:
                tau_cav = 1e-8
            cavity_mean = nu_cav / tau_cav
            cav_var = 1.0 / tau_cav

            # --- Moment matching ---
            Z_i, m_hat, v_hat = _ep_1d_moments(
                soft_pdfs[i], cavity_mean, cav_var)

            if Z_i <= 0 or v_hat <= 0:
                continue

            # New site parameters (precision parameterisation)
            tau_new = max(1.0 / v_hat - tau_cav, 1e-10)
            nu_new = m_hat / v_hat - nu_cav

            # Damped update
            tau_upd = damp * tau_new + (1.0 - damp) * tau[i]
            nu_upd = damp * nu_new + (1.0 - damp) * nu[i]

            max_delta = max(max_delta,
                            abs(tau_upd - tau[i]),
                            abs(nu_upd - nu[i]))
            tau[i] = tau_upd
            nu[i] = nu_upd

        if max_delta < tol:
            break

    # --- Compute the EP marginal-likelihood approximation ---
    # Posterior precision / mean:
    #   Λ_q = Λ + diag(τ),  η_q = η + ν,  Σ_q = Λ_q⁻¹,  μ_q = Σ_q η_q
    Lambda_q = Lambda.copy()
    Lambda_q[np.diag_indices(ns)] += tau
    eta_q = eta + nu

    try:
        sign, logdet_q = np.linalg.slogdet(Lambda_q)
        if sign <= 0:
            return 1e-300
        Sigma_q = np.linalg.inv(Lambda_q)
        mu_q = Sigma_q @ eta_q
    except np.linalg.LinAlgError:
        return 1e-300

    _, logdet_prior = np.linalg.slogdet(Lambda)

    # Compute site log-scales from the *final* cavities.
    # log ŝᵢ = log Zᵢ + ½ log(v_cav/v̂) + ½ m_cav²/v_cav − ½ m̂²/v̂
    # where Zᵢ, m̂, v̂ come from moment matching against the final cavity.
    log_s_sum = 0.0
    for i in range(ns):
        post_var_i = Sigma_q[i, i]
        if post_var_i <= 0:
            continue
        tau_cav = 1.0 / post_var_i - tau[i]
        nu_cav = mu_q[i] / post_var_i - nu[i]
        if tau_cav <= 0:
            tau_cav = 1e-8
        cav_var = 1.0 / tau_cav
        cav_mean = nu_cav / tau_cav

        Z_i, m_hat, v_hat = _ep_1d_moments(
            soft_pdfs[i], cav_mean, cav_var)
        if Z_i <= 0 or v_hat <= 0:
            continue

        log_s_sum += (math.log(max(Z_i, 1e-300))
                      + 0.5 * math.log(max(cav_var / v_hat, 1e-300))
                      + 0.5 * cav_mean ** 2 / cav_var
                      - 0.5 * m_hat ** 2 / v_hat)

    # log Z_EP = Σ log ŝᵢ  +  ½ (log|Λ| − log|Λ_q|)
    #          + ½ (η_q^T Σ_q η_q  −  η^T Σ η)
    log_Z = log_s_sum
    log_Z += 0.5 * (logdet_prior - logdet_q)
    log_Z += 0.5 * (float(eta_q @ Sigma_q @ eta_q)
                     - float(eta @ cov @ eta))

    return max(math.exp(min(log_Z, 500.0)), 1e-300)


def integrate_soft_ep_batch(soft_pdfs: List[SoftPDF],
                            mu_grid: np.ndarray,
                            cov: np.ndarray,
                            max_iter: int = 50,
                            damp: float = 0.8,
                            tol: float = 1e-6) -> np.ndarray:
    """Batch EP integration for multiple conditional means.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF  (length ns)
    mu_grid   : (M, ns) array — one mean per row
    cov       : (ns, ns) conditional covariance
    max_iter  : EP sweeps per evaluation
    damp      : damping factor
    tol       : convergence tolerance

    Returns
    -------
    (M,) array ≥ 1e-300
    """
    M = mu_grid.shape[0]
    ns = len(soft_pdfs)
    if ns == 0:
        return np.ones(M)
    result = np.empty(M)
    for j in range(M):
        result[j] = integrate_soft_ep(soft_pdfs, mu_grid[j], cov,
                                      max_iter=max_iter, damp=damp,
                                      tol=tol)
    return np.maximum(result, 1e-300)


# ════════════════════════════════════════════════════════════════
# LAPLACE IMPORTANCE SAMPLING  (LIS)
# ════════════════════════════════════════════════════════════════
#
# 1. Run the existing Laplace machinery to find the mode x* and
#    Hessian H of log g(x) = log N(x|μ,Σ) + Σ log fᵢ(xᵢ).
# 2. Build a Gaussian proposal  q(x) = N(x*; (−H)⁻¹).
# 3. Draw N importance samples from q, compute weights
#    w_j = g(x_j) / q(x_j),  and return  mean(w_j) × Z_prior.
#
# The estimate is *unbiased* (unlike Laplace alone) and has much
# lower variance than raw MC because the proposal is centred on
# the mode.
#
# Cost: O(ns³) for mode+Hessian  +  O(N × ns²) for sampling.

def integrate_soft_lis(soft_pdfs: List[SoftPDF],
                       mu: np.ndarray, cov: np.ndarray,
                       n_samples: int = 4096) -> float:
    """Laplace Importance Sampling for  E_{x ~ N(μ, Σ)}[ ∏ fᵢ(xᵢ) ].

    Unbiased correction to Laplace: uses the Laplace mode & Hessian
    as a Gaussian importance-sampling proposal.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF  (length ns)
    mu        : (ns,) conditional mean
    cov       : (ns, ns) conditional covariance
    n_samples : importance samples to draw

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

    # --- Laplace mode + Hessian ---
    try:
        x_star = _find_mode(soft_pdfs, mu, Q)
        H = _log_target_hessian(x_star, soft_pdfs, mu, Q)
        neg_H = -H
        ev = np.linalg.eigvalsh(neg_H)
        if ev.min() <= 0:
            neg_H += (abs(ev.min()) + 1e-6) * np.eye(ns)
        Sigma_prop = np.linalg.inv(neg_H)        # proposal covariance
        Sigma_prop = 0.5 * (Sigma_prop + Sigma_prop.T)
        L_prop = np.linalg.cholesky(Sigma_prop)
    except Exception:
        # Fall back to plain MC if Laplace mode-finding fails
        try:
            L_prior = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            ev2, Q2 = np.linalg.eigh(cov)
            L_prior = Q2 @ np.diag(np.sqrt(np.maximum(ev2, 1e-10)))
        u = np.random.randn(n_samples, ns)
        x = mu[None, :] + u @ L_prior.T
        prod_f = np.ones(n_samples)
        for i, sp in enumerate(soft_pdfs):
            prod_f *= sp.evaluate(x[:, i])
        return max(float(np.mean(prod_f)), 1e-300)

    # --- Importance sampling from proposal q = N(x*; Σ_prop) ---
    u = np.random.randn(n_samples, ns)
    x = x_star[None, :] + u @ L_prop.T    # (N, ns)

    # log g(x_j) = log N(x_j | mu, cov) + Σ log fᵢ(x_ji)  [unnormalised]
    # log q(x_j) = log N(x_j | x*, Σ_prop)
    # But E[∏fᵢ] = (1/(2π)^{ns/2} |Σ|^{1/2})
    #              × ∫ exp(-½(x-μ)^T Q (x-μ)) ∏fᵢ dx
    # Weight = [N_prior(x) × ∏fᵢ(x)] / q(x)
    # and we return mean(weights).

    # Compute log-weights stably:
    # log_w_j = log_target(x_j) - log_q(x_j)
    # where log_target = -0.5 (x-mu)^T Q (x-mu) + sum log f_i
    #       log_q      = -0.5 (x-x*)^T negH (x-x*)  + const_q
    # The 1/(2π)^{ns/2} |Σ|^{1/2} from prior and analogous term
    # from proposal cancel partially; easier to compute the ratio:

    _, logdet_cov = np.linalg.slogdet(cov)
    _, logdet_prop = np.linalg.slogdet(Sigma_prop)

    log_w = np.empty(n_samples)
    for j in range(n_samples):
        xj = x[j]
        lt = _log_target(xj, soft_pdfs, mu, Q)  # -0.5 d^T Q d + Σlog f
        dq = xj - x_star
        lq = -0.5 * float(dq @ neg_H @ dq)     # log q (up to const)
        log_w[j] = lt - lq
    # Normalising constant ratio: (|Σ_prop|/|Σ|)^{1/2}
    log_w += 0.5 * (logdet_prop - logdet_cov)

    # Stable mean of exp(log_w)
    max_lw = np.max(log_w)
    if not np.isfinite(max_lw):
        return 1e-300
    result = math.exp(min(max_lw, 500.0)) * float(np.mean(np.exp(log_w - max_lw)))
    return max(result, 1e-300)


def integrate_soft_lis_batch(soft_pdfs: List[SoftPDF],
                             mu_grid: np.ndarray,
                             cov: np.ndarray,
                             n_samples: int = 4096) -> np.ndarray:
    """Batch LIS integration for multiple conditional means.

    Parameters
    ----------
    soft_pdfs : list of SoftPDF  (length ns)
    mu_grid   : (M, ns) array — one mean per row
    cov       : (ns, ns) conditional covariance
    n_samples : importance samples per evaluation

    Returns
    -------
    (M,) array ≥ 1e-300
    """
    M = mu_grid.shape[0]
    ns = len(soft_pdfs)
    if ns == 0:
        return np.ones(M)
    result = np.empty(M)
    for j in range(M):
        result[j] = integrate_soft_lis(soft_pdfs, mu_grid[j], cov,
                                       n_samples=n_samples)
    return np.maximum(result, 1e-300)
