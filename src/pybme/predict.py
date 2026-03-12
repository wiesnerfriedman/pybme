"""BME and kriging prediction — posterior PDF, mode, moments, CI.

This module corresponds to the MATLAB functions BMEprobaPdf, BMEprobaMode,
BMEprobaMoments, BMEprobaCI combined into a single efficient call.

Scalability features (v0.2)
---------------------------
* KD-tree neighbourhood via ``SpatialIndex`` / ``SpatialTemporalIndex``
  — O(K log N) instead of O(K N).
* Vectorised z-grid integration loop (whole grid evaluated by one
  ``integrate_soft_product_batch`` call).
* Optional ``joblib`` parallel backend (``n_jobs>1``).
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import linalg
from scipy.stats import norm

from .covariance import build_cov_matrix, build_cov_matrix_st, eval_cov
from .distance import coord2dist
from .integration import (
    integrate_soft_product, integrate_soft_product_batch,
    integrate_soft_laplace, integrate_soft_laplace_batch,
    integrate_soft_qmc, integrate_soft_qmc_batch,
    integrate_soft_ep, integrate_soft_ep_batch,
    integrate_soft_lis, integrate_soft_lis_batch,
)
from .neighborhood import (
    select_neighbors, select_neighbors_st,
    SpatialIndex, SpatialTemporalIndex,
)
from .soft_data import SoftPDF
from .trend import estimate_trend

# NumPy compat
_trapz = getattr(np, "trapezoid", np.trapz)


# ── Result container ─────────────────────────────────────────

@dataclass
class BMEResult:
    """Container for BME prediction results at a single estimation point."""
    mode: float = np.nan
    mean: float = np.nan
    variance: float = np.nan
    skewness: float = np.nan
    z_grid: Optional[np.ndarray] = None
    pdf: Optional[np.ndarray] = None
    ci_lower: float = np.nan
    ci_upper: float = np.nan
    ci_prob: float = 0.95
    kriging_mean: float = np.nan
    kriging_var: float = np.nan
    n_hard: int = 0
    n_soft: int = 0
    info: str = ""


# ── Main spatial BME ─────────────────────────────────────────

def bme_predict(ck, ch, zh,
                cs=None, soft_pdfs=None,
                model="exponential", params=None,
                nhmax=20, nsmax=8, dmax=np.inf,
                order=0, n_grid=200, ci_prob=0.95,
                n_quad=15, mean_prior=0.0,
                n_jobs=1, method="auto") -> List[BMEResult]:
    """Full BME prediction at one or more estimation points.

    Combines the MATLAB functions ``BMEprobaPdf``, ``BMEprobaMode``,
    ``BMEprobaMoments`` and ``BMEprobaCI`` into one call.

    Parameters
    ----------
    ck         : (nk, d) estimation coordinates
    ch         : (nh, d) hard-data coordinates
    zh         : (nh,) hard-data values
    cs         : (ns, d) soft-data coordinates (optional)
    soft_pdfs  : list[SoftPDF] one per soft point (optional)
    model      : covariance model name or list for nested
    params     : model parameters or list-of-lists for nested
    nhmax      : max hard neighbours
    nsmax      : max soft neighbours (≤ 8 recommended)
    dmax       : max distance for neighbourhood
    order      : NaN = simple, 0 = ordinary, 1 = linear, 2 = quadratic
    n_grid     : z-grid resolution for posterior PDF
    ci_prob    : confidence-interval probability
    n_quad     : base Gauss-Hermite quadrature points
    mean_prior : prior mean (used when order = NaN)
    n_jobs     : number of parallel workers (1 = serial; -1 = all CPUs).
                 Requires ``joblib`` when > 1.
    method     : integration method for soft data: ``'auto'`` (default),
                 ``'gauss_hermite'``, ``'laplace'``, ``'mc'``,
                 ``'ep'`` (Expectation Propagation),
                 ``'qmc'`` (Quasi-Monte Carlo), or
                 ``'lis'`` (Laplace Importance Sampling).
                 ``'auto'`` selects Laplace when ns >= 6, GH otherwise.

    Returns
    -------
    list[BMEResult]
    """
    ck = np.atleast_2d(ck)
    ch = np.atleast_2d(ch)
    zh = np.asarray(zh, dtype=np.float64)
    d_dim = ck.shape[1]
    if cs is None or soft_pdfs is None:
        cs = np.empty((0, d_dim))
        soft_pdfs = []
    else:
        cs = np.atleast_2d(cs) if len(cs) > 0 else np.empty((0, d_dim))
    if params is None:
        params = [1.0, 1.0]

    # Build KD-tree indices once  (O(N log N))
    _idx_h = SpatialIndex(ch)
    _idx_s = SpatialIndex(cs) if len(soft_pdfs) > 0 else None

    nk = ck.shape[0]

    def _worker(i):
        return _bme_point(ck[i], ch, zh, cs, soft_pdfs,
                          model, params, nhmax, nsmax, dmax,
                          order, n_grid, ci_prob, n_quad, mean_prior,
                          _idx_h, _idx_s, method=method)

    if n_jobs == 1 or nk <= 4:
        return [_worker(i) for i in range(nk)]

    # Parallel via joblib
    try:
        from joblib import Parallel, delayed
    except ImportError:
        warnings.warn("joblib not installed — falling back to serial. "
                      "Install joblib for parallel BME:  pip install joblib",
                      stacklevel=2)
        return [_worker(i) for i in range(nk)]

    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker)(i) for i in range(nk)
    )


def _bme_point(ck, ch, zh, cs, soft_pdfs,
               model, params, nhmax, nsmax, dmax,
               order, n_grid, ci_prob, n_quad, mean_prior,
               idx_h=None, idx_s=None, method="auto"):
    """BME at a single estimation point.

    If *idx_h* / *idx_s* are ``SpatialIndex`` objects they are used for
    O(log N) neighbour lookup; otherwise falls back to brute-force.
    """
    res = BMEResult(ci_prob=ci_prob)

    # ── neighbourhood ──
    if idx_h is not None:
        idxh = idx_h.query(ck, nmax=nhmax, dmax=dmax)
    else:
        idxh = select_neighbors(ck, ch, len(zh), nhmax, dmax)
    if idx_s is not None:
        idxs = idx_s.query(ck, nmax=nsmax, dmax=dmax)
    else:
        idxs = select_neighbors(ck, cs, len(soft_pdfs), nsmax, dmax)
    ch_l, zh_l = ch[idxh], zh[idxh]
    cs_l = cs[idxs] if len(idxs) > 0 else np.empty((0, ck.shape[-1]))
    sp_l = [soft_pdfs[i] for i in idxs]
    nh, ns = len(idxh), len(idxs)
    res.n_hard, res.n_soft = nh, ns

    # ── duplicate check ──
    if nh > 0:
        dd = coord2dist(ck.reshape(1, -1), ch_l).ravel()
        dup = np.where(dd < 1e-10)[0]
        if len(dup) > 0:
            v = zh_l[dup[0]]
            res.mode = res.mean = res.kriging_mean = v
            res.variance = res.kriging_var = 0.0
            res.info = "duplicate"
            return res

    # ── trend removal ──
    zh_dt, sp_dt, mk, _, _ = estimate_trend(
        ch_l, zh_l, cs_l, sp_l, ck, order, mean_prior
    )

    # ── covariance ──
    ck2 = ck.reshape(1, -1)
    sig2 = float(build_cov_matrix(ck2, ck2, model, params)[0, 0])

    if nh == 0 and ns == 0:
        # When no data, mk may be 0 from trend estimation; use mean_prior
        return _fill_prior(res, mean_prior, sig2, n_grid, ci_prob)

    # hard blocks
    if nh > 0:
        Ckh = build_cov_matrix(ck2, ch_l, model, params)
        Chh = build_cov_matrix(ch_l, ch_l, model, params) + np.eye(nh) * 1e-10
        Lhh = linalg.cholesky(Chh, lower=True)

    # soft blocks
    if ns > 0:
        Cks = build_cov_matrix(ck2, cs_l, model, params)
        Css = build_cov_matrix(cs_l, cs_l, model, params) + np.eye(ns) * 1e-10
        if nh > 0:
            Chs = build_cov_matrix(ch_l, cs_l, model, params)

    # ── kriging (hard only) ──
    if nh > 0:
        alpha_h = linalg.cho_solve((Lhh, True), zh_dt)
        k_mu = float((Ckh @ alpha_h).ravel()[0])
        k_var = max(float((sig2 - Ckh @ linalg.cho_solve((Lhh, True), Ckh.T)).ravel()[0]), 1e-12)
    else:
        k_mu, k_var = 0.0, sig2
    res.kriging_mean = k_mu + mk
    res.kriging_var = k_var

    if ns == 0:
        return _fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob)

    # ────────────────────────────────────────────────────────
    # FULL BME — non-Gaussian posterior via numerical integration
    # ────────────────────────────────────────────────────────

    nkh = 1 + nh
    C_kh = np.zeros((nkh, nkh))
    C_kh[0, 0] = sig2
    if nh > 0:
        C_kh[0, 1:] = C_kh[1:, 0] = Ckh.ravel()
        C_kh[1:, 1:] = Chh
    C_kh += np.eye(nkh) * 1e-10

    C_s_kh = np.zeros((ns, nkh))
    C_s_kh[:, 0] = Cks.ravel()
    if nh > 0:
        C_s_kh[:, 1:] = Chs.T

    # Denominator integral:  E [ ∏ fS | z_h ]
    if nh > 0:
        mu_s_h = (Chs.T @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()
        K_s_h = Css - Chs.T @ linalg.cho_solve((Lhh, True), Chs)
    else:
        mu_s_h = np.zeros(ns)
        K_s_h = Css.copy()
    K_s_h = 0.5 * (K_s_h + K_s_h.T) + np.eye(ns) * 1e-10

    # Numerator integral:  E [ ∏ fS | z_k, z_h ]  — varies with z_k
    L_kh = linalg.cholesky(C_kh, lower=True)
    K_s_kh = Css - C_s_kh @ linalg.cho_solve((L_kh, True), C_s_kh.T)
    K_s_kh = 0.5 * (K_s_kh + K_s_kh.T) + np.eye(ns) * 1e-10

    B = C_s_kh @ np.linalg.inv(C_kh)
    b_k = B[:, 0]
    b_const = B[:, 1:] @ zh_dt if nh > 0 else np.zeros(ns)

    # ── evaluate posterior on z-grid (vectorised) ──
    sigma_k = math.sqrt(k_var)
    z_lo = k_mu - 5.0 * sigma_k
    z_hi = k_mu + 5.0 * sigma_k
    for sp in sp_dt:
        z_lo = min(z_lo, sp.support[0] - 2 * sigma_k)
        z_hi = max(z_hi, sp.support[1] + 2 * sigma_k)
    zg = np.linspace(z_lo, z_hi, n_grid)

    prior = norm.pdf(zg, k_mu, sigma_k)
    # mu_s_kh(z_k) = b_k * z_k + b_const  → (n_grid, ns)
    mu_all = np.outer(zg, b_k) + b_const[None, :]

    # Select integration method
    _method = method
    if _method == "auto":
        _method = "laplace" if ns >= 6 else "gauss_hermite"
    if _method == "laplace":
        I_num = integrate_soft_laplace_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "ep":
        I_num = integrate_soft_ep_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "qmc":
        I_num = integrate_soft_qmc_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "lis":
        I_num = integrate_soft_lis_batch(sp_dt, mu_all, K_s_kh)
    else:
        I_num = integrate_soft_product_batch(sp_dt, mu_all, K_s_kh, n_quad)
    pdf_raw = prior * I_num
    pdf_raw[prior < 1e-300] = 0.0

    pdf_n = _normalise_posterior_pdf(pdf_raw, zg, k_mu, sigma_k)
    if not np.isfinite(float(_trapz(pdf_raw, zg))) or float(_trapz(pdf_raw, zg)) <= 1e-300:
        res.info = "integration_fallback"

    zg_final = zg + mk

    # ── extract statistics ──
    res.mode = zg_final[int(np.argmax(pdf_n))]
    res.mean = float(_trapz(zg_final * pdf_n, zg))
    res.variance = max(float(_trapz((zg_final - res.mean) ** 2 * pdf_n, zg)), 1e-12)
    m3 = float(_trapz((zg_final - res.mean) ** 3 * pdf_n, zg))
    res.skewness = m3 / res.variance ** 1.5 if res.variance > 1e-14 else 0.0

    cdf = _posterior_cdf(pdf_n, zg)
    al = (1 - ci_prob) / 2
    res.ci_lower = zg_final[np.clip(np.searchsorted(cdf, al), 0, n_grid - 1)]
    res.ci_upper = zg_final[np.clip(np.searchsorted(cdf, 1 - al), 0, n_grid - 1)]
    res.z_grid = zg_final
    res.pdf = pdf_n
    if not res.info:
        res.info = f"full_bme nh={nh} ns={ns}"
    return res


# ── Space-time BME ───────────────────────────────────────────

def bme_predict_st(ck, tk, ch, th, zh,
                   cs=None, ts=None, soft_pdfs=None,
                   model_s="exponential", params_s=None,
                   model_t="exponential", params_t=None,
                   sigma2=1.0,
                   nhmax=20, nsmax=8,
                   dmax_s=np.inf, dmax_t=np.inf,
                   order=0, n_grid=200, ci_prob=0.95,
                   n_quad=15, mean_prior=0.0,
                   n_jobs=1, method="auto") -> List[BMEResult]:
    """Separable space-time BME prediction.

    C((x,t),(x',t')) = σ² · Cs(‖x−x'‖) · Ct(|t−t'|)

    Spatial and temporal models should have sill = 1 (overall sill in *sigma2*).

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers (1 = serial; -1 = all CPUs).
        Requires ``joblib`` when > 1.
    method : str
        Integration method for soft data: ``'auto'`` (default),
        ``'gauss_hermite'``, ``'laplace'``, ``'mc'``,
        ``'ep'`` (Expectation Propagation),
        ``'qmc'`` (Quasi-Monte Carlo), or
        ``'lis'`` (Laplace Importance Sampling).
        ``'auto'`` selects Laplace when ns >= 6, GH otherwise.
    """
    ck = np.atleast_2d(ck)
    ch = np.atleast_2d(ch)
    zh = np.asarray(zh, dtype=np.float64)
    tk = np.atleast_1d(tk).astype(np.float64)
    th = np.atleast_1d(th).astype(np.float64)
    d_dim = ck.shape[1]
    if cs is None or soft_pdfs is None:
        cs, ts, soft_pdfs = np.empty((0, d_dim)), np.array([]), []
    else:
        cs = np.atleast_2d(cs) if len(cs) else np.empty((0, d_dim))
        ts = np.atleast_1d(ts).astype(np.float64)
    if params_s is None:
        params_s = [1.0, 1.0]
    if params_t is None:
        params_t = [1.0, 1.0]

    # Build KD-tree indices once
    _st_idx_h = SpatialTemporalIndex(ch, th)
    _st_idx_s = SpatialTemporalIndex(cs, ts) if len(soft_pdfs) > 0 else None

    def _stcov(c1, t1, c2, t2):
        return build_cov_matrix_st(c1, t1, c2, t2,
                                   model_s, params_s, model_t, params_t, sigma2)

    nk = ck.shape[0]

    def _worker(ik):
        return _bme_st_point(
            ck[ik], tk[ik], ch, th, zh, cs, ts, soft_pdfs,
            _stcov, nhmax, nsmax, dmax_s, dmax_t,
            order, n_grid, ci_prob, n_quad, mean_prior, d_dim,
            _st_idx_h, _st_idx_s, method=method,
        )

    if n_jobs == 1 or nk <= 4:
        results = [_worker(ik) for ik in range(nk)]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            warnings.warn("joblib not installed — falling back to serial.",
                          stacklevel=2)
            return [_worker(ik) for ik in range(nk)]
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_worker)(ik) for ik in range(nk)
        )
    return results


def _bme_st_point(ck_i, tk_i, ch, th, zh, cs, ts, soft_pdfs,
                  stcov_fn, nhmax, nsmax, dmax_s, dmax_t,
                  order, n_grid, ci_prob, n_quad, mean_prior, d_dim,
                  idx_h=None, idx_s=None, method="auto"):
    """Space-time BME at a single point, with optional KD-tree indices."""
    res = BMEResult(ci_prob=ci_prob)
    ck_2d = ck_i.reshape(1, -1)
    tk_1d = np.atleast_1d(tk_i)

    # ── neighbourhood ──
    if idx_h is not None:
        idxh = idx_h.query(ck_i, tk_i, nmax=nhmax, dmax_s=dmax_s, dmax_t=dmax_t)
    else:
        idxh = select_neighbors_st(ck_i, tk_i, ch, th, nhmax, dmax_s, dmax_t)
    if idx_s is not None:
        idxs = idx_s.query(ck_i, tk_i, nmax=nsmax, dmax_s=dmax_s, dmax_t=dmax_t)
    elif len(soft_pdfs):
        idxs = select_neighbors_st(ck_i, tk_i, cs, ts, nsmax, dmax_s, dmax_t)
    else:
        idxs = np.array([], dtype=int)

    ch_l, th_l, zh_l = ch[idxh], th[idxh], zh[idxh]
    cs_l = cs[idxs] if len(idxs) else np.empty((0, d_dim))
    ts_l = ts[idxs] if len(idxs) else np.array([])
    sp_l = [soft_pdfs[i] for i in idxs]
    nh, ns = len(idxh), len(idxs)
    res.n_hard, res.n_soft = nh, ns

    sig2_loc = float(stcov_fn(ck_2d, tk_1d, ck_2d, tk_1d)[0, 0])

    if nh == 0 and ns == 0:
        return _fill_prior(res, mean_prior, sig2_loc, n_grid, ci_prob)

    zh_dt, sp_dt, mk, _, _ = estimate_trend(
        ch_l, zh_l, cs_l, sp_l, ck_i, order, mean_prior
    )

    if nh > 0:
        Ckh = stcov_fn(ck_2d, tk_1d, ch_l, th_l)
        Chh = stcov_fn(ch_l, th_l, ch_l, th_l) + np.eye(nh) * 1e-10
        Lhh = linalg.cholesky(Chh, lower=True)
    if ns > 0:
        Cks = stcov_fn(ck_2d, tk_1d, cs_l, ts_l)
        Css = stcov_fn(cs_l, ts_l, cs_l, ts_l) + np.eye(ns) * 1e-10
        if nh > 0:
            Chs = stcov_fn(ch_l, th_l, cs_l, ts_l)

    if nh > 0:
        k_mu = float((Ckh @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()[0])
        k_var = max(float((sig2_loc - Ckh @ linalg.cho_solve((Lhh, True), Ckh.T)).ravel()[0]), 1e-12)
    else:
        k_mu, k_var = 0.0, sig2_loc
    res.kriging_mean, res.kriging_var = k_mu + mk, k_var

    if ns == 0:
        return _fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob)

    # full BME integration
    nkh = 1 + nh
    C_kh_kh = np.zeros((nkh, nkh))
    C_kh_kh[0, 0] = sig2_loc
    if nh > 0:
        C_kh_kh[0, 1:] = C_kh_kh[1:, 0] = Ckh.ravel()
        C_kh_kh[1:, 1:] = Chh
    C_kh_kh += np.eye(nkh) * 1e-10
    C_s_kh = np.zeros((ns, nkh))
    C_s_kh[:, 0] = Cks.ravel()
    if nh > 0:
        C_s_kh[:, 1:] = Chs.T

    if nh > 0:
        mu_s_h = (Chs.T @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()
        K_s_h = Css - Chs.T @ linalg.cho_solve((Lhh, True), Chs)
    else:
        mu_s_h, K_s_h = np.zeros(ns), Css.copy()
    K_s_h = 0.5 * (K_s_h + K_s_h.T) + np.eye(ns) * 1e-10

    L_kh = linalg.cholesky(C_kh_kh, lower=True)
    K_s_kh = Css - C_s_kh @ linalg.cho_solve((L_kh, True), C_s_kh.T)
    K_s_kh = 0.5 * (K_s_kh + K_s_kh.T) + np.eye(ns) * 1e-10

    B = C_s_kh @ np.linalg.inv(C_kh_kh)
    b_k = B[:, 0]
    b_c = B[:, 1:] @ zh_dt if nh > 0 else np.zeros(ns)

    sigma_k = math.sqrt(k_var)
    z_lo = k_mu - 5 * sigma_k
    z_hi = k_mu + 5 * sigma_k
    for sp in sp_dt:
        z_lo = min(z_lo, sp.support[0] - 2 * sigma_k)
        z_hi = max(z_hi, sp.support[1] + 2 * sigma_k)
    zg = np.linspace(z_lo, z_hi, n_grid)

    # Vectorised z-grid integration
    prior = norm.pdf(zg, k_mu, sigma_k)
    mu_all = np.outer(zg, b_k) + b_c[None, :]

    # Select integration method
    _method = method
    if _method == "auto":
        _method = "laplace" if ns >= 6 else "gauss_hermite"
    if _method == "laplace":
        I_num = integrate_soft_laplace_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "ep":
        I_num = integrate_soft_ep_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "qmc":
        I_num = integrate_soft_qmc_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "lis":
        I_num = integrate_soft_lis_batch(sp_dt, mu_all, K_s_kh)
    else:
        I_num = integrate_soft_product_batch(sp_dt, mu_all, K_s_kh, n_quad)
    pdf_raw = prior * I_num
    pdf_raw[prior < 1e-300] = 0.0

    pdf_n = _normalise_posterior_pdf(pdf_raw, zg, k_mu, sigma_k)
    zg_f = zg + mk

    res.mode = zg_f[int(np.argmax(pdf_n))]
    res.mean = float(_trapz(zg_f * pdf_n, zg))
    res.variance = max(float(_trapz((zg_f - res.mean) ** 2 * pdf_n, zg)), 1e-12)
    m3 = float(_trapz((zg_f - res.mean) ** 3 * pdf_n, zg))
    res.skewness = m3 / res.variance ** 1.5 if res.variance > 1e-14 else 0.0
    cdf = _posterior_cdf(pdf_n, zg)
    al = (1 - ci_prob) / 2
    res.ci_lower = zg_f[np.clip(np.searchsorted(cdf, al), 0, n_grid - 1)]
    res.ci_upper = zg_f[np.clip(np.searchsorted(cdf, 1 - al), 0, n_grid - 1)]
    res.z_grid, res.pdf = zg_f, pdf_n
    res.info = f"full_st_bme nh={nh} ns={ns}"
    return res


# ── helpers ──────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════
# NETWORK BME  — spatial
# ════════════════════════════════════════════════════════════════

def bme_predict_network(
    ck_nodes, ch_nodes, zh,
    cs_nodes=None, soft_pdfs=None,
    net_cov=None,
    nhmax=20, nsmax=8,
    order=0, n_grid=200, ci_prob=0.95,
    n_quad=15, mean_prior=0.0,
    method="auto",
) -> List[BMEResult]:
    """BME prediction on a network domain (spatial, graph-Laplacian covariance).

    Instead of coordinates + Euclidean model, the covariance blocks are
    extracted from a ``NetworkCovariance`` object that encodes the graph
    topology.

    Parameters
    ----------
    ck_nodes   : (nk,) integer array — estimation node indices.
    ch_nodes   : (nh,) integer array — hard-data node indices.
    zh         : (nh,) hard-data values.
    cs_nodes   : (ns,) integer array — soft-data node indices (optional).
    soft_pdfs  : list[SoftPDF] one per soft point (optional).
    net_cov    : NetworkCovariance instance.
    nhmax      : max hard neighbours (by graph proximity — NOT used here;
                 all provided hard data within the graph is used unless
                 you pre-filter externally).
    nsmax      : max soft neighbours (same caveat).
    order      : trend order (0 = ordinary kriging with constant mean).
                 Only 0 and NaN (simple) are supported for network BME.
    n_grid     : z-grid resolution for posterior PDF.
    ci_prob    : confidence-interval probability.
    n_quad     : Gauss-Hermite quadrature points (GH method).
    mean_prior : prior mean (used when order = NaN).
    method     : integration method: ``'auto'``, ``'gauss_hermite'``,
                 ``'laplace'``, ``'ep'``, ``'qmc'``, ``'lis'``, ``'mc'``.

    Returns
    -------
    list[BMEResult]
    """
    if net_cov is None:
        raise ValueError("net_cov (NetworkCovariance) is required.")

    ck_nodes = np.atleast_1d(np.asarray(ck_nodes, dtype=int))
    ch_nodes = np.atleast_1d(np.asarray(ch_nodes, dtype=int))
    zh = np.asarray(zh, dtype=np.float64)

    if cs_nodes is None or soft_pdfs is None:
        cs_nodes = np.array([], dtype=int)
        soft_pdfs = []
    else:
        cs_nodes = np.atleast_1d(np.asarray(cs_nodes, dtype=int))

    nk = len(ck_nodes)
    nh_all = len(ch_nodes)
    ns_all = len(cs_nodes)
    results = []
    for i in range(nk):
        ck_i = ck_nodes[i]
        ck_arr = np.array([ck_i])

        # --- neighbourhood selection via covariance magnitude ---
        if nh_all > nhmax:
            cov_h = np.abs(net_cov.covariance_block(ck_arr, ch_nodes).ravel())
            idx_h = np.argsort(-cov_h)[:nhmax]
            ch_sel, zh_sel = ch_nodes[idx_h], zh[idx_h]
        else:
            ch_sel, zh_sel = ch_nodes, zh

        if ns_all > nsmax:
            cov_s = np.abs(net_cov.covariance_block(ck_arr, cs_nodes).ravel())
            idx_s = np.argsort(-cov_s)[:nsmax]
            cs_sel = cs_nodes[idx_s]
            sp_sel = [soft_pdfs[j] for j in idx_s]
        else:
            cs_sel, sp_sel = cs_nodes, soft_pdfs

        results.append(_bme_network_point(
            ck_i, ch_sel, zh_sel, cs_sel, sp_sel,
            net_cov, order, n_grid, ci_prob, n_quad, mean_prior,
            method=method,
        ))
    return results


def _bme_network_point(
    ck_node, ch_nodes, zh, cs_nodes, soft_pdfs,
    net_cov, order, n_grid, ci_prob, n_quad, mean_prior,
    method="auto",
):
    """BME at a single network node."""
    res = BMEResult(ci_prob=ci_prob)
    nh = len(ch_nodes)
    ns = len(cs_nodes)
    res.n_hard, res.n_soft = nh, ns

    # ── duplicate check ──
    if nh > 0:
        dup = np.where(ch_nodes == ck_node)[0]
        if len(dup) > 0:
            v = zh[dup[0]]
            res.mode = res.mean = res.kriging_mean = v
            res.variance = res.kriging_var = 0.0
            res.info = "duplicate"
            return res

    # ── trend removal (simplified for network: constant mean or simple) ──
    if order is None or (isinstance(order, float) and np.isnan(order)):
        mk = mean_prior
        zh_dt = zh - mk
        sp_dt = [sp.shifted(mk) for sp in soft_pdfs]
    else:
        mk = float(np.mean(zh)) if nh > 0 else mean_prior
        zh_dt = zh - mk
        sp_dt = [sp.shifted(mk) for sp in soft_pdfs]

    # ── covariance blocks ──
    ck_arr = np.array([ck_node])
    sig2 = float(net_cov.covariance_block(ck_arr, ck_arr)[0, 0])

    if nh == 0 and ns == 0:
        return _fill_prior(res, mean_prior, sig2, n_grid, ci_prob)

    if nh > 0:
        Ckh = net_cov.covariance_block(ck_arr, ch_nodes)          # (1, nh)
        Chh = net_cov.covariance_block(ch_nodes, ch_nodes)        # (nh, nh)
        Chh += np.eye(nh) * 1e-10
        Lhh = linalg.cholesky(Chh, lower=True)

    if ns > 0:
        Cks = net_cov.covariance_block(ck_arr, cs_nodes)          # (1, ns)
        Css = net_cov.covariance_block(cs_nodes, cs_nodes)        # (ns, ns)
        Css += np.eye(ns) * 1e-10
        if nh > 0:
            Chs = net_cov.covariance_block(ch_nodes, cs_nodes)    # (nh, ns)

    # ── kriging (hard only) ──
    if nh > 0:
        alpha_h = linalg.cho_solve((Lhh, True), zh_dt)
        k_mu = float((Ckh @ alpha_h).ravel()[0])
        k_var = max(float((sig2 - Ckh @ linalg.cho_solve((Lhh, True), Ckh.T)).ravel()[0]), 1e-12)
    else:
        k_mu, k_var = 0.0, sig2
    res.kriging_mean = k_mu + mk
    res.kriging_var = k_var

    if ns == 0:
        return _fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob)

    # ── full BME integration ──
    nkh = 1 + nh
    C_kh = np.zeros((nkh, nkh))
    C_kh[0, 0] = sig2
    if nh > 0:
        C_kh[0, 1:] = C_kh[1:, 0] = Ckh.ravel()
        C_kh[1:, 1:] = Chh
    C_kh += np.eye(nkh) * 1e-10

    C_s_kh = np.zeros((ns, nkh))
    C_s_kh[:, 0] = Cks.ravel()
    if nh > 0:
        C_s_kh[:, 1:] = Chs.T

    if nh > 0:
        mu_s_h = (Chs.T @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()
        K_s_h = Css - Chs.T @ linalg.cho_solve((Lhh, True), Chs)
    else:
        mu_s_h = np.zeros(ns)
        K_s_h = Css.copy()
    K_s_h = 0.5 * (K_s_h + K_s_h.T) + np.eye(ns) * 1e-10

    L_kh = linalg.cholesky(C_kh, lower=True)
    K_s_kh = Css - C_s_kh @ linalg.cho_solve((L_kh, True), C_s_kh.T)
    K_s_kh = 0.5 * (K_s_kh + K_s_kh.T) + np.eye(ns) * 1e-10

    B = linalg.cho_solve((L_kh, True), C_s_kh.T).T
    b_k = B[:, 0]
    b_const = B[:, 1:] @ zh_dt if nh > 0 else np.zeros(ns)

    sigma_k = math.sqrt(k_var)
    z_lo = k_mu - 5.0 * sigma_k
    z_hi = k_mu + 5.0 * sigma_k
    for sp in sp_dt:
        z_lo = min(z_lo, sp.support[0] - 2 * sigma_k)
        z_hi = max(z_hi, sp.support[1] + 2 * sigma_k)
    zg = np.linspace(z_lo, z_hi, n_grid)

    prior = norm.pdf(zg, k_mu, sigma_k)
    mu_all = np.outer(zg, b_k) + b_const[None, :]

    _method = method
    if _method == "auto":
        _method = "laplace" if ns >= 6 else "gauss_hermite"
    if _method == "laplace":
        I_num = integrate_soft_laplace_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "ep":
        I_num = integrate_soft_ep_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "qmc":
        I_num = integrate_soft_qmc_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "lis":
        I_num = integrate_soft_lis_batch(sp_dt, mu_all, K_s_kh)
    else:
        I_num = integrate_soft_product_batch(sp_dt, mu_all, K_s_kh, n_quad)
    pdf_raw = prior * I_num
    pdf_raw[prior < 1e-300] = 0.0

    pdf_n = _normalise_posterior_pdf(pdf_raw, zg, k_mu, sigma_k)
    if not np.isfinite(float(_trapz(pdf_raw, zg))) or float(_trapz(pdf_raw, zg)) <= 1e-300:
        res.info = "integration_fallback"

    zg_final = zg + mk

    res.mode = zg_final[int(np.argmax(pdf_n))]
    res.mean = float(_trapz(zg_final * pdf_n, zg))
    res.variance = max(float(_trapz((zg_final - res.mean) ** 2 * pdf_n, zg)), 1e-12)
    m3 = float(_trapz((zg_final - res.mean) ** 3 * pdf_n, zg))
    res.skewness = m3 / res.variance ** 1.5 if res.variance > 1e-14 else 0.0

    cdf = _posterior_cdf(pdf_n, zg)
    al = (1 - ci_prob) / 2
    res.ci_lower = zg_final[np.clip(np.searchsorted(cdf, al), 0, n_grid - 1)]
    res.ci_upper = zg_final[np.clip(np.searchsorted(cdf, 1 - al), 0, n_grid - 1)]
    res.z_grid = zg_final
    res.pdf = pdf_n
    if not res.info:
        res.info = f"full_network_bme nh={nh} ns={ns}"
    return res


# ════════════════════════════════════════════════════════════════
# NETWORK BME  — space-time
# ════════════════════════════════════════════════════════════════

def bme_predict_network_st(
    ck_nodes, tk, ch_nodes, th, zh,
    cs_nodes=None, ts=None, soft_pdfs=None,
    net_cov_st=None,
    nhmax=20, nsmax=8,
    order=0, n_grid=200, ci_prob=0.95,
    n_quad=15, mean_prior=0.0,
    method="auto",
    n_jobs=1,
) -> List[BMEResult]:
    """Separable space-time BME on a network domain.

    Uses ``NetworkCovarianceST`` for the covariance blocks — separable
    product of graph-Laplacian spatial covariance and parametric temporal
    covariance.

    Parameters
    ----------
    ck_nodes    : (nk,) estimation node indices.
    tk          : (nk,) estimation times.
    ch_nodes    : (nh,) hard-data node indices.
    th          : (nh,) hard-data times.
    zh          : (nh,) hard-data values.
    cs_nodes    : (ns,) soft-data node indices (optional).
    ts          : (ns,) soft-data times (optional).
    soft_pdfs   : list[SoftPDF] one per soft point (optional).
    net_cov_st  : NetworkCovarianceST instance.
    nhmax, nsmax, order, n_grid, ci_prob, n_quad, mean_prior, method :
        Same as ``bme_predict_network``.
    n_jobs      : number of parallel workers (1 = serial; -1 = all CPUs).
                  Requires ``concurrent.futures`` (stdlib) or ``joblib``.

    Returns
    -------
    list[BMEResult]
    """
    if net_cov_st is None:
        raise ValueError("net_cov_st (NetworkCovarianceST) is required.")

    ck_nodes = np.atleast_1d(np.asarray(ck_nodes, dtype=int))
    tk = np.atleast_1d(np.asarray(tk, dtype=np.float64))
    ch_nodes = np.atleast_1d(np.asarray(ch_nodes, dtype=int))
    th = np.atleast_1d(np.asarray(th, dtype=np.float64))
    zh = np.asarray(zh, dtype=np.float64)

    if cs_nodes is None or soft_pdfs is None:
        cs_nodes = np.array([], dtype=int)
        ts = np.array([], dtype=np.float64)
        soft_pdfs = []
    else:
        cs_nodes = np.atleast_1d(np.asarray(cs_nodes, dtype=int))
        ts = np.atleast_1d(np.asarray(ts, dtype=np.float64))

    nk = len(ck_nodes)
    nh_all = len(ch_nodes)
    ns_all = len(cs_nodes)

    def _worker(i):
        """Process a single prediction point (neighbourhood + BME)."""
        ck_i = ck_nodes[i]
        tk_i = tk[i]
        ck_arr = np.array([ck_i])
        tk_arr = np.array([tk_i])

        # --- neighbourhood selection via covariance magnitude ---
        if nh_all > nhmax:
            cov_h = np.abs(net_cov_st(ck_arr, tk_arr, ch_nodes, th).ravel())
            idx_h = np.argsort(-cov_h)[:nhmax]
            ch_sel, th_sel, zh_sel = ch_nodes[idx_h], th[idx_h], zh[idx_h]
        else:
            ch_sel, th_sel, zh_sel = ch_nodes, th, zh

        if ns_all > nsmax:
            cov_s = np.abs(net_cov_st(ck_arr, tk_arr, cs_nodes, ts).ravel())
            idx_s = np.argsort(-cov_s)[:nsmax]
            cs_sel = cs_nodes[idx_s]
            ts_sel = ts[idx_s]
            sp_sel = [soft_pdfs[j] for j in idx_s]
        else:
            cs_sel, ts_sel, sp_sel = cs_nodes, ts, soft_pdfs

        return _bme_network_st_point(
            ck_i, tk_i, ch_sel, th_sel, zh_sel,
            cs_sel, ts_sel, sp_sel,
            net_cov_st, order, n_grid, ci_prob, n_quad, mean_prior,
            method=method,
        )

    # ── parallel dispatch ───────────────────────────────────────
    if n_jobs != 1 and nk > 4:
        import os
        n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        try:
            from concurrent.futures import ProcessPoolExecutor
            import sys as _sys
            print(f"    BME ST: dispatching {nk} points to {n_workers} workers ...",
                  file=_sys.stderr, flush=True)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_worker, range(nk), chunksize=max(1, nk // (n_workers * 4))))
            print(f"\r    BME ST: {nk}/{nk} (100%)    ", file=_sys.stderr)
            return results
        except Exception:
            pass  # fall back to serial loop

    # ── serial loop with progress ───────────────────────────────
    _log_interval = max(1, nk // 10)
    results = []
    for i in range(nk):
        if i % _log_interval == 0 and nk > 20:
            import sys as _sys
            print(f"\r    BME ST: {i}/{nk} ({100*i//nk}%) ...", end="", flush=True, file=_sys.stderr)
        results.append(_worker(i))
    if nk > 20:
        import sys as _sys
        print(f"\r    BME ST: {nk}/{nk} (100%)    ", file=_sys.stderr)
    return results


def _bme_network_st_point(
    ck_node, tk_i, ch_nodes, th, zh,
    cs_nodes, ts, soft_pdfs,
    net_cov_st, order, n_grid, ci_prob, n_quad, mean_prior,
    method="auto",
):
    """Space-time BME at a single network node + time."""
    res = BMEResult(ci_prob=ci_prob)
    nh = len(ch_nodes)
    ns = len(cs_nodes)
    res.n_hard, res.n_soft = nh, ns

    ck_arr = np.array([ck_node])
    tk_arr = np.array([tk_i])

    sig2 = float(net_cov_st(ck_arr, tk_arr, ck_arr, tk_arr)[0, 0])

    if nh == 0 and ns == 0:
        return _fill_prior(res, mean_prior, sig2, n_grid, ci_prob)

    # ── trend removal ──
    if order is None or (isinstance(order, float) and np.isnan(order)):
        mk = mean_prior
    else:
        mk = float(np.mean(zh)) if nh > 0 else mean_prior
    zh_dt = zh - mk
    sp_dt = [sp.shifted(mk) for sp in soft_pdfs]

    # ── covariance blocks ──
    if nh > 0:
        Ckh = net_cov_st(ck_arr, tk_arr, ch_nodes, th)         # (1, nh)
        Chh = net_cov_st(ch_nodes, th, ch_nodes, th)            # (nh, nh)
        Chh += np.eye(nh) * 1e-10
        Lhh = linalg.cholesky(Chh, lower=True)

    if ns > 0:
        Cks = net_cov_st(ck_arr, tk_arr, cs_nodes, ts)         # (1, ns)
        Css = net_cov_st(cs_nodes, ts, cs_nodes, ts)            # (ns, ns)
        Css += np.eye(ns) * 1e-10
        if nh > 0:
            Chs = net_cov_st(ch_nodes, th, cs_nodes, ts)       # (nh, ns)

    # ── kriging ──
    if nh > 0:
        k_mu = float((Ckh @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()[0])
        k_var = max(float((sig2 - Ckh @ linalg.cho_solve((Lhh, True), Ckh.T)).ravel()[0]), 1e-12)
    else:
        k_mu, k_var = 0.0, sig2
    res.kriging_mean, res.kriging_var = k_mu + mk, k_var

    if ns == 0:
        return _fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob)

    # ── full BME ──
    nkh = 1 + nh
    C_kh_kh = np.zeros((nkh, nkh))
    C_kh_kh[0, 0] = sig2
    if nh > 0:
        C_kh_kh[0, 1:] = C_kh_kh[1:, 0] = Ckh.ravel()
        C_kh_kh[1:, 1:] = Chh
    C_kh_kh += np.eye(nkh) * 1e-10

    C_s_kh = np.zeros((ns, nkh))
    C_s_kh[:, 0] = Cks.ravel()
    if nh > 0:
        C_s_kh[:, 1:] = Chs.T

    if nh > 0:
        mu_s_h = (Chs.T @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()
        K_s_h = Css - Chs.T @ linalg.cho_solve((Lhh, True), Chs)
    else:
        mu_s_h, K_s_h = np.zeros(ns), Css.copy()
    K_s_h = 0.5 * (K_s_h + K_s_h.T) + np.eye(ns) * 1e-10

    L_kh = linalg.cholesky(C_kh_kh, lower=True)
    K_s_kh = Css - C_s_kh @ linalg.cho_solve((L_kh, True), C_s_kh.T)
    K_s_kh = 0.5 * (K_s_kh + K_s_kh.T) + np.eye(ns) * 1e-10

    B = linalg.cho_solve((L_kh, True), C_s_kh.T).T
    b_k = B[:, 0]
    b_c = B[:, 1:] @ zh_dt if nh > 0 else np.zeros(ns)

    sigma_k = math.sqrt(k_var)
    z_lo = k_mu - 5 * sigma_k
    z_hi = k_mu + 5 * sigma_k
    for sp in sp_dt:
        z_lo = min(z_lo, sp.support[0] - 2 * sigma_k)
        z_hi = max(z_hi, sp.support[1] + 2 * sigma_k)
    zg = np.linspace(z_lo, z_hi, n_grid)

    prior = norm.pdf(zg, k_mu, sigma_k)
    mu_all = np.outer(zg, b_k) + b_c[None, :]

    _method = method
    if _method == "auto":
        _method = "laplace" if ns >= 6 else "gauss_hermite"
    if _method == "laplace":
        I_num = integrate_soft_laplace_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "ep":
        I_num = integrate_soft_ep_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "qmc":
        I_num = integrate_soft_qmc_batch(sp_dt, mu_all, K_s_kh)
    elif _method == "lis":
        I_num = integrate_soft_lis_batch(sp_dt, mu_all, K_s_kh)
    else:
        I_num = integrate_soft_product_batch(sp_dt, mu_all, K_s_kh, n_quad)
    pdf_raw = prior * I_num
    pdf_raw[prior < 1e-300] = 0.0

    pdf_n = _normalise_posterior_pdf(pdf_raw, zg, k_mu, sigma_k)
    zg_f = zg + mk

    res.mode = zg_f[int(np.argmax(pdf_n))]
    res.mean = float(_trapz(zg_f * pdf_n, zg))
    res.variance = max(float(_trapz((zg_f - res.mean) ** 2 * pdf_n, zg)), 1e-12)
    m3 = float(_trapz((zg_f - res.mean) ** 3 * pdf_n, zg))
    res.skewness = m3 / res.variance ** 1.5 if res.variance > 1e-14 else 0.0
    cdf = _posterior_cdf(pdf_n, zg)
    al = (1 - ci_prob) / 2
    res.ci_lower = zg_f[np.clip(np.searchsorted(cdf, al), 0, n_grid - 1)]
    res.ci_upper = zg_f[np.clip(np.searchsorted(cdf, 1 - al), 0, n_grid - 1)]
    res.z_grid, res.pdf = zg_f, pdf_n
    res.info = f"full_network_st_bme nh={nh} ns={ns}"
    return res


# ── helpers ──────────────────────────────────────────────────

def _fill_prior(res, mk, sig2, n_grid, ci_prob):
    sig = math.sqrt(sig2)
    res.kriging_mean = res.mean = res.mode = mk
    res.kriging_var = res.variance = sig2
    zg = np.linspace(mk - 4 * sig, mk + 4 * sig, n_grid)
    res.z_grid, res.pdf = zg, norm.pdf(zg, mk, sig)
    zc = norm.ppf((1 + ci_prob) / 2)
    res.ci_lower, res.ci_upper = mk - zc * sig, mk + zc * sig
    res.info = "no_data"
    return res


def _normalise_posterior_pdf(pdf_raw, zg, mean, sigma):
    """Return a finite, unit-area posterior PDF on ``zg``."""
    pdf_n = np.asarray(pdf_raw, dtype=np.float64)
    pdf_n = np.where(np.isfinite(pdf_n) & (pdf_n >= 0.0), pdf_n, 0.0)

    area = float(_trapz(pdf_n, zg))
    if not (area > 1e-300 and np.isfinite(area)):
        pdf_n = norm.pdf(zg, mean, sigma)
        pdf_n = np.where(np.isfinite(pdf_n) & (pdf_n >= 0.0), pdf_n, 0.0)
        area = float(_trapz(pdf_n, zg))

    if not (area > 1e-300 and np.isfinite(area)):
        pdf_n = np.zeros_like(zg, dtype=np.float64)
        pdf_n[np.argmin(np.abs(zg - mean))] = 1.0 / max(float(np.mean(np.diff(zg))), 1e-12)
        area = float(_trapz(pdf_n, zg))

    return pdf_n / max(area, 1e-300)


def _posterior_cdf(pdf_n, zg):
    """Return a monotone CDF for a posterior PDF grid."""
    dz = max(float(np.mean(np.diff(zg))), 1e-12)
    cdf = np.cumsum(np.asarray(pdf_n, dtype=np.float64)) * dz
    total = float(cdf[-1]) if len(cdf) else 0.0
    if not (total > 1e-300 and np.isfinite(total)):
        cdf = np.linspace(0.0, 1.0, len(pdf_n), dtype=np.float64)
    else:
        cdf /= total
    if len(cdf):
        cdf[0] = max(cdf[0], 0.0)
        cdf[-1] = 1.0
    return cdf


def _fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob):
    sig = math.sqrt(k_var)
    res.mean = res.mode = k_mu + mk
    res.variance = k_var
    res.skewness = 0.0  # Gaussian posterior is symmetric
    zg = np.linspace(k_mu + mk - 4 * sig, k_mu + mk + 4 * sig, n_grid)
    res.z_grid, res.pdf = zg, norm.pdf(zg, k_mu + mk, sig)
    zc = norm.ppf((1 + ci_prob) / 2)
    res.ci_lower = k_mu + mk - zc * sig
    res.ci_upper = k_mu + mk + zc * sig
    res.info = f"kriging_only nh={res.n_hard}"
    return res
