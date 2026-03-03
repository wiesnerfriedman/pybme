"""BME and kriging prediction — posterior PDF, mode, moments, CI.

This module corresponds to the MATLAB functions BMEprobaPdf, BMEprobaMode,
BMEprobaMoments, BMEprobaCI combined into a single efficient call.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import linalg
from scipy.stats import norm

from .covariance import build_cov_matrix, build_cov_matrix_st
from .distance import coord2dist
from .integration import integrate_soft_product
from .neighborhood import select_neighbors, select_neighbors_st
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
                n_quad=15, mean_prior=0.0) -> List[BMEResult]:
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

    return [
        _bme_point(ck[i], ch, zh, cs, soft_pdfs,
                   model, params, nhmax, nsmax, dmax,
                   order, n_grid, ci_prob, n_quad, mean_prior)
        for i in range(ck.shape[0])
    ]


def _bme_point(ck, ch, zh, cs, soft_pdfs,
               model, params, nhmax, nsmax, dmax,
               order, n_grid, ci_prob, n_quad, mean_prior):
    """BME at a single estimation point."""
    res = BMEResult(ci_prob=ci_prob)

    # ── neighbourhood ──
    idx_h = select_neighbors(ck, ch, len(zh), nhmax, dmax)
    idx_s = select_neighbors(ck, cs, len(soft_pdfs), nsmax, dmax)
    ch_l, zh_l = ch[idx_h], zh[idx_h]
    cs_l = cs[idx_s] if len(idx_s) > 0 else np.empty((0, ck.shape[-1]))
    sp_l = [soft_pdfs[i] for i in idx_s]
    nh, ns = len(idx_h), len(idx_s)
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
    I_den = integrate_soft_product(sp_dt, mu_s_h, K_s_h, n_quad)

    # Numerator integral:  E [ ∏ fS | z_k, z_h ]  — varies with z_k
    L_kh = linalg.cholesky(C_kh, lower=True)
    K_s_kh = Css - C_s_kh @ linalg.cho_solve((L_kh, True), C_s_kh.T)
    K_s_kh = 0.5 * (K_s_kh + K_s_kh.T) + np.eye(ns) * 1e-10

    B = C_s_kh @ np.linalg.inv(C_kh)
    b_k = B[:, 0]
    b_const = B[:, 1:] @ zh_dt if nh > 0 else np.zeros(ns)

    # ── evaluate posterior on z-grid ──
    sigma_k = math.sqrt(k_var)
    z_lo = k_mu - 5.0 * sigma_k
    z_hi = k_mu + 5.0 * sigma_k
    for sp in sp_dt:
        z_lo = min(z_lo, sp.support[0] - 2 * sigma_k)
        z_hi = max(z_hi, sp.support[1] + 2 * sigma_k)
    zg = np.linspace(z_lo, z_hi, n_grid)

    pdf_raw = np.empty(n_grid)
    for iz, zk in enumerate(zg):
        prior = norm.pdf(zk, k_mu, sigma_k)
        if prior < 1e-300:
            pdf_raw[iz] = 0.0
            continue
        mu_s_kh = b_k * zk + b_const
        I_num = integrate_soft_product(sp_dt, mu_s_kh, K_s_kh, n_quad)
        pdf_raw[iz] = prior * I_num

    area = float(_trapz(pdf_raw, zg))
    if area > 1e-300:
        pdf_n = pdf_raw / area
    else:
        pdf_n = norm.pdf(zg, k_mu, sigma_k)
        res.info = "integration_fallback"

    zg_final = zg + mk

    # ── extract statistics ──
    res.mode = zg_final[int(np.argmax(pdf_n))]
    res.mean = float(_trapz(zg_final * pdf_n, zg))
    res.variance = max(float(_trapz((zg_final - res.mean) ** 2 * pdf_n, zg)), 1e-12)
    m3 = float(_trapz((zg_final - res.mean) ** 3 * pdf_n, zg))
    res.skewness = m3 / res.variance ** 1.5 if res.variance > 1e-14 else 0.0

    cdf = np.cumsum(pdf_n) * np.mean(np.diff(zg))
    cdf /= cdf[-1]
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
                   n_quad=15, mean_prior=0.0) -> List[BMEResult]:
    """Separable space-time BME prediction.

    C((x,t),(x',t')) = σ² · Cs(‖x−x'‖) · Ct(|t−t'|)

    Spatial and temporal models should have sill = 1 (overall sill in *sigma2*).
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

    def _stcov(c1, t1, c2, t2):
        return build_cov_matrix_st(c1, t1, c2, t2,
                                   model_s, params_s, model_t, params_t, sigma2)

    results = []
    for ik in range(ck.shape[0]):
        res = BMEResult(ci_prob=ci_prob)
        ck_i, tk_i = ck[ik:ik + 1], tk[ik:ik + 1]

        idx_h = select_neighbors_st(ck[ik], tk[ik], ch, th, nhmax, dmax_s, dmax_t)
        idx_s = (
            select_neighbors_st(ck[ik], tk[ik], cs, ts, nsmax, dmax_s, dmax_t)
            if len(soft_pdfs) else np.array([], dtype=int)
        )
        ch_l, th_l, zh_l = ch[idx_h], th[idx_h], zh[idx_h]
        cs_l = cs[idx_s] if len(idx_s) else np.empty((0, d_dim))
        ts_l = ts[idx_s] if len(idx_s) else np.array([])
        sp_l = [soft_pdfs[i] for i in idx_s]
        nh, ns = len(idx_h), len(idx_s)
        res.n_hard, res.n_soft = nh, ns

        sig2_loc = float(_stcov(ck_i, tk_i, ck_i, tk_i)[0, 0])

        if nh == 0 and ns == 0:
            results.append(_fill_prior(res, mean_prior, sig2_loc, n_grid, ci_prob))
            continue

        zh_dt, sp_dt, mk, _, _ = estimate_trend(
            ch_l, zh_l, cs_l, sp_l, ck[ik], order, mean_prior
        )

        if nh > 0:
            Ckh = _stcov(ck_i, tk_i, ch_l, th_l)
            Chh = _stcov(ch_l, th_l, ch_l, th_l) + np.eye(nh) * 1e-10
            Lhh = linalg.cholesky(Chh, lower=True)
        if ns > 0:
            Cks = _stcov(ck_i, tk_i, cs_l, ts_l)
            Css = _stcov(cs_l, ts_l, cs_l, ts_l) + np.eye(ns) * 1e-10
            if nh > 0:
                Chs = _stcov(ch_l, th_l, cs_l, ts_l)

        if nh > 0:
            k_mu = float((Ckh @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()[0])
            k_var = max(float((sig2_loc - Ckh @ linalg.cho_solve((Lhh, True), Ckh.T)).ravel()[0]), 1e-12)
        else:
            k_mu, k_var = 0.0, sig2_loc
        res.kriging_mean, res.kriging_var = k_mu + mk, k_var

        if ns == 0:
            results.append(_fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob))
            continue

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
        integrate_soft_product(sp_dt, mu_s_h, K_s_h, n_quad)  # denominator

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

        pdf_raw = np.empty(n_grid)
        for iz, zk in enumerate(zg):
            pr = norm.pdf(zk, k_mu, sigma_k)
            if pr < 1e-300:
                pdf_raw[iz] = 0.0
                continue
            pdf_raw[iz] = pr * integrate_soft_product(
                sp_dt, b_k * zk + b_c, K_s_kh, n_quad
            )

        area = float(_trapz(pdf_raw, zg))
        pdf_n = pdf_raw / area if area > 1e-300 else norm.pdf(zg, k_mu, sigma_k)
        zg_f = zg + mk

        res.mode = zg_f[int(np.argmax(pdf_n))]
        res.mean = float(_trapz(zg_f * pdf_n, zg))
        res.variance = max(float(_trapz((zg_f - res.mean) ** 2 * pdf_n, zg)), 1e-12)
        m3 = float(_trapz((zg_f - res.mean) ** 3 * pdf_n, zg))
        res.skewness = m3 / res.variance ** 1.5 if res.variance > 1e-14 else 0.0
        cdf = np.cumsum(pdf_n) * np.mean(np.diff(zg))
        cdf /= cdf[-1]
        al = (1 - ci_prob) / 2
        res.ci_lower = zg_f[np.clip(np.searchsorted(cdf, al), 0, n_grid - 1)]
        res.ci_upper = zg_f[np.clip(np.searchsorted(cdf, 1 - al), 0, n_grid - 1)]
        res.z_grid, res.pdf = zg_f, pdf_n
        res.info = f"full_st_bme nh={nh} ns={ns}"
        results.append(res)
    return results


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
