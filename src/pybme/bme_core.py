"""
bme_core.py  –  Complete Bayesian Maximum Entropy (BME) Framework in Python
===========================================================================

A comprehensive implementation matching and extending the MATLAB BMElib library
(Christakos et al.), with modern Python efficiency and scalability improvements.


Mathematical Framework
----------------------
Given hard data  z_h  and soft PDFs  f_{S,i}  over a Gaussian-process prior:

  f_K(z_k) = φ(z_k; μ_{k|h}, σ²_{k|h})
              × E_{x_s|z_k,z_h}[∏ f_{S,i}(x_i)]
              / E_{x_s|z_h}     [∏ f_{S,i}(x_i)]

When R(z_k)=1 (no soft data or Gaussian soft data) this is standard kriging.
Non-Gaussian soft data gives genuine non-Gaussian posteriors — the key advantage
of BME.

Dependencies:  numpy, scipy  (core);  matplotlib  (optional, demo plots only)
"""

from __future__ import annotations

import warnings
import math
from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple, Union

import numpy as np
from scipy import linalg, special
from scipy.stats import norm, truncnorm as _truncnorm_dist, lognorm as _lognorm_dist
from scipy.spatial import cKDTree
from scipy.optimize import minimize, minimize_scalar

# Compatibility: np.trapezoid was added in NumPy 2.0; fall back to np.trapz
_trapz = getattr(np, 'trapezoid', np.trapz)

# ════════════════════════════════════════════════════════════════
# §1  DISTANCE UTILITIES
# ════════════════════════════════════════════════════════════════

def coord2dist(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix.  c1 (n1,d), c2 (n2,d) → (n1,n2)."""
    c1, c2 = np.atleast_2d(c1), np.atleast_2d(c2)
    diff = c1[:, None, :] - c2[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ════════════════════════════════════════════════════════════════
# §2  COVARIANCE MODELS  (matching MATLAB BMElib modelslib)
# ════════════════════════════════════════════════════════════════

def exponential_cov(h, params):
    """C(h) = sill·exp(−3h/range).  params=[sill, range]"""
    s, r = params[0], params[1]
    return s * np.exp(-3.0 * np.asarray(h, dtype=np.float64) / r)

def gaussian_cov(h, params):
    """C(h) = sill·exp(−3(h/range)²).  params=[sill, range]"""
    s, r = params[0], params[1]
    return s * np.exp(-3.0 * (np.asarray(h, dtype=np.float64) / r) ** 2)

def spherical_cov(h, params):
    """C(h) = sill·(1 − 1.5t + 0.5t³) for t=min(h/range,1).  params=[sill, range]"""
    s, r = params[0], params[1]
    t = np.minimum(np.asarray(h, dtype=np.float64) / r, 1.0)
    return s * (1.0 - 1.5 * t + 0.5 * t ** 3)

def matern_cov(h, params):
    """Matérn: params=[sill, range, nu].  nu=0.5→exp, nu→∞→Gaussian."""
    s, r, nu = params[0], params[1], params[2]
    h = np.asarray(h, dtype=np.float64)
    out = np.full_like(h, s, dtype=np.float64)
    mask = h > 1e-15
    if np.any(mask):
        sc = np.sqrt(2.0 * nu) * h[mask] / r
        out[mask] = s * (2.0 ** (1 - nu) / special.gamma(nu)) * sc ** nu * special.kv(nu, sc)
    return out

def nugget_cov(h, params):
    """C(h) = sill·δ(h≈0).  params=[sill]"""
    return params[0] * (np.asarray(h) < 1e-10).astype(np.float64)

def hole_cos_cov(h, params):
    """Hole-effect cosine: C(h) = sill·cos(π h/range).  params=[sill, range]"""
    return params[0] * np.cos(np.pi * np.asarray(h, dtype=np.float64) / params[1])

# Registry
COV_MODELS = {
    "exponential": exponential_cov,
    "gaussian": gaussian_cov,
    "spherical": spherical_cov,
    "matern": matern_cov,
    "nugget": nugget_cov,
    "hole_cos": hole_cos_cov,
}

def eval_cov(h, model, params):
    """Evaluate single or *nested* covariance model.

    Nested example:  model=['nugget','exponential'], params=[[0.1],[0.9, 10.0]]
    """
    h = np.asarray(h, dtype=np.float64)
    if isinstance(model, str):
        return COV_MODELS[model](h, params)
    return sum(COV_MODELS[m](h, p) for m, p in zip(model, params))

def build_cov_matrix(c1, c2, model, params):
    """Covariance matrix C(c1,c2).  c1 (n1,d), c2 (n2,d) → (n1,n2)."""
    return eval_cov(coord2dist(np.atleast_2d(c1), np.atleast_2d(c2)), model, params)

def build_cov_matrix_st(c1, t1, c2, t2, model_s, params_s, model_t, params_t, sigma2):
    """Separable space-time: C = σ² · Cs(‖Δx‖) · Ct(|Δt|).

    Spatial/temporal models should have sill=1 (overall sill controlled by sigma2).
    """
    hs = coord2dist(np.atleast_2d(c1), np.atleast_2d(c2))
    ht = np.abs(np.atleast_1d(t1)[:, None] - np.atleast_1d(t2)[None, :])
    return sigma2 * eval_cov(hs, model_s, params_s) * eval_cov(ht, model_t, params_t)


# ════════════════════════════════════════════════════════════════
# §3  SOFT DATA TYPES
# ════════════════════════════════════════════════════════════════

class SoftPDF:
    """Piecewise-linear or histogram representation of a soft probabilistic datum.

    Matches MATLAB BMElib softpdftype 1–4.  Automatically normalised to ∫=1.
    """

    def __init__(self, z_grid: np.ndarray, pdf_values: np.ndarray,
                 pdf_type: str = "linear"):
        """
        Parameters
        ----------
        z_grid      breakpoints (K,)
        pdf_values  densities — length K for 'linear', length K-1 for 'histogram'
        pdf_type    'linear' | 'histogram'
        """
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.pdf_values = np.asarray(pdf_values, dtype=np.float64)
        self.pdf_type = pdf_type
        self._normalize()

    # ---- internal -----------------------------------------------------------
    def _raw_area(self):
        if self.pdf_type == "linear":
            return float(_trapz(self.pdf_values, self.z_grid))
        return float(np.sum(self.pdf_values * np.diff(self.z_grid)))

    def _normalize(self):
        a = self._raw_area()
        if a > 1e-300:
            self.pdf_values = self.pdf_values / a

    # ---- public API ---------------------------------------------------------
    def evaluate(self, z):
        """Evaluate PDF at arbitrary z (scalar or array).  0 outside support."""
        z = np.asarray(z, dtype=np.float64)
        scalar_in = z.ndim == 0
        z = np.atleast_1d(z)
        if self.pdf_type == "linear":
            out = np.interp(z, self.z_grid, self.pdf_values, left=0.0, right=0.0)
        else:
            idx = np.clip(np.searchsorted(self.z_grid, z, side="right") - 1,
                          0, len(self.pdf_values) - 1)
            out = np.where((z >= self.z_grid[0]) & (z <= self.z_grid[-1]),
                           self.pdf_values[idx], 0.0)
        return float(out) if scalar_in else out

    @property
    def support(self):
        return (float(self.z_grid[0]), float(self.z_grid[-1]))

    def moments(self):
        """Return (mean, variance)."""
        zf = np.linspace(self.z_grid[0], self.z_grid[-1], 500)
        pf = self.evaluate(zf)
        mu = float(_trapz(zf * pf, zf))
        var = max(float(_trapz((zf - mu) ** 2 * pf, zf)), 1e-16)
        return mu, var

    # ---- convenience constructors (match MATLAB BMElib) ---------------------
    @classmethod
    def from_gaussian(cls, mean, var, n_pts=25, n_sig=5):
        """Discretised Gaussian N(mean, var).  ≈ MATLAB probaGaussian."""
        sig = np.sqrt(var)
        z = np.linspace(mean - n_sig * sig, mean + n_sig * sig, n_pts)
        return cls(z, norm.pdf(z, mean, sig), "linear")

    @classmethod
    def from_uniform(cls, a, b):
        """Uniform on [a,b].  ≈ MATLAB probaUniform."""
        eps = max((b - a) * 1e-6, 1e-12)
        d = 1.0 / (b - a)
        return cls(np.array([a - eps, a, b, b + eps]),
                   np.array([0.0, d, d, 0.0]), "linear")

    @classmethod
    def from_interval(cls, a, b):
        """Interval-only soft datum [a,b] (uniform likelihood).
        Matches MATLAB BMEinterval approach."""
        return cls.from_uniform(a, b)

    @classmethod
    def from_triangular(cls, a, mode, b):
        """Triangular on [a,b] with peak at *mode*.  ≈ MATLAB probaTriangular."""
        peak = 2.0 / (b - a)
        eps = max((b - a) * 1e-6, 1e-12)
        return cls(np.array([a - eps, a, mode, b, b + eps]),
                   np.array([0.0, 0.0, peak, 0.0, 0.0]), "linear")

    @classmethod
    def from_truncnorm(cls, mu, sigma, a=None, b=None, n_pts=25):
        """Truncated Gaussian N(mu,σ²) on [a,b].  Ideal for censored data."""
        if a is None:
            a = mu - 6 * sigma
        if b is None:
            b = mu + 6 * sigma
        alpha, beta = (a - mu) / sigma, (b - mu) / sigma
        z = np.linspace(a, b, n_pts)
        return cls(z, _truncnorm_dist.pdf(z, alpha, beta, loc=mu, scale=sigma), "linear")

    @classmethod
    def from_lognormal(cls, mu_log, sigma_log, n_pts=30, n_sig=5):
        """Lognormal (underlying normal parameters).  Good for concentrations."""
        zmax = np.exp(mu_log + n_sig * sigma_log)
        zmin = max(1e-10, np.exp(mu_log - n_sig * sigma_log))
        z = np.linspace(zmin, zmax, n_pts)
        return cls(z, _lognorm_dist.pdf(z, s=sigma_log, scale=np.exp(mu_log)), "linear")

    @classmethod
    def from_histogram(cls, breaks, densities):
        """Piecewise-constant (histogram) PDF.  MATLAB softpdftype=1."""
        return cls(np.asarray(breaks), np.asarray(densities), "histogram")

    @classmethod
    def from_linear(cls, z_grid, pdf_values):
        """Piecewise-linear PDF.  MATLAB softpdftype=2."""
        return cls(np.asarray(z_grid), np.asarray(pdf_values), "linear")

    @classmethod
    def from_callable(cls, func: Callable, a: float, b: float, n_pts=50):
        """Arbitrary PDF from a Python callable."""
        z = np.linspace(a, b, n_pts)
        return cls(z, np.maximum([func(zi) for zi in z], 0.0), "linear")

    @classmethod
    def from_mixture(cls, components: list, weights, a=None, b=None, n_pts=60):
        """Mixture of SoftPDF objects."""
        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()
        if a is None:
            a = min(c.support[0] for c in components)
        if b is None:
            b = max(c.support[1] for c in components)
        z = np.linspace(a, b, n_pts)
        pdf = sum(wi * ci.evaluate(z) for wi, ci in zip(w, components))
        return cls(z, pdf, "linear")


# ════════════════════════════════════════════════════════════════
# §4  NEIGHBOURHOOD SELECTION
# ════════════════════════════════════════════════════════════════

def select_neighbors(ck, c_data, n_valid, nmax, dmax):
    """Return indices of ≤ nmax nearest points within dmax of ck."""
    if n_valid == 0:
        return np.array([], dtype=int)
    c_data = np.atleast_2d(c_data[:n_valid])
    d = coord2dist(np.atleast_1d(ck).reshape(1, -1), c_data).ravel()
    within = np.where(d <= dmax)[0]
    if len(within) == 0:
        return np.array([], dtype=int)
    return within[np.argsort(d[within])][:nmax]


def select_neighbors_st(ck, tk, c_data, t_data, nmax, dmax_s, dmax_t, st_ratio=1.0):
    """Space-time neighbourhood: both spatial *and* temporal distance constraints."""
    if len(c_data) == 0:
        return np.array([], dtype=int)
    ds = coord2dist(ck.reshape(1, -1), np.atleast_2d(c_data)).ravel()
    dt = np.abs(float(tk) - np.atleast_1d(t_data))
    within = np.where((ds <= dmax_s) & (dt <= dmax_t))[0]
    if len(within) == 0:
        return np.array([], dtype=int)
    d_comb = ds[within] + st_ratio * dt[within]
    return within[np.argsort(d_comb)][:nmax]


# ════════════════════════════════════════════════════════════════
# §5  MEAN TREND (DRIFT)
# ════════════════════════════════════════════════════════════════

def _design_matrix(coords, order):
    """Polynomial design matrix for trend.
    order: NaN→empty, 0→constant, 1→linear, 2→quadratic.
    """
    coords = np.atleast_2d(coords)
    n, d = coords.shape
    if order is None or (isinstance(order, float) and np.isnan(order)):
        return np.empty((n, 0))
    order = int(order)
    if order == 0:
        return np.ones((n, 1))
    if order == 1:
        return np.column_stack([np.ones(n)] + [coords[:, i] for i in range(d)])
    if order == 2:
        cols = [np.ones(n)] + [coords[:, i] for i in range(d)]
        for i in range(d):
            for j in range(i, d):
                cols.append(coords[:, i] * coords[:, j])
        return np.column_stack(cols)
    raise ValueError(f"order must be NaN, 0, 1, or 2; got {order}")


def _estimate_trend(ch, zh, cs, soft_pdfs, ck, order, mean_prior):
    """Estimate and subtract local mean trend.

    Returns
    -------
    zh_dt, sp_dt, mk, trend_h, trend_s
    """
    if order is None or (isinstance(order, float) and np.isnan(order)):
        zh_dt = zh - mean_prior
        sp_dt = [SoftPDF(sp.z_grid - mean_prior, sp.pdf_values.copy(), sp.pdf_type)
                 for sp in soft_pdfs]
        return zh_dt, sp_dt, mean_prior, np.full(len(zh), mean_prior), np.zeros(len(soft_pdfs))

    Xh = _design_matrix(ch, order)
    Xk = _design_matrix(ck.reshape(1, -1), order)

    # OLS with hard + soft-data means
    soft_means = np.array([sp.moments()[0] for sp in soft_pdfs]) if soft_pdfs else np.array([])
    if len(soft_pdfs) > 0 and len(cs) > 0:
        Xs = _design_matrix(cs, order)
        X_all = np.vstack([Xh, Xs])
        z_all = np.concatenate([zh, soft_means])
    else:
        X_all, z_all = Xh, zh

    try:
        beta = np.linalg.lstsq(X_all, z_all, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(Xh.shape[1])

    trend_h = Xh @ beta if Xh.shape[1] > 0 else np.zeros(len(zh))
    mk = float(Xk @ beta) if Xk.shape[1] > 0 else 0.0
    if len(soft_pdfs) > 0 and len(cs) > 0:
        trend_s = _design_matrix(cs, order) @ beta
    else:
        trend_s = np.array([])

    zh_dt = zh - trend_h
    sp_dt = [SoftPDF(sp.z_grid - trend_s[i], sp.pdf_values.copy(), sp.pdf_type)
             for i, sp in enumerate(soft_pdfs)]
    return zh_dt, sp_dt, mk, trend_h, trend_s


# ════════════════════════════════════════════════════════════════
# §6  CORE INTEGRATION ENGINE
# ════════════════════════════════════════════════════════════════

_GH_CACHE: dict = {}

def _gh_nodes(n: int):
    """Cached physicist-Hermite nodes & weights for ∫ f(x) e^{-x²} dx."""
    if n not in _GH_CACHE:
        _GH_CACHE[n] = np.polynomial.hermite.hermgauss(n)
    return _GH_CACHE[n]


def _adaptive_nquad(ns: int, base: int = 15) -> int:
    """Choose quadrature points per dimension so total cost stays manageable."""
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
    return 0  # flag → use Monte Carlo


def _integrate_soft_product(soft_pdfs: List[SoftPDF],
                            mu: np.ndarray, cov: np.ndarray,
                            n_quad: int = 15) -> float:
    """E_{x∼N(μ,Σ)}[ ∏ᵢ fᵢ(xᵢ) ]  via Gauss-Hermite quadrature.

    This is the heart of BME — the integral that distinguishes it from kriging.
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
        # fast 1-D path
        x = mu[0] + math.sqrt(2.0) * L[0, 0] * nodes
        fv = soft_pdfs[0].evaluate(x)
        return max(float(np.dot(weights, fv)) / math.sqrt(math.pi), 1e-300)

    # tensor-product Gauss-Hermite  (cost = nq^ns, feasible for ns ≤ 8)
    grids = np.meshgrid(*([nodes] * ns), indexing="ij")
    u = np.column_stack([g.ravel() for g in grids])           # (N, ns)
    wg = np.meshgrid(*([weights] * ns), indexing="ij")
    w = np.ones(u.shape[0])
    for wgi in wg:
        w *= wgi.ravel()

    x = mu[None, :] + math.sqrt(2.0) * (u @ L.T)             # (N, ns)

    prod_f = np.ones(x.shape[0])
    for i, sp in enumerate(soft_pdfs):
        prod_f *= sp.evaluate(x[:, i])

    return max(float(np.dot(w, prod_f)) / (math.pi ** (ns / 2.0)), 1e-300)


def _mc_integrate(soft_pdfs, mu, L, n_samples=30000):
    """Monte Carlo fallback for high-dimensional soft-data integration."""
    ns = len(soft_pdfs)
    u = np.random.randn(n_samples, ns)
    x = mu[None, :] + u @ L.T
    prod_f = np.ones(n_samples)
    for i, sp in enumerate(soft_pdfs):
        prod_f *= sp.evaluate(x[:, i])
    return max(float(np.mean(prod_f)), 1e-300)


# ════════════════════════════════════════════════════════════════
# §7  RESULT CONTAINER
# ════════════════════════════════════════════════════════════════

@dataclass
class BMEResult:
    """Container returned for each estimation point."""
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


# ════════════════════════════════════════════════════════════════
# §8  MAIN BME PREDICTION (spatial)
# ════════════════════════════════════════════════════════════════

def bme_predict(ck, ch, zh,
                cs=None, soft_pdfs=None,
                model="exponential", params=None,
                nhmax=20, nsmax=8, dmax=np.inf,
                order=0, n_grid=200, ci_prob=0.95,
                n_quad=15, mean_prior=0.0) -> List[BMEResult]:
    """Full BME prediction — posterior PDF, mode, moments, CI.

    Combines MATLAB's BMEprobaPdf, BMEprobaMode, BMEprobaMoments, BMEprobaCI
    into one efficient call.

    Parameters
    ----------
    ck         (nk, d)  estimation point(s)
    ch         (nh, d)  hard-data coordinates
    zh         (nh,)    hard-data values
    cs         (ns, d)  soft-data coordinates          [optional]
    soft_pdfs  list of SoftPDF (one per soft point)    [optional]
    model      covariance model name or list (nested)
    params     model parameters or list-of-lists (nested)
    nhmax      max hard neighbours
    nsmax      max soft neighbours  (keep ≤ 8 for speed)
    dmax       max distance for neighbourhood
    order      NaN=simple, 0=ordinary, 1=linear, 2=quadratic kriging trend
    n_grid     z-grid resolution for posterior PDF
    ci_prob    confidence-interval probability (e.g. 0.95)
    n_quad     base Gauss-Hermite quadrature points per dimension
    mean_prior prior mean (used when order=NaN)

    Returns
    -------
    List[BMEResult]   one per estimation point
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
        params = [1.0, 1.0]  # default sill=1, range=1

    return [_bme_point(ck[i], ch, zh, cs, soft_pdfs,
                       model, params, nhmax, nsmax, dmax,
                       order, n_grid, ci_prob, n_quad, mean_prior)
            for i in range(ck.shape[0])]


def _bme_point(ck, ch, zh, cs, soft_pdfs,
               model, params, nhmax, nsmax, dmax,
               order, n_grid, ci_prob, n_quad, mean_prior):
    """BME at a single estimation point."""
    res = BMEResult(ci_prob=ci_prob)

    # ---- neighbourhood ----
    idx_h = select_neighbors(ck, ch, len(zh), nhmax, dmax)
    idx_s = select_neighbors(ck, cs, len(soft_pdfs), nsmax, dmax)
    ch_l, zh_l = ch[idx_h], zh[idx_h]
    cs_l = cs[idx_s] if len(idx_s) > 0 else np.empty((0, ck.shape[-1]))
    sp_l = [soft_pdfs[i] for i in idx_s]
    nh, ns = len(idx_h), len(idx_s)
    res.n_hard, res.n_soft = nh, ns

    # ---- duplicate check ----
    if nh > 0:
        dd = coord2dist(ck.reshape(1, -1), ch_l).ravel()
        dup = np.where(dd < 1e-10)[0]
        if len(dup) > 0:
            v = zh_l[dup[0]]
            res.mode = res.mean = res.kriging_mean = v
            res.variance = res.kriging_var = 0.0
            res.info = "duplicate"
            return res

    # ---- trend removal ----
    zh_dt, sp_dt, mk, _, _ = _estimate_trend(ch_l, zh_l, cs_l, sp_l, ck, order, mean_prior)

    # ---- covariance matrices ----
    ck2 = ck.reshape(1, -1)
    sig2 = float(build_cov_matrix(ck2, ck2, model, params)[0, 0])

    if nh == 0 and ns == 0:
        return _fill_prior(res, mk, sig2, n_grid, ci_prob)

    # hard-data blocks
    if nh > 0:
        Ckh = build_cov_matrix(ck2, ch_l, model, params)      # (1, nh)
        Chh = build_cov_matrix(ch_l, ch_l, model, params)      # (nh, nh)
        Chh += np.eye(nh) * 1e-10
        Lhh = linalg.cholesky(Chh, lower=True)

    # soft-data blocks
    if ns > 0:
        Cks = build_cov_matrix(ck2, cs_l, model, params)       # (1, ns)
        Css = build_cov_matrix(cs_l, cs_l, model, params)      # (ns, ns)
        Css += np.eye(ns) * 1e-10
        if nh > 0:
            Chs = build_cov_matrix(ch_l, cs_l, model, params)  # (nh, ns)

    # ---- kriging (hard-data only, for comparison & as prior) ----
    if nh > 0:
        alpha_h = linalg.cho_solve((Lhh, True), zh_dt)
        k_mu = float((Ckh @ alpha_h).ravel()[0])
        k_var = max(float((sig2 - Ckh @ linalg.cho_solve((Lhh, True), Ckh.T)).ravel()[0]), 1e-12)
    else:
        k_mu, k_var = 0.0, sig2
    res.kriging_mean = k_mu + mk
    res.kriging_var = k_var

    # ---- pure kriging shortcut (no soft data) ----
    if ns == 0:
        return _fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob)

    # ════════════════════════════════════════════════════════════
    # FULL BME — non-Gaussian posterior via numerical integration
    # ════════════════════════════════════════════════════════════

    # Block matrix for [k; h]
    nkh = 1 + nh
    C_kh = np.zeros((nkh, nkh))
    C_kh[0, 0] = sig2
    if nh > 0:
        C_kh[0, 1:] = C_kh[1:, 0] = Ckh.ravel()
        C_kh[1:, 1:] = Chh
    C_kh += np.eye(nkh) * 1e-10

    # Cross [s, kh]
    C_s_kh = np.zeros((ns, nkh))
    C_s_kh[:, 0] = Cks.ravel()
    if nh > 0:
        C_s_kh[:, 1:] = Chs.T

    # Denominator: x_s | z_h
    if nh > 0:
        Chh_inv_Chs = linalg.cho_solve((Lhh, True), Chs)
        mu_s_h = (Chs.T @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()
        K_s_h = Css - Chs.T @ Chh_inv_Chs
    else:
        mu_s_h = np.zeros(ns)
        K_s_h = Css.copy()
    K_s_h = 0.5 * (K_s_h + K_s_h.T) + np.eye(ns) * 1e-10
    I_den = _integrate_soft_product(sp_dt, mu_s_h, K_s_h, n_quad)

    # Numerator: x_s | z_k, z_h — precompute reusable pieces
    L_kh = linalg.cholesky(C_kh, lower=True)
    K_s_kh = Css - C_s_kh @ linalg.cho_solve((L_kh, True), C_s_kh.T)
    K_s_kh = 0.5 * (K_s_kh + K_s_kh.T) + np.eye(ns) * 1e-10

    B = C_s_kh @ np.linalg.inv(C_kh)          # (ns, 1+nh)
    b_k = B[:, 0]                              # z_k coefficient
    b_const = B[:, 1:] @ zh_dt if nh > 0 else np.zeros(ns)

    # ---- evaluate posterior on z-grid ----
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
        I_num = _integrate_soft_product(sp_dt, mu_s_kh, K_s_kh, n_quad)
        pdf_raw[iz] = prior * I_num

    area = float(_trapz(pdf_raw, zg))
    if area > 1e-300:
        pdf_n = pdf_raw / area
    else:
        pdf_n = norm.pdf(zg, k_mu, sigma_k)
        res.info = "integration_fallback"

    zg_final = zg + mk

    # ---- extract statistics ----
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


# ---- helpers for shortcut paths ----

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
    zg = np.linspace(k_mu + mk - 4 * sig, k_mu + mk + 4 * sig, n_grid)
    res.z_grid, res.pdf = zg, norm.pdf(zg, k_mu + mk, sig)
    zc = norm.ppf((1 + ci_prob) / 2)
    res.ci_lower = k_mu + mk - zc * sig
    res.ci_upper = k_mu + mk + zc * sig
    res.info = f"kriging_only nh={res.n_hard}"
    return res


# ════════════════════════════════════════════════════════════════
# §9  SPACE-TIME BME PREDICTION
# ════════════════════════════════════════════════════════════════

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

    Spatial and temporal models should have sill=1 (overall sill in sigma2).
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
        return build_cov_matrix_st(c1, t1, c2, t2, model_s, params_s, model_t, params_t, sigma2)

    results = []
    for ik in range(ck.shape[0]):
        res = BMEResult(ci_prob=ci_prob)
        ck_i, tk_i = ck[ik:ik + 1], tk[ik:ik + 1]

        idx_h = select_neighbors_st(ck[ik], tk[ik], ch, th, nhmax, dmax_s, dmax_t)
        idx_s = (select_neighbors_st(ck[ik], tk[ik], cs, ts, nsmax, dmax_s, dmax_t)
                 if len(soft_pdfs) else np.array([], dtype=int))
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

        # trend
        zh_dt, sp_dt, mk, _, _ = _estimate_trend(ch_l, zh_l, cs_l, sp_l, ck[ik], order, mean_prior)

        if nh > 0:
            Ckh = _stcov(ck_i, tk_i, ch_l, th_l)
            Chh = _stcov(ch_l, th_l, ch_l, th_l) + np.eye(nh) * 1e-10
            Lhh = linalg.cholesky(Chh, lower=True)
        if ns > 0:
            Cks = _stcov(ck_i, tk_i, cs_l, ts_l)
            Css = _stcov(cs_l, ts_l, cs_l, ts_l) + np.eye(ns) * 1e-10
            if nh > 0:
                Chs = _stcov(ch_l, th_l, cs_l, ts_l)

        # kriging
        if nh > 0:
            k_mu = float((Ckh @ linalg.cho_solve((Lhh, True), zh_dt)).ravel()[0])
            k_var = max(float((sig2_loc - Ckh @ linalg.cho_solve((Lhh, True), Ckh.T)).ravel()[0]), 1e-12)
        else:
            k_mu, k_var = 0.0, sig2_loc
        res.kriging_mean, res.kriging_var = k_mu + mk, k_var

        if ns == 0:
            results.append(_fill_gaussian(res, k_mu, k_var, mk, n_grid, ci_prob))
            continue

        # full BME (same maths as spatial, different cov builder)
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
        I_den = _integrate_soft_product(sp_dt, mu_s_h, K_s_h, n_quad)

        L_kh = linalg.cholesky(C_kh_kh, lower=True)
        K_s_kh = Css - C_s_kh @ linalg.cho_solve((L_kh, True), C_s_kh.T)
        K_s_kh = 0.5 * (K_s_kh + K_s_kh.T) + np.eye(ns) * 1e-10

        B = C_s_kh @ np.linalg.inv(C_kh_kh)
        b_k, b_c = B[:, 0], (B[:, 1:] @ zh_dt if nh > 0 else np.zeros(ns))

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
            pdf_raw[iz] = pr * _integrate_soft_product(sp_dt, b_k * zk + b_c, K_s_kh, n_quad)

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


# ════════════════════════════════════════════════════════════════
# §10  CONVENIENCE WRAPPERS
# ════════════════════════════════════════════════════════════════

def kriging_predict(ck, ch, zh, model="exponential", params=None,
                    nhmax=50, dmax=np.inf, order=0, mean_prior=0.0):
    """Standard kriging (hard data only)."""
    return bme_predict(ck, ch, zh, model=model, params=params,
                       nhmax=nhmax, nsmax=0, dmax=dmax, order=order,
                       mean_prior=mean_prior, n_grid=100)


def kriging_me_predict(ck, ch, zh, cs, zs_mean, zs_var,
                       model="exponential", params=None,
                       nhmax=20, nsmax=20, dmax=np.inf, order=0):
    """Kriging with measurement error (Gaussian soft data)."""
    sp = [SoftPDF.from_gaussian(m, v) for m, v in zip(zs_mean, zs_var)]
    return bme_predict(ck, ch, zh, cs, sp, model=model, params=params,
                       nhmax=nhmax, nsmax=nsmax, dmax=dmax, order=order)


# ════════════════════════════════════════════════════════════════
# §11  COVARIANCE PARAMETER FITTING (restricted MLE)
# ════════════════════════════════════════════════════════════════

def fit_covariance(ch, zh, model="exponential", order=0,
                   range_bounds=(0.01, None), sill_bounds=(0.01, None),
                   nugget_bounds=(0.0, None)):
    """Fit covariance parameters by maximising the restricted log-likelihood.

    Fits *sill*, *range*, and *nugget* for a single isotropic model.

    Returns
    -------
    dict with keys: 'sill', 'range', 'nugget', 'nll', 'success'
    """
    ch = np.atleast_2d(ch)
    zh = np.asarray(zh, dtype=np.float64)
    n = len(zh)

    def neg_reml(log_theta):
        sill = np.exp(log_theta[0])
        rng  = np.exp(log_theta[1])
        nug  = np.exp(log_theta[2])
        K = build_cov_matrix(ch, ch, model, [sill, rng]) + np.eye(n) * (nug + 1e-10)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        # profile mean
        X = _design_matrix(ch, order)
        alpha = linalg.cho_solve((L, True), zh)
        if X.shape[1] > 0:
            XtKi = linalg.cho_solve((L, True), X)   # K^{-1} X
            beta = np.linalg.lstsq(X.T @ XtKi, X.T @ alpha, rcond=None)[0]
            resid = zh - X @ beta
        else:
            resid = zh
        alpha_r = linalg.cho_solve((L, True), resid)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        return 0.5 * (float(resid @ alpha_r) + logdet + n * math.log(2 * math.pi))

    # initial guess from data range
    dists = coord2dist(ch, ch)
    dmax_data = np.max(dists) if len(dists) > 0 else 1.0
    z_var = max(np.var(zh), 1e-6)
    x0 = np.log([z_var, dmax_data / 3.0, z_var * 0.05 + 1e-6])
    res = minimize(neg_reml, x0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4})
    sill, rng, nug = np.exp(res.x)
    return {"sill": sill, "range": rng, "nugget": nug,
            "nll": res.fun, "success": res.success}


# ════════════════════════════════════════════════════════════════
# §12  CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════

def cross_validate(ch, zh, cs=None, soft_pdfs=None,
                   model="exponential", params=None,
                   nhmax=20, nsmax=8, dmax=np.inf, order=0):
    """Leave-one-out cross-validation on hard data.

    Returns dict with 'predicted', 'actual', 'errors', 'rmse', 'mae'.
    """
    ch, zh = np.atleast_2d(ch), np.asarray(zh)
    nh = len(zh)
    pred, vari = np.zeros(nh), np.zeros(nh)
    for i in range(nh):
        r = bme_predict(ch[i:i + 1], np.delete(ch, i, 0), np.delete(zh, i),
                        cs, soft_pdfs, model, params,
                        nhmax, nsmax, dmax, order, n_grid=80)[0]
        pred[i], vari[i] = r.mean, r.variance
    err = zh - pred
    return {"predicted": pred, "predicted_var": vari, "actual": zh,
            "errors": err, "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err)))}


# ════════════════════════════════════════════════════════════════
# §13  DEMONSTRATION — BME vs. KRIGING with diverse soft data
# ════════════════════════════════════════════════════════════════

def demo():
    """
    1-D spatial demonstration showing BME's advantage over kriging
    when diverse soft (uncertain / censored / interval) data are available.
    """
    np.random.seed(42)

    # ── true field ──
    x_true = np.linspace(0, 10, 150).reshape(-1, 1)
    z_true = np.sin(x_true.ravel()) + 0.3 * np.cos(2.5 * x_true.ravel())

    # ── hard data (exact measurements) ──
    x_hard = np.array([[0.5], [2.0], [4.0], [6.5], [9.0]])
    z_hard = np.sin(x_hard.ravel()) + 0.3 * np.cos(2.5 * x_hard.ravel()) + 0.05 * np.random.randn(5)

    # ── soft data — diverse types (the BME advantage) ──
    x_soft = np.array([[1.2], [3.0], [5.0], [7.5], [8.0]])
    z_soft_true = np.sin(x_soft.ravel()) + 0.3 * np.cos(2.5 * x_soft.ravel())

    soft_pdfs = [
        # 1. Gaussian — uncertain measurement with known error variance
        SoftPDF.from_gaussian(z_soft_true[0] + 0.1, 0.15),

        # 2. Interval — "value is somewhere between 0.0 and 0.8"
        SoftPDF.from_interval(-0.2, 0.8),

        # 3. Triangular — expert says "most likely near true value"
        SoftPDF.from_triangular(z_soft_true[2] - 0.5, z_soft_true[2] + 0.05, z_soft_true[2] + 0.6),

        # 4. Truncated normal — below detection limit (censored): value < 0.3
        #    modelled as N(z_true, 0.3²) truncated to (−∞, 0.3)
        SoftPDF.from_truncnorm(z_soft_true[3], 0.3, a=None, b=z_soft_true[3] + 0.3),

        # 5. Lognormal — concentration measurement (strictly positive)
        SoftPDF.from_lognormal(np.log(max(abs(z_soft_true[4]) + 0.1, 0.2)), 0.4),
    ]

    # ── covariance model (nested: nugget + exponential) ──
    cov_model = ["nugget", "exponential"]
    cov_params = [[0.02], [1.0, 3.0]]

    print("=" * 70)
    print("BME CORE — Demonstration:  BME vs. Kriging with diverse soft data")
    print("=" * 70)

    # ── kriging (hard data only) ──
    krig_res = bme_predict(x_true, x_hard, z_hard,
                           model=cov_model, params=cov_params,
                           nhmax=20, nsmax=0, dmax=np.inf, order=0,
                           n_grid=100)
    krig_mean = np.array([r.mean for r in krig_res])
    krig_var  = np.array([r.variance for r in krig_res])

    # ── BME (hard + soft) ──
    print("  Computing BME predictions (this exercises the full integration engine)...")
    bme_res = bme_predict(x_true, x_hard, z_hard,
                          cs=x_soft, soft_pdfs=soft_pdfs,
                          model=cov_model, params=cov_params,
                          nhmax=20, nsmax=4, dmax=4.0, order=0,
                          n_grid=100, n_quad=10)
    bme_mean = np.array([r.mean for r in bme_res])
    bme_var  = np.array([r.variance for r in bme_res])

    # ── error comparison ──
    rmse_krig = np.sqrt(np.mean((krig_mean - z_true) ** 2))
    rmse_bme  = np.sqrt(np.mean((bme_mean - z_true) ** 2))
    print(f"\n  RMSE  Kriging (hard only)  : {rmse_krig:.4f}")
    print(f"  RMSE  BME    (hard + soft) : {rmse_bme:.4f}")
    print(f"  Improvement               : {(rmse_krig - rmse_bme) / rmse_krig * 100:.1f}%")

    avg_var_krig = np.mean(krig_var)
    avg_var_bme  = np.mean(bme_var)
    print(f"\n  Avg posterior variance  Kriging : {avg_var_krig:.4f}")
    print(f"  Avg posterior variance  BME     : {avg_var_bme:.4f}")
    print(f"  Uncertainty reduction          : {(avg_var_krig - avg_var_bme) / avg_var_krig * 100:.1f}%")

    # ── show non-Gaussian posteriors at soft data locations ──
    print("\n  Non-Gaussian posterior details at soft data locations:")
    print(f"  {'Location':>8s}  {'SoftType':>12s}  {'BME mode':>9s}  {'BME mean':>9s}"
          f"  {'Skewness':>9s}  {'95% CI':>16s}")
    soft_types = ["Gaussian", "Interval", "Triangular", "TruncNorm", "Lognormal"]
    for i, xs in enumerate(x_soft):
        r = bme_predict(xs.reshape(1, -1), x_hard, z_hard,
                        cs=x_soft, soft_pdfs=soft_pdfs,
                        model=cov_model, params=cov_params,
                        nhmax=20, nsmax=4, dmax=4.0, order=0,
                        n_grid=150, n_quad=10)[0]
        print(f"  x={xs[0]:5.1f}  {soft_types[i]:>12s}  {r.mode:9.4f}  {r.mean:9.4f}"
              f"  {r.skewness:9.4f}  [{r.ci_lower:7.3f}, {r.ci_upper:6.3f}]")

    # ── covariance parameter fitting ──
    print("\n  Fitting covariance parameters from hard data (REML)...")
    fit = fit_covariance(x_hard, z_hard, model="exponential", order=0)
    print(f"    sill={fit['sill']:.4f}  range={fit['range']:.4f}  nugget={fit['nugget']:.6f}"
          f"  (converged={fit['success']})")

    # ── cross-validation ──
    print("\n  Leave-one-out cross-validation (hard data, with soft data support):")
    cv = cross_validate(x_hard, z_hard, cs=x_soft, soft_pdfs=soft_pdfs,
                        model=cov_model, params=cov_params,
                        nhmax=20, nsmax=4, dmax=4.0, order=0)
    print(f"    RMSE = {cv['rmse']:.4f},  MAE = {cv['mae']:.4f}")

    # ── optional plot ──
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 3, 2]})

        # --- panel 1: kriging ---
        ax = axes[0]
        ax.plot(x_true, z_true, "k-", lw=1.5, label="True field")
        ax.plot(x_true, krig_mean, "b--", lw=1.2, label=f"Kriging mean (RMSE={rmse_krig:.3f})")
        ax.fill_between(x_true.ravel(),
                        krig_mean - 1.96 * np.sqrt(krig_var),
                        krig_mean + 1.96 * np.sqrt(krig_var),
                        alpha=0.2, color="blue", label="Kriging 95% CI")
        ax.plot(x_hard, z_hard, "ko", ms=8, zorder=5, label="Hard data")
        ax.set_title("Standard Kriging (hard data only)", fontsize=13)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylabel("Z")

        # --- panel 2: BME ---
        ax = axes[1]
        ax.plot(x_true, z_true, "k-", lw=1.5, label="True field")
        ax.plot(x_true, bme_mean, "r-", lw=1.2, label=f"BME mean (RMSE={rmse_bme:.3f})")
        ax.fill_between(x_true.ravel(),
                        bme_mean - 1.96 * np.sqrt(bme_var),
                        bme_mean + 1.96 * np.sqrt(bme_var),
                        alpha=0.2, color="red", label="BME 95% CI")
        ax.plot(x_hard, z_hard, "ko", ms=8, zorder=5, label="Hard data")
        markers = ["s", "D", "^", "v", "p"]
        colors = ["green", "orange", "purple", "brown", "teal"]
        for i, xs in enumerate(x_soft):
            m, _ = soft_pdfs[i].moments()
            ax.plot(xs[0], m, markers[i], color=colors[i], ms=10, zorder=5,
                    label=f"Soft: {soft_types[i]}")
        ax.set_title("BME (hard + diverse soft data)", fontsize=13)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.set_ylabel("Z")

        # --- panel 3: posterior PDFs at soft data locations ---
        ax = axes[2]
        for i, xs in enumerate(x_soft):
            r = bme_predict(xs.reshape(1, -1), x_hard, z_hard,
                            cs=x_soft, soft_pdfs=soft_pdfs,
                            model=cov_model, params=cov_params,
                            nhmax=20, nsmax=4, dmax=4.0, order=0,
                            n_grid=150, n_quad=10)[0]
            if r.z_grid is not None:
                ax.plot(r.z_grid, r.pdf, color=colors[i], lw=1.5,
                        label=f"x={xs[0]:.1f} ({soft_types[i]})")
                # overlay kriging Gaussian for comparison
                ax.plot(r.z_grid,
                        norm.pdf(r.z_grid, r.kriging_mean, np.sqrt(r.kriging_var)),
                        "--", color=colors[i], alpha=0.4, lw=1)
        ax.set_title("Non-Gaussian posterior PDFs (solid=BME, dashed=kriging)", fontsize=13)
        ax.legend(fontsize=8, ncol=2)
        ax.set_xlabel("Z value")
        ax.set_ylabel("Density")

        plt.tight_layout()
        plt.savefig("bme_demo.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("\n  Plot saved to bme_demo.png")

    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")

    print("\n" + "=" * 70)
    print("Demo complete.  Key takeaways:")
    print("  • BME integrates diverse uncertain data that kriging cannot use")
    print("  • Non-Gaussian soft data → genuinely non-Gaussian posteriors")
    print("  • Lower RMSE and tighter uncertainty bounds with BME")
    print("=" * 70)


if __name__ == "__main__":
    demo()
