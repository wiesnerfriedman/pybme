"""Soft probabilistic data types for BME.

Provides the :class:`SoftPDF` class with convenience constructors matching
every MATLAB BMElib soft-data type (softpdftype 1–4, probaGaussian,
probaUniform, probaTriangular, etc.) plus additional distributions
(truncated-normal, lognormal, mixture, arbitrary callable).
"""

from __future__ import annotations
from typing import Callable, List

import numpy as np
from scipy.stats import (
    norm,
    truncnorm as _truncnorm_dist,
    lognorm as _lognorm_dist,
)

# NumPy ≥ 2.0 renamed trapz → trapezoid
_trapz = getattr(np, "trapezoid", np.trapz)


class SoftPDF:
    """Piecewise-linear or histogram representation of a single soft datum.

    Automatically normalised so that the total area equals 1.

    Parameters
    ----------
    z_grid     : 1-D array of breakpoints  (length K)
    pdf_values : density values —
                   * length K   for ``pdf_type='linear'``
                   * length K-1 for ``pdf_type='histogram'``
    pdf_type   : ``'linear'`` (piecewise-linear) or ``'histogram'`` (piecewise-constant)
    """

    def __init__(self, z_grid: np.ndarray, pdf_values: np.ndarray,
                 pdf_type: str = "linear"):
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.pdf_values = np.asarray(pdf_values, dtype=np.float64)
        self.pdf_type = pdf_type
        self._analytic = None  # set by from_truncnorm / from_gaussian
        self._normalize()

    # ── internal ──────────────────────────────────────────────

    def _raw_area(self):
        if self.pdf_type == "linear":
            return float(_trapz(self.pdf_values, self.z_grid))
        return float(np.sum(self.pdf_values * np.diff(self.z_grid)))

    def _normalize(self):
        a = self._raw_area()
        if a > 1e-300:
            self.pdf_values = self.pdf_values / a

    # ── public API ────────────────────────────────────────────

    def evaluate(self, z):
        """Evaluate the PDF at arbitrary *z* values.  Returns 0 outside support."""
        z = np.asarray(z, dtype=np.float64)
        scalar_in = z.ndim == 0
        z = np.atleast_1d(z)
        if self.pdf_type == "linear":
            out = np.interp(z, self.z_grid, self.pdf_values, left=0.0, right=0.0)
        else:
            idx = np.clip(
                np.searchsorted(self.z_grid, z, side="right") - 1,
                0, len(self.pdf_values) - 1,
            )
            out = np.where(
                (z >= self.z_grid[0]) & (z <= self.z_grid[-1]),
                self.pdf_values[idx], 0.0,
            )
        return float(out.item()) if scalar_in else out

    @property
    def support(self):
        """(lower, upper) bounds of the support."""
        return (float(self.z_grid[0]), float(self.z_grid[-1]))

    # ── analytic derivatives (for Laplace / Newton) ───────────

    @property
    def has_analytic_deriv(self):
        """True when closed-form log-PDF derivatives are available."""
        return self._analytic is not None

    def log_pdf(self, x: float) -> float:
        """Evaluate log f(x). Fast analytic path when available."""
        if self._analytic is not None:
            p = self._analytic
            mu, sigma = p['mu'], p['sigma']
            a, b = p['a'], p['b']
            if x < a or x > b:
                return -np.inf
            z = (x - mu) / sigma
            return -0.5 * z * z - p['_log_norm']
        fi = self.evaluate(x)
        if isinstance(fi, np.ndarray):
            fi = float(fi.item())
        return np.log(fi) if fi > 0.0 else -np.inf

    def d_log_pdf(self, x: float) -> float:
        """First derivative of log f(x). Requires analytic params."""
        p = self._analytic
        return -(x - p['mu']) / (p['sigma'] * p['sigma'])

    def d2_log_pdf(self) -> float:
        """Second derivative of log f(x) (constant for Gaussian kernel)."""
        sigma = self._analytic['sigma']
        return -1.0 / (sigma * sigma)

    def shifted(self, offset: float) -> "SoftPDF":
        """Return a copy with z-grid shifted by *offset* (z → z − offset)."""
        new = SoftPDF(self.z_grid - offset, self.pdf_values.copy(), self.pdf_type)
        if self._analytic is not None:
            ap = self._analytic.copy()
            ap['mu'] -= offset
            ap['a'] -= offset
            ap['b'] -= offset
            # _log_norm stays the same (normalization constant is shift-invariant)
            new._analytic = ap
        return new

    def moments(self):
        """Numerically compute (mean, variance)."""
        zf = np.linspace(self.z_grid[0], self.z_grid[-1], 500)
        pf = self.evaluate(zf)
        mu = float(_trapz(zf * pf, zf))
        var = max(float(_trapz((zf - mu) ** 2 * pf, zf)), 1e-16)
        return mu, var

    # ── convenience constructors ──────────────────────────────

    @classmethod
    def from_gaussian(cls, mean: float, var: float,
                      n_pts: int = 25, n_sig: float = 5):
        """Discretised Gaussian N(mean, var).  ≈ MATLAB ``probaGaussian``."""
        sig = np.sqrt(var)
        a = mean - n_sig * sig
        b = mean + n_sig * sig
        z = np.linspace(a, b, n_pts)
        obj = cls(z, norm.pdf(z, mean, sig), "linear")
        import math as _m
        obj._analytic = {
            'mu': float(mean), 'sigma': float(sig),
            'a': float(a), 'b': float(b),
            '_log_norm': 0.5 * _m.log(2.0 * _m.pi) + _m.log(float(sig)),
        }
        return obj

    @classmethod
    def from_uniform(cls, a: float, b: float):
        """Uniform on [a, b].  ≈ MATLAB ``probaUniform``."""
        eps = max((b - a) * 1e-6, 1e-12)
        d = 1.0 / (b - a)
        return cls(np.array([a - eps, a, b, b + eps]),
                   np.array([0.0, d, d, 0.0]), "linear")

    @classmethod
    def from_interval(cls, a: float, b: float):
        """Interval-only soft datum [a, b].  Equivalent to Uniform(a, b).

        Matches MATLAB ``BMEinterval*`` approach.
        """
        return cls.from_uniform(a, b)

    @classmethod
    def from_triangular(cls, a: float, mode: float, b: float):
        """Triangular distribution on [a, b] with peak at *mode*.

        ≈ MATLAB ``probaTriangular``.
        """
        peak = 2.0 / (b - a)
        eps = max((b - a) * 1e-6, 1e-12)
        return cls(
            np.array([a - eps, a, mode, b, b + eps]),
            np.array([0.0, 0.0, peak, 0.0, 0.0]),
            "linear",
        )

    @classmethod
    def from_truncnorm(cls, mu: float, sigma: float,
                       a=None, b=None, n_pts: int = 25):
        """Truncated Gaussian N(mu, σ²) on [a, b].  Ideal for censored data.

        Use ``a=None`` for left-unbounded (−∞)  and ``b=None`` for right-unbounded.
        """
        if a is None:
            a = mu - 6 * sigma
        if b is None:
            b = mu + 6 * sigma
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        z = np.linspace(a, b, n_pts)
        obj = cls(z, _truncnorm_dist.pdf(z, alpha, beta, loc=mu, scale=sigma), "linear")
        import math as _m
        # Pre-compute log normalisation constant (shift-invariant)
        log_Z = _m.log(max(norm.cdf(beta) - norm.cdf(alpha), 1e-300))
        obj._analytic = {
            'mu': float(mu), 'sigma': float(sigma),
            'a': float(a), 'b': float(b),
            '_log_norm': 0.5 * _m.log(2.0 * _m.pi) + _m.log(float(sigma)) + log_Z,
        }
        return obj

    @classmethod
    def from_lognormal(cls, mu_log: float, sigma_log: float,
                       n_pts: int = 30, n_sig: float = 5):
        """Lognormal distribution (underlying normal parameters).

        Good for strictly-positive variables like concentrations.
        """
        zmax = np.exp(mu_log + n_sig * sigma_log)
        zmin = max(1e-10, np.exp(mu_log - n_sig * sigma_log))
        z = np.linspace(zmin, zmax, n_pts)
        return cls(z, _lognorm_dist.pdf(z, s=sigma_log, scale=np.exp(mu_log)), "linear")

    @classmethod
    def from_histogram(cls, breaks: np.ndarray, densities: np.ndarray):
        """Piecewise-constant (histogram) PDF.  MATLAB ``softpdftype=1``."""
        return cls(np.asarray(breaks), np.asarray(densities), "histogram")

    @classmethod
    def from_linear(cls, z_grid: np.ndarray, pdf_values: np.ndarray):
        """Piecewise-linear PDF.  MATLAB ``softpdftype=2``."""
        return cls(np.asarray(z_grid), np.asarray(pdf_values), "linear")

    @classmethod
    def from_callable(cls, func: Callable, a: float, b: float, n_pts: int = 50):
        """Construct from an arbitrary Python callable ``func(z) → density``."""
        z = np.linspace(a, b, n_pts)
        return cls(z, np.maximum([func(zi) for zi in z], 0.0), "linear")

    @classmethod
    def from_mixture(cls, components: "List[SoftPDF]", weights,
                     a=None, b=None, n_pts: int = 60):
        """Mixture of several :class:`SoftPDF` objects."""
        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()
        if a is None:
            a = min(c.support[0] for c in components)
        if b is None:
            b = max(c.support[1] for c in components)
        z = np.linspace(a, b, n_pts)
        pdf = sum(wi * ci.evaluate(z) for wi, ci in zip(w, components))
        return cls(z, pdf, "linear")
