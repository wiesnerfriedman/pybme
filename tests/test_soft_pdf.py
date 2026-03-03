"""Tests for soft PDF constructors and integration.

Validates normalisation, moments, and support for every soft-data type.
Corresponds to MATLAB probaGenerationTest / softpdftype checks.
"""

import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import SoftPDF

# NumPy compat
_trapz = getattr(np, "trapezoid", np.trapz)


# ── 1. All constructors produce a unit-area PDF ──────────────

_CONSTRUCTORS = [
    ("gaussian",  lambda: SoftPDF.from_gaussian(3.0, 2.0)),
    ("uniform",   lambda: SoftPDF.from_uniform(1.0, 5.0)),
    ("interval",  lambda: SoftPDF.from_interval(-2.0, 2.0)),
    ("triangular", lambda: SoftPDF.from_triangular(0, 3, 8)),
    ("truncnorm", lambda: SoftPDF.from_truncnorm(0, 1, -2, 2)),
    ("lognormal", lambda: SoftPDF.from_lognormal(0.5, 0.3)),
    ("histogram", lambda: SoftPDF.from_histogram(
        np.array([0, 1, 2, 3]), np.array([1.0, 2.0, 1.0])
    )),
    ("linear",    lambda: SoftPDF.from_linear(
        np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 1, 0])
    )),
]


@pytest.mark.parametrize("name,ctor", _CONSTRUCTORS, ids=[c[0] for c in _CONSTRUCTORS])
def test_unit_area(name, ctor):
    """Every SoftPDF integrates to 1."""
    sp = ctor()
    z = np.linspace(sp.support[0], sp.support[1], 500)
    area = float(_trapz(sp.evaluate(z), z))
    assert_allclose(area, 1.0, atol=0.02, err_msg=f"{name} area = {area}")


# ── 2. Gaussian moments ──────────────────────────────────────

def test_gaussian_moments():
    sp = SoftPDF.from_gaussian(5.0, 4.0, n_pts=50)
    mu, var = sp.moments()
    assert_allclose(mu, 5.0, atol=0.05)
    assert_allclose(var, 4.0, atol=0.2)


# ── 3. Uniform support and mean ──────────────────────────────

def test_uniform_properties():
    sp = SoftPDF.from_uniform(2.0, 6.0)
    mu, var = sp.moments()
    assert_allclose(mu, 4.0, atol=0.1)
    assert_allclose(var, (6.0 - 2.0) ** 2 / 12, atol=0.15)


# ── 4. Lognormal mean matches theory ─────────────────────────

def test_lognormal_mean():
    mu_log, sigma_log = 1.0, 0.4
    sp = SoftPDF.from_lognormal(mu_log, sigma_log, n_pts=60)
    analytic_mean = np.exp(mu_log + 0.5 * sigma_log ** 2)
    mu, _ = sp.moments()
    assert_allclose(mu, analytic_mean, rtol=0.05)


# ── 5. Truncated normal limits ───────────────────────────────

def test_truncnorm_support():
    sp = SoftPDF.from_truncnorm(0, 1, a=-1, b=2)
    lo, hi = sp.support
    assert lo >= -1 - 0.01
    assert hi <= 2 + 0.01
    # density should be zero well outside truncation
    assert sp.evaluate(-3.0) < 1e-10


# ── 6. Mixture of 2 Gaussians has bimodal shape ──────────────

def test_mixture_bimodal():
    g1 = SoftPDF.from_gaussian(-3, 0.5)
    g2 = SoftPDF.from_gaussian(3, 0.5)
    mx = SoftPDF.from_mixture([g1, g2], [0.5, 0.5])
    z = np.linspace(*mx.support, 300)
    p = mx.evaluate(z)
    # two local maxima
    peaks = np.diff(np.sign(np.diff(p)))
    n_peaks = np.sum(peaks < 0)
    assert n_peaks >= 2, f"Expected bimodal, found {n_peaks} peak(s)"


# ── 7. from_callable matches analytical ──────────────────────

def test_from_callable():
    """Construct from lambda and verify normalisation + support."""
    sp = SoftPDF.from_callable(lambda z: np.exp(-z ** 2 / 2), -5, 5, n_pts=80)
    z = np.linspace(-5, 5, 500)
    area = float(_trapz(sp.evaluate(z), z))
    assert_allclose(area, 1.0, atol=0.02)


# ── 8. Triangular peak is at mode ────────────────────────────

def test_triangular_peak():
    sp = SoftPDF.from_triangular(0, 2, 5)
    z = np.linspace(0, 5, 500)
    peak_z = z[np.argmax(sp.evaluate(z))]
    assert_allclose(peak_z, 2.0, atol=0.05)


# ── 9. Histogram type ────────────────────────────────────────

def test_histogram_flat():
    """A uniform histogram should have constant density."""
    sp = SoftPDF.from_histogram(np.array([0, 1, 2, 3]), np.array([1.0, 1.0, 1.0]))
    z_mid = np.array([0.5, 1.5, 2.5])
    vals = sp.evaluate(z_mid)
    # all bins should have the same density
    assert_allclose(vals, vals[0] * np.ones(3), atol=1e-10)
