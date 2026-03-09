"""Tests for numerical integration — matching MATLAB MVNLIBtest.m.

Validates the Gauss-Hermite tensor-product integrator by checking
known analytic expectations.
"""

import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import SoftPDF
from pybme.integration import integrate_soft_product


# ── 1. Trivial: 0 soft dims → 1.0 ───────────────────────────

def test_zero_soft_returns_one():
    assert integrate_soft_product([], np.array([]), np.array([[]]), 15) == 1.0


# ── 2. Single Gaussian soft: known result ────────────────────

def test_single_gaussian_soft():
    """E[f(x)] where x ~ N(mu, σ²) and f = N(mu, σ²) → 1/(2√(π σ²)).
    
    The exact integral of  N(x; μf, σf²) · N(x; μ, σ²)  over x  is
    N(μ; μf, σ² + σf²)  =  1/√(2π(σ²+σf²)) when  μ = μf.
    """
    mu, var, var_f = 3.0, 2.0, 2.0
    sp = SoftPDF.from_gaussian(mu, var_f, n_pts=60)
    result = integrate_soft_product([sp], np.array([mu]), np.array([[var]]))
    expected = 1.0 / math.sqrt(2 * math.pi * (var + var_f))
    assert_allclose(result, expected, rtol=0.05)


# ── 3. Uniform soft → CDF difference ─────────────────────────

def test_uniform_soft_integral():
    """E[U(a,b)(x)] where x ~ N(mu, σ²) — the discretised uniform has edge effects
    so we use a wider interval and more quadrature points."""
    from scipy.stats import norm
    mu, sigma = 0.0, 1.0
    a, b = -2.0, 2.0
    sp = SoftPDF.from_uniform(a, b)
    I = integrate_soft_product([sp], np.array([mu]), np.array([[sigma ** 2]]), n_quad=25)
    # Expected = (1/(b-a)) * [Φ(b/σ) - Φ(a/σ)]
    expected = (1.0 / (b - a)) * (norm.cdf(b / sigma) - norm.cdf(a / sigma))
    assert_allclose(I, expected, rtol=0.15)


# ── 4. Two independent Gaussians → product factorises ────────

def test_two_independent_gaussians():
    """Two independent soft → integral = product of individual integrals."""
    mu1, mu2, v1, v2 = 0.0, 0.0, 1.0, 1.5
    sp1 = SoftPDF.from_gaussian(mu1, v1, n_pts=40)
    sp2 = SoftPDF.from_gaussian(mu2, v2, n_pts=40)
    sigma2 = np.diag([1.0, 1.0])
    mu = np.array([mu1, mu2])
    I = integrate_soft_product([sp1, sp2], mu, sigma2)
    # Each marginal: N(0; 0, σ² + v) = 1/√(2π(1+v))
    expected = (1.0 / math.sqrt(2 * math.pi * (1 + v1))) * \
               (1.0 / math.sqrt(2 * math.pi * (1 + v2)))
    assert_allclose(I, expected, rtol=0.10)


# ── 5. Positive-definiteness guard ───────────────────────────

def test_integration_with_near_singular_cov():
    """Integration should not crash on a near-singular covariance matrix."""
    sp = SoftPDF.from_gaussian(0, 1, n_pts=30)
    mu = np.array([0.0])
    cov = np.array([[1e-14]])  # nearly 0
    I = integrate_soft_product([sp], mu, cov)
    assert np.isfinite(I) and I > 0


# ── 6. SPD covariance in multi-dimensional integration ───────

def test_multivariate_cov_spd_integration():
    """Verify integration works correctly with a verified-SPD 3×3 covariance."""
    cov = np.array([[1.0, 0.5, 0.2],
                    [0.5, 1.0, 0.4],
                    [0.2, 0.4, 1.0]])
    # Confirm SPD before integration
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), "Test covariance should be PD"

    sp1 = SoftPDF.from_gaussian(0, 1, n_pts=30)
    sp2 = SoftPDF.from_gaussian(0, 1, n_pts=30)
    sp3 = SoftPDF.from_gaussian(0, 1, n_pts=30)
    mu = np.zeros(3)
    I = integrate_soft_product([sp1, sp2, sp3], mu, cov)
    assert np.isfinite(I) and I > 0
