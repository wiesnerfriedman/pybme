"""Tests for Laplace approximation integration.

Validates:
  1. Laplace matches exact GH for Gaussian soft PDFs (known analytic result)
  2. Laplace handles multiple soft dimensions
  3. Laplace vs GH agreement for near-Gaussian soft PDFs
  4. Batch Laplace consistency
  5. Integration method dispatch in predict
"""

import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import SoftPDF
from pybme.integration import (
    integrate_soft_product,
    integrate_soft_laplace,
    integrate_soft_laplace_batch,
)


# ── §1 Single Gaussian soft — analytic check ────────────────

class TestLaplaceSingleGaussian:
    def test_matches_analytic(self):
        """E[f(x)] where x ~ N(μ, σ²) and f = N(μ, σf²).
           Exact: 1 / sqrt(2π(σ² + σf²))
        """
        mu, var, var_f = 3.0, 2.0, 2.0
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=60)
        result = integrate_soft_laplace([sp], np.array([mu]), np.array([[var]]))
        expected = 1.0 / math.sqrt(2 * math.pi * (var + var_f))
        assert_allclose(result, expected, rtol=0.15)

    def test_matches_gh(self):
        """Laplace should match GH for a Gaussian soft PDF."""
        mu, var, var_f = 1.0, 3.0, 1.5
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=60)
        mu_arr = np.array([mu])
        cov = np.array([[var]])
        gh = integrate_soft_product([sp], mu_arr, cov, n_quad=25)
        lap = integrate_soft_laplace([sp], mu_arr, cov)
        assert_allclose(lap, gh, rtol=0.20)


# ── §2 Two independent Gaussians ────────────────────────────

class TestLaplaceTwoGaussians:
    def test_product_factorises(self):
        """For independent soft PDFs the integral = product of marginals."""
        mu1, mu2 = 0.0, 0.0
        v1, v2 = 1.0, 1.5
        sp1 = SoftPDF.from_gaussian(mu1, v1, n_pts=40)
        sp2 = SoftPDF.from_gaussian(mu2, v2, n_pts=40)
        cov = np.diag([1.0, 1.0])
        mu = np.array([mu1, mu2])

        lap = integrate_soft_laplace([sp1, sp2], mu, cov)
        expected = (1.0 / math.sqrt(2 * math.pi * (1 + v1))) * \
                   (1.0 / math.sqrt(2 * math.pi * (1 + v2)))
        assert_allclose(lap, expected, rtol=0.20)

    def test_matches_gh_2d(self):
        sp1 = SoftPDF.from_gaussian(1.0, 2.0, n_pts=40)
        sp2 = SoftPDF.from_gaussian(-1.0, 1.0, n_pts=40)
        cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        mu = np.array([1.0, -1.0])

        gh = integrate_soft_product([sp1, sp2], mu, cov, n_quad=15)
        lap = integrate_soft_laplace([sp1, sp2], mu, cov)
        assert_allclose(lap, gh, rtol=0.25)


# ── §3 Near-Gaussian soft PDFs ──────────────────────────────

class TestLaplaceNearGaussian:
    def test_truncnorm_close_to_gh(self):
        """Truncated-normal is close to Gaussian → Laplace should work well."""
        sp = SoftPDF.from_truncnorm(mu=5.0, sigma=2.0, a=0.0, b=20.0,
                                     n_pts=60)
        mu = np.array([5.0])
        cov = np.array([[3.0]])
        gh = integrate_soft_product([sp], mu, cov, n_quad=25)
        lap = integrate_soft_laplace([sp], mu, cov)
        assert_allclose(lap, gh, rtol=0.25)


# ── §4 Batch Laplace ────────────────────────────────────────

class TestLaplaceBatch:
    def test_batch_matches_individual(self):
        """integrate_soft_laplace_batch should match individual calls."""
        sp1 = SoftPDF.from_gaussian(0, 1.0, n_pts=40)
        sp2 = SoftPDF.from_gaussian(0, 2.0, n_pts=40)
        cov = np.array([[1.5, 0.3], [0.3, 2.0]])

        M = 5
        mu_grid = np.random.RandomState(42).randn(M, 2)
        batch = integrate_soft_laplace_batch([sp1, sp2], mu_grid, cov)

        individual = np.array([
            integrate_soft_laplace([sp1, sp2], mu_grid[i], cov)
            for i in range(M)
        ])
        assert_allclose(batch, individual, rtol=1e-6)

    def test_batch_shape(self):
        sp = SoftPDF.from_gaussian(0, 1.0, n_pts=30)
        cov = np.array([[2.0]])
        mu_grid = np.linspace(-3, 3, 20).reshape(-1, 1)
        result = integrate_soft_laplace_batch([sp], mu_grid, cov)
        assert result.shape == (20,)
        assert np.all(result > 0)


# ── §5 Zero soft dims ───────────────────────────────────────

class TestLaplaceEdgeCases:
    def test_zero_soft_returns_one(self):
        assert integrate_soft_laplace([], np.array([]), np.array([[]]).reshape(0, 0)) == 1.0

    def test_positive_result(self):
        """Result should always be positive."""
        sp = SoftPDF.from_uniform(-5, 5)
        mu = np.array([0.0])  # within soft support
        cov = np.array([[1.0]])
        r = integrate_soft_laplace([sp], mu, cov)
        assert r > 0


# ── §6 Predict dispatch ─────────────────────────────────────

class TestPredictMethodDispatch:
    def test_method_laplace_runs(self):
        """bme_predict with method='laplace' should work."""
        from pybme import bme_predict

        rng = np.random.RandomState(123)
        ch = rng.rand(10, 2) * 10
        zh = rng.randn(10) * 2
        cs = rng.rand(3, 2) * 10
        soft_pdfs = [SoftPDF.from_gaussian(rng.randn(), 2.0) for _ in range(3)]
        ck = np.array([[5.0, 5.0]])

        results = bme_predict(
            ck, ch, zh, cs=cs, soft_pdfs=soft_pdfs,
            model="exponential", params=[1.0, 3.0],
            method="laplace"
        )
        assert len(results) == 1
        assert np.isfinite(results[0].mean)

    def test_method_gh_vs_laplace_comparable(self):
        """GH and Laplace should give comparable results for Gaussian soft."""
        from pybme import bme_predict

        rng = np.random.RandomState(456)
        ch = rng.rand(8, 2) * 10
        zh = rng.randn(8) * 2
        cs = rng.rand(2, 2) * 10
        soft_pdfs = [SoftPDF.from_gaussian(rng.randn(), 2.0, n_pts=50) for _ in range(2)]
        ck = np.array([[5.0, 5.0]])

        r_gh = bme_predict(
            ck, ch, zh, cs=cs, soft_pdfs=soft_pdfs,
            model="exponential", params=[1.0, 3.0],
            method="gauss_hermite"
        )[0]
        r_lap = bme_predict(
            ck, ch, zh, cs=cs, soft_pdfs=soft_pdfs,
            model="exponential", params=[1.0, 3.0],
            method="laplace"
        )[0]

        # Means should be within ~20%
        assert abs(r_gh.mean - r_lap.mean) < max(0.5, 0.3 * abs(r_gh.mean))
