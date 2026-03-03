"""Tests for advanced integration methods: EP, QMC, and LIS.

Validates:
  1. QMC agrees with GH for Gaussian soft PDFs
  2. EP agrees with GH for Gaussian soft PDFs
  3. LIS agrees with GH for Gaussian soft PDFs
  4. Batch consistency (batch == individual calls)
  5. Multi-dimensional tests (2-D)
  6. Non-Gaussian soft PDFs (uniform, truncated-normal)
  7. Predict dispatch wires all three methods correctly
  8. Edge cases (zero soft dims, etc.)
"""

import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import SoftPDF
from pybme.integration import (
    integrate_soft_product,
    integrate_soft_qmc, integrate_soft_qmc_batch,
    integrate_soft_ep, integrate_soft_ep_batch,
    integrate_soft_lis, integrate_soft_lis_batch,
)


# ════════════════════════════════════════════════════════════════
#  QUASI-MONTE CARLO  (QMC)
# ════════════════════════════════════════════════════════════════

class TestQMCSingleGaussian:
    def test_matches_analytic(self):
        """E[f(x)] where x ~ N(μ, σ²) and f = N(μ, σf²).
           Exact: 1 / sqrt(2π(σ² + σf²))
        """
        mu, var, var_f = 3.0, 2.0, 2.0
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=60)
        result = integrate_soft_qmc([sp], np.array([mu]),
                                    np.array([[var]]), n_samples=8192)
        expected = 1.0 / math.sqrt(2 * math.pi * (var + var_f))
        assert_allclose(result, expected, rtol=0.15)

    def test_matches_gh(self):
        mu, var, var_f = 1.0, 3.0, 1.5
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=60)
        mu_arr = np.array([mu])
        cov = np.array([[var]])
        gh = integrate_soft_product([sp], mu_arr, cov, n_quad=25)
        qmc = integrate_soft_qmc([sp], mu_arr, cov, n_samples=8192)
        assert_allclose(qmc, gh, rtol=0.15)


class TestQMCTwoDim:
    def test_product_factorises(self):
        sp1 = SoftPDF.from_gaussian(0.0, 1.0, n_pts=40)
        sp2 = SoftPDF.from_gaussian(0.0, 1.5, n_pts=40)
        cov = np.diag([1.0, 1.0])
        mu = np.array([0.0, 0.0])
        qmc = integrate_soft_qmc([sp1, sp2], mu, cov, n_samples=8192)
        expected = (1.0 / math.sqrt(2 * math.pi * 2.0)) * \
                   (1.0 / math.sqrt(2 * math.pi * 2.5))
        assert_allclose(qmc, expected, rtol=0.20)


class TestQMCBatch:
    def test_batch_matches_individual(self):
        sp1 = SoftPDF.from_gaussian(0, 1.0, n_pts=40)
        sp2 = SoftPDF.from_gaussian(0, 2.0, n_pts=40)
        cov = np.array([[1.5, 0.3], [0.3, 2.0]])
        M = 5
        mu_grid = np.random.RandomState(42).randn(M, 2)
        batch = integrate_soft_qmc_batch([sp1, sp2], mu_grid, cov,
                                         n_samples=8192)
        individual = np.array([
            integrate_soft_qmc([sp1, sp2], mu_grid[i], cov, n_samples=8192)
            for i in range(M)
        ])
        # QMC is randomised (scrambled), so relax tolerance
        assert_allclose(batch, individual, rtol=0.25)

    def test_batch_shape(self):
        sp = SoftPDF.from_gaussian(0, 1.0, n_pts=30)
        cov = np.array([[2.0]])
        mu_grid = np.linspace(-3, 3, 20).reshape(-1, 1)
        result = integrate_soft_qmc_batch([sp], mu_grid, cov)
        assert result.shape == (20,)
        assert np.all(result > 0)


class TestQMCEdgeCases:
    def test_zero_soft_returns_one(self):
        assert integrate_soft_qmc([], np.array([]),
                                  np.array([[]]).reshape(0, 0)) == 1.0

    def test_positive_result(self):
        sp = SoftPDF.from_uniform(-5, 5)
        r = integrate_soft_qmc([sp], np.array([0.0]), np.array([[1.0]]))
        assert r > 0


# ════════════════════════════════════════════════════════════════
#  EXPECTATION PROPAGATION  (EP)
# ════════════════════════════════════════════════════════════════

class TestEPSingleGaussian:
    def test_matches_analytic(self):
        mu, var, var_f = 3.0, 2.0, 2.0
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=80)
        result = integrate_soft_ep([sp], np.array([mu]), np.array([[var]]))
        expected = 1.0 / math.sqrt(2 * math.pi * (var + var_f))
        assert_allclose(result, expected, rtol=0.20)

    def test_matches_gh(self):
        mu, var, var_f = 1.0, 3.0, 1.5
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=80)
        mu_arr = np.array([mu])
        cov = np.array([[var]])
        gh = integrate_soft_product([sp], mu_arr, cov, n_quad=25)
        ep = integrate_soft_ep([sp], mu_arr, cov)
        assert_allclose(ep, gh, rtol=0.25)


class TestEPTwoDim:
    def test_product_factorises(self):
        sp1 = SoftPDF.from_gaussian(0.0, 1.0, n_pts=60)
        sp2 = SoftPDF.from_gaussian(0.0, 1.5, n_pts=60)
        cov = np.diag([1.0, 1.0])
        mu = np.array([0.0, 0.0])
        ep = integrate_soft_ep([sp1, sp2], mu, cov)
        expected = (1.0 / math.sqrt(2 * math.pi * 2.0)) * \
                   (1.0 / math.sqrt(2 * math.pi * 2.5))
        assert_allclose(ep, expected, rtol=0.30)


class TestEPNonGaussian:
    def test_uniform_soft_runs(self):
        """EP should handle uniform soft PDFs without crashing.
        Note: EP approximation quality degrades for hard-edged PDFs
        like uniforms — we only check it produces a positive result.
        """
        sp = SoftPDF.from_uniform(-2, 2)
        mu = np.array([0.0])
        cov = np.array([[1.0]])
        result = integrate_soft_ep([sp], mu, cov)
        assert result > 0

    def test_truncnorm_soft(self):
        sp = SoftPDF.from_truncnorm(mu=5.0, sigma=2.0, a=0.0, b=20.0,
                                     n_pts=80)
        mu = np.array([5.0])
        cov = np.array([[3.0]])
        ep = integrate_soft_ep([sp], mu, cov)
        gh = integrate_soft_product([sp], mu, cov, n_quad=25)
        assert_allclose(ep, gh, rtol=0.30)


class TestEPBatch:
    def test_batch_matches_individual(self):
        sp1 = SoftPDF.from_gaussian(0, 1.0, n_pts=40)
        sp2 = SoftPDF.from_gaussian(0, 2.0, n_pts=40)
        cov = np.array([[1.5, 0.3], [0.3, 2.0]])
        M = 5
        mu_grid = np.random.RandomState(42).randn(M, 2)
        batch = integrate_soft_ep_batch([sp1, sp2], mu_grid, cov)
        individual = np.array([
            integrate_soft_ep([sp1, sp2], mu_grid[i], cov)
            for i in range(M)
        ])
        assert_allclose(batch, individual, rtol=1e-6)

    def test_batch_shape(self):
        sp = SoftPDF.from_gaussian(0, 1.0, n_pts=30)
        cov = np.array([[2.0]])
        mu_grid = np.linspace(-3, 3, 20).reshape(-1, 1)
        result = integrate_soft_ep_batch([sp], mu_grid, cov)
        assert result.shape == (20,)
        assert np.all(result > 0)


class TestEPEdgeCases:
    def test_zero_soft_returns_one(self):
        assert integrate_soft_ep([], np.array([]),
                                 np.array([[]]).reshape(0, 0)) == 1.0


# ════════════════════════════════════════════════════════════════
#  LAPLACE IMPORTANCE SAMPLING  (LIS)
# ════════════════════════════════════════════════════════════════

class TestLISSingleGaussian:
    def test_matches_analytic(self):
        mu, var, var_f = 3.0, 2.0, 2.0
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=60)
        result = integrate_soft_lis([sp], np.array([mu]),
                                    np.array([[var]]), n_samples=8192)
        expected = 1.0 / math.sqrt(2 * math.pi * (var + var_f))
        assert_allclose(result, expected, rtol=0.15)

    def test_matches_gh(self):
        mu, var, var_f = 1.0, 3.0, 1.5
        sp = SoftPDF.from_gaussian(mu, var_f, n_pts=60)
        mu_arr = np.array([mu])
        cov = np.array([[var]])
        gh = integrate_soft_product([sp], mu_arr, cov, n_quad=25)
        lis = integrate_soft_lis([sp], mu_arr, cov, n_samples=8192)
        assert_allclose(lis, gh, rtol=0.15)


class TestLISTwoDim:
    def test_product_factorises(self):
        sp1 = SoftPDF.from_gaussian(0.0, 1.0, n_pts=40)
        sp2 = SoftPDF.from_gaussian(0.0, 1.5, n_pts=40)
        cov = np.diag([1.0, 1.0])
        mu = np.array([0.0, 0.0])
        lis = integrate_soft_lis([sp1, sp2], mu, cov, n_samples=8192)
        expected = (1.0 / math.sqrt(2 * math.pi * 2.0)) * \
                   (1.0 / math.sqrt(2 * math.pi * 2.5))
        assert_allclose(lis, expected, rtol=0.25)


class TestLISNonGaussian:
    def test_uniform_soft(self):
        sp = SoftPDF.from_uniform(-2, 2)
        mu = np.array([0.0])
        cov = np.array([[1.0]])
        result = integrate_soft_lis([sp], mu, cov, n_samples=8192)
        gh = integrate_soft_product([sp], mu, cov, n_quad=25)
        assert_allclose(result, gh, rtol=0.25)


class TestLISBatch:
    def test_batch_matches_individual(self):
        sp1 = SoftPDF.from_gaussian(0, 1.0, n_pts=40)
        sp2 = SoftPDF.from_gaussian(0, 2.0, n_pts=40)
        cov = np.array([[1.5, 0.3], [0.3, 2.0]])
        M = 5
        rng = np.random.RandomState(42)
        mu_grid = rng.randn(M, 2)
        batch = integrate_soft_lis_batch([sp1, sp2], mu_grid, cov,
                                         n_samples=8192)
        individual = np.array([
            integrate_soft_lis([sp1, sp2], mu_grid[i], cov, n_samples=8192)
            for i in range(M)
        ])
        # LIS is stochastic, so allow generous tolerance
        assert_allclose(batch, individual, rtol=0.30)

    def test_batch_shape(self):
        sp = SoftPDF.from_gaussian(0, 1.0, n_pts=30)
        cov = np.array([[2.0]])
        mu_grid = np.linspace(-3, 3, 20).reshape(-1, 1)
        result = integrate_soft_lis_batch([sp], mu_grid, cov)
        assert result.shape == (20,)
        assert np.all(result > 0)


class TestLISEdgeCases:
    def test_zero_soft_returns_one(self):
        assert integrate_soft_lis([], np.array([]),
                                  np.array([[]]).reshape(0, 0)) == 1.0

    def test_positive_result(self):
        sp = SoftPDF.from_uniform(-5, 5)
        r = integrate_soft_lis([sp], np.array([0.0]), np.array([[1.0]]))
        assert r > 0


# ════════════════════════════════════════════════════════════════
#  PREDICT METHOD DISPATCH
# ════════════════════════════════════════════════════════════════

class TestPredictDispatchAdvanced:
    """All three new methods should work through bme_predict."""

    @pytest.fixture
    def data(self):
        rng = np.random.RandomState(789)
        ch = rng.rand(10, 2) * 10
        zh = rng.randn(10) * 2
        cs = rng.rand(3, 2) * 10
        soft_pdfs = [SoftPDF.from_gaussian(rng.randn(), 2.0, n_pts=50)
                     for _ in range(3)]
        ck = np.array([[5.0, 5.0]])
        return ck, ch, zh, cs, soft_pdfs

    @pytest.mark.parametrize("method", ["ep", "qmc", "lis"])
    def test_method_runs(self, method, data):
        from pybme import bme_predict
        ck, ch, zh, cs, soft_pdfs = data
        results = bme_predict(
            ck, ch, zh, cs=cs, soft_pdfs=soft_pdfs,
            model="exponential", params=[1.0, 3.0],
            method=method
        )
        assert len(results) == 1
        assert np.isfinite(results[0].mean)

    @pytest.mark.parametrize("method", ["ep", "qmc", "lis"])
    def test_comparable_to_gh(self, method, data):
        from pybme import bme_predict
        ck, ch, zh, cs, soft_pdfs = data
        r_gh = bme_predict(
            ck, ch, zh, cs=cs, soft_pdfs=soft_pdfs,
            model="exponential", params=[1.0, 3.0],
            method="gauss_hermite"
        )[0]
        r_new = bme_predict(
            ck, ch, zh, cs=cs, soft_pdfs=soft_pdfs,
            model="exponential", params=[1.0, 3.0],
            method=method
        )[0]
        # Means should be in the same ballpark
        assert abs(r_gh.mean - r_new.mean) < max(1.0, 0.5 * abs(r_gh.mean))
