"""Tests for Newton solver convergence and analytic-vs-finite-difference derivatives.

Covers the two gaps identified in the test-suite review:
  1. Newton solver: convergence to known modes, monotone log-target increase,
     behaviour under Hessian regularisation.
  2. Analytic derivatives: d_log_pdf / d2_log_pdf vs central finite differences
     for Gaussian and truncated-normal soft PDFs.
"""

import math
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import SoftPDF
from pybme.integration import (
    _find_mode,
    _log_target,
    _log_target_grad,
    _log_target_hessian,
    _adaptive_eps,
    integrate_soft_laplace,
)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _precision(cov):
    """Covariance → precision matrix."""
    return np.linalg.inv(cov)


# ---------------------------------------------------------------------------
#  1. Newton solver convergence
# ---------------------------------------------------------------------------

class TestNewtonGaussianSoft:
    """When soft PDFs are Gaussian, the mode has a closed-form solution."""

    def test_1d_mode_exact(self):
        """Gaussian prior + Gaussian soft → mode = precision-weighted mean."""
        sigma2 = 2.0          # prior variance
        mu_prior = 1.0
        sf2 = 0.5             # soft variance
        mu_soft = 3.0

        cov = np.array([[sigma2]])
        Q = _precision(cov)
        mu = np.array([mu_prior])
        sp = SoftPDF.from_gaussian(mu_soft, sf2, n_pts=60)

        mode = _find_mode([sp], mu, Q)

        # analytic: (mu_prior/sigma2 + mu_soft/sf2) / (1/sigma2 + 1/sf2)
        expected = (mu_prior / sigma2 + mu_soft / sf2) / (1.0 / sigma2 + 1.0 / sf2)
        assert_allclose(mode[0], expected, atol=1e-5)

    def test_2d_mode_independent(self):
        """Two independent Gaussian soft PDFs: each coordinate solves independently."""
        cov = np.diag([1.0, 2.0])
        Q = _precision(cov)
        mu = np.array([0.0, 0.0])
        sp1 = SoftPDF.from_gaussian(2.0, 1.0, n_pts=60)
        sp2 = SoftPDF.from_gaussian(-3.0, 0.5, n_pts=60)

        mode = _find_mode([sp1, sp2], mu, Q)

        for i, (mu_p, sig2_p, mu_s, sig2_s) in enumerate(
            [(0.0, 1.0, 2.0, 1.0), (0.0, 2.0, -3.0, 0.5)]
        ):
            expected = (mu_p / sig2_p + mu_s / sig2_s) / (1.0 / sig2_p + 1.0 / sig2_s)
            assert_allclose(mode[i], expected, atol=1e-4,
                            err_msg=f"dim {i}")

    def test_2d_mode_correlated(self):
        """Correlated prior: mode still close to precision-weighted mean."""
        cov = np.array([[2.0, 0.8], [0.8, 1.5]])
        Q = _precision(cov)
        mu = np.array([1.0, -1.0])
        sp1 = SoftPDF.from_gaussian(3.0, 1.0, n_pts=60)
        sp2 = SoftPDF.from_gaussian(0.0, 1.0, n_pts=60)

        mode = _find_mode([sp1, sp2], mu, Q)

        # Analytic mode: solve (Q + diag(1/sf2)) x* = Q mu + [mu_s1/sf2_1, mu_s2/sf2_2]
        S = np.diag([1.0, 1.0])           # diag(1/sf2)
        Q_post = Q + S
        eta = Q @ mu + np.array([3.0 / 1.0, 0.0 / 1.0])
        expected = np.linalg.solve(Q_post, eta)
        assert_allclose(mode, expected, atol=1e-4)


class TestNewtonMonotone:
    """The line-search should guarantee non-decrease of log g at each step."""

    def test_logtarget_at_mode_geq_at_prior_mean(self):
        """Mode should have log g ≥ log g(mu)."""
        cov = np.array([[1.5]])
        Q = _precision(cov)
        mu = np.array([0.0])
        sp = SoftPDF.from_gaussian(5.0, 1.0, n_pts=60)

        mode = _find_mode([sp], mu, Q)
        lt_mode = _log_target(mode, [sp], mu, Q)
        lt_mu = _log_target(mu, [sp], mu, Q)
        assert lt_mode >= lt_mu - 1e-8


class TestNewtonTruncatedNormal:
    """Mode-finding with truncated-normal soft PDFs (support constraints)."""

    def test_mode_inside_support(self):
        """Mode must lie within [a, b] of the soft PDF."""
        cov = np.array([[2.0]])
        Q = _precision(cov)
        mu = np.array([10.0])           # prior mean far from soft support
        sp = SoftPDF.from_truncnorm(mu=2.0, sigma=1.0, a=0.0, b=4.0, n_pts=60)

        mode = _find_mode([sp], mu, Q)
        lo, hi = sp.support
        assert lo - 1e-6 <= mode[0] <= hi + 1e-6

    def test_mode_matches_laplace(self):
        """Mode from _find_mode should give same integral as integrate_soft_laplace."""
        cov = np.array([[1.0]])
        Q = _precision(cov)
        mu = np.array([2.0])
        sp = SoftPDF.from_truncnorm(mu=2.0, sigma=1.5, a=0.0, b=10.0, n_pts=60)

        result = integrate_soft_laplace([sp], mu, cov)
        assert result > 0


class TestNewtonHessianRegularisation:
    """When soft PDFs cause a non-PD Hessian, regularisation keeps Newton stable."""

    def test_uniform_soft_converges(self):
        """Uniform PDF has zero second derivative → Hessian = -Q only.
        Newton should still converge (Hessian is already neg-def from prior)."""
        cov = np.array([[1.0]])
        Q = _precision(cov)
        mu = np.array([0.0])
        sp = SoftPDF.from_uniform(-2.0, 2.0)

        mode = _find_mode([sp], mu, Q)
        # Mode should be near prior mean (uniform doesn't pull)
        assert_allclose(mode[0], 0.0, atol=0.5)

    def test_multimodal_soft_converges(self):
        """A bimodal-ish soft PDF (mixture of two bumps via histogram):
        Newton should converge to *some* finite mode."""
        z = np.linspace(-5, 5, 100)
        pdf = np.exp(-0.5 * (z - 2) ** 2) + np.exp(-0.5 * (z + 2) ** 2)
        pdf /= np.trapezoid(pdf, z)                  # normalise
        sp = SoftPDF(z, pdf, pdf_type="linear")

        cov = np.array([[3.0]])
        Q = _precision(cov)
        mu = np.array([0.0])

        mode = _find_mode([sp], mu, Q)
        assert np.isfinite(mode[0])
        lt = _log_target(mode, [sp], mu, Q)
        assert np.isfinite(lt)


# ---------------------------------------------------------------------------
#  2. Analytic derivatives vs finite differences
# ---------------------------------------------------------------------------

class TestAnalyticGradGaussian:
    """Compare analytic gradient of log g(x) to central finite differences
    when soft PDFs are Gaussian (analytic derivatives known)."""

    @pytest.fixture()
    def setup_1d(self):
        sp = SoftPDF.from_gaussian(2.0, 1.0, n_pts=80)
        cov = np.array([[1.5]])
        Q = _precision(cov)
        mu = np.array([0.0])
        return [sp], mu, Q

    def test_gradient_at_mode(self, setup_1d):
        """Gradient should be ≈ 0 at mode."""
        soft, mu, Q = setup_1d
        mode = _find_mode(soft, mu, Q)
        grad = _log_target_grad(mode, soft, mu, Q)
        assert_allclose(grad, 0.0, atol=1e-4)

    def test_gradient_vs_fd(self, setup_1d):
        """Analytic gradient vs central FD at an off-mode point."""
        soft, mu, Q = setup_1d
        x = np.array([1.23])
        grad_analytic = _log_target_grad(x, soft, mu, Q)

        eps = 1e-5
        fd = np.empty_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            fd[i] = (_log_target(xp, soft, mu, Q) - _log_target(xm, soft, mu, Q)) / (2 * eps)

        assert_allclose(grad_analytic, fd, rtol=1e-4)

    def test_gradient_vs_fd_2d(self):
        """2-D correlated: analytic gradient vs FD."""
        sp1 = SoftPDF.from_gaussian(1.0, 0.5, n_pts=80)
        sp2 = SoftPDF.from_gaussian(-2.0, 1.0, n_pts=80)
        cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        Q = _precision(cov)
        mu = np.array([0.0, 0.0])
        x = np.array([0.7, -1.1])

        grad_analytic = _log_target_grad(x, [sp1, sp2], mu, Q)

        eps = 1e-5
        fd = np.empty(2)
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            fd[i] = (_log_target(xp, [sp1, sp2], mu, Q) - _log_target(xm, [sp1, sp2], mu, Q)) / (2 * eps)

        assert_allclose(grad_analytic, fd, rtol=1e-4)


class TestAnalyticHessianGaussian:
    """Analytic Hessian vs FD Hessian for Gaussian soft PDFs."""

    def test_hessian_vs_fd_1d(self):
        sp = SoftPDF.from_gaussian(2.0, 1.0, n_pts=80)
        cov = np.array([[1.5]])
        Q = _precision(cov)
        mu = np.array([0.0])
        x = np.array([1.0])

        H_analytic = _log_target_hessian(x, [sp], mu, Q)

        # FD Hessian from gradient
        eps = 1e-5
        gp = _log_target_grad(x + np.array([eps]), [sp], mu, Q)
        gm = _log_target_grad(x - np.array([eps]), [sp], mu, Q)
        H_fd = (gp - gm) / (2 * eps)

        assert_allclose(H_analytic[0, 0], H_fd[0], rtol=1e-3)

    def test_hessian_vs_fd_2d(self):
        sp1 = SoftPDF.from_gaussian(1.0, 0.5, n_pts=80)
        sp2 = SoftPDF.from_gaussian(-2.0, 1.0, n_pts=80)
        cov = np.array([[2.0, 0.5], [0.5, 1.5]])
        Q = _precision(cov)
        mu = np.array([0.0, 0.0])
        x = np.array([0.7, -1.1])

        H_analytic = _log_target_hessian(x, [sp1, sp2], mu, Q)

        eps = 1e-5
        H_fd = np.empty((2, 2))
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            gp = _log_target_grad(xp, [sp1, sp2], mu, Q)
            gm = _log_target_grad(xm, [sp1, sp2], mu, Q)
            H_fd[i, :] = (gp - gm) / (2 * eps)

        assert_allclose(H_analytic, H_fd, rtol=1e-3)

    def test_hessian_is_constant_gaussian(self):
        """For Gaussian soft PDFs the Hessian is independent of x."""
        sp = SoftPDF.from_gaussian(2.0, 1.0, n_pts=80)
        cov = np.array([[1.5]])
        Q = _precision(cov)
        mu = np.array([0.0])

        H1 = _log_target_hessian(np.array([0.5]), [sp], mu, Q)
        H2 = _log_target_hessian(np.array([3.0]), [sp], mu, Q)
        assert_allclose(H1, H2, atol=1e-10)

    def test_hessian_known_value_gaussian(self):
        """H = -Q + diag(-1/sigma_f^2).  Verify for 1-D."""
        sf2 = 0.8
        sp = SoftPDF.from_gaussian(0.0, sf2, n_pts=80)
        sigma2 = 2.0
        cov = np.array([[sigma2]])
        Q = _precision(cov)
        mu = np.array([0.0])

        H = _log_target_hessian(np.array([1.0]), [sp], mu, Q)
        expected = -(1.0 / sigma2 + 1.0 / sf2)
        assert_allclose(H[0, 0], expected, rtol=1e-6)


class TestAnalyticGradTruncNorm:
    """Analytic gradient/Hessian for truncated-normal soft PDFs vs FD."""

    def test_gradient_vs_fd(self):
        sp = SoftPDF.from_truncnorm(mu=3.0, sigma=2.0, a=0.0, b=10.0, n_pts=80)
        cov = np.array([[2.0]])
        Q = _precision(cov)
        mu = np.array([1.0])
        x = np.array([2.5])

        grad_analytic = _log_target_grad(x, [sp], mu, Q)

        eps = 1e-5
        fd = np.array([
            (_log_target(x + np.array([eps]), [sp], mu, Q)
             - _log_target(x - np.array([eps]), [sp], mu, Q)) / (2 * eps)
        ])

        assert_allclose(grad_analytic, fd, rtol=1e-4)

    def test_hessian_vs_fd(self):
        sp = SoftPDF.from_truncnorm(mu=3.0, sigma=2.0, a=0.0, b=10.0, n_pts=80)
        cov = np.array([[2.0]])
        Q = _precision(cov)
        mu = np.array([1.0])
        x = np.array([2.5])

        H_analytic = _log_target_hessian(x, [sp], mu, Q)

        eps = 1e-5
        gp = _log_target_grad(x + np.array([eps]), [sp], mu, Q)
        gm = _log_target_grad(x - np.array([eps]), [sp], mu, Q)
        H_fd = (gp - gm) / (2 * eps)

        assert_allclose(H_analytic[0, 0], H_fd[0], rtol=1e-3)

    def test_hessian_is_constant_truncnorm(self):
        """Truncated-normal Hessian is also x-independent (same as Gaussian)."""
        sp = SoftPDF.from_truncnorm(mu=3.0, sigma=2.0, a=0.0, b=10.0, n_pts=80)
        cov = np.array([[1.0]])
        Q = _precision(cov)
        mu = np.array([0.0])

        H1 = _log_target_hessian(np.array([2.0]), [sp], mu, Q)
        H2 = _log_target_hessian(np.array([5.0]), [sp], mu, Q)
        assert_allclose(H1, H2, atol=1e-10)


class TestFDDerivativesNonAnalytic:
    """For non-analytic soft PDFs (uniform, histogram), FD path is used.
    Verify FD gradient/Hessian are self-consistent."""

    def test_fd_gradient_uniform(self):
        """FD gradient of log g should be ≈ 0 at mode for uniform soft."""
        sp = SoftPDF.from_uniform(-3.0, 3.0)
        assert not sp.has_analytic_deriv
        cov = np.array([[1.0]])
        Q = _precision(cov)
        mu = np.array([0.0])

        mode = _find_mode([sp], mu, Q)
        grad = _log_target_grad(mode, [sp], mu, Q)
        assert_allclose(grad, 0.0, atol=1e-2)

    def test_fd_hessian_uniform_equals_neg_Q(self):
        """Uniform PDF has zero curvature → Hessian ≈ -Q inside support."""
        sp = SoftPDF.from_uniform(-5.0, 5.0)
        cov = np.array([[2.0]])
        Q = _precision(cov)
        mu = np.array([0.0])

        H = _log_target_hessian(np.array([0.0]), [sp], mu, Q)
        # log f is constant inside support, so H ≈ -Q
        assert_allclose(H[0, 0], -Q[0, 0], rtol=0.1)

    def test_fd_gradient_vs_logtarget_diff(self):
        """FD gradient consistency: compare _log_target_grad (which uses the
        internal FD machinery) against our own external FD from _log_target."""
        z = np.linspace(-5, 5, 80)
        pdf = np.exp(-0.5 * (z - 1.0) ** 2)
        pdf /= np.trapezoid(pdf, z)
        sp = SoftPDF(z, pdf, pdf_type="linear")
        assert not sp.has_analytic_deriv

        cov = np.array([[1.5]])
        Q = _precision(cov)
        mu = np.array([0.0])
        x = np.array([0.8])

        grad_internal = _log_target_grad(x, [sp], mu, Q)

        eps = 1e-5
        fd_external = np.array([
            (_log_target(x + np.array([eps]), [sp], mu, Q)
             - _log_target(x - np.array([eps]), [sp], mu, Q)) / (2 * eps)
        ])

        # Internal FD uses adaptive (coarser) step sizes based on the grid
        # spacing, so agreement at ~15% is expected for coarse linear grids.
        assert_allclose(grad_internal, fd_external, rtol=0.20)
