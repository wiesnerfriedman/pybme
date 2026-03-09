"""Tests for covariance models — roughly matching MATLAB MODELSLIBtest.m.

Validates:
  - Each covariance model is positive, monotone decreasing (where expected),
    and returns ``sill`` at distance 0.
  - Nested (additive) models work correctly.
  - Covariance matrices are symmetric positive-semidefinite.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import (
    coord2dist,
    exponential_cov, gaussian_cov, spherical_cov,
    matern_cov, nugget_cov, hole_cos_cov,
    eval_cov, build_cov_matrix,
)


# ── 1. Model evaluation at h = 0 returns sill ────────────────

_MODELS = [
    ("exponential", [2.5, 10.0]),
    ("gaussian",    [3.0, 8.0]),
    ("spherical",   [1.5, 5.0]),
    ("matern",      [2.0, 6.0, 1.5]),
    ("nugget",      [4.0]),
    ("hole_cos",    [1.0, 12.0]),
]


@pytest.mark.parametrize("name,params", _MODELS, ids=[m[0] for m in _MODELS])
def test_sill_at_zero(name, params):
    """C(0) == sill for all standard models."""
    sill = params[0]
    assert_allclose(eval_cov(0.0, name, params), sill, rtol=1e-12)


# ── 2. Positivity & monotonicity ─────────────────────────────

@pytest.mark.parametrize("name,params", _MODELS[:4], ids=[m[0] for m in _MODELS[:4]])
def test_positive_and_decreasing(name, params):
    """C(h) > 0 and non-increasing for monotone models."""
    h = np.linspace(0, 50, 200)
    c = eval_cov(h, name, params)
    assert np.all(c >= -1e-15), f"{name} returned negative values"
    diffs = np.diff(c)
    assert np.all(diffs <= 1e-10), f"{name} is not non-increasing"


# ── 3. Exponential model matches known formula ───────────────

def test_exponential_formula():
    """Verify exponential_cov against sill * exp(-3 h / range)."""
    h = np.array([0, 1, 5, 10, 20, 50])
    sill, rng = 2.0, 15.0
    expected = sill * np.exp(-3.0 * h / rng)
    assert_allclose(exponential_cov(h, [sill, rng]), expected, rtol=1e-14)


def test_gaussian_formula():
    h = np.array([0, 1, 5, 10])
    sill, rng = 1.5, 8.0
    expected = sill * np.exp(-3.0 * (h / rng) ** 2)
    assert_allclose(gaussian_cov(h, [sill, rng]), expected, rtol=1e-14)


def test_spherical_at_range():
    """Spherical model should reach 0 exactly at the range distance."""
    sill, rng = 1.0, 10.0
    assert_allclose(spherical_cov(rng, [sill, rng]), 0.0, atol=1e-14)


def test_nugget_off_diagonal():
    """Nugget is non-zero only when h ≈ 0."""
    assert nugget_cov(0.0, [3.0]) == 3.0
    assert nugget_cov(0.1, [3.0]) == 0.0


# ── 4. Nested (additive) model ───────────────────────────────

def test_nested_model():
    """Nested nugget + exponential sums correctly."""
    h = np.array([0.0, 1.0, 5.0])
    c = eval_cov(h, ["nugget", "exponential"], [[0.5], [1.5, 10.0]])
    expected = nugget_cov(h, [0.5]) + exponential_cov(h, [1.5, 10.0])
    assert_allclose(c, expected, rtol=1e-14)


# ── 5. Covariance matrix is SPD ──────────────────────────────

def test_cov_matrix_spd():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 20, (15, 2))
    K = build_cov_matrix(coords, coords, "exponential", [1.0, 5.0])

    # Symmetric
    assert_allclose(K, K.T, atol=1e-14)
    # Positive-semidefinite
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"


# ── 5b. SPD for all models with various point configurations ─

_SPD_CONFIGS = [
    # (model_name, params, n_points, n_dims, seed)
    ("exponential", [1.0, 5.0],  20, 1, 10),
    ("exponential", [2.0, 3.0],  40, 2, 11),
    ("exponential", [1.5, 8.0],  30, 3, 12),
    ("gaussian",    [2.0, 5.0],  20, 1, 13),
    ("gaussian",    [3.0, 8.0],  25, 2, 14),
    ("spherical",   [1.0, 10.0], 20, 1, 15),
    ("spherical",   [2.5, 6.0],  25, 2, 16),
    ("matern",      [1.0, 5.0, 0.5], 20, 2, 17),
    ("matern",      [2.0, 8.0, 1.5], 25, 2, 18),
    ("matern",      [1.5, 4.0, 2.5], 30, 2, 19),
    ("nugget",      [3.0],       15, 2, 20),
    # NOTE: hole_cos is intentionally excluded — the cosine / hole-effect
    # model is not guaranteed positive-definite in 2-D or 3-D.
]


@pytest.mark.parametrize(
    "name,params,n,ndim,seed", _SPD_CONFIGS,
    ids=[f"{c[0]}_{c[2]}pts_{c[3]}d" for c in _SPD_CONFIGS],
)
def test_cov_matrix_spd_all_models(name, params, n, ndim, seed):
    """Covariance matrix must be SPD for every standard model."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 20, (n, ndim))
    K = build_cov_matrix(coords, coords, name, params)
    # Symmetric
    assert_allclose(K, K.T, atol=1e-12)
    # Positive semi-definite: all eigenvalues ≥ 0
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), \
        f"{name}: min eigenvalue = {eigvals.min():.2e} (n={n}, ndim={ndim})"


def test_cov_matrix_spd_nested():
    """Nested (additive) covariance matrix must be SPD."""
    rng = np.random.default_rng(99)
    coords = rng.uniform(0, 15, (30, 2))
    K = build_cov_matrix(
        coords, coords,
        ["nugget", "exponential", "gaussian"],
        [[0.5], [1.0, 8.0], [0.8, 5.0]],
    )
    assert_allclose(K, K.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), \
        f"Nested model: min eigenvalue = {eigvals.min():.2e}"


def test_cov_matrix_spd_large():
    """SPD check with a larger matrix (n=100)."""
    rng = np.random.default_rng(77)
    coords = rng.uniform(0, 50, (100, 2))
    K = build_cov_matrix(coords, coords, "exponential", [1.0, 10.0])
    assert_allclose(K, K.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-10), \
        f"Large matrix: min eigenvalue = {eigvals.min():.2e}"
    # Condition number should not be astronomical
    cond = eigvals.max() / max(eigvals.min(), 1e-15)
    assert cond < 1e14, f"Condition number too large: {cond:.2e}"


# ── 6. Distance matrix ───────────────────────────────────────

def test_coord2dist_basic():
    c1 = np.array([[0, 0], [3, 4]])
    c2 = np.array([[0, 0], [1, 0]])
    D = coord2dist(c1, c2)
    assert_allclose(D[0, 0], 0.0)
    assert_allclose(D[0, 1], 1.0)
    assert_allclose(D[1, 0], 5.0)             # 3-4-5 triangle
    assert_allclose(D[1, 1], np.sqrt(4 + 16))


def test_coord2dist_self_symmetric():
    rng = np.random.default_rng(7)
    c = rng.uniform(size=(10, 3))
    D = coord2dist(c, c)
    assert_allclose(D, D.T, atol=1e-14)
    assert_allclose(np.diag(D), 0.0, atol=1e-14)
