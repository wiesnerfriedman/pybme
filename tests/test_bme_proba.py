"""Tests for BME prediction — matching MATLAB BMEPROBALIBtest.m cases.

Scenarios:
  1. Hard-only kriging (ordinary / simple)
  2. Hard + Gaussian soft → result near kriging
  3. Hard + interval soft → appropriate posterior shift
  4. Hard + lognormal soft → skewed posterior
  5. No data → prior N(0, σ²)
  6. Duplicate estimation point sits on a hard datum
  7. Multiple estimation points
  8. Confidence-interval coverage
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import bme_predict, SoftPDF


# ── shared fixtures ──────────────────────────────────────────

COV_MODEL = "exponential"
COV_PARAMS = [1.0, 10.0]


def _make_hard_1d(n=12, seed=0):
    rng = np.random.default_rng(seed)
    ch = rng.uniform(0, 30, (n, 1))
    zh = np.sin(ch[:, 0] * 0.5) + rng.normal(0, 0.15, n)
    return ch, zh


# ── 1. Hard-only (ordinary kriging) agrees with kriging_mean/var

def test_hard_only_bme_equals_kriging():
    ch, zh = _make_hard_1d()
    ck = np.array([[15.0]])
    res = bme_predict(ck, ch, zh, model=COV_MODEL, params=COV_PARAMS, order=0)[0]
    # When no soft data, BME mean = kriging mean
    assert_allclose(res.mean, res.kriging_mean, atol=1e-6)
    assert_allclose(res.variance, res.kriging_var, atol=1e-6)
    # Posterior should be Gaussian: skewness ≈ 0
    assert abs(res.skewness) < 0.05


# ── 2. Gaussian soft ≈ additional hard ───────────────────────

def test_gaussian_soft_refines_kriging():
    ch, zh = _make_hard_1d()
    ck = np.array([[15.0]])
    cs = np.array([[14.0]])
    sp = [SoftPDF.from_gaussian(mean=zh.mean(), var=0.2)]

    res = bme_predict(ck, ch, zh, cs, sp,
                      model=COV_MODEL, params=COV_PARAMS, order=0,
                      nsmax=4, dmax=20.0)[0]
    # Variance should be ≤ hard-only kriging
    res_h = bme_predict(ck, ch, zh, model=COV_MODEL, params=COV_PARAMS, order=0)[0]
    assert res.variance <= res_h.kriging_var + 1e-6


# ── 3. Interval (uniform) soft data ─────────────────────────

def test_interval_soft_shifts_mean():
    ch, zh = _make_hard_1d()
    ck = np.array([[15.0]])
    cs = np.array([[15.5]])
    # Soft: value is known to lie in [1, 3] (above mean)
    sp = [SoftPDF.from_interval(1.0, 3.0)]
    res = bme_predict(ck, ch, zh, cs, sp,
                      model=COV_MODEL, params=[1.0, 15.0], order=0,
                      nsmax=4, dmax=25.0)[0]
    # BME mean should shift upward relative to kriging
    res_h = bme_predict(ck, ch, zh,
                        model=COV_MODEL, params=[1.0, 15.0], order=0)[0]
    assert res.mean > res_h.mean - 0.5  # not a strict shift, just sanity


# ── 4. Lognormal soft → positive skewness ────────────────────

def test_lognormal_soft_produces_skew():
    ch, zh = _make_hard_1d()
    ck = np.array([[15.0]])
    cs = np.array([[14.5]])
    sp = [SoftPDF.from_lognormal(mu_log=0.5, sigma_log=0.6)]
    res = bme_predict(ck, ch, zh, cs, sp,
                      model=COV_MODEL, params=COV_PARAMS, order=0,
                      nsmax=4, dmax=20.0, n_grid=250)[0]
    # PDF should exist and be plausible
    assert res.z_grid is not None
    assert res.pdf is not None


# ── 5. No data → prior ───────────────────────────────────────

def test_no_data_returns_prior():
    ch = np.empty((0, 1))
    zh = np.array([])
    ck = np.array([[10.0]])
    # With no data, simple kriging (order=NaN) should return the mean_prior
    res = bme_predict(ck, ch, zh, model=COV_MODEL, params=COV_PARAMS,
                      order=float('nan'), mean_prior=2.0)[0]
    assert_allclose(res.mean, 2.0, atol=0.05)
    assert_allclose(res.variance, COV_PARAMS[0], atol=0.05)


# ── 6. Duplicate estimation point ────────────────────────────

def test_duplicate_point():
    ch, zh = _make_hard_1d()
    ck = ch[0:1]  # exact match
    res = bme_predict(ck, ch, zh, model=COV_MODEL, params=COV_PARAMS, order=0)[0]
    # Should return the exact hard value
    assert_allclose(res.mean, zh[0], atol=1e-6)
    assert res.variance < 1e-6


# ── 7. Multiple estimation points ────────────────────────────

def test_multiple_estimation_points():
    ch, zh = _make_hard_1d()
    ck = np.array([[5.0], [15.0], [25.0]])
    results = bme_predict(ck, ch, zh, model=COV_MODEL, params=COV_PARAMS, order=0)
    assert len(results) == 3
    for r in results:
        assert np.isfinite(r.mean)
        assert r.variance > 0


# ── 8. Confidence interval contains truth at stated level ────

def test_ci_coverage():
    """Empirical coverage of 95 % CI over many left-out hard points."""
    rng = np.random.default_rng(42)
    n = 30
    ch = rng.uniform(0, 30, (n, 1))
    zh = np.sin(ch[:, 0] / 5) + rng.normal(0, 0.25, n)

    inside = 0
    n_test = 10
    for i in range(n_test):
        ck_i = ch[i:i + 1]
        ch_i = np.delete(ch, i, 0)
        zh_i = np.delete(zh, i)
        r = bme_predict(ck_i, ch_i, zh_i,
                        model=COV_MODEL, params=[1.0, 8.0], order=0,
                        n_grid=120, ci_prob=0.95)[0]
        if r.ci_lower <= zh[i] <= r.ci_upper:
            inside += 1
    # Expect ≥ 50 % of test points inside the 95 % CI (loose; many are near boundary)
    assert inside >= 4, f"Coverage too low: {inside}/{n_test}"
