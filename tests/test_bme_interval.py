"""Tests for interval soft data — matching MATLAB BMEINTLIBtest.m.

Validates that interval (uniform) soft data properly constrains
the posterior.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybme import bme_predict, SoftPDF


def _setup():
    """Hard data on a regular 1-D grid with known signal."""
    ch = np.linspace(0, 20, 8).reshape(-1, 1)
    zh = np.sin(ch[:, 0] / 3.0)
    return ch, zh


# ── 1. Interval soft restricts posterior support ─────────────

def test_interval_constrains_posterior():
    ch, zh = _setup()
    ck = np.array([[10.0]])
    cs = np.array([[10.5]])
    lo, hi = 0.5, 1.5
    sp = [SoftPDF.from_interval(lo, hi)]
    res = bme_predict(ck, ch, zh, cs, sp,
                      model="exponential", params=[1.0, 8.0],
                      nsmax=4, dmax=15.0, n_grid=200, order=0)[0]
    # Most of the posterior mass should lie within or near [lo, hi]
    assert res.mean >= lo - 0.5
    assert res.mean <= hi + 0.5


# ── 2. Narrow interval → near-deterministic ──────────────────

def test_narrow_interval():
    ch, zh = _setup()
    ck = np.array([[10.0]])
    cs = np.array([[10.0]])
    val = 0.8
    sp = [SoftPDF.from_interval(val - 0.01, val + 0.01)]
    res = bme_predict(ck, ch, zh, cs, sp,
                      model="exponential", params=[1.0, 8.0],
                      nsmax=4, dmax=15.0, n_grid=200, order=0)[0]
    # With a very narrow interval collocated, result should be close to val
    assert_allclose(res.mean, val, atol=0.3)


# ── 3. Wide interval has little effect ───────────────────────

def test_wide_interval_minimal_effect():
    ch, zh = _setup()
    ck = np.array([[10.0]])
    res_h = bme_predict(ck, ch, zh,
                        model="exponential", params=[1.0, 8.0], order=0)[0]
    cs = np.array([[10.5]])
    sp = [SoftPDF.from_interval(-100, 100)]  # very wide → uninformative
    res = bme_predict(ck, ch, zh, cs, sp,
                      model="exponential", params=[1.0, 8.0],
                      nsmax=4, dmax=15.0, order=0)[0]
    # Means should be very similar since wide interval is uninformative
    assert_allclose(res.mean, res_h.mean, atol=0.3)


# ── 4. Multiple intervals ────────────────────────────────────

def test_multiple_intervals():
    ch, zh = _setup()
    ck = np.array([[10.0]])
    cs = np.array([[9.0], [11.0]])
    sp = [SoftPDF.from_interval(0.0, 1.0), SoftPDF.from_interval(0.0, 1.0)]
    res = bme_predict(ck, ch, zh, cs, sp,
                      model="exponential", params=[1.0, 8.0],
                      nsmax=4, dmax=15.0, n_grid=200, order=0)[0]
    assert res.n_soft == 2
    assert np.isfinite(res.mean)
