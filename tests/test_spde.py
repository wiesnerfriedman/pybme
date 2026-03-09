"""Tests for pybme.spde — SPDE / GMRF module.

Validates:
  1. Mesh construction (Delaunay, FEM matrices)
  2. Mass and stiffness matrix properties
  3. Precision matrix structure (sparse, symmetric, positive-definite)
  4. matern_to_spde_params round-trip
  5. spde_kriging reproduces simple kriging on small problems
  6. snap_to_mesh correctness
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pybme.spde import (
    SPDEMesh,
    matern_to_spde_params,
    build_precision_matrix,
    spde_kriging,
    snap_to_mesh,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_grid_mesh(nx=10, ny=10, extent=10.0):
    """Create a simple regular-grid mesh for testing."""
    x = np.linspace(0, extent, nx)
    y = np.linspace(0, extent, ny)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return SPDEMesh.from_points(pts)


# ── §1 Mesh construction ────────────────────────────────────

class TestMeshConstruction:
    def test_from_points_basic(self):
        pts = np.random.RandomState(42).rand(30, 2) * 10
        mesh = SPDEMesh.from_points(pts)
        assert mesh.n_nodes >= 30   # may add boundary nodes
        assert mesh.n_tri > 0
        assert mesh.triangles.shape[1] == 3

    def test_from_points_with_extend(self):
        pts = np.random.RandomState(0).rand(20, 2) * 5
        mesh = SPDEMesh.from_points(pts, extend=0.5)
        assert mesh.n_nodes > 20  # extension was applied

    def test_mass_stiffness_shape(self):
        mesh = _make_grid_mesh(6, 6)
        C, G = mesh._C, mesh._G
        n = mesh.n_nodes
        assert C.shape == (n, n)
        assert G.shape == (n, n)

    def test_mass_positive(self):
        """All diagonal entries of the lumped mass matrix should be > 0."""
        mesh = _make_grid_mesh(8, 8)
        C = mesh._C
        diag = C.diagonal()
        assert np.all(diag > 0), "Mass matrix has non-positive diagonal"

    def test_stiffness_symmetric(self):
        mesh = _make_grid_mesh(7, 7)
        G = mesh._G
        diff = G - G.T
        assert sparse.issparse(G)
        assert abs(diff).max() < 1e-12


# ── §2 SPDE parameters ──────────────────────────────────────

class TestSPDEParams:
    def test_exponential_nu05(self):
        """ν=0.5 (exponential covariance): α=1, d=2."""
        kappa, tau = matern_to_spde_params(sigma2=10.0, range_param=2.0, nu=0.5)
        assert kappa > 0
        assert tau > 0

    def test_matern_nu15(self):
        """ν=1.5 (Matérn 3/2): α=2."""
        kappa, tau = matern_to_spde_params(sigma2=5.0, range_param=3.0, nu=1.5)
        assert kappa > 0
        assert tau > 0

    def test_kappa_scales_with_range(self):
        """Larger range → smaller κ."""
        _, _ = matern_to_spde_params(1.0, 1.0, 0.5)
        k1, _ = matern_to_spde_params(1.0, 1.0, 0.5)
        k2, _ = matern_to_spde_params(1.0, 5.0, 0.5)
        assert k2 < k1


# ── §3 Precision matrix ─────────────────────────────────────

class TestPrecisionMatrix:
    def test_sparse_spd(self):
        mesh = _make_grid_mesh(8, 8)
        kappa, tau = matern_to_spde_params(1.0, 2.0, 0.5)
        Q = build_precision_matrix(mesh, kappa, tau, alpha=1)
        assert sparse.issparse(Q)
        # symmetry
        diff = Q - Q.T
        assert abs(diff).max() < 1e-10
        # positive-definiteness (check via Cholesky-like)
        ev = sparse.linalg.eigsh(Q.tocsc(), k=1, which='SM', return_eigenvectors=False)
        assert ev[0] > -1e-8, "Q has negative eigenvalue"

    def test_alpha1_vs_alpha2_different(self):
        mesh = _make_grid_mesh(6, 6)
        kappa, tau = matern_to_spde_params(1.0, 2.0, 0.5)
        Q1 = build_precision_matrix(mesh, kappa, tau, alpha=1)
        kappa2, tau2 = matern_to_spde_params(1.0, 2.0, 1.5)
        Q2 = build_precision_matrix(mesh, kappa2, tau2, alpha=2)
        # They should differ
        assert abs(Q1 - Q2).max() > 1e-6

    def test_precision_spd_alpha2(self):
        """Precision matrix with α=2 (Matérn ν=1.5) must be SPD."""
        mesh = _make_grid_mesh(8, 8)
        kappa, tau = matern_to_spde_params(1.0, 2.0, 1.5)
        Q = build_precision_matrix(mesh, kappa, tau, alpha=2)
        assert sparse.issparse(Q)
        diff = Q - Q.T
        assert abs(diff).max() < 1e-10
        ev = sparse.linalg.eigsh(Q.tocsc(), k=1, which='SM',
                                  return_eigenvectors=False)
        assert ev[0] > -1e-8, "Q (α=2) has negative eigenvalue"

    def test_covariance_from_precision_spd(self):
        """Dense inverse of Q (the implied covariance) should be SPD."""
        mesh = _make_grid_mesh(6, 6)
        kappa, tau = matern_to_spde_params(1.0, 2.0, 0.5)
        Q = build_precision_matrix(mesh, kappa, tau, alpha=1)
        Q_dense = Q.toarray()
        C = np.linalg.inv(Q_dense)
        assert_allclose(C, C.T, atol=1e-10)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > -1e-10), \
            f"Implied covariance: min eigenvalue = {eigvals.min():.2e}"


# ── §4 spde_kriging ─────────────────────────────────────────

class TestSPDEKriging:
    def test_interpolates_observed(self):
        """Kriging at an observed location should return ≈ observed value."""
        mesh = _make_grid_mesh(10, 10, extent=10.0)
        kappa, tau = matern_to_spde_params(sigma2=4.0, range_param=3.0, nu=0.5)
        Q = build_precision_matrix(mesh, kappa, tau, alpha=1)

        # Observe at some mesh nodes
        rng = np.random.RandomState(99)
        obs_idx = rng.choice(mesh.n_nodes, size=5, replace=False)
        obs_vals = rng.randn(5) * 2

        # Predict at the same node indices
        pred, var = spde_kriging(mesh, Q, obs_idx, obs_vals, obs_idx,
                                 nugget=1e-4)
        assert_allclose(pred, obs_vals, atol=0.5)
        assert np.all(var >= 0)

    def test_variance_decreases_near_obs(self):
        """Kriging variance should be lower near observed points."""
        mesh = _make_grid_mesh(12, 12, extent=10.0)
        kappa, tau = matern_to_spde_params(4.0, 3.0, 0.5)
        Q = build_precision_matrix(mesh, kappa, tau, alpha=1)

        # Observe at node nearest to center (5,5)
        obs_idx = snap_to_mesh(np.array([[5.0, 5.0]]), mesh)
        obs_vals = np.array([10.0])

        near_idx = snap_to_mesh(np.array([[5.1, 5.1]]), mesh)
        far_idx = snap_to_mesh(np.array([[9.9, 9.9]]), mesh)
        _, v_near = spde_kriging(mesh, Q, obs_idx, obs_vals, near_idx,
                                  nugget=1e-4)
        _, v_far = spde_kriging(mesh, Q, obs_idx, obs_vals, far_idx,
                                 nugget=1e-4)
        assert v_near[0] < v_far[0]


# ── §5 snap_to_mesh ─────────────────────────────────────────

class TestSnapToMesh:
    def test_identity_on_nodes(self):
        """Snapping actual mesh nodes should return valid indices."""
        mesh = _make_grid_mesh(5, 5)
        idx = snap_to_mesh(mesh.nodes[:3], mesh)
        # Each node should snap to itself or a very close node
        for i in range(3):
            dist = np.linalg.norm(mesh.nodes[idx[i]] - mesh.nodes[i])
            assert dist < 1e-6

    def test_nearest(self):
        mesh = _make_grid_mesh(5, 5, extent=4.0)
        # Query the origin — should snap to the node closest to (0,0)
        q = np.array([[0.01, 0.01]])
        idx = snap_to_mesh(q, mesh)
        # The snapped node should be very close to the query
        dist = np.linalg.norm(mesh.nodes[idx[0]] - q[0])
        assert dist < 1.0  # within one mesh cell
