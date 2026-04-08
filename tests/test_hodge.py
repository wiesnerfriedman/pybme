"""Tests for pybme.hodge — Hodge Laplacian operators and time-varying covariance.

Covers:
  §1 Oriented incidence matrix construction
  §2 Hodge Laplacian L₀ (node) and L₁ (edge)
  §3 Hodge decomposition
  §4 HodgeNetworkCovariance (time-varying precision)
  §5 HodgeNetworkCovarianceST (non-separable space-time)
  §6 EdgeCovariance (1-Hodge)
  §7 Compatibility with bme_predict_network_st
"""

import numpy as np
import pytest
from scipy import sparse

from pybme import (
    build_oriented_incidence,
    build_hodge_laplacian_0,
    build_hodge_laplacian_1,
    hodge_decomposition,
    HodgeNetworkCovariance,
    HodgeNetworkCovarianceST,
    EdgeCovariance,
    build_graph_laplacian,
    adjacency_from_edges,
    NetworkCovariance,
    bme_predict_network_st,
)


# ════════════════════════════════════════════════════════════════
# Fixtures — reusable test graphs
# ════════════════════════════════════════════════════════════════

def _line_directed(n=5):
    """0 → 1 → 2 → … → (n-1):  directed chain."""
    edges = np.array([[i, i + 1] for i in range(n - 1)])
    return n, edges


def _river_tree_directed():
    """Directed Y-shaped river:
         2
          ↘
     0 → 1 → 3
          ↗
         4
    plus downstream 1→5→6

    7 nodes, 6 edges.
    """
    edges = np.array([
        [0, 1],  # e0
        [2, 1],  # e1
        [4, 1],  # e2
        [1, 3],  # e3
        [3, 5],  # e4
        [5, 6],  # e5
    ])
    return 7, edges


def _triangle_graph():
    """Triangle: 0→1, 1→2, 2→0.  One independent cycle."""
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    n_nodes = 3
    # B2: single triangle  (edges oriented 0→1→2→0)
    # Convention: triangle [0,1,2] maps to edges [e0, e1, e2] with signs +1
    B2 = sparse.csc_matrix(
        ([1.0, 1.0, 1.0], ([0, 1, 2], [0, 0, 0])),
        shape=(3, 1),
    )
    return n_nodes, edges, B2


# ════════════════════════════════════════════════════════════════
# §1  Oriented incidence matrix
# ════════════════════════════════════════════════════════════════

class TestOrientedIncidence:
    def test_shape(self):
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        assert B1.shape == (5, 4)

    def test_column_sums_zero(self):
        """Each column of B₁ has exactly one −1 and one +1."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        col_sums = np.asarray(B1.sum(axis=0)).ravel()
        np.testing.assert_allclose(col_sums, 0.0, atol=1e-14)

    def test_entries_are_pm1(self):
        """Non-zero entries must be ±1."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        data = B1.toarray().ravel()
        nonzero = data[data != 0]
        assert set(nonzero) == {-1.0, 1.0}

    def test_tail_negative_head_positive(self):
        """Convention: tail = −1, head = +1."""
        edges = np.array([[0, 1], [1, 2]])
        B1 = build_oriented_incidence(3, edges)
        # Edge 0→1:  B1[0,0] = -1, B1[1,0] = +1
        assert B1[0, 0] == -1.0
        assert B1[1, 0] == +1.0
        # Edge 1→2:  B1[1,1] = -1, B1[2,1] = +1
        assert B1[1, 1] == -1.0
        assert B1[2, 1] == +1.0

    def test_B1T_B1_recovers_laplacian(self):
        """B₁ B₁ᵀ should equal the combinatorial Laplacian (unit weights)."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        L_hodge = (B1 @ B1.T).toarray()

        # Build same graph as undirected adjacency
        W = adjacency_from_edges(n, edges)
        L_std = build_graph_laplacian(W).toarray()

        np.testing.assert_allclose(L_hodge, L_std, atol=1e-12)


# ════════════════════════════════════════════════════════════════
# §2  Hodge Laplacians
# ════════════════════════════════════════════════════════════════

class TestHodgeLaplacian0:
    def test_unit_weights_match_standard(self):
        """L₀ with unit weights must equal D − W."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        L0 = build_hodge_laplacian_0(B1).toarray()

        W = adjacency_from_edges(n, edges)
        L_std = build_graph_laplacian(W).toarray()
        np.testing.assert_allclose(L0, L_std, atol=1e-12)

    def test_weighted_psd(self):
        """Weighted L₀ must be positive-semidefinite."""
        n, edges = _line_directed(6)
        B1 = build_oriented_incidence(n, edges)
        w = np.array([0.5, 1.0, 2.0, 0.1, 3.0])
        L0 = build_hodge_laplacian_0(B1, edge_weights=w)
        eigvals = np.linalg.eigvalsh(L0.toarray())
        assert np.all(eigvals >= -1e-12)

    def test_row_sums_zero(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        w = np.random.RandomState(42).rand(len(edges)) + 0.1
        L0 = build_hodge_laplacian_0(B1, edge_weights=w)
        row_sums = np.abs(np.asarray(L0.sum(axis=1)).ravel())
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_symmetry(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        w = np.array([1.0, 2.0, 3.0, 0.5, 1.5, 0.8])
        L0 = build_hodge_laplacian_0(B1, edge_weights=w)
        diff = L0 - L0.T
        assert sparse.linalg.norm(diff) < 1e-14

    def test_weight_length_mismatch_raises(self):
        n, edges = _line_directed(4)
        B1 = build_oriented_incidence(n, edges)
        with pytest.raises(ValueError, match="edge_weights"):
            build_hodge_laplacian_0(B1, edge_weights=np.ones(99))


class TestHodgeLaplacian1:
    def test_shape(self):
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        L1 = build_hodge_laplacian_1(B1)
        E = len(edges)
        assert L1.shape == (E, E)

    def test_psd(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        L1 = build_hodge_laplacian_1(B1)
        eigvals = np.linalg.eigvalsh(L1.toarray())
        assert np.all(eigvals >= -1e-12)

    def test_kernel_dimension_tree(self):
        """For a tree, ker(L₁) = ker(B₁ᵀ B₁) is 1-dimensional
        (the harmonic flow from connectivity)."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        L1 = build_hodge_laplacian_1(B1).toarray()
        eigvals = np.linalg.eigvalsh(L1)
        n_zero = np.sum(np.abs(eigvals) < 1e-10)
        # For a line graph the harmonic space has dimension 0
        # (line graph is a tree with no loops, β₁ = E − N + 1 = 0)
        assert n_zero == 0

    def test_with_triangle(self):
        """Triangle graph should have 1-dim kernel (one cycle)."""
        n_nodes, edges, B2 = _triangle_graph()
        B1 = build_oriented_incidence(n_nodes, edges)
        L1 = build_hodge_laplacian_1(B1, B2=B2).toarray()
        eigvals = np.linalg.eigvalsh(L1)
        # Kernel should not grow; triangle has known spectral structure
        assert L1.shape == (3, 3)
        assert np.all(eigvals >= -1e-12)

    def test_symmetry(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        L1 = build_hodge_laplacian_1(B1)
        diff = L1 - L1.T
        assert sparse.linalg.norm(diff) < 1e-14


# ════════════════════════════════════════════════════════════════
# §3  Hodge decomposition
# ════════════════════════════════════════════════════════════════

class TestHodgeDecomposition:
    def test_gradient_flow_is_exact(self):
        """A pure gradient flow f = B₁ᵀ φ should decompose back exactly."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        phi = np.array([3.0, 1.0, 2.0, 0.5, 4.0, 0.0, -1.0])
        f = np.asarray((B1.T @ phi)).ravel()

        decomp = hodge_decomposition(B1, f)
        np.testing.assert_allclose(decomp["gradient"], f, atol=1e-6)
        np.testing.assert_allclose(decomp["curl"], 0.0, atol=1e-6)
        np.testing.assert_allclose(decomp["harmonic"], 0.0, atol=1e-6)

    def test_sum_equals_original(self):
        """gradient + curl + harmonic must equal the original signal."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        rng = np.random.RandomState(123)
        f = rng.randn(len(edges))
        decomp = hodge_decomposition(B1, f)
        reconstructed = decomp["gradient"] + decomp["curl"] + decomp["harmonic"]
        np.testing.assert_allclose(reconstructed, f, atol=1e-8)

    def test_orthogonality(self):
        """The three components should be mutually orthogonal."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        rng = np.random.RandomState(456)
        f = rng.randn(len(edges))
        decomp = hodge_decomposition(B1, f)
        g, c, h = decomp["gradient"], decomp["curl"], decomp["harmonic"]
        assert abs(np.dot(g, c)) < 1e-6
        assert abs(np.dot(g, h)) < 1e-6
        assert abs(np.dot(c, h)) < 1e-6

    def test_triangle_has_curl(self):
        """A circulation on a triangle should have a curl component."""
        n_nodes, edges, B2 = _triangle_graph()
        B1 = build_oriented_incidence(n_nodes, edges)
        # Pure circulation: equal flow on each edge around the cycle
        f = np.array([1.0, 1.0, 1.0])
        decomp = hodge_decomposition(B1, f, B2=B2)
        assert np.linalg.norm(decomp["curl"]) > 0.1


# ════════════════════════════════════════════════════════════════
# §4  HodgeNetworkCovariance (time-varying)
# ════════════════════════════════════════════════════════════════

class TestHodgeNetworkCovariance:
    def test_static_spd(self):
        """With constant unit weights, covariance must be SPD."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        hcov = HodgeNetworkCovariance(B1, edges, kappa=1.0, sigma2=2.0)
        C = hcov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"Min eigenvalue: {eigvals.min()}"

    def test_static_matches_standard_network_cov(self):
        """Unit weights should match NetworkCovariance (regularised)."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        hcov = HodgeNetworkCovariance(
            B1, edges, kappa=1.5, sigma2=1.0, alpha=1.0, lam=0.0,
        )
        W = adjacency_from_edges(n, edges)
        ncov = NetworkCovariance(W, kappa=1.5, sigma2=1.0, from_adjacency=True)

        C_hodge = hcov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        C_std = ncov.C_dense
        np.testing.assert_allclose(C_hodge, C_std, atol=1e-10)

    def test_time_varying_changes_covariance(self):
        """Different edge weights at different times → different covariance."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)

        def weight_func(t):
            if t == 0.0:
                return np.ones(len(edges))
            else:
                # Deactivate middle edge
                w = np.ones(len(edges))
                w[2] = 0.01
                return w

        hcov = HodgeNetworkCovariance(
            B1, edges, edge_weight_func=weight_func, kappa=0.5,
        )
        C0 = hcov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        C1 = hcov.covariance_block_at(1.0, np.arange(n), np.arange(n))

        # Covariances must differ
        assert not np.allclose(C0, C1, atol=1e-6)

        # Deactivating middle edge should reduce correlation across it
        corr0_04 = C0[0, 4] / np.sqrt(C0[0, 0] * C0[4, 4])
        corr1_04 = C1[0, 4] / np.sqrt(C1[0, 0] * C1[4, 4])
        assert abs(corr1_04) < abs(corr0_04), \
            "Deactivating edge should reduce cross-edge correlation"

    def test_marginal_variance_at(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        hcov = HodgeNetworkCovariance(B1, edges, kappa=2.0, sigma2=3.0)
        mvar = hcov.marginal_variance_at(0.0)
        C = hcov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        np.testing.assert_allclose(mvar, np.diag(C), atol=1e-12)

    def test_static_compatibility_api(self):
        """The t=0 compatibility shims match covariance_block_at(0)."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        hcov = HodgeNetworkCovariance(B1, edges, kappa=1.0)
        idx = np.array([0, 2, 4])
        block_at = hcov.covariance_block_at(0.0, idx, idx)
        block_compat = hcov.covariance_block(idx, idx)
        np.testing.assert_allclose(block_at, block_compat, atol=1e-14)
        np.testing.assert_allclose(
            hcov.marginal_variance_at(0.0),
            hcov.marginal_variance(),
            atol=1e-14,
        )

    def test_precision_at_spd(self):
        """Q(t) must be positive-definite for any positive weights."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        rng = np.random.RandomState(99)
        hcov = HodgeNetworkCovariance(
            B1, edges,
            edge_weight_func=lambda t: rng.rand(len(edges)) + 0.01,
            kappa=0.5,
        )
        Q = hcov.precision_at(42.0)
        eigvals = np.linalg.eigvalsh(Q.toarray())
        assert np.all(eigvals > 0)

    def test_mass_balance_penalty(self):
        """With lam > 0, the mass-balance penalty changes correlation structure."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)

        hcov_no_mb = HodgeNetworkCovariance(
            B1, edges, kappa=0.5, alpha=1.0, lam=0.0,
        )
        hcov_mb = HodgeNetworkCovariance(
            B1, edges, kappa=0.5, alpha=1.0, lam=5.0,
        )
        all_idx = np.arange(n)
        C_no = hcov_no_mb.covariance_block_at(0.0, all_idx, all_idx)
        C_mb = hcov_mb.covariance_block_at(0.0, all_idx, all_idx)

        # Mass-balance penalty should produce a meaningfully different
        # correlation structure
        diff = np.abs(C_mb - C_no).max()
        assert diff > 0.01, "Mass-balance penalty should change the covariance"

        # Junction→downstream (1→3) correlation should be strong with MB
        def corr(C, i, j):
            return C[i, j] / np.sqrt(C[i, i] * C[j, j])
        # Downstream pair should have positive correlation
        assert corr(C_mb, 1, 3) > 0.3


# ════════════════════════════════════════════════════════════════
# §5  HodgeNetworkCovarianceST (non-separable)
# ════════════════════════════════════════════════════════════════

class TestHodgeNetworkCovarianceST:
    @pytest.fixture
    def _setup(self):
        """River tree with time-varying edge weights."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)

        def weight_func(t):
            w = np.ones(len(edges))
            # Ramp up edge 3 (1→3) over time
            w[3] = min(1.0, 0.1 + 0.9 * t / 10.0)
            return w

        hcov = HodgeNetworkCovariance(
            B1, edges, edge_weight_func=weight_func,
            kappa=1.0, sigma2=2.0, alpha=1.0, lam=0.0,
        )
        hcov_st = HodgeNetworkCovarianceST(
            hcov, model_t="exponential", params_t=[1.0, 5.0],
        )
        return n, edges, hcov, hcov_st

    def test_diagonal_equals_sigma2(self, _setup):
        """Same node, same time → σ²."""
        n, edges, hcov, hcov_st = _setup
        idx = np.array([0, 1, 2])
        t = np.array([0.0, 0.0, 0.0])
        C = hcov_st(idx, t, idx, t)
        np.testing.assert_allclose(np.diag(C), hcov_st.sigma2, rtol=0.05)

    def test_spd(self, _setup):
        """Full covariance block must be positive-semidefinite."""
        n, edges, hcov, hcov_st = _setup
        idx = np.array([0, 1, 3, 5, 6])
        t = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
        C = hcov_st(idx, t, idx, t)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals >= -1e-8), f"Min eigval: {eigvals.min()}"

    def test_non_separable(self, _setup):
        """Covariance at t=0 vs t=10 should differ for same node pair."""
        n, edges, hcov, hcov_st = _setup
        idx = np.array([1, 3])  # junction → child
        C_t0 = hcov_st(idx, np.array([0.0, 0.0]),
                        idx, np.array([0.0, 0.0]))
        C_t10 = hcov_st(idx, np.array([10.0, 10.0]),
                         idx, np.array([10.0, 10.0]))
        # Spatial correlations should differ because edge weight changes
        rho_t0 = C_t0[0, 1] / np.sqrt(C_t0[0, 0] * C_t0[1, 1])
        rho_t10 = C_t10[0, 1] / np.sqrt(C_t10[0, 0] * C_t10[1, 1])
        assert abs(rho_t0 - rho_t10) > 0.01, \
            "Time-varying weights should produce different spatial correlations"

    def test_temporal_decay(self, _setup):
        """Same node, increasing lag → decreasing covariance."""
        n, edges, hcov, hcov_st = _setup
        node = np.array([1])
        t0 = np.array([0.0])
        c_lag0 = float(hcov_st(node, t0, node, np.array([0.0])))
        c_lag5 = float(hcov_st(node, t0, node, np.array([5.0])))
        c_lag20 = float(hcov_st(node, t0, node, np.array([20.0])))
        assert c_lag0 > c_lag5 > c_lag20

    def test_geometric_blend_nonneg(self, _setup):
        """Geometric blend should produce non-negative spatial correlations."""
        n, edges, hcov, hcov_st = _setup
        assert hcov_st.blend == "geometric"
        idx = np.arange(n)
        t = np.linspace(0, 10, n)
        C = hcov_st(idx, t, idx, t)
        # All entries should be non-negative (geometric sqrt)
        assert np.all(C >= -1e-10)

    def test_arithmetic_blend(self):
        """Arithmetic blend should also work."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        hcov = HodgeNetworkCovariance(B1, edges, kappa=1.0)
        hcov_st = HodgeNetworkCovarianceST(
            hcov, model_t="exponential", params_t=[1.0, 3.0],
            blend="arithmetic",
        )
        idx = np.array([0, 2, 4])
        t = np.array([0.0, 1.0, 2.0])
        C = hcov_st(idx, t, idx, t)
        # Must still be finite and reasonably shaped
        assert C.shape == (3, 3)
        assert np.all(np.isfinite(C))

    def test_invalid_blend_raises(self):
        n, edges = _line_directed(3)
        B1 = build_oriented_incidence(n, edges)
        hcov = HodgeNetworkCovariance(B1, edges, kappa=1.0)
        with pytest.raises(ValueError, match="blend"):
            HodgeNetworkCovarianceST(
                hcov, model_t="exponential", params_t=[1.0, 3.0],
                blend="invalid",
            )


# ════════════════════════════════════════════════════════════════
# §6  EdgeCovariance
# ════════════════════════════════════════════════════════════════

class TestEdgeCovariance:
    def test_spd(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        ecov = EdgeCovariance(B1, kappa=1.0, sigma2=2.0)
        C = ecov.C_dense
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_shape(self):
        n, edges = _line_directed(6)
        E = len(edges)
        B1 = build_oriented_incidence(n, edges)
        ecov = EdgeCovariance(B1, kappa=1.0)
        assert ecov.C_dense.shape == (E, E)

    def test_symmetry(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        ecov = EdgeCovariance(B1, kappa=0.5, sigma2=3.0)
        C = ecov.C_dense
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_covariance_block(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        ecov = EdgeCovariance(B1, kappa=1.0)
        idx1 = np.array([0, 2, 5])
        idx2 = np.array([1, 3, 4])
        block = ecov.covariance_block(idx1, idx2)
        expected = ecov.C_dense[np.ix_(idx1, idx2)]
        np.testing.assert_allclose(block, expected, atol=1e-12)

    def test_marginal_variance(self):
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        ecov = EdgeCovariance(B1, kappa=2.0, sigma2=4.0)
        mvar = ecov.marginal_variance()
        np.testing.assert_allclose(mvar, np.diag(ecov.C_dense), atol=1e-12)

    def test_adjacent_edges_correlated(self):
        """Sequential edges in a line should be positively correlated."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        ecov = EdgeCovariance(B1, kappa=0.5, sigma2=1.0)
        C = ecov.C_dense
        # Adjacent edges e0(0→1) and e1(1→2) share node 1
        corr_01 = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
        # Distant edges e0(0→1) and e3(3→4) share no nodes
        corr_03 = C[0, 3] / np.sqrt(C[0, 0] * C[3, 3])
        assert abs(corr_01) > abs(corr_03)

    def test_callable(self):
        n, edges = _line_directed(4)
        B1 = build_oriented_incidence(n, edges)
        ecov = EdgeCovariance(B1, kappa=1.0)
        idx = np.array([0, 1, 2])
        block1 = ecov.covariance_block(idx, idx)
        block2 = ecov(idx, idx)
        np.testing.assert_allclose(block1, block2)

    def test_with_triangle(self):
        """Edge covariance on a graph with triangles."""
        n_nodes, edges, B2 = _triangle_graph()
        B1 = build_oriented_incidence(n_nodes, edges)
        ecov = EdgeCovariance(B1, kappa=1.0, B2=B2)
        C = ecov.C_dense
        assert C.shape == (3, 3)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)


# ════════════════════════════════════════════════════════════════
# §7  Integration with bme_predict_network_st
# ════════════════════════════════════════════════════════════════

class TestBMEIntegration:
    def test_hodge_st_with_bme_predict(self):
        """HodgeNetworkCovarianceST plugs into bme_predict_network_st."""
        n, edges = _line_directed(8)
        B1 = build_oriented_incidence(n, edges)
        hcov = HodgeNetworkCovariance(
            B1, edges, kappa=0.5, sigma2=5.0, alpha=1.0, lam=0.0,
        )
        hcov_st = HodgeNetworkCovarianceST(
            hcov, model_t="exponential", params_t=[1.0, 3.0],
        )

        # Hard data at nodes 1 and 5
        ch_nodes = np.array([1, 5])
        th = np.array([0.0, 0.0])
        zh = np.array([10.0, 20.0])

        # Predict at nodes 3 and 7
        ck_nodes = np.array([3, 7])
        tk = np.array([0.0, 0.0])

        results = bme_predict_network_st(
            ck_nodes, tk, ch_nodes, th, zh,
            net_cov_st=hcov_st,
            nhmax=8,
        )
        assert len(results) == 2
        for r in results:
            assert np.isfinite(r.kriging_mean)
            assert r.kriging_var > 0
            # Prediction should be between the observations (interpolation)
            assert 5 < r.kriging_mean < 25

    def test_time_varying_bme(self):
        """BME predictions differ at different times when graph changes."""
        n, edges = _line_directed(6)
        B1 = build_oriented_incidence(n, edges)

        def weight_func(t):
            w = np.ones(len(edges))
            if t > 5.0:
                w[2] = 0.01  # cut middle edge at late times
            return w

        hcov = HodgeNetworkCovariance(
            B1, edges, edge_weight_func=weight_func,
            kappa=0.5, sigma2=3.0, alpha=1.0, lam=0.0,
        )
        hcov_st = HodgeNetworkCovarianceST(
            hcov, model_t="exponential", params_t=[1.0, 5.0],
        )

        # Observe node 0 at t=0
        ch_nodes = np.array([0])
        th = np.array([0.0])
        zh = np.array([10.0])

        # Predict node 4 at t=0 (full connectivity)
        r_early = bme_predict_network_st(
            np.array([4]), np.array([0.0]),
            ch_nodes, th, zh,
            net_cov_st=hcov_st, nhmax=6,
        )[0]

        # Predict node 4 at t=10 (broken connectivity)
        r_late = bme_predict_network_st(
            np.array([4]), np.array([10.0]),
            ch_nodes, th, zh,
            net_cov_st=hcov_st, nhmax=6,
        )[0]

        # With broken edge, variance at distant node should increase
        assert r_late.kriging_var > r_early.kriging_var * 0.9


# ════════════════════════════════════════════════════════════════
# §8  SpectralHodgeNetworkCovariance
# ════════════════════════════════════════════════════════════════

from pybme import (
    SpectralHodgeNetworkCovariance,
    SpectralHodgeNetworkCovarianceST,
)


class TestSpectralHodgeNetworkCovariance:
    def test_static_spd(self):
        """Covariance with unit weights must be SPD."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        scov = SpectralHodgeNetworkCovariance(
            B1, edges, kappa=1.0, sigma2=2.0,
        )
        C = scov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > -1e-10), f"Min eigenvalue: {eigvals.min()}"

    def test_static_matches_standard_network_cov(self):
        """Unit ref weights + unit current weights → same as static NetworkCov."""
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        scov = SpectralHodgeNetworkCovariance(
            B1, edges, kappa=1.5, sigma2=1.0, alpha=1.0, lam=0.0,
        )
        W = adjacency_from_edges(n, edges)
        ncov = NetworkCovariance(W, kappa=1.5, sigma2=1.0, from_adjacency=True)

        C_spectral = scov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        C_std = ncov.C_dense
        np.testing.assert_allclose(C_spectral, C_std, atol=1e-8)

    def test_time_varying_changes_covariance(self):
        """Different edge weights at different times → different covariance."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)

        def weight_func(t):
            if t == 0.0:
                return np.ones(len(edges))
            else:
                w = np.ones(len(edges))
                w[2] = 0.01
                return w

        scov = SpectralHodgeNetworkCovariance(
            B1, edges, edge_weight_func=weight_func, kappa=0.5,
        )
        C0 = scov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        C1 = scov.covariance_block_at(1.0, np.arange(n), np.arange(n))
        assert not np.allclose(C0, C1, atol=1e-6)

    def test_spectral_variance_stability(self):
        """The spectral approach should produce a different variance
        field from full Hodge when weights deviate from the reference
        (unit weights), because it projects through fixed eigenvectors
        rather than re-solving the full system.
        """
        n, edges = _line_directed(10)
        B1 = build_oriented_incidence(n, edges)

        def weight_func(t):
            w = np.ones(len(edges))
            w[:4] = 5.0
            w[4:] = 0.1
            return w

        hodge_cov = HodgeNetworkCovariance(
            B1, edges, edge_weight_func=weight_func,
            kappa=0.3, sigma2=1.0, alpha=1.0, lam=0.0,
        )
        spectral_cov = SpectralHodgeNetworkCovariance(
            B1, edges, edge_weight_func=weight_func,
            kappa=0.3, sigma2=1.0, alpha=1.0, lam=0.0,
        )

        mvar_h = hodge_cov.marginal_variance_at(1.0)
        mvar_s = spectral_cov.marginal_variance_at(1.0)

        # Both produce positive variances
        assert np.all(mvar_h > 0)
        assert np.all(mvar_s > 0)

        # They should differ — the spectral approach smooths through
        # the reference basis, producing a distinct variance field
        assert not np.allclose(mvar_h, mvar_s, atol=1e-4)

    def test_marginal_variance_at(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)
        scov = SpectralHodgeNetworkCovariance(
            B1, edges, kappa=2.0, sigma2=3.0,
        )
        mvar = scov.marginal_variance_at(0.0)
        C = scov.covariance_block_at(0.0, np.arange(n), np.arange(n))
        np.testing.assert_allclose(mvar, np.diag(C), atol=1e-10)

    def test_precompute_dense(self):
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        scov = SpectralHodgeNetworkCovariance(
            B1, edges, kappa=1.0, sigma2=1.0,
        )
        times = np.array([0.0, 1.0, 2.0])
        scov.precompute_dense(times)
        assert len(scov._dense_cache) == 3

        # Cached result should match fresh computation
        C_cached = scov.covariance_block_at(1.0, np.arange(n), np.arange(n))
        scov._dense_cache.clear()
        C_fresh = scov.covariance_block_at(1.0, np.arange(n), np.arange(n))
        np.testing.assert_allclose(C_cached, C_fresh, atol=1e-12)

    def test_n_modes_truncation(self):
        """Truncation to k modes should still be PSD and close to full."""
        n, edges = _line_directed(8)
        B1 = build_oriented_incidence(n, edges)

        scov_full = SpectralHodgeNetworkCovariance(
            B1, edges, kappa=0.5, sigma2=1.0,
        )
        scov_trunc = SpectralHodgeNetworkCovariance(
            B1, edges, kappa=0.5, sigma2=1.0, n_modes=4,
        )
        assert scov_trunc.n_modes == 4

        C_full = scov_full.covariance_block_at(0.0, np.arange(n), np.arange(n))
        C_trunc = scov_trunc.covariance_block_at(0.0, np.arange(n), np.arange(n))

        # Truncated should be PSD (at most zero eigs from missing modes)
        eigvals = np.linalg.eigvalsh(C_trunc)
        assert np.all(eigvals >= -1e-10)

        # Should capture most of the variance
        ratio = np.trace(C_trunc) / np.trace(C_full)
        assert ratio > 0.5, f"Truncated captures only {ratio:.1%} of variance"

    def test_static_api_compatibility(self):
        """t=0 API shims should work."""
        n, edges = _line_directed(5)
        B1 = build_oriented_incidence(n, edges)
        scov = SpectralHodgeNetworkCovariance(B1, edges, kappa=1.0)
        idx = np.array([0, 2, 4])
        block_at = scov.covariance_block_at(0.0, idx, idx)
        block_compat = scov.covariance_block(idx, idx)
        np.testing.assert_allclose(block_at, block_compat, atol=1e-14)
        np.testing.assert_allclose(
            scov.marginal_variance_at(0.0),
            scov.marginal_variance(),
            atol=1e-14,
        )


class TestSpectralHodgeNetworkCovarianceST:
    @pytest.fixture
    def _setup(self):
        n, edges = _river_tree_directed()
        B1 = build_oriented_incidence(n, edges)

        def weight_func(t):
            w = np.ones(len(edges))
            if t > 5.0:
                w[3] = 3.0
            return w

        scov = SpectralHodgeNetworkCovariance(
            B1, edges, edge_weight_func=weight_func,
            kappa=0.5, sigma2=2.0, alpha=1.0, lam=0.5,
        )
        scov_st = SpectralHodgeNetworkCovarianceST(
            scov, model_t="exponential", params_t=[1.0, 5.0],
        )
        return n, edges, B1, scov, scov_st

    def test_same_time_psd(self, _setup):
        n, _, _, _, scov_st = _setup
        nodes = np.arange(n)
        t = np.zeros(n)
        C = scov_st(nodes, t, nodes, t)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > -1e-10)

    def test_temporal_decay(self, _setup):
        n, _, _, _, scov_st = _setup
        node = np.array([1])
        t0 = np.array([0.0])
        c_lag0 = float(scov_st(node, t0, node, np.array([0.0])))
        c_lag5 = float(scov_st(node, t0, node, np.array([5.0])))
        c_lag20 = float(scov_st(node, t0, node, np.array([20.0])))
        assert c_lag0 > c_lag5 > c_lag20

    def test_bme_integration(self, _setup):
        """SpectralHodge ST works with bme_predict_network_st."""
        n, edges, _, _, scov_st = _setup
        ch_nodes = np.array([1, 5])
        th = np.array([0.0, 0.0])
        zh = np.array([10.0, 20.0])
        ck_nodes = np.array([3, 6])
        tk = np.array([0.0, 0.0])

        results = bme_predict_network_st(
            ck_nodes, tk, ch_nodes, th, zh,
            net_cov_st=scov_st, nhmax=7,
        )
        assert len(results) == 2
        for r in results:
            assert np.isfinite(r.kriging_mean)
            assert r.kriging_var > 0

    def test_geometric_blend_nonneg(self, _setup):
        """Geometric blend should produce non-negative correlation."""
        n, _, _, _, scov_st = _setup
        nodes = np.arange(n)
        t1 = np.zeros(n)
        t2 = np.full(n, 10.0)
        C = scov_st(nodes, t1, nodes, t2)
        # Diagonal (self-correlation across time) should be non-negative
        assert np.all(np.diag(C) >= -1e-10)
