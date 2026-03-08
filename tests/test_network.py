"""Tests for the network-domain covariance and BME prediction module.

Covers:
  §1 Graph Laplacian construction
  §2 NetworkCovariance (regularised, diffusion, precision)
  §3 Network kriging (hard data only)
  §4 Network BME with soft data (spatial)
  §5 NetworkCovarianceST and space-time BME
  §6 Edge cases and validation
"""

import numpy as np
import pytest
from scipy import sparse

from pybme import (
    NetworkCovariance,
    NetworkCovarianceST,
    build_graph_laplacian,
    adjacency_from_edges,
    network_kriging_precision,
    bme_predict_network,
    bme_predict_network_st,
    SoftPDF,
    BMEResult,
)


# ════════════════════════════════════════════════════════════════
# Fixtures — reusable test graphs
# ════════════════════════════════════════════════════════════════

def _line_graph(n=10):
    """0 — 1 — 2 — … — (n-1):  linear chain."""
    edges = np.array([[i, i + 1] for i in range(n - 1)])
    W = adjacency_from_edges(n, edges)
    return W, edges


def _star_graph(n=6):
    """Hub (node 0) connected to leaves 1..n-1."""
    edges = np.array([[0, i] for i in range(1, n)])
    W = adjacency_from_edges(n, edges)
    return W, edges


def _river_tree():
    """Simple Y-shaped river:
         2
          \\
     0 — 1 — 3
          /
         4
    plus downstream 1→5→6
    """
    edges = np.array([
        [0, 1], [1, 2], [1, 3], [1, 4], [1, 5], [5, 6],
    ])
    W = adjacency_from_edges(7, edges)
    return W, edges


def _cycle_graph(n=6):
    """Closed cycle:  0—1—2—…—(n-1)—0."""
    edges = np.array([[i, (i + 1) % n] for i in range(n)])
    W = adjacency_from_edges(n, edges)
    return W, edges


# ════════════════════════════════════════════════════════════════
# §1  Graph Laplacian
# ════════════════════════════════════════════════════════════════

class TestGraphLaplacian:
    def test_line_laplacian_shape(self):
        W, _ = _line_graph(5)
        L = build_graph_laplacian(W)
        assert L.shape == (5, 5)

    def test_laplacian_row_sum_zero(self):
        """Rows of L must sum to zero."""
        W, _ = _line_graph(8)
        L = build_graph_laplacian(W)
        row_sums = np.abs(np.asarray(L.sum(axis=1)).ravel())
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-14)

    def test_laplacian_symmetry(self):
        W, _ = _river_tree()
        L = build_graph_laplacian(W)
        diff = L - L.T
        assert sparse.linalg.norm(diff) < 1e-14

    def test_normalised_laplacian_diagonal_one(self):
        W, _ = _star_graph(5)
        L = build_graph_laplacian(W, normalised=True)
        diag = L.diagonal()
        np.testing.assert_allclose(diag, 1.0, atol=1e-14)

    def test_laplacian_positive_semidefinite(self):
        W, _ = _cycle_graph(6)
        L = build_graph_laplacian(W)
        eigvals = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigvals >= -1e-12)

    def test_adjacency_from_edges_symmetry(self):
        W, _ = _line_graph(5)
        diff = W - W.T
        assert sparse.linalg.norm(diff) < 1e-14

    def test_adjacency_from_edges_weighted(self):
        edges = np.array([[0, 1], [1, 2]])
        weights = np.array([2.0, 3.0])
        W = adjacency_from_edges(3, edges, weights)
        assert W[0, 1] == 2.0
        assert W[1, 2] == 3.0
        assert W[1, 0] == 2.0  # symmetric


# ════════════════════════════════════════════════════════════════
# §2  NetworkCovariance
# ════════════════════════════════════════════════════════════════

class TestNetworkCovariance:
    def test_regularised_spd(self):
        """Regularised covariance must be SPD."""
        W, _ = _line_graph(8)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=2.0,
                               method="regularised", from_adjacency=True)
        C = nc.C_dense
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"Min eigenvalue: {eigvals.min()}"

    def test_diffusion_spd(self):
        """Diffusion kernel must be SPD."""
        W, _ = _star_graph(5)
        nc = NetworkCovariance(W, kappa=0.5, sigma2=1.0,
                               method="diffusion", from_adjacency=True)
        C = nc.C_dense
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_precision_matches_regularised(self):
        """Precision mode should yield same covariance as regularised."""
        W, _ = _river_tree()
        nc_r = NetworkCovariance(W, kappa=1.5, sigma2=1.0,
                                  method="regularised", from_adjacency=True)
        nc_p = NetworkCovariance(W, kappa=1.5, sigma2=1.0,
                                  method="precision", from_adjacency=True)
        np.testing.assert_allclose(nc_r.C_dense, nc_p.C_dense, atol=1e-10)

    def test_covariance_block_matches_dense(self):
        """Sub-block extraction must match full dense slicing."""
        W, _ = _line_graph(10)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=1.5,
                               method="regularised", from_adjacency=True)
        idx1 = np.array([0, 3, 7])
        idx2 = np.array([1, 5, 9])
        block = nc.covariance_block(idx1, idx2)
        expected = nc.C_dense[np.ix_(idx1, idx2)]
        np.testing.assert_allclose(block, expected, atol=1e-12)

    def test_marginal_variance(self):
        W, _ = _star_graph(5)
        nc = NetworkCovariance(W, kappa=2.0, sigma2=3.0,
                               method="regularised", from_adjacency=True)
        mvar = nc.marginal_variance()
        diag = np.diag(nc.C_dense)
        np.testing.assert_allclose(mvar, diag, atol=1e-12)

    def test_callable_interface(self):
        W, _ = _line_graph(5)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        block1 = nc.covariance_block(np.array([0, 1]), np.array([2, 3]))
        block2 = nc(np.array([0, 1]), np.array([2, 3]))
        np.testing.assert_allclose(block1, block2)

    def test_kappa_controls_decorrelation(self):
        """Higher kappa → faster decorrelation (lower off-diagonal)."""
        W, _ = _line_graph(10)
        nc_slow = NetworkCovariance(W, kappa=0.5, from_adjacency=True)
        nc_fast = NetworkCovariance(W, kappa=3.0, from_adjacency=True)
        # Correlation between nodes 0 and 9
        corr_slow = nc_slow.C_dense[0, 9] / np.sqrt(nc_slow.C_dense[0, 0] * nc_slow.C_dense[9, 9])
        corr_fast = nc_fast.C_dense[0, 9] / np.sqrt(nc_fast.C_dense[0, 0] * nc_fast.C_dense[9, 9])
        assert abs(corr_slow) > abs(corr_fast), "Higher kappa should decorrelate faster"

    def test_invalid_method_raises(self):
        W, _ = _line_graph(3)
        with pytest.raises(ValueError, match="Unknown method"):
            NetworkCovariance(W, kappa=1.0, method="bogus", from_adjacency=True)

    def test_american_spelling_accepted(self):
        W, _ = _line_graph(3)
        nc = NetworkCovariance(W, kappa=1.0, method="regularized", from_adjacency=True)
        assert nc.method == "regularised"

    def test_symmetry_of_covariance(self):
        W, _ = _river_tree()
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        C = nc.C_dense
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_from_laplacian_directly(self):
        """Can pass a pre-built Laplacian instead of adjacency."""
        W, _ = _line_graph(5)
        L = build_graph_laplacian(W)
        nc = NetworkCovariance(L, kappa=1.0, from_adjacency=False)
        assert nc.n_nodes == 5
        assert nc.C_dense.shape == (5, 5)


# ════════════════════════════════════════════════════════════════
# §3  Network kriging (hard data only)
# ════════════════════════════════════════════════════════════════

class TestNetworkKriging:
    def test_interpolation_at_observation(self):
        """Kriging must reproduce observed values at observation nodes."""
        W, _ = _line_graph(10)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)
        obs = np.array([2, 5, 8])
        z = np.array([1.0, 3.0, -1.0])
        mu, var = network_kriging_precision(nc, obs, z, est_nodes=obs, nugget=1e-8)
        np.testing.assert_allclose(mu, z, atol=0.05)

    def test_kriging_reduces_variance(self):
        """Posterior variance must be less than prior variance near data."""
        W, _ = _line_graph(10)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=2.0, from_adjacency=True)
        obs = np.array([5])
        z = np.array([1.0])
        mu, var = network_kriging_precision(nc, obs, z, est_nodes=np.array([4, 5, 6]),
                                            nugget=1e-6)
        prior_var = nc.marginal_variance(np.array([4, 5, 6]))
        assert np.all(var < prior_var + 1e-10)

    def test_kriging_all_nodes(self):
        """est_nodes=None should return predictions at all nodes."""
        W, _ = _star_graph(5)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        obs = np.array([0])
        z = np.array([2.0])
        mu, var = network_kriging_precision(nc, obs, z, nugget=1e-6)
        assert len(mu) == 5
        assert len(var) == 5

    def test_bme_predict_network_kriging_only(self):
        """bme_predict_network with no soft data should match kriging."""
        W, _ = _line_graph(10)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)
        ch_nodes = np.array([2, 5, 8])
        zh = np.array([1.0, 3.0, -1.0])
        results = bme_predict_network(
            ck_nodes=np.array([4]),
            ch_nodes=ch_nodes, zh=zh,
            net_cov=nc,
        )
        assert len(results) == 1
        res = results[0]
        assert np.isfinite(res.kriging_mean)
        assert res.kriging_var > 0
        assert "kriging_only" in res.info

    def test_duplicate_node(self):
        """Predicting at an observation node should return exact value."""
        W, _ = _line_graph(5)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        results = bme_predict_network(
            ck_nodes=np.array([2]),
            ch_nodes=np.array([1, 2, 3]),
            zh=np.array([1.0, 5.0, 3.0]),
            net_cov=nc,
        )
        assert results[0].mean == pytest.approx(5.0)
        assert results[0].info == "duplicate"


# ════════════════════════════════════════════════════════════════
# §4  Network BME with soft data
# ════════════════════════════════════════════════════════════════

class TestNetworkBME:
    def test_soft_data_shifts_posterior(self):
        """Adding soft data should shift the posterior toward the soft value."""
        W, _ = _line_graph(10)
        nc = NetworkCovariance(W, kappa=0.8, sigma2=1.0, from_adjacency=True)
        ch = np.array([0, 9])
        zh = np.array([0.0, 0.0])

        # Without soft data
        res_hard = bme_predict_network(
            ck_nodes=np.array([5]), ch_nodes=ch, zh=zh, net_cov=nc)[0]

        # With soft data near node 5, biasing toward +3
        soft = [SoftPDF.from_gaussian(3.0, 0.5)]
        res_soft = bme_predict_network(
            ck_nodes=np.array([5]), ch_nodes=ch, zh=zh,
            cs_nodes=np.array([4]), soft_pdfs=soft,
            net_cov=nc)[0]

        assert res_soft.mean > res_hard.mean, \
            "Soft data biasing up should increase posterior mean"

    def test_full_bme_info_string(self):
        W, _ = _line_graph(6)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        soft = [SoftPDF.from_uniform(0.0, 2.0)]
        results = bme_predict_network(
            ck_nodes=np.array([3]),
            ch_nodes=np.array([0, 5]),
            zh=np.array([1.0, 1.0]),
            cs_nodes=np.array([2]),
            soft_pdfs=soft,
            net_cov=nc,
        )
        assert "full_network_bme" in results[0].info

    def test_multiple_estimation_points(self):
        W, _ = _line_graph(10)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        results = bme_predict_network(
            ck_nodes=np.array([2, 4, 6, 8]),
            ch_nodes=np.array([0, 5, 9]),
            zh=np.array([1.0, 2.0, 0.5]),
            net_cov=nc,
        )
        assert len(results) == 4
        for r in results:
            assert np.isfinite(r.mean)

    def test_all_integration_methods(self):
        """Every integration method should run without error on a network."""
        W, _ = _line_graph(8)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        soft = [SoftPDF.from_gaussian(1.0, 0.5)]
        for method in ("gauss_hermite", "laplace", "ep", "qmc", "lis", "mc"):
            res = bme_predict_network(
                ck_nodes=np.array([4]),
                ch_nodes=np.array([0, 7]),
                zh=np.array([0.5, 1.5]),
                cs_nodes=np.array([3]),
                soft_pdfs=soft,
                net_cov=nc,
                method=method,
            )
            assert np.isfinite(res[0].mean), f"method={method} failed"

    def test_no_data(self):
        """With no hard or soft data, should return prior."""
        W, _ = _line_graph(5)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=2.0, from_adjacency=True)
        results = bme_predict_network(
            ck_nodes=np.array([2]),
            ch_nodes=np.array([], dtype=int),
            zh=np.array([]),
            net_cov=nc,
            mean_prior=5.0,
        )
        assert results[0].mean == pytest.approx(5.0, abs=0.1)
        assert "no_data" in results[0].info

    def test_river_tree_topology(self):
        """BME should work on a Y-shaped river network."""
        W, _ = _river_tree()
        nc = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)
        results = bme_predict_network(
            ck_nodes=np.array([6]),
            ch_nodes=np.array([0, 2, 4]),
            zh=np.array([1.0, 2.0, 3.0]),
            net_cov=nc,
        )
        assert np.isfinite(results[0].mean)

    def test_cycle_graph(self):
        """BME should work on a graph with cycles."""
        W, _ = _cycle_graph(8)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        soft = [SoftPDF.from_uniform(0.5, 1.5)]
        results = bme_predict_network(
            ck_nodes=np.array([0]),
            ch_nodes=np.array([2, 6]),
            zh=np.array([1.0, 1.0]),
            cs_nodes=np.array([4]),
            soft_pdfs=soft,
            net_cov=nc,
        )
        assert np.isfinite(results[0].mean)


# ════════════════════════════════════════════════════════════════
# §5  Space-time network BME
# ════════════════════════════════════════════════════════════════

class TestNetworkST:
    def test_st_covariance_spd(self):
        """Separable S/T covariance block must be SPD."""
        W, _ = _line_graph(6)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)
        nc_st = NetworkCovarianceST(nc, model_t="exponential",
                                     params_t=[1.0, 5.0], sigma2=2.0)
        nodes = np.array([0, 1, 2, 3, 4, 5])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        C = nc_st(nodes, times, nodes, times)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > -1e-10)

    def test_st_covariance_separability(self):
        """C(i,t,j,t') = σ² ρ_s(i,j) ρ_t(|t-t'|)."""
        W, _ = _line_graph(4)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)
        nc_st = NetworkCovarianceST(nc, model_t="exponential",
                                     params_t=[1.0, 3.0], sigma2=1.0)
        # Same time → temporal factor = 1
        C_same_t = nc_st(np.array([0, 1]), np.array([0.0, 0.0]),
                          np.array([0, 1]), np.array([0.0, 0.0]))
        # Different time
        C_diff_t = nc_st(np.array([0, 1]), np.array([0.0, 0.0]),
                          np.array([0, 1]), np.array([5.0, 5.0]))
        # Ratio should be the temporal correlation at lag 5
        from pybme.covariance import exponential_cov
        rho_t = float(exponential_cov(5.0, [1.0, 3.0]))
        # Diagonal ratio
        ratio = C_diff_t[0, 0] / C_same_t[0, 0]
        assert ratio == pytest.approx(rho_t, abs=0.01)

    def test_st_bme_kriging_only(self):
        W, _ = _line_graph(6)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        nc_st = NetworkCovarianceST(nc, model_t="exponential",
                                     params_t=[1.0, 3.0])
        results = bme_predict_network_st(
            ck_nodes=np.array([3]),
            tk=np.array([2.0]),
            ch_nodes=np.array([0, 2, 5]),
            th=np.array([0.0, 1.0, 3.0]),
            zh=np.array([1.0, 2.0, 0.5]),
            net_cov_st=nc_st,
        )
        assert len(results) == 1
        assert np.isfinite(results[0].mean)

    def test_st_bme_with_soft_data(self):
        W, _ = _line_graph(6)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        nc_st = NetworkCovarianceST(nc, model_t="exponential",
                                     params_t=[1.0, 3.0])
        soft = [SoftPDF.from_gaussian(2.0, 0.3)]
        results = bme_predict_network_st(
            ck_nodes=np.array([3]),
            tk=np.array([2.0]),
            ch_nodes=np.array([0, 5]),
            th=np.array([0.0, 3.0]),
            zh=np.array([1.0, 1.5]),
            cs_nodes=np.array([2]),
            ts=np.array([1.5]),
            soft_pdfs=soft,
            net_cov_st=nc_st,
        )
        assert "full_network_st_bme" in results[0].info
        assert np.isfinite(results[0].mean)

    def test_st_multiple_estimation_points(self):
        W, _ = _line_graph(8)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        nc_st = NetworkCovarianceST(nc, model_t="exponential",
                                     params_t=[1.0, 2.0])
        results = bme_predict_network_st(
            ck_nodes=np.array([1, 3, 6]),
            tk=np.array([1.0, 2.0, 3.0]),
            ch_nodes=np.array([0, 4, 7]),
            th=np.array([0.0, 1.5, 2.5]),
            zh=np.array([1.0, 2.0, 0.5]),
            net_cov_st=nc_st,
        )
        assert len(results) == 3
        for r in results:
            assert np.isfinite(r.mean)
            assert r.kriging_var > 0

    def test_st_temporal_decorrelation(self):
        """Points at distant times should have wider posterior."""
        W, _ = _line_graph(5)
        nc = NetworkCovariance(W, kappa=1.0, sigma2=1.0, from_adjacency=True)
        nc_st = NetworkCovarianceST(nc, model_t="exponential",
                                     params_t=[1.0, 2.0])
        # Close in time
        res_close = bme_predict_network_st(
            ck_nodes=np.array([2]), tk=np.array([1.0]),
            ch_nodes=np.array([2]), th=np.array([0.0]),
            zh=np.array([3.0]),
            net_cov_st=nc_st,
        )[0]
        # Far in time
        res_far = bme_predict_network_st(
            ck_nodes=np.array([2]), tk=np.array([100.0]),
            ch_nodes=np.array([2]), th=np.array([0.0]),
            zh=np.array([3.0]),
            net_cov_st=nc_st,
        )[0]
        assert res_far.kriging_var > res_close.kriging_var, \
            "Further in time → more uncertainty"

    def test_st_no_data(self):
        W, _ = _line_graph(4)
        nc = NetworkCovariance(W, kappa=1.0, from_adjacency=True)
        nc_st = NetworkCovarianceST(nc, model_t="exponential",
                                     params_t=[1.0, 2.0])
        res = bme_predict_network_st(
            ck_nodes=np.array([1]),
            tk=np.array([0.0]),
            ch_nodes=np.array([], dtype=int),
            th=np.array([]),
            zh=np.array([]),
            net_cov_st=nc_st,
            mean_prior=7.0,
        )[0]
        assert res.mean == pytest.approx(7.0, abs=0.1)
        assert "no_data" in res.info


# ════════════════════════════════════════════════════════════════
# §6  Edge cases
# ════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_node_graph(self):
        """Degenerate: one node, no edges."""
        W = sparse.csc_matrix((1, 1))
        nc = NetworkCovariance(W, kappa=1.0, sigma2=2.0, from_adjacency=True)
        assert nc.C_dense.shape == (1, 1)
        assert nc.C_dense[0, 0] == pytest.approx(2.0, rel=0.01)

    def test_disconnected_graph(self):
        """Two disconnected components — covariance between them should be ~0."""
        # Nodes 0-2 form one component, nodes 3-5 form another
        edges = np.array([[0, 1], [1, 2], [3, 4], [4, 5]])
        W = adjacency_from_edges(6, edges)
        nc = NetworkCovariance(W, kappa=2.0, from_adjacency=True)
        # Cross-component correlation
        cross = nc.C_dense[0, 5] / np.sqrt(nc.C_dense[0, 0] * nc.C_dense[5, 5])
        # Should be very small (κ²I + L is block-diagonal + κ²I connects them)
        # With kappa=2 the regularisation does create a tiny cross-covariance
        assert abs(cross) < 0.5, f"Cross-component correlation {cross} too large"

    def test_weighted_edges(self):
        """Edge weights should affect covariance structure."""
        edges = np.array([[0, 1], [1, 2]])
        W_unif = adjacency_from_edges(3, edges, weights=np.array([1.0, 1.0]))
        W_asym = adjacency_from_edges(3, edges, weights=np.array([1.0, 10.0]))
        nc_u = NetworkCovariance(W_unif, kappa=1.0, from_adjacency=True)
        nc_a = NetworkCovariance(W_asym, kappa=1.0, from_adjacency=True)
        # Stronger edge 1→2 should give higher correlation between 1 and 2
        corr_u_12 = nc_u.C_dense[1, 2] / np.sqrt(nc_u.C_dense[1, 1] * nc_u.C_dense[2, 2])
        corr_a_12 = nc_a.C_dense[1, 2] / np.sqrt(nc_a.C_dense[1, 1] * nc_a.C_dense[2, 2])
        assert corr_a_12 > corr_u_12

    def test_missing_net_cov_raises(self):
        with pytest.raises(ValueError, match="net_cov"):
            bme_predict_network(
                ck_nodes=np.array([0]),
                ch_nodes=np.array([1]),
                zh=np.array([1.0]),
            )

    def test_missing_net_cov_st_raises(self):
        with pytest.raises(ValueError, match="net_cov_st"):
            bme_predict_network_st(
                ck_nodes=np.array([0]), tk=np.array([0.0]),
                ch_nodes=np.array([1]), th=np.array([0.0]),
                zh=np.array([1.0]),
            )

    def test_invalid_edges_shape(self):
        with pytest.raises(ValueError, match="edges must be"):
            adjacency_from_edges(3, np.array([0, 1, 2]))
