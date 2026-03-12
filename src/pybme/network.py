"""Network-domain covariance via graph Laplacian / diffusion operators.

Original contribution by Corinne Wiesner-Friedman (not part of MATLAB BMElib).

On network domains (rivers, sewers, road networks, pipe systems) Euclidean
distance is inappropriate.  Classical covariance models C(h) evaluated at
network shortest-path distances are not guaranteed positive-definite for
most model families (Gaussian, spherical, Matérn ν > 0.5).

This module constructs valid (symmetric positive-definite) covariance
matrices directly from the *graph Laplacian* of the network, bypassing
the need for a distance-based covariance model altogether.  The approach
is the natural extension of the SPDE framework to graph domains
(Borovitskiy et al., 2021).

Three covariance constructions are available:

1. **Regularised inverse** (default):
       C = σ² (κ²I + L)⁻¹
   SPD because κ²I + L is strictly positive-definite.  Equivalent to
   the Matérn(ν=1) SPDE on the graph.  Exponential-like decay along
   branches; valid on arbitrary graph topologies (trees, cycles, DAGs).

2. **Diffusion kernel**:
       C = σ² exp(−β L)
   Always SPD (matrix exponential of a negative semi-definite matrix).
   Smoother than the regularised inverse; analogous to a Gaussian model.

3. **Precision (GMRF) mode**:
       Q = σ⁻² (κ²I + β L)
   Returns the sparse precision matrix directly for use with sparse
   Cholesky solvers.  Most efficient when the full dense covariance
   is not needed (see ``network_kriging_precision``).

Separable space-time support
----------------------------
``NetworkCovarianceST`` pairs a spatial ``NetworkCovariance`` with any
parametric temporal covariance model from ``pybme.covariance``:

    C((i,t),(j,t')) = σ² · ρ_network(i,j) · ρ_temporal(|t−t'|)

This plugs directly into ``bme_predict_network_st`` for full BME with
soft data on space-time network domains.

References
----------
Borovitskiy V., Azangulov I., Terenin A., Mostowsky P., Deisenroth M.,
Durrande N. (2021).  Matérn Gaussian processes on graphs.  AISTATS.

Kondor R.I. & Lafferty J. (2002).  Diffusion kernels on graphs and other
discrete structures.  ICML.

Smola A.J. & Kondor R. (2003).  Kernels and regularization on graphs.
Learning Theory and Kernel Machines (COLT/KW 2003), Springer LNAI 2777.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy import sparse, linalg as dense_linalg
from scipy.sparse.linalg import splu, spsolve


# ════════════════════════════════════════════════════════════════
# §1  GRAPH LAPLACIAN CONSTRUCTION
# ════════════════════════════════════════════════════════════════

def build_graph_laplacian(
    adjacency: Union[np.ndarray, sparse.spmatrix],
    *,
    normalised: bool = False,
) -> sparse.csc_matrix:
    """Construct the combinatorial (or normalised) graph Laplacian.

    Parameters
    ----------
    adjacency : (N, N) weighted adjacency matrix (symmetric, non-negative).
                Accepted as dense ndarray or any scipy sparse format.
    normalised : if True, return the symmetric normalised Laplacian
                 L_sym = D^{-1/2} L D^{-1/2}.

    Returns
    -------
    L : (N, N) sparse CSC Laplacian.
    """
    W = sparse.csc_matrix(adjacency, dtype=np.float64)
    n = W.shape[0]
    if W.shape[0] != W.shape[1]:
        raise ValueError("Adjacency matrix must be square.")

    # Symmetrise (in case user passes an upper- or lower-triangular matrix)
    W = 0.5 * (W + W.T)

    # Degree matrix
    d = np.asarray(W.sum(axis=1)).ravel()

    L = sparse.diags(d, format="csc") - W

    if normalised:
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = sparse.diags(d_inv_sqrt, format="csc")
        L = D_inv_sqrt @ L @ D_inv_sqrt

    return L.tocsc()


def adjacency_from_edges(
    n_nodes: int,
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> sparse.csc_matrix:
    """Build a sparse adjacency matrix from an edge list.

    Parameters
    ----------
    n_nodes : total number of nodes in the graph.
    edges   : (E, 2) integer array — each row is (i, j).
    weights : (E,) optional edge weights (default: 1.0).

    Returns
    -------
    W : (n_nodes, n_nodes) sparse symmetric adjacency matrix.
    """
    edges = np.asarray(edges, dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be (E, 2).")
    if weights is None:
        weights = np.ones(len(edges), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.concatenate([weights, weights])

    W = sparse.csc_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    return W


# ════════════════════════════════════════════════════════════════
# §2  NETWORK COVARIANCE CLASS
# ════════════════════════════════════════════════════════════════

class NetworkCovariance:
    """Graph-Laplacian covariance for network-constrained random fields.

    Parameters
    ----------
    laplacian : (N, N) sparse graph Laplacian, or (N, N) dense/sparse
                weighted adjacency matrix when ``from_adjacency=True``.
    kappa     : spatial scale parameter (κ).  Larger κ → faster decorrelation.
    sigma2    : sill / marginal variance.
    method    : ``'regularised'`` (default), ``'diffusion'``, or ``'precision'``.

                * ``'regularised'`` — C = σ²(κ²I + L)⁻¹.  Network analog of
                  Matérn(ν=1) via SPDE.
                * ``'diffusion'``   — C = σ² exp(−κ L).  Smoother; Gaussian-like.
                * ``'precision'``   — stores Q = σ⁻²(κ²I + L) only.
                  Dense covariance computed lazily on first access.

    from_adjacency : if True, ``laplacian`` is treated as an adjacency matrix
                     and the Laplacian is computed internally.
    normalised     : whether to use the symmetric normalised Laplacian (only
                     when ``from_adjacency=True``).

    Attributes
    ----------
    L        : sparse Laplacian
    Q        : sparse precision matrix (always available)
    C_dense  : full dense covariance (computed lazily for regularised/diffusion;
               always available for small graphs).  ``None`` until requested.
    n_nodes  : number of nodes
    """

    def __init__(
        self,
        laplacian: Union[np.ndarray, sparse.spmatrix],
        kappa: float,
        sigma2: float = 1.0,
        method: str = "regularised",
        *,
        from_adjacency: bool = False,
        normalised: bool = False,
    ):
        self.kappa = float(kappa)
        self.sigma2 = float(sigma2)
        self.method = method.lower()
        if self.method not in ("regularised", "regularized", "diffusion", "precision"):
            raise ValueError(f"Unknown method '{method}'.")
        # Accept American spelling
        if self.method == "regularized":
            self.method = "regularised"

        if from_adjacency:
            self.L = build_graph_laplacian(laplacian, normalised=normalised)
        else:
            self.L = sparse.csc_matrix(laplacian, dtype=np.float64)

        self.n_nodes = self.L.shape[0]

        # Precision matrix — always built (sparse)
        self.Q: sparse.csc_matrix = self._build_precision()

        # Dense covariance — lazily cached
        self._C_dense: Optional[np.ndarray] = None

        # Sparse LU factor — lazily cached
        self._Q_factor = None

    # ── internal builders ────────────────────────────────────

    def _build_precision(self) -> sparse.csc_matrix:
        """Q = (1/σ²)(κ²I + L)  — sparse precision."""
        k2 = self.kappa ** 2
        Q = (k2 * sparse.eye(self.n_nodes, format="csc") + self.L) / self.sigma2
        # Symmetrise
        Q = 0.5 * (Q + Q.T)
        return Q.tocsc()

    def _get_factor(self):
        """Sparse LU (SuperLU) factorisation of Q — cached."""
        if self._Q_factor is None:
            self._Q_factor = splu(self.Q.tocsc())
        return self._Q_factor

    @property
    def C_dense(self) -> np.ndarray:
        """Full dense covariance matrix (computed on first access).

        For the regularised method:  C = σ²(κ²I + L)⁻¹
        For the diffusion method:    C = σ² exp(−κ L)

        Warning: O(N²) memory.  Use ``covariance_block`` for large graphs.
        """
        if self._C_dense is None:
            if self.method in ("regularised", "precision"):
                self._C_dense = self.sigma2 * np.linalg.inv(
                    (self.kappa ** 2 * sparse.eye(self.n_nodes) + self.L).toarray()
                )
            elif self.method == "diffusion":
                L_dense = self.L.toarray()
                self._C_dense = self.sigma2 * dense_linalg.expm(-self.kappa * L_dense)
            # Symmetrise
            self._C_dense = 0.5 * (self._C_dense + self._C_dense.T)
        return self._C_dense

    # ── sub-block extraction ─────────────────────────────────

    def covariance_block(
        self,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        """Extract the covariance sub-matrix C[idx1, :][:, idx2].

        For small graphs (N ≤ 5000) the full dense inverse is cached;
        for larger graphs each column of the sub-block is obtained via
        a sparse solve Q x = e_j.

        Parameters
        ----------
        idx1, idx2 : 1-D integer arrays of node indices.

        Returns
        -------
        (len(idx1), len(idx2)) dense covariance sub-matrix.
        """
        idx1 = np.atleast_1d(np.asarray(idx1, dtype=int))
        idx2 = np.atleast_1d(np.asarray(idx2, dtype=int))

        if self.n_nodes <= 5000 or self._C_dense is not None:
            return self.C_dense[np.ix_(idx1, idx2)]

        # Large graph — sparse solves
        factor = self._get_factor()
        n = self.n_nodes
        block = np.empty((len(idx1), len(idx2)))
        for j_out, j_node in enumerate(idx2):
            e = np.zeros(n)
            e[j_node] = 1.0
            col = self.sigma2 * factor.solve(e)  # Q⁻¹ e_j * σ²
            block[:, j_out] = col[idx1]
        return block

    def __call__(
        self,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        """Convenience: ``cov(idx1, idx2)`` ↔ ``cov.covariance_block(idx1, idx2)``."""
        return self.covariance_block(idx1, idx2)

    # ── marginal variances ───────────────────────────────────

    def marginal_variance(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Diagonal of the covariance matrix at selected nodes.

        Parameters
        ----------
        idx : (k,) node indices.  If None, all nodes.

        Returns
        -------
        (k,) array of marginal variances.
        """
        if idx is None:
            idx = np.arange(self.n_nodes)
        idx = np.atleast_1d(np.asarray(idx, dtype=int))

        if self.n_nodes <= 5000 or self._C_dense is not None:
            return np.diag(self.C_dense)[idx]

        factor = self._get_factor()
        var = np.empty(len(idx))
        for j, i_node in enumerate(idx):
            e = np.zeros(self.n_nodes)
            e[i_node] = 1.0
            var[j] = self.sigma2 * factor.solve(e)[i_node]
        return var


# ════════════════════════════════════════════════════════════════
# §3  SEPARABLE SPACE-TIME NETWORK COVARIANCE
# ════════════════════════════════════════════════════════════════

class NetworkCovarianceST:
    """Separable space-time covariance on a network domain.

    C((i,t), (j,t')) = σ² · ρ_s(i,j) · ρ_t(|t − t'|)

    where ρ_s is the *normalised* network covariance (marginal variance = 1)
    and ρ_t is a temporal covariance model from ``pybme.covariance``.

    Parameters
    ----------
    net_cov : NetworkCovariance
        Spatial network covariance.  Will be internally normalised so that
        diagonal entries are 1 (the overall sill is controlled by ``sigma2``).
    model_t : str or list[str]
        Temporal covariance model name(s) — e.g. ``'exponential'``.
    params_t : list or list[list]
        Temporal model parameters.  The temporal model sill should be 1
        (the overall sill is ``sigma2``).
    sigma2 : float
        Overall sill controlling total variance.  Defaults to the network
        covariance's σ².

    Notes
    -----
    The network covariance is normalised internally: C_s(i,j) / √(C_s(i,i)·C_s(j,j)).
    """

    def __init__(
        self,
        net_cov: NetworkCovariance,
        model_t: Union[str, list],
        params_t,
        sigma2: Optional[float] = None,
    ):
        self.net_cov = net_cov
        self.model_t = model_t
        self.params_t = params_t
        self.sigma2 = sigma2 if sigma2 is not None else net_cov.sigma2

        # Pre-compute marginal standard deviations for normalisation
        self._marg_std = np.sqrt(net_cov.marginal_variance())

    def covariance_block(
        self,
        idx1: np.ndarray, t1: np.ndarray,
        idx2: np.ndarray, t2: np.ndarray,
    ) -> np.ndarray:
        """Separable S/T covariance sub-matrix.

        Parameters
        ----------
        idx1, idx2 : (n1,), (n2,) node index arrays.
        t1, t2     : (n1,), (n2,) time coordinate arrays.

        Returns
        -------
        (n1, n2) covariance matrix.
        """
        from .covariance import eval_cov

        idx1 = np.atleast_1d(np.asarray(idx1, dtype=int))
        idx2 = np.atleast_1d(np.asarray(idx2, dtype=int))
        t1 = np.atleast_1d(np.asarray(t1, dtype=np.float64))
        t2 = np.atleast_1d(np.asarray(t2, dtype=np.float64))

        # Spatial block (normalised)
        Cs_raw = self.net_cov.covariance_block(idx1, idx2)
        std1 = self._marg_std[idx1][:, None]
        std2 = self._marg_std[idx2][None, :]
        denom = std1 * std2
        denom = np.where(denom > 1e-30, denom, 1.0)
        rho_s = Cs_raw / denom

        # Temporal block
        ht = np.abs(t1[:, None] - t2[None, :])
        rho_t = eval_cov(ht, self.model_t, self.params_t)

        return self.sigma2 * rho_s * rho_t

    def __call__(
        self,
        idx1: np.ndarray, t1: np.ndarray,
        idx2: np.ndarray, t2: np.ndarray,
    ) -> np.ndarray:
        """Convenience: ``cov(idx1, t1, idx2, t2)``."""
        return self.covariance_block(idx1, t1, idx2, t2)


# ════════════════════════════════════════════════════════════════
# §4  NETWORK KRIGING (PRECISION-BASED)
# ════════════════════════════════════════════════════════════════

def network_kriging_precision(
    net_cov: NetworkCovariance,
    obs_nodes: np.ndarray,
    z_obs: np.ndarray,
    est_nodes: Optional[np.ndarray] = None,
    nugget: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple kriging on a network via the sparse precision matrix.

    Analogous to ``spde_kriging`` but for graph-domain fields.

    Solves  (Q + Aᵀ Σ_ε⁻¹ A) x = Aᵀ Σ_ε⁻¹ z

    Parameters
    ----------
    net_cov   : NetworkCovariance
    obs_nodes : (m,) node indices with observations.
    z_obs     : (m,) observed values (zero-mean assumed).
    est_nodes : (k,) node indices for prediction.  If None, all nodes.
    nugget    : measurement noise variance.

    Returns
    -------
    mu  : (k,) posterior mean at estimation nodes.
    var : (k,) posterior marginal variance at estimation nodes.
    """
    n = net_cov.n_nodes
    m = len(obs_nodes)
    obs_nodes = np.asarray(obs_nodes, dtype=int)
    z_obs = np.asarray(z_obs, dtype=np.float64)

    A = sparse.csc_matrix(
        (np.ones(m), (np.arange(m), obs_nodes)),
        shape=(m, n),
    )

    noise_prec = 1.0 / max(nugget, 1e-10)
    Q_post = (net_cov.Q + noise_prec * (A.T @ A)).tocsc()

    rhs = noise_prec * A.T @ z_obs

    try:
        factor = splu(Q_post)
        mu_all = factor.solve(rhs)
    except Exception:
        warnings.warn("splu failed; falling back to spsolve.", stacklevel=2)
        mu_all = spsolve(Q_post, rhs)

    if est_nodes is None:
        est_nodes = np.arange(n)
    est_nodes = np.asarray(est_nodes, dtype=int)
    mu = mu_all[est_nodes]

    # Marginal variances
    var = np.empty(len(est_nodes))
    try:
        for j, idx in enumerate(est_nodes):
            e = np.zeros(n)
            e[idx] = 1.0
            var[j] = factor.solve(e)[idx]
    except Exception:
        var = np.full(len(est_nodes), np.nan)

    return mu, var


# ════════════════════════════════════════════════════════════════
# §5  MASS-BALANCE OPERATOR & PHYSICS-INFORMED PRECISION
# ════════════════════════════════════════════════════════════════

def build_mass_balance_operator(
    n_nodes: int,
    directed_edges: np.ndarray,
    *,
    routing_weights: Optional[np.ndarray] = None,
) -> sparse.csc_matrix:
    """Build the directed mass-balance (flow-conservation) operator H.

    For each non-source node *j* with directed parents {p₁, … , pₖ},
    one row of H encodes the constraint

        x_j − Σᵢ wᵢ · x_{pᵢ}  ≈ 0        (no lateral inflow)

    where wᵢ are optional routing fractions (default 1.0).

    Parameters
    ----------
    n_nodes : int
        Total number of nodes.
    directed_edges : (E, 2) int array
        Each row ``(i, j)`` is a directed edge from *i* → *j*.
    routing_weights : (E,) optional
        Per-edge routing/attenuation weights.  Default is 1.0 for every edge.

    Returns
    -------
    H : (n_constraints, n_nodes) sparse CSC matrix.
        One row per non-source node (nodes with at least one incoming edge).

    Notes
    -----
    The symmetric product ``M = H.T @ H`` is positive semi-definite and its
    quadratic form ``x.T @ M @ x = Σⱼ (xⱼ − Σᵢ wᵢ xᵢ)²`` sums the squared
    mass-balance residuals over all junction nodes.
    """
    directed_edges = np.asarray(directed_edges, dtype=int)
    if routing_weights is None:
        routing_weights = np.ones(len(directed_edges), dtype=np.float64)
    else:
        routing_weights = np.asarray(routing_weights, dtype=np.float64)

    # Find parent list for each node
    parents: dict = {}  # node_j -> list of (parent_i, weight)
    for k, (i, j) in enumerate(directed_edges):
        parents.setdefault(int(j), []).append((int(i), routing_weights[k]))

    # Build H: one row per node that has at least one parent
    rows, cols, vals = [], [], []
    constraint_idx = 0
    constraint_nodes = []
    for j in sorted(parents.keys()):
        # +1 for the node itself
        rows.append(constraint_idx)
        cols.append(j)
        vals.append(1.0)
        # −w for each parent
        for p, w in parents[j]:
            rows.append(constraint_idx)
            cols.append(p)
            vals.append(-w)
        constraint_nodes.append(j)
        constraint_idx += 1

    H = sparse.csc_matrix(
        (vals, (rows, cols)),
        shape=(constraint_idx, n_nodes),
    )
    return H


class PhysicsInformedNetworkCovariance:
    """Network covariance with mass-balance penalty built into the prior.

    Precision matrix:

        Q = (1/σ²)(κ²I  +  α·L  +  λ·HᵀH)

    where

    * **κ²I** — regularisation that keeps Q strictly positive-definite.
    * **α·L** — standard undirected-Laplacian smoothness
      (set ``alpha=0`` to remove smoothness entirely).
    * **λ·HᵀH** — mass-balance penalty.  ``H`` is the directed
      flow-conservation operator from :func:`build_mass_balance_operator`.
      Higher ``lam`` penalises conservation violations more heavily,
      so the prior concentrates on flow fields that satisfy
      downstream accumulation.

    The interface (``covariance_block``, ``marginal_variance``, ``Q``)
    matches :class:`NetworkCovariance` so it can be used directly in
    :func:`~pybme.predict.bme_predict_network` and
    :class:`NetworkCovarianceST`.

    Parameters
    ----------
    laplacian : (N, N) sparse graph Laplacian **or** adjacency matrix when
                ``from_adjacency=True``.
    directed_edges : (E, 2) int array of directed edges (i → j).
    kappa : float
        Regularisation / spatial-scale parameter.
    sigma2 : float
        Overall sill.
    alpha : float
        Weight on the undirected-Laplacian smoothness term (default 1.0).
    lam : float
        Weight on the mass-balance penalty ``HᵀH`` (default 1.0).
    routing_weights : (E,) optional per-edge mass-balance weights.
    from_adjacency : bool
        If True, treat *laplacian* as an adjacency matrix.
    normalised : bool
        Use the normalised Laplacian (only when ``from_adjacency=True``).
    """

    def __init__(
        self,
        laplacian: Union[np.ndarray, sparse.spmatrix],
        directed_edges: np.ndarray,
        kappa: float,
        sigma2: float = 1.0,
        alpha: float = 1.0,
        lam: float = 1.0,
        *,
        routing_weights: Optional[np.ndarray] = None,
        from_adjacency: bool = False,
        normalised: bool = False,
    ):
        self.kappa = float(kappa)
        self.sigma2 = float(sigma2)
        self.alpha = float(alpha)
        self.lam = float(lam)

        if from_adjacency:
            self.L = build_graph_laplacian(laplacian, normalised=normalised)
        else:
            self.L = sparse.csc_matrix(laplacian, dtype=np.float64)

        self.n_nodes = self.L.shape[0]
        self.H = build_mass_balance_operator(
            self.n_nodes, directed_edges, routing_weights=routing_weights,
        )
        self.M = (self.H.T @ self.H).tocsc()  # mass-balance gram matrix

        self.method = "physics_informed"

        self.Q: sparse.csc_matrix = self._build_precision()
        self._C_dense: Optional[np.ndarray] = None
        self._Q_factor = None

    # ── internal builders ────────────────────────────────────

    def _build_precision(self) -> sparse.csc_matrix:
        """Q = (1/σ²)(κ²I + α L + λ HᵀH)."""
        k2 = self.kappa ** 2
        I_n = sparse.eye(self.n_nodes, format="csc")
        Q = (k2 * I_n + self.alpha * self.L + self.lam * self.M) / self.sigma2
        Q = 0.5 * (Q + Q.T)
        return Q.tocsc()

    def _get_factor(self):
        if self._Q_factor is None:
            self._Q_factor = splu(self.Q.tocsc())
        return self._Q_factor

    @property
    def C_dense(self) -> np.ndarray:
        if self._C_dense is None:
            self._C_dense = self.sigma2 * np.linalg.inv(
                (self.kappa ** 2 * sparse.eye(self.n_nodes)
                 + self.alpha * self.L
                 + self.lam * self.M).toarray()
            )
            self._C_dense = 0.5 * (self._C_dense + self._C_dense.T)
        return self._C_dense

    # ── sub-block extraction ─────────────────────────────────

    def covariance_block(
        self,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        idx1 = np.atleast_1d(np.asarray(idx1, dtype=int))
        idx2 = np.atleast_1d(np.asarray(idx2, dtype=int))

        if self.n_nodes <= 5000 or self._C_dense is not None:
            return self.C_dense[np.ix_(idx1, idx2)]

        factor = self._get_factor()
        n = self.n_nodes
        block = np.empty((len(idx1), len(idx2)))
        for j_out, j_node in enumerate(idx2):
            e = np.zeros(n)
            e[j_node] = 1.0
            col = self.sigma2 * factor.solve(e)
            block[:, j_out] = col[idx1]
        return block

    def __call__(
        self,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        return self.covariance_block(idx1, idx2)

    def marginal_variance(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        if idx is None:
            idx = np.arange(self.n_nodes)
        idx = np.atleast_1d(np.asarray(idx, dtype=int))

        if self.n_nodes <= 5000 or self._C_dense is not None:
            return np.diag(self.C_dense)[idx]

        factor = self._get_factor()
        var = np.empty(len(idx))
        for j, i_node in enumerate(idx):
            e = np.zeros(self.n_nodes)
            e[i_node] = 1.0
            var[j] = self.sigma2 * factor.solve(e)[i_node]
        return var
