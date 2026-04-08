"""Hodge Laplacian operators and time-varying network covariance.

Extends the graph-Laplacian framework in :mod:`pybme.network` with
*oriented incidence matrices*, *Hodge Laplacians* of arbitrary order,
and — critically — **time-varying graph operators** that produce
non-separable space-time covariance on network domains.

Background
----------
For a graph (1-dimensional simplicial complex) with *N* nodes and
*E* directed edges, the two Hodge Laplacians are:

    L₀  =  B₁ Wₑ B₁ᵀ          (node Laplacian — acts on node signals)
    L₁  =  B₁ᵀ B₁  +  B₂ B₂ᵀ  (edge Laplacian — acts on edge signals)

where ``B₁`` is the (N, E) oriented node-edge incidence matrix and
``B₂`` encodes triangles (zero for trees / DAGs).

When the edge weights ``Wₑ`` are constant, ``L₀ = D − W`` reduces to the
standard combinatorial Laplacian already used in
:class:`~pybme.network.NetworkCovariance`.  The key extension here is
making ``Wₑ = Wₑ(t)`` **time-dependent** — e.g. driven by hydraulic
state — so that the precision (and therefore the covariance) varies
across timesteps.  This breaks the separability assumption of
:class:`~pybme.network.NetworkCovarianceST` and captures phenomena like
pump cycling, dry-weather branch deactivation, and backwater-induced
correlation changes.

Hodge decomposition of edge flows:

    f  =  B₁ᵀ ∇φ   +   B₂ ψ   +   h
          ↑gradient     ↑curl     ↑harmonic

For tree-like sewer/river networks ``B₂ = 0``, curl vanishes, and
every flow is gradient + harmonic (topological component).

Classes
-------
HodgeNetworkCovariance
    Node-valued covariance with time-varying edge weights.
    Precision: ``Q(t) = (1/σ²)(κ²I + α L₀(t) + λ M(t))``

HodgeNetworkCovarianceST
    Non-separable space-time covariance that evaluates the spatial
    operator at the time of each data/estimation point.

EdgeCovariance
    Edge-valued covariance via the 1-Hodge Laplacian ``L₁(t)``.

Functions
---------
build_oriented_incidence
    Build the (N, E) signed incidence matrix ``B₁``.
build_hodge_laplacian_0 / build_hodge_laplacian_1
    Weighted Hodge Laplacians from ``B₁`` and optional ``B₂``.

References
----------
Lim L.-H. (2020).  Hodge Laplacians on graphs. SIAM Review, 62(3).

Schaub M.T. et al. (2021).  Signal processing on higher-order networks.
  Signal Processing, 187.

Borovitskiy V. et al. (2021).  Matérn Gaussian processes on graphs.
  AISTATS.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from .network import build_graph_laplacian, build_mass_balance_operator

from scipy import linalg as dense_linalg


# ════════════════════════════════════════════════════════════════
# §1  INCIDENCE & HODGE LAPLACIAN CONSTRUCTION
# ════════════════════════════════════════════════════════════════

def build_oriented_incidence(
    n_nodes: int,
    directed_edges: np.ndarray,
) -> sparse.csc_matrix:
    """Build the oriented node-edge incidence matrix B₁.

    For each directed edge ``e = (i → j)`` with index *k*:

        B₁[i, k] = −1   (tail / source)
        B₁[j, k] = +1   (head / target)

    The standard graph Laplacian satisfies ``L₀ = B₁ B₁ᵀ`` when all
    edge weights are 1.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes.
    directed_edges : (E, 2) int array
        Each row ``(i, j)`` is a directed edge from node *i* to node *j*.

    Returns
    -------
    B1 : (N, E) sparse CSC matrix.
    """
    directed_edges = np.asarray(directed_edges, dtype=int)
    E = len(directed_edges)
    tails = directed_edges[:, 0]
    heads = directed_edges[:, 1]

    rows = np.concatenate([tails, heads])
    cols = np.concatenate([np.arange(E), np.arange(E)])
    vals = np.concatenate([-np.ones(E), np.ones(E)])

    B1 = sparse.csc_matrix((vals, (rows, cols)), shape=(n_nodes, E))
    return B1


def build_hodge_laplacian_0(
    B1: sparse.spmatrix,
    edge_weights: Optional[np.ndarray] = None,
) -> sparse.csc_matrix:
    """Weighted 0-Hodge Laplacian (node Laplacian).

        L₀ = B₁ Wₑ B₁ᵀ

    When ``edge_weights`` are all 1, this equals the standard combinatorial
    Laplacian ``D − W``.

    Parameters
    ----------
    B1 : (N, E) oriented incidence matrix from :func:`build_oriented_incidence`.
    edge_weights : (E,) positive weights.  Default: all 1.

    Returns
    -------
    L0 : (N, N) sparse CSC, symmetric positive-semidefinite.
    """
    E = B1.shape[1]
    if edge_weights is None:
        L0 = (B1 @ B1.T).tocsc()
    else:
        edge_weights = np.asarray(edge_weights, dtype=np.float64)
        if len(edge_weights) != E:
            raise ValueError(
                f"edge_weights length ({len(edge_weights)}) != "
                f"number of edges ({E})."
            )
        We = sparse.diags(edge_weights, format="csc")
        L0 = (B1 @ We @ B1.T).tocsc()

    # Guarantee exact symmetry
    L0 = 0.5 * (L0 + L0.T)
    return L0.tocsc()


def build_hodge_laplacian_1(
    B1: sparse.spmatrix,
    edge_weights: Optional[np.ndarray] = None,
    B2: Optional[sparse.spmatrix] = None,
    triangle_weights: Optional[np.ndarray] = None,
) -> sparse.csc_matrix:
    """Weighted 1-Hodge Laplacian (edge Laplacian).

        L₁ = Wₑ⁻¹ B₁ᵀ B₁ + B₂ Wₜ B₂ᵀ

    For tree-like networks, ``B₂ = 0`` and
    ``L₁ = Wₑ⁻¹ B₁ᵀ B₁`` (the "down" Laplacian only).

    Parameters
    ----------
    B1 : (N, E) oriented incidence matrix.
    edge_weights : (E,) positive weights for the inverse weighting.
        Default: all 1 (so L₁_down = B₁ᵀ B₁).
    B2 : (E, T) edge-triangle incidence matrix (optional).
        Required for networks with independent cycles.
    triangle_weights : (T,) weights for triangles.  Default: all 1.

    Returns
    -------
    L1 : (E, E) sparse CSC, symmetric positive-semidefinite.
    """
    E = B1.shape[1]

    # Down Laplacian: B₁ᵀ B₁  (optionally with Wₑ⁻¹ scaling)
    if edge_weights is None:
        L1_down = (B1.T @ B1).tocsc()
    else:
        edge_weights = np.asarray(edge_weights, dtype=np.float64)
        We_inv = sparse.diags(1.0 / np.maximum(edge_weights, 1e-30), format="csc")
        L1_down = (We_inv @ B1.T @ B1).tocsc()

    # Up Laplacian: B₂ Wₜ B₂ᵀ
    if B2 is not None:
        if triangle_weights is None:
            L1_up = (B2 @ B2.T).tocsc()
        else:
            Wt = sparse.diags(
                np.asarray(triangle_weights, dtype=np.float64),
                format="csc",
            )
            L1_up = (B2 @ Wt @ B2.T).tocsc()
    else:
        L1_up = sparse.csc_matrix((E, E))

    L1 = L1_down + L1_up
    L1 = 0.5 * (L1 + L1.T)
    return L1.tocsc()


def hodge_decomposition(
    B1: sparse.spmatrix,
    f: np.ndarray,
    B2: Optional[sparse.spmatrix] = None,
    kappa: float = 1e-8,
) -> dict:
    """Orthogonal Hodge decomposition of an edge signal.

        f = B₁ᵀ ∇φ  +  B₂ ψ  +  h

    Returns dict with keys ``'gradient'``, ``'curl'``, ``'harmonic'``.

    Parameters
    ----------
    B1 : (N, E) oriented incidence matrix.
    f : (E,) edge signal (e.g. conduit flows).
    B2 : (E, T) optional edge-triangle incidence matrix.
    kappa : float
        Tikhonov regularisation for the least-squares solves.

    Returns
    -------
    dict with keys:
        ``'gradient'`` : (E,) = B₁ᵀ φ
        ``'curl'``     : (E,) = B₂ ψ      (zero if B₂ is None)
        ``'harmonic'`` : (E,) = f − gradient − curl
    """
    f = np.asarray(f, dtype=np.float64)
    N, E = B1.shape

    # Gradient component: solve (B₁ B₁ᵀ + κI) φ = B₁ f
    L0 = (B1 @ B1.T).tocsc()
    reg = kappa * sparse.eye(N, format="csc")
    phi = spsolve((L0 + reg).tocsc(), B1 @ f)
    grad = np.asarray((B1.T @ phi)).ravel()

    # Curl component
    if B2 is not None:
        T = B2.shape[1]
        L1_up = (B2 @ B2.T).tocsc()
        # Project f onto curl subspace via B₂ᵀ
        b2tf = np.asarray((B2.T @ f)).ravel()
        reg_t = kappa * sparse.eye(T, format="csc")
        psi = spsolve((B2.T @ B2 + reg_t).tocsc(), b2tf)
        curl = np.asarray((B2 @ psi)).ravel()
    else:
        curl = np.zeros(E)

    harmonic = f - grad - curl
    return {"gradient": grad, "curl": curl, "harmonic": harmonic}


# ════════════════════════════════════════════════════════════════
# §2  TIME-VARYING NODE COVARIANCE  (HODGE L₀)
# ════════════════════════════════════════════════════════════════

class HodgeNetworkCovariance:
    """Node-valued covariance with time-varying graph operator.

    At each time *t* the precision is:

        Q(t) = (1/σ²)(κ²I  +  α · L₀(t)  +  λ · M)

    where ``L₀(t) = B₁ Wₑ(t) B₁ᵀ`` depends on edge weights that vary
    with the hydraulic state, and ``M = HᵀH`` is the (time-invariant)
    mass-balance penalty.

    The edge weights are supplied via a callable
    ``edge_weight_func(t) → (E,)`` that returns positive weights for
    every edge at time *t*.  Typical sources:

    * SWMM conduit depth/flow at time *t* (normalised to [0, 1]).
    * Binary pump state: 0 (off) or 1 (on).
    * Manning-equation conductance.
    * Constant 1.0 everywhere — recovers the static case.

    The interface (``covariance_block``, ``marginal_variance``, ``Q``)
    matches :class:`~pybme.network.NetworkCovariance` for plug-in use
    with :func:`~pybme.predict.bme_predict_network`.

    Parameters
    ----------
    B1 : (N, E) oriented incidence matrix.
    directed_edges : (E, 2) int array   — for the mass-balance operator.
    edge_weight_func : callable(t) → (E,) array of positive edge weights.
        If ``None``, unit weights at all times (static Laplacian).
    kappa : float
        Regularisation / spatial-scale parameter.
    sigma2 : float
        Overall sill.
    alpha : float
        Weight on the (time-varying) Laplacian smoothness term.
    lam : float
        Weight on the mass-balance penalty ``HᵀH``.
    routing_weights : (E,) optional per-edge mass-balance weights.

    Attributes
    ----------
    B1      : oriented incidence matrix
    n_nodes : number of nodes
    n_edges : number of edges
    H       : mass-balance operator
    M       : mass-balance Gram matrix  HᵀH

    Notes
    -----
    Each call to ``precision_at(t)`` is O(nnz(B₁)) to assemble and
    O(N^{3/2}) for the sparse LU factor.  Factors are cached by time key.
    """

    def __init__(
        self,
        B1: Union[np.ndarray, sparse.spmatrix],
        directed_edges: np.ndarray,
        edge_weight_func: Optional[Callable[[float], np.ndarray]] = None,
        kappa: float = 1.0,
        sigma2: float = 1.0,
        alpha: float = 1.0,
        lam: float = 1.0,
        *,
        routing_weights: Optional[np.ndarray] = None,
    ):
        self.B1 = sparse.csc_matrix(B1, dtype=np.float64)
        self.n_nodes, self.n_edges = self.B1.shape
        self.kappa = float(kappa)
        self.sigma2 = float(sigma2)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.method = "hodge"

        if edge_weight_func is None:
            self._edge_weight_func: Callable = lambda t: np.ones(
                self.n_edges, dtype=np.float64
            )
        else:
            self._edge_weight_func = edge_weight_func

        self.H = build_mass_balance_operator(
            self.n_nodes, directed_edges, routing_weights=routing_weights,
        )
        self.M = (self.H.T @ self.H).tocsc()

        # Cache: rounded-time → splu factor
        self._factor_cache: Dict[float, object] = {}
        # Cache: rounded-time → dense covariance matrix (N×N)
        self._dense_cache: Dict[float, np.ndarray] = {}
        # Max cache entries to prevent unbounded memory growth
        self._max_cache = 256

    # ── precision assembly ───────────────────────────────────

    def precision_at(
        self,
        t: float,
        edge_weights: Optional[np.ndarray] = None,
    ) -> sparse.csc_matrix:
        """Assemble Q(t) = (1/σ²)(κ²I + α L₀(t) + λ M).

        Parameters
        ----------
        t : float
            Time (only used to query ``edge_weight_func`` if
            ``edge_weights`` is not supplied directly).
        edge_weights : (E,) optional — override ``edge_weight_func(t)``.
        """
        if edge_weights is None:
            edge_weights = self._edge_weight_func(t)
        edge_weights = np.asarray(edge_weights, dtype=np.float64)

        L0_t = build_hodge_laplacian_0(self.B1, edge_weights)
        I_n = sparse.eye(self.n_nodes, format="csc")
        Q = (self.kappa ** 2 * I_n + self.alpha * L0_t + self.lam * self.M) / self.sigma2
        Q = 0.5 * (Q + Q.T)
        return Q.tocsc()

    def _get_factor(self, t: float):
        """LU factor of Q(t), cached by rounded time key."""
        key = round(t, 10)
        if key not in self._factor_cache:
            if len(self._factor_cache) >= self._max_cache:
                # Evict oldest entry
                oldest = next(iter(self._factor_cache))
                del self._factor_cache[oldest]
            Q_t = self.precision_at(t)
            self._factor_cache[key] = splu(Q_t)
        return self._factor_cache[key]

    def precompute_dense(self, times: np.ndarray) -> None:
        """Pre-compute and cache full dense covariance matrices.

        For networks of moderate size (< ~2000 nodes) this is much
        faster than repeated per-column sparse solves when many
        calls to ``covariance_block_at`` share the same time steps.

        Parameters
        ----------
        times : 1-D array of time values to pre-compute.
        """
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        for t in times:
            key = round(float(t), 10)
            if key in self._dense_cache:
                continue
            factor = self._get_factor(float(t))
            I_n = np.eye(self.n_nodes)
            C = self.sigma2 * factor.solve(I_n)
            # Symmetrise
            C = 0.5 * (C + C.T)
            if len(self._dense_cache) >= self._max_cache:
                oldest = next(iter(self._dense_cache))
                del self._dense_cache[oldest]
            self._dense_cache[key] = C

    # ── covariance extraction ────────────────────────────────

    def covariance_block_at(
        self,
        t: float,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        """Extract C(t)[idx1, idx2] via sparse solve of Q(t).

        Parameters
        ----------
        t : float
            Time at which to evaluate the graph operator.
        idx1, idx2 : 1-D int arrays of node indices.

        Returns
        -------
        (len(idx1), len(idx2)) dense covariance sub-matrix.
        """
        idx1 = np.atleast_1d(np.asarray(idx1, dtype=int))
        idx2 = np.atleast_1d(np.asarray(idx2, dtype=int))

        key = round(t, 10)
        if key in self._dense_cache:
            return self._dense_cache[key][np.ix_(idx1, idx2)]

        factor = self._get_factor(t)

        block = np.empty((len(idx1), len(idx2)))
        for j, node in enumerate(idx2):
            e = np.zeros(self.n_nodes)
            e[node] = 1.0
            col = self.sigma2 * factor.solve(e)
            block[:, j] = col[idx1]
        return block

    def marginal_variance_at(
        self,
        t: float,
        idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Diagonal of C(t) at selected nodes.

        Parameters
        ----------
        t : float
            Time at which to evaluate.
        idx : (k,) node indices.  If None, all nodes.
        """
        if idx is None:
            idx = np.arange(self.n_nodes)
        idx = np.atleast_1d(np.asarray(idx, dtype=int))

        key = round(t, 10)
        if key in self._dense_cache:
            return np.diag(self._dense_cache[key])[idx]

        factor = self._get_factor(t)

        var = np.empty(len(idx))
        for j, node in enumerate(idx):
            e = np.zeros(self.n_nodes)
            e[node] = 1.0
            var[j] = self.sigma2 * factor.solve(e)[node]
        return var

    # ── static-time compatibility layer ──────────────────────
    # These allow HodgeNetworkCovariance to be used as a drop-in for
    # NetworkCovariance when a fixed time is acceptable (e.g. for
    # bme_predict_network with a single snapshot).

    @property
    def Q(self) -> sparse.csc_matrix:
        """Precision at t=0 (static fallback for the NetworkCovariance API)."""
        return self.precision_at(0.0)

    def covariance_block(self, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
        """Static covariance block at t=0."""
        return self.covariance_block_at(0.0, idx1, idx2)

    def __call__(self, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
        return self.covariance_block(idx1, idx2)

    def marginal_variance(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Static marginal variance at t=0."""
        return self.marginal_variance_at(0.0, idx)


# ════════════════════════════════════════════════════════════════
# §3  NON-SEPARABLE SPACE-TIME COVARIANCE
# ════════════════════════════════════════════════════════════════

class HodgeNetworkCovarianceST:
    """Non-separable space-time covariance with time-varying graph operator.

    The cross-covariance between ``(node i, time t)`` and
    ``(node j, time t')`` is:

        C((i,t), (j,t'))  =  σ² · ρ_s(i,j; t,t') · ρ_t(|t−t'|)

    where the spatial correlation ``ρ_s`` depends on the graph state at
    **both** times.  Two blending strategies are provided:

    * ``'geometric'`` (default):
        ``ρ_s(i,j; t,t') = √max(ρ(i,j|W(t)) · ρ(i,j|W(t')), 0)``
      Guaranteed non-negative; PSD by the Schur-product theorem.

    * ``'arithmetic'``:
        ``ρ_s(i,j; t,t') = ½[ρ(i,j|W(t)) + ρ(i,j|W(t'))]``
      Simpler; PSD when both snapshot covariances are PSD.

    Parameters
    ----------
    hodge_cov : HodgeNetworkCovariance
        Time-varying spatial covariance object.
    model_t : str or list[str]
        Temporal covariance model name(s) (e.g. ``'exponential'``).
    params_t : list or list[list]
        Temporal model parameters (sill should be 1.0).
    sigma2 : float
        Overall sill.  Default: ``hodge_cov.sigma2``.
    blend : ``'geometric'`` or ``'arithmetic'``
        How to combine spatial correlations from two time snapshots.
    """

    def __init__(
        self,
        hodge_cov: HodgeNetworkCovariance,
        model_t: Union[str, list],
        params_t,
        sigma2: Optional[float] = None,
        blend: str = "geometric",
    ):
        self.hodge_cov = hodge_cov
        self.model_t = model_t
        self.params_t = params_t
        self.sigma2 = sigma2 if sigma2 is not None else hodge_cov.sigma2
        if blend not in ("geometric", "arithmetic"):
            raise ValueError(f"blend must be 'geometric' or 'arithmetic', got '{blend}'.")
        self.blend = blend

    def _spatial_correlation(
        self,
        t: float,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        """Normalised spatial correlation block ρ_s(idx1, idx2 | W(t))."""
        C_raw = self.hodge_cov.covariance_block_at(t, idx1, idx2)
        std1 = np.sqrt(self.hodge_cov.marginal_variance_at(t, idx1))
        std2 = np.sqrt(self.hodge_cov.marginal_variance_at(t, idx2))
        denom = std1[:, None] * std2[None, :]
        denom = np.where(denom > 1e-30, denom, 1.0)
        return C_raw / denom

    def covariance_block(
        self,
        idx1: np.ndarray, t1: np.ndarray,
        idx2: np.ndarray, t2: np.ndarray,
    ) -> np.ndarray:
        """Non-separable S/T covariance sub-matrix.

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

        n1, n2 = len(idx1), len(idx2)

        # Temporal component
        ht = np.abs(t1[:, None] - t2[None, :])
        rho_t = eval_cov(ht, self.model_t, self.params_t)

        # Spatial component — blended across both time snapshots
        # Group by unique times to minimise redundant factorisations
        unique_t = np.unique(np.concatenate([t1, t2]))
        rho_cache: Dict[float, np.ndarray] = {}
        for t_val in unique_t:
            key = round(float(t_val), 10)
            if key not in rho_cache:
                rho_cache[key] = self._spatial_correlation(
                    float(t_val), idx1, idx2,
                )

        rho_s = np.empty((n1, n2))
        for i in range(n1):
            k1 = round(float(t1[i]), 10)
            for j in range(n2):
                k2 = round(float(t2[j]), 10)
                r1 = rho_cache[k1][i, j]
                r2 = rho_cache[k2][i, j]
                if self.blend == "geometric":
                    rho_s[i, j] = np.sqrt(np.maximum(r1 * r2, 0.0))
                else:
                    rho_s[i, j] = 0.5 * (r1 + r2)

        return self.sigma2 * rho_s * rho_t

    def __call__(
        self,
        idx1: np.ndarray, t1: np.ndarray,
        idx2: np.ndarray, t2: np.ndarray,
    ) -> np.ndarray:
        return self.covariance_block(idx1, t1, idx2, t2)


# ════════════════════════════════════════════════════════════════
# §4  EDGE-VALUED COVARIANCE (1-HODGE LAPLACIAN)
# ════════════════════════════════════════════════════════════════

class EdgeCovariance:
    """Covariance for edge-valued (flow) random fields via L₁.

    Precision:

        Q_edge = (1/σ²)(κ²Iₑ + L₁)

    where ``L₁ = B₁ᵀ B₁ + B₂ B₂ᵀ`` is the 1-Hodge Laplacian.  For
    tree-like networks (no triangles), ``L₁ = B₁ᵀ B₁``.

    This models correlations *between conduit flows* directly, without
    projecting to nodes first.

    Parameters
    ----------
    B1 : (N, E) oriented incidence matrix.
    kappa : float
        Regularisation / decorrelation parameter.
    sigma2 : float
        Overall sill / marginal variance.
    B2 : (E, T) optional edge-triangle incidence matrix.
    """

    def __init__(
        self,
        B1: Union[np.ndarray, sparse.spmatrix],
        kappa: float = 1.0,
        sigma2: float = 1.0,
        B2: Optional[Union[np.ndarray, sparse.spmatrix]] = None,
    ):
        self.B1 = sparse.csc_matrix(B1, dtype=np.float64)
        self.n_nodes, self.n_edges = self.B1.shape
        self.kappa = float(kappa)
        self.sigma2 = float(sigma2)
        self.B2 = sparse.csc_matrix(B2) if B2 is not None else None

        self.L1 = build_hodge_laplacian_1(self.B1, B2=self.B2)
        self.Q: sparse.csc_matrix = self._build_precision()
        self._Q_factor = None
        self._C_dense: Optional[np.ndarray] = None

    def _build_precision(self) -> sparse.csc_matrix:
        k2 = self.kappa ** 2
        Q = (k2 * sparse.eye(self.n_edges, format="csc") + self.L1) / self.sigma2
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
                (self.kappa ** 2 * sparse.eye(self.n_edges) + self.L1).toarray()
            )
            self._C_dense = 0.5 * (self._C_dense + self._C_dense.T)
        return self._C_dense

    def covariance_block(
        self,
        idx1: np.ndarray,
        idx2: np.ndarray,
    ) -> np.ndarray:
        """Extract edge-covariance sub-matrix C[idx1, idx2].

        Parameters
        ----------
        idx1, idx2 : 1-D int arrays of *edge* indices.
        """
        idx1 = np.atleast_1d(np.asarray(idx1, dtype=int))
        idx2 = np.atleast_1d(np.asarray(idx2, dtype=int))

        if self.n_edges <= 5000 or self._C_dense is not None:
            return self.C_dense[np.ix_(idx1, idx2)]

        factor = self._get_factor()
        block = np.empty((len(idx1), len(idx2)))
        for j, edge in enumerate(idx2):
            e = np.zeros(self.n_edges)
            e[edge] = 1.0
            col = self.sigma2 * factor.solve(e)
            block[:, j] = col[idx1]
        return block

    def __call__(self, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
        return self.covariance_block(idx1, idx2)

    def marginal_variance(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        if idx is None:
            idx = np.arange(self.n_edges)
        idx = np.atleast_1d(np.asarray(idx, dtype=int))

        if self.n_edges <= 5000 or self._C_dense is not None:
            return np.diag(self.C_dense)[idx]

        factor = self._get_factor()
        var = np.empty(len(idx))
        for j, edge in enumerate(idx):
            e = np.zeros(self.n_edges)
            e[edge] = 1.0
            var[j] = self.sigma2 * factor.solve(e)[edge]
        return var


# ════════════════════════════════════════════════════════════════
# §5  SPECTRAL GRAPH FILTER COVARIANCE
# ════════════════════════════════════════════════════════════════

class SpectralHodgeNetworkCovariance:
    r"""Node covariance via fixed spectral basis with time-varying eigenvalues.

    Addresses a key limitation of :class:`HodgeNetworkCovariance`: when
    the Hodge Laplacian ``L₀(t)`` varies with time, its eigenvectors
    rotate, destroying the persistent spectral structure that links
    metered and unmetered nodes.  The result is over-homogeneous
    uncertainty that fails to propagate information from observations
    through spectrally-similar parts of the network.

    **Approach** — spectral graph filter (Galerkin projection):

    1. Compute a **reference** operator
       ``A_ref = α L₀^{ref} + λ M`` and its eigendecomposition
       ``A_ref = V Λ_ref V^T``.  The eigenvectors ``V`` define the
       persistent spectral identity of each node.

    2. At time *t*, project the current operator into the reference
       basis via Galerkin diagonal approximation:

       .. math::
           \lambda_i^{\mathrm{eff}}(t) = v_i^T (α L_0(t) + λ M) v_i

    3. Covariance:

       .. math::
           C(t) = σ^2 \, V \,
                  \mathrm{diag}\!\big(1/(κ^2 + λ_i^{\mathrm{eff}}(t))\big)
                  \, V^T

    This preserves the spectral similarity structure (which nodes
    project onto the same low-frequency modes) while adapting the
    covariance strength to the current hydraulic state.  When
    ``L₀(t) = L₀^{ref}``, the result is identical to the static
    :class:`~pybme.network.NetworkCovariance`.

    Parameters
    ----------
    B1 : (N, E) oriented incidence matrix.
    directed_edges : (E, 2) int array.
    edge_weight_func : callable(t) → (E,) positive weights.
    kappa, sigma2, alpha, lam : float
        Same meaning as :class:`HodgeNetworkCovariance`.
    n_modes : int or None
        Number of spectral modes to retain. ``None`` = all (exact).
        Truncating to the *k* lowest-frequency modes speeds up
        evaluation and concentrates on long-range structure.
    ref_weights : (E,) array or None
        Edge weights for the reference Laplacian.
        ``None`` → unit weights (static combinatorial Laplacian).
    routing_weights : (E,) optional per-edge mass-balance weights.
    """

    def __init__(
        self,
        B1: Union[np.ndarray, sparse.spmatrix],
        directed_edges: np.ndarray,
        edge_weight_func: Optional[Callable[[float], np.ndarray]] = None,
        kappa: float = 1.0,
        sigma2: float = 1.0,
        alpha: float = 1.0,
        lam: float = 1.0,
        *,
        n_modes: Optional[int] = None,
        ref_weights: Optional[np.ndarray] = None,
        routing_weights: Optional[np.ndarray] = None,
    ):
        self.B1 = sparse.csc_matrix(B1, dtype=np.float64)
        self.n_nodes, self.n_edges = self.B1.shape
        self.kappa = float(kappa)
        self.sigma2 = float(sigma2)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.method = "spectral_hodge"

        if edge_weight_func is None:
            self._edge_weight_func: Callable = lambda t: np.ones(
                self.n_edges, dtype=np.float64
            )
        else:
            self._edge_weight_func = edge_weight_func

        # Mass-balance operator (time-invariant)
        H = build_mass_balance_operator(
            self.n_nodes, directed_edges, routing_weights=routing_weights,
        )
        self.M = (H.T @ H).tocsc()

        # ── Build reference operator and eigendecompose ──────────
        if ref_weights is not None:
            ref_weights = np.asarray(ref_weights, dtype=np.float64)
        L0_ref = build_hodge_laplacian_0(self.B1, ref_weights)
        A_ref = self.alpha * L0_ref + self.lam * self.M
        A_ref_dense = A_ref.toarray()
        # Symmetric → eigh (real eigenvalues, orthogonal eigenvectors)
        eigenvalues, eigenvectors = dense_linalg.eigh(A_ref_dense)

        # Truncate if requested
        if n_modes is not None and n_modes < self.n_nodes:
            k = max(1, min(n_modes, self.n_nodes))
            self.V = eigenvectors[:, :k].copy()
            self.lam_ref = eigenvalues[:k].copy()
        else:
            self.V = eigenvectors
            self.lam_ref = eigenvalues
        self.n_modes = self.V.shape[1]

        # Cache: rounded-time → effective eigenvalues (k,)
        self._lam_cache: Dict[float, np.ndarray] = {}
        # Cache: rounded-time → dense covariance (N, N)
        self._dense_cache: Dict[float, np.ndarray] = {}
        self._max_cache = 512

    # ── spectral eigenvalue computation ──────────────────────

    def _spectral_eigenvalues_at(self, t: float) -> np.ndarray:
        r"""Galerkin-projected eigenvalues at time *t*.

        .. math::
            \lambda_i^{\mathrm{eff}}(t) = v_i^T (α L_0(t) + λ M) v_i

        Returns
        -------
        (n_modes,) non-negative effective eigenvalues.
        """
        key = round(t, 10)
        if key in self._lam_cache:
            return self._lam_cache[key]

        w = self._edge_weight_func(t)
        L0_t = build_hodge_laplacian_0(self.B1, w)
        A_t = self.alpha * L0_t + self.lam * self.M

        # Efficient diagonal of V^T A V:  sum(V * (A @ V), axis=0)
        AV = A_t @ self.V  # (N, k)  — sparse @ dense
        lam_eff = np.sum(self.V * AV, axis=0)  # (k,)
        lam_eff = np.maximum(lam_eff, 0.0)

        if len(self._lam_cache) >= self._max_cache:
            oldest = next(iter(self._lam_cache))
            del self._lam_cache[oldest]
        self._lam_cache[key] = lam_eff
        return lam_eff

    # ── covariance construction ──────────────────────────────

    def _build_dense_at(self, t: float) -> np.ndarray:
        r"""C(t) = σ² V diag(1/(κ² + λ_eff(t))) V^T."""
        lam_eff = self._spectral_eigenvalues_at(t)
        h = self.sigma2 / (self.kappa ** 2 + lam_eff)  # (k,)
        # (V * h) @ V^T  avoids forming diag(h) explicitly
        C = (self.V * h[np.newaxis, :]) @ self.V.T
        C = 0.5 * (C + C.T)  # exact symmetry
        return C

    def precompute_dense(self, times: np.ndarray) -> None:
        """Pre-compute and cache dense covariance at given times.

        Much faster than :meth:`HodgeNetworkCovariance.precompute_dense`
        because no LU factorisation is needed — just a matrix multiply.
        """
        times = np.atleast_1d(np.asarray(times, dtype=np.float64))
        for t in times:
            key = round(float(t), 10)
            if key in self._dense_cache:
                continue
            C = self._build_dense_at(float(t))
            if len(self._dense_cache) >= self._max_cache:
                oldest = next(iter(self._dense_cache))
                del self._dense_cache[oldest]
            self._dense_cache[key] = C

    def covariance_block_at(
        self, t: float, idx1: np.ndarray, idx2: np.ndarray,
    ) -> np.ndarray:
        """Extract C(t)[idx1, idx2]."""
        idx1 = np.atleast_1d(np.asarray(idx1, dtype=int))
        idx2 = np.atleast_1d(np.asarray(idx2, dtype=int))
        key = round(t, 10)
        if key in self._dense_cache:
            return self._dense_cache[key][np.ix_(idx1, idx2)]
        C = self._build_dense_at(t)
        return C[np.ix_(idx1, idx2)]

    def marginal_variance_at(
        self, t: float, idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Diagonal of C(t) at selected nodes."""
        if idx is None:
            idx = np.arange(self.n_nodes)
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        key = round(t, 10)
        if key in self._dense_cache:
            return np.diag(self._dense_cache[key])[idx]
        lam_eff = self._spectral_eigenvalues_at(t)
        h = self.sigma2 / (self.kappa ** 2 + lam_eff)
        # diag(C) = sum(V**2 * h, axis=1)
        diag = np.sum(self.V[idx, :] ** 2 * h[np.newaxis, :], axis=1)
        return diag

    # ── static-time API (drop-in for NetworkCovariance) ──────

    @property
    def Q(self) -> sparse.csc_matrix:
        """Precision at t=0."""
        w = self._edge_weight_func(0.0)
        L0 = build_hodge_laplacian_0(self.B1, w)
        I_n = sparse.eye(self.n_nodes, format="csc")
        Q = (self.kappa ** 2 * I_n + self.alpha * L0 + self.lam * self.M) / self.sigma2
        return 0.5 * (Q + Q.T).tocsc()

    def covariance_block(self, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
        return self.covariance_block_at(0.0, idx1, idx2)

    def __call__(self, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
        return self.covariance_block(idx1, idx2)

    def marginal_variance(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        return self.marginal_variance_at(0.0, idx)


class SpectralHodgeNetworkCovarianceST:
    r"""Non-separable space-time covariance using the spectral graph filter.

    Same interface and blending logic as :class:`HodgeNetworkCovarianceST`
    but backed by :class:`SpectralHodgeNetworkCovariance` to preserve the
    persistent spectral structure of the network.

    Parameters
    ----------
    spectral_cov : SpectralHodgeNetworkCovariance
    model_t : str or list[str]
        Temporal covariance model(s).
    params_t : parameters for temporal model (sill should be 1.0).
    sigma2 : float
        Overall sill.
    blend : ``'geometric'`` or ``'arithmetic'``
    """

    def __init__(
        self,
        spectral_cov: SpectralHodgeNetworkCovariance,
        model_t: Union[str, list],
        params_t,
        sigma2: Optional[float] = None,
        blend: str = "geometric",
    ):
        self.hodge_cov = spectral_cov  # duck-type compatible name
        self.model_t = model_t
        self.params_t = params_t
        self.sigma2 = sigma2 if sigma2 is not None else spectral_cov.sigma2
        if blend not in ("geometric", "arithmetic"):
            raise ValueError(f"blend must be 'geometric' or 'arithmetic', got '{blend}'.")
        self.blend = blend

    def _spatial_correlation(
        self, t: float, idx1: np.ndarray, idx2: np.ndarray,
    ) -> np.ndarray:
        C_raw = self.hodge_cov.covariance_block_at(t, idx1, idx2)
        std1 = np.sqrt(self.hodge_cov.marginal_variance_at(t, idx1))
        std2 = np.sqrt(self.hodge_cov.marginal_variance_at(t, idx2))
        denom = std1[:, None] * std2[None, :]
        denom = np.where(denom > 1e-30, denom, 1.0)
        return C_raw / denom

    def covariance_block(
        self,
        idx1: np.ndarray, t1: np.ndarray,
        idx2: np.ndarray, t2: np.ndarray,
    ) -> np.ndarray:
        from .covariance import eval_cov

        idx1 = np.atleast_1d(np.asarray(idx1, dtype=int))
        idx2 = np.atleast_1d(np.asarray(idx2, dtype=int))
        t1 = np.atleast_1d(np.asarray(t1, dtype=np.float64))
        t2 = np.atleast_1d(np.asarray(t2, dtype=np.float64))

        n1, n2 = len(idx1), len(idx2)

        ht = np.abs(t1[:, None] - t2[None, :])
        rho_t = eval_cov(ht, self.model_t, self.params_t)

        unique_t = np.unique(np.concatenate([t1, t2]))
        rho_cache: Dict[float, np.ndarray] = {}
        for t_val in unique_t:
            key = round(float(t_val), 10)
            if key not in rho_cache:
                rho_cache[key] = self._spatial_correlation(
                    float(t_val), idx1, idx2,
                )

        rho_s = np.empty((n1, n2))
        for i in range(n1):
            k1 = round(float(t1[i]), 10)
            for j in range(n2):
                k2 = round(float(t2[j]), 10)
                r1 = rho_cache[k1][i, j]
                r2 = rho_cache[k2][i, j]
                if self.blend == "geometric":
                    rho_s[i, j] = np.sqrt(np.maximum(r1 * r2, 0.0))
                else:
                    rho_s[i, j] = 0.5 * (r1 + r2)

        return self.sigma2 * rho_s * rho_t

    def __call__(
        self, idx1: np.ndarray, t1: np.ndarray,
        idx2: np.ndarray, t2: np.ndarray,
    ) -> np.ndarray:
        return self.covariance_block(idx1, t1, idx2, t2)
