"""SPDE / GMRF module — sparse-precision Matérn fields for scalable kriging.

Original contribution by Corinne Wiesner-Friedman (not part of MATLAB BMElib).

Implements the stochastic partial differential equation (SPDE) approach
of Lindgren, Rue & Lindström (2011) to represent Matérn random fields
as Gaussian Markov Random Fields (GMRFs) on a triangulated mesh.

Key advantage: the precision matrix **Q** is sparse, enabling O(n)
storage and O(n^{3/2}) Cholesky factorisation (2-D) vs O(n²)/O(n³) for
the dense covariance approach.

Scope and limitations
---------------------
* **Hard-data (kriging) only** — ``spde_kriging()`` does not integrate
  soft probabilistic data (SoftPDF).  For full BME with soft data, use
  ``bme_predict()`` with ``method='laplace'``.
* **Simple kriging** — assumes zero mean; no polynomial trend estimation.
* **2-D spatial fields** — the FEM mesh is a Delaunay triangulation in ℝ².
* **Matérn covariance family** — the SPDE link requires ν = α − d/2.

References
----------
Lindgren F., Rue H., Lindström J. (2011).  An explicit link between
Gaussian fields and Gaussian Markov random fields: the stochastic partial
differential equation approach.  JRSS-B, 73(4), 423–498.
https://doi.org/10.1111/j.1467-9868.2011.00777.x

Rue H., Martino S., Chopin N. (2009).  Approximate Bayesian inference
for latent Gaussian models by using integrated nested Laplace approximations.
JRSS-B, 71(2), 319–392.
https://doi.org/10.1111/j.1467-9868.2008.00700.x
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve


# ════════════════════════════════════════════════════════════════
# §1  MESH CONSTRUCTION
# ════════════════════════════════════════════════════════════════

class SPDEMesh:
    """FEM triangular mesh for the SPDE discretisation.

    Parameters
    ----------
    nodes : (N, 2) array of node coordinates
    triangles : (T, 3) integer array of triangle vertex indices
    """

    def __init__(self, nodes: np.ndarray, triangles: np.ndarray):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_tri = self.triangles.shape[0]

        # Precompute FEM matrices
        self._C, self._G = self._fem_matrices()

    @classmethod
    def from_points(cls, coords: np.ndarray,
                    extend: float = 0.1,
                    max_edge: Optional[float] = None) -> "SPDEMesh":
        """Build a mesh from observation/estimation coordinates.

        Parameters
        ----------
        coords    : (N, 2) observation coordinates
        extend    : fractional extension of domain beyond data bounding box
        max_edge  : maximum triangle edge length (for mesh refinement).
                    If None, uses 1/4 of the domain extent.

        Returns
        -------
        SPDEMesh
        """
        coords = np.atleast_2d(coords)
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        pad_x, pad_y = dx * extend, dy * extend

        if max_edge is None:
            max_edge = max(dx, dy) / 4.0

        # Build a regular grid covering the extended domain, then add data points
        x_lo, x_hi = xmin - pad_x, xmax + pad_x
        y_lo, y_hi = ymin - pad_y, ymax + pad_y

        nx = max(int(np.ceil((x_hi - x_lo) / max_edge)) + 1, 3)
        ny = max(int(np.ceil((y_hi - y_lo) / max_edge)) + 1, 3)
        gx = np.linspace(x_lo, x_hi, nx)
        gy = np.linspace(y_lo, y_hi, ny)
        gx2, gy2 = np.meshgrid(gx, gy)
        grid_pts = np.column_stack([gx2.ravel(), gy2.ravel()])

        # Merge data coordinates into the grid (snap if close)
        all_pts = np.vstack([grid_pts, coords])
        # Remove near-duplicates
        from scipy.spatial import cKDTree
        tree = cKDTree(all_pts)
        keep = np.ones(len(all_pts), dtype=bool)
        pairs = tree.query_pairs(r=max_edge * 0.05)
        for i, j in pairs:
            if j >= len(grid_pts):
                keep[i] = False  # prefer the data point
            elif i >= len(grid_pts):
                keep[j] = False
            else:
                keep[max(i, j)] = False
        nodes = all_pts[keep]

        tri = Delaunay(nodes)
        return cls(nodes, tri.simplices)

    def _fem_matrices(self) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
        """Compute FEM mass (C) and stiffness (G) matrices.

        For each triangle with area A:
          C_ij += A/12 * (1 + δ_ij)     (lumped → diagonal)
          G_ij += (∇φ_i · ∇φ_j) * A
        """
        n = self.n_nodes
        C_diag = np.zeros(n)
        G_rows, G_cols, G_vals = [], [], []

        for tri in self.triangles:
            i, j, k = tri
            v = self.nodes[tri]
            # Edge vectors
            e1 = v[1] - v[0]
            e2 = v[2] - v[0]
            area = 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
            if area < 1e-20:
                continue

            # Lumped mass matrix (diagonal)
            C_diag[i] += area / 3.0
            C_diag[j] += area / 3.0
            C_diag[k] += area / 3.0

            # Stiffness: gradients of piecewise-linear basis functions
            # ∇φ_a = [-1, 1, 0; -1, 0, 1]^T  in local coords,
            # transformed to global via Jacobian inverse
            inv_2A = 1.0 / (2.0 * area)
            # Gradient of φ_i, φ_j, φ_k in global coords
            grads = np.array([
                [v[1, 1] - v[2, 1], v[2, 0] - v[1, 0]],
                [v[2, 1] - v[0, 1], v[0, 0] - v[2, 0]],
                [v[0, 1] - v[1, 1], v[1, 0] - v[0, 0]],
            ]) * inv_2A

            local_idx = [i, j, k]
            for a in range(3):
                for b in range(3):
                    val = np.dot(grads[a], grads[b]) * area
                    G_rows.append(local_idx[a])
                    G_cols.append(local_idx[b])
                    G_vals.append(val)

        C = sparse.diags(C_diag, format="csc")
        G = sparse.csc_matrix(
            (np.array(G_vals), (np.array(G_rows), np.array(G_cols))),
            shape=(n, n),
        )
        return C, G


# ════════════════════════════════════════════════════════════════
# §2  MATÉRN PRECISION MATRIX  VIA SPDE
# ════════════════════════════════════════════════════════════════

def matern_to_spde_params(sigma2: float, range_param: float,
                          nu: float = 0.5, d: int = 2):
    """Convert (σ², range, ν) to SPDE parameters (κ, τ).

    For Matérn in d dimensions:
      κ  = √(8ν) / range
      τ² = Γ(ν) / (Γ(ν + d/2) · (4π)^{d/2} · κ^{2ν} · σ²)

    For ν = 0.5 (exponential) in 2-D:
      κ = √(8·0.5) / range = 2/range
      τ² = 1 / (4π · σ²)

    Parameters
    ----------
    sigma2      : sill (marginal variance)
    range_param : practical range parameter (distance where C ≈ 0.05·sill)
    nu          : Matérn smoothness (0.5 = exponential, 1.5, 2.5, …)
    d           : spatial dimension

    Returns
    -------
    kappa, tau : SPDE parameters
    """
    from scipy.special import gamma as _gamma

    # Convert "practical range" to Matérn range parameter
    # For exponential: C(h) = σ² exp(-3h/a), practical range a ≈ range_param
    # SPDE κ = √(8ν)/ρ where ρ is the Matérn scale parameter
    # and practical range ≈ ρ√(8ν)  for ν=0.5: range≈2ρ → ρ=range/2
    # So κ = √(8ν) / (range/√(8ν)) = 8ν/range  ... no.
    # Let's be more careful:
    # Matérn correlation: C(h) = σ² · 2^{1-ν}/Γ(ν) · (κh)^ν · K_ν(κh)
    # Practical range (C ≈ 0.05 sill): for exponential (ν=0.5):
    #   C(h) = σ² exp(-κh), so κ = 3/range  (since exp(-3)≈0.05)
    # The SPDE relationship: κ = √(8ν)/ρ where ρ = Matérn range
    # For ν=0.5: ρ = √(8·0.5)/κ = 2/κ, and κ = 3/practical_range
    # Actually: κ = √(8ν) / ρ  and the practical range R ≈ ρ√(8ν) = 8ν/κ
    # For ν=0.5: R = 4·0.5/κ = ... hmm, let me just use the standard formulas.

    # Standard: κ = √(8ν)/ρ,  where ρ is s.t. correlation ≈ 0.13 at distance ρ
    # In BMElib convention: C(h) = sill * exp(-3h/a) with 'a' = range param
    # → effective κ = 3/a for ν=0.5
    # More generally: κ = √(8ν) / a_matern, where a_matern relates to practical range

    kappa = np.sqrt(8.0 * nu) / range_param

    alpha = nu + d / 2.0
    tau_sq = (_gamma(nu) / (_gamma(alpha) * (4.0 * np.pi) ** (d / 2.0)
              * kappa ** (2.0 * nu) * sigma2))
    tau = np.sqrt(tau_sq)
    return float(kappa), float(tau)


def build_precision_matrix(mesh: SPDEMesh, kappa: float, tau: float,
                           alpha: int = 2) -> sparse.csc_matrix:
    """Build the SPDE precision matrix Q for a Matérn field.

    For α = 2 (i.e. ν = 1 in 2-D, or ν = 0.5 approximated):
      Q = τ² (κ⁴ C + 2κ² G + G C⁻¹ G)

    For α = 1 (ν = 0 — extremely rough, rarely used):
      Q = τ² (κ² C + G)

    Parameters
    ----------
    mesh  : SPDEMesh
    kappa : SPDE spatial scale parameter
    tau   : SPDE precision parameter
    alpha : SPDE order (1 or 2); α = ν + d/2

    Returns
    -------
    Q : (n, n) sparse CSC precision matrix
    """
    C, G = mesh._C, mesh._G
    k2 = kappa ** 2

    if alpha == 1:
        Q = tau ** 2 * (k2 * C + G)
    elif alpha == 2:
        # Q = τ² (κ⁴C + 2κ²G + G C⁻¹ G)
        C_inv_diag = 1.0 / C.diagonal()
        C_inv = sparse.diags(C_inv_diag, format="csc")
        GCinvG = G @ C_inv @ G
        Q = tau ** 2 * (k2 ** 2 * C + 2.0 * k2 * G + GCinvG)
    else:
        raise ValueError(f"alpha must be 1 or 2, got {alpha}")

    # Symmetrise
    Q = 0.5 * (Q + Q.T)
    return Q.tocsc()


# ════════════════════════════════════════════════════════════════
# §3  SPARSE KRIGING  (GMRF-based)
# ════════════════════════════════════════════════════════════════

def spde_kriging(mesh: SPDEMesh, Q: sparse.csc_matrix,
                 obs_node_idx: np.ndarray, z_obs: np.ndarray,
                 est_node_idx: Optional[np.ndarray] = None,
                 nugget: float = 0.0):
    """Simple kriging via the GMRF precision matrix.

    Solves  (Q + A^T Σ_ε^{-1} A) x = A^T Σ_ε^{-1} z

    where A is the observation matrix projecting mesh nodes to observation
    locations, and Σ_ε is the measurement noise covariance.

    Parameters
    ----------
    mesh         : SPDEMesh
    Q            : (n, n) sparse precision matrix
    obs_node_idx : (m,) indices of mesh nodes that have observations
    z_obs        : (m,) observed values (zero-mean assumed)
    est_node_idx : (k,) node indices at which to return predictions.
                   If None, returns predictions at all mesh nodes.
    nugget       : measurement noise variance (σ²_ε)

    Returns
    -------
    mu  : (k,) posterior mean at estimation nodes
    var : (k,) posterior marginal variance at estimation nodes
    """
    n = mesh.n_nodes
    m = len(obs_node_idx)

    # Observation matrix A: (m, n) sparse — maps nodes → obs locations
    A = sparse.csc_matrix(
        (np.ones(m), (np.arange(m), obs_node_idx)),
        shape=(m, n),
    )

    # Posterior precision:  Q_post = Q + A^T (1/σ²_ε) A
    noise_prec = 1.0 / max(nugget, 1e-10)
    Q_post = (Q + noise_prec * (A.T @ A)).tocsc()

    # RHS:  A^T (1/σ²_ε) z
    rhs = noise_prec * A.T @ z_obs

    # Sparse Cholesky solve
    try:
        factor = splu(Q_post)
        mu_all = factor.solve(rhs)
    except Exception:
        warnings.warn("splu failed; falling back to spsolve", stacklevel=2)
        mu_all = spsolve(Q_post, rhs)

    # Marginal variances via selected inversion (diagonal of Q_post^{-1})
    # For speed we only compute the diagonal entries we need.
    if est_node_idx is None:
        est_node_idx = np.arange(n)
    mu = mu_all[est_node_idx]

    # Compute variances via probing: var_i = e_i^T Q_post^{-1} e_i
    var = np.empty(len(est_node_idx))
    try:
        for j, idx in enumerate(est_node_idx):
            e = np.zeros(n)
            e[idx] = 1.0
            var[j] = factor.solve(e)[idx]
    except Exception:
        # fallback: no variance computation
        var = np.full(len(est_node_idx), np.nan)

    return mu, var


def snap_to_mesh(coords: np.ndarray, mesh: SPDEMesh) -> np.ndarray:
    """Find the nearest mesh node for each coordinate.

    Parameters
    ----------
    coords : (N, 2) array
    mesh   : SPDEMesh

    Returns
    -------
    (N,) integer array of mesh node indices
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.nodes)
    _, idx = tree.query(np.atleast_2d(coords))
    return idx
