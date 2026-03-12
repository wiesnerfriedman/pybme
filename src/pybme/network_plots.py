"""Plotting utilities for network-domain BME results.

All functions accept pre-computed arrays and return ``(fig, ax)`` or
``(fig, axes)`` tuples so callers can further customise or save.
Matplotlib is imported lazily — the module can be imported even if
matplotlib is not installed (functions raise at call time).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _import_plt():
    """Lazy-import matplotlib.pyplot; raise ImportError with a helpful message."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting.  "
                          "Install with:  pip install matplotlib")


def _draw_edges(ax, edges, coords, **kwargs):
    """Draw network edges as line segments.

    Parameters
    ----------
    ax     : matplotlib Axes
    edges  : list of (from_name, to_name, ...) tuples.
    coords : dict  node_name → (x, y).
    **kwargs : forwarded to ``ax.plot``.
    """
    kw = dict(color="grey", linewidth=0.3, alpha=0.4)
    kw.update(kwargs)
    for edge in edges:
        fn, tn = edge[0], edge[1]
        if fn in coords and tn in coords:
            x0, y0 = coords[fn]
            x1, y1 = coords[tn]
            ax.plot([x0, x1], [y0, y1], **kw)


# ════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════

def plot_network_observations(
    node_names: Sequence[str],
    coords: Dict[str, Tuple[float, float]],
    edges: Sequence[Tuple[str, str, float]],
    obs_nodes: Sequence[int],
    obs_values: np.ndarray,
    obs_labels: Optional[Sequence[str]] = None,
    *,
    title: str = "Network — Observation Stations",
    cmap: str = "YlOrRd",
    units: str = "",
    figsize: Tuple[float, float] = (10, 8),
    ax=None,
) -> Tuple:
    """Plot the network graph with observation stations coloured by value.

    Parameters
    ----------
    node_names : ordered list of *all* node names (index → name).
    coords     : dict mapping node name → (x, y).
    edges      : iterable of (from_name, to_name, ...) for drawing links.
    obs_nodes  : integer indices into *node_names* for observed nodes.
    obs_values : values at observed nodes (same length as *obs_nodes*).
    obs_labels : optional text labels for each observed node.
    title      : figure title.
    cmap       : colour-map name for observation markers.
    units      : label for the colour-bar (e.g. "MGD").
    figsize    : figure size if creating a new figure.
    ax         : optional existing Axes; if None a new figure is created.

    Returns
    -------
    (fig, ax)
    """
    plt = _import_plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _draw_edges(ax, edges, coords, alpha=0.5)

    # All nodes (faint)
    plot_x = [coords[node_names[i]][0] for i in range(len(node_names))
              if node_names[i] in coords]
    plot_y = [coords[node_names[i]][1] for i in range(len(node_names))
              if node_names[i] in coords]
    ax.scatter(plot_x, plot_y, s=3, c="lightgray", zorder=2,
               label=f"All nodes ({len(plot_x)})")

    # Observed stations
    ox = [coords[node_names[ni]][0] for ni in obs_nodes
          if node_names[ni] in coords]
    oy = [coords[node_names[ni]][1] for ni in obs_nodes
          if node_names[ni] in coords]
    ov = [obs_values[j] for j, ni in enumerate(obs_nodes)
          if node_names[ni] in coords]
    sc = ax.scatter(ox, oy, c=ov, cmap=cmap, s=80,
                    edgecolor="k", linewidth=0.5, zorder=5, vmin=0)
    cb_label = f"Observed ({units})" if units else "Observed"
    plt.colorbar(sc, ax=ax, label=cb_label, shrink=0.7)

    if obs_labels is not None:
        for j, ni in enumerate(obs_nodes):
            nn = node_names[ni]
            if nn in coords:
                cx, cy = coords[nn]
                ax.annotate(obs_labels[j], (cx, cy), fontsize=5.5,
                            xytext=(4, 4), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.15",
                                      fc="white", ec="gray", alpha=0.8))

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_network_field(
    node_names: Sequence[str],
    coords: Dict[str, Tuple[float, float]],
    edges: Sequence[Tuple[str, str, float]],
    values: np.ndarray,
    *,
    obs_nodes: Optional[Sequence[int]] = None,
    title: str = "",
    cmap: str = "YlOrRd",
    units: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmax_percentile: Optional[float] = 95,
    clip_label: bool = True,
    marker_size: float = 10,
    obs_marker: str = "^",
    obs_color: str = "blue",
    obs_label: str = "Meters",
    figsize: Tuple[float, float] = (10, 8),
    ax=None,
) -> Tuple:
    """Scatter plot of a scalar field on the network nodes.

    Suitable for BME posterior mean, std. dev., or any per-node quantity.

    Parameters
    ----------
    node_names  : ordered list of node names.
    coords      : dict node_name → (x, y).
    edges       : link list for drawing.
    values      : (n_nodes,) array — one value per node.
    obs_nodes   : optional integer indices of observed nodes (overlaid as markers).
    title       : figure title.
    cmap        : colour-map name.
    units       : colour-bar label.
    vmin, vmax  : explicit colour-scale bounds.  If *vmax* is None and
                  *vmax_percentile* is set, vmax is computed from data.
    vmax_percentile : percentile (0–100) used to cap colour scale
                      when *vmax* is None.  Prevents outlier compression.
    clip_label  : if True and vmax was clipped, annotate the colour-bar.
    marker_size : size of node dots.
    obs_marker  : marker style for observation overlay.
    obs_color   : colour for observation overlay markers.
    obs_label   : legend label for observation overlay.
    figsize     : figure size.
    ax          : optional existing Axes.

    Returns
    -------
    (fig, ax)
    """
    plt = _import_plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _draw_edges(ax, edges, coords)

    # map node index → plot coordinate
    plot_idx = [i for i in range(len(node_names)) if node_names[i] in coords]
    px = np.array([coords[node_names[i]][0] for i in plot_idx])
    py = np.array([coords[node_names[i]][1] for i in plot_idx])
    pv = values[np.array(plot_idx)]

    if vmax is None and vmax_percentile is not None:
        positive = pv[pv > 0] if (pv > 0).any() else pv
        vmax = float(np.percentile(positive, vmax_percentile))
    if vmin is None:
        vmin = 0

    sc = ax.scatter(px, py, c=pv, cmap=cmap, s=marker_size,
                    edgecolor="none", zorder=3, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sc, ax=ax, label=units, shrink=0.7)
    if clip_label and vmax is not None and (pv > vmax).any():
        cb.ax.set_title(f"clipped at\n{vmax:.2g}", fontsize=7)

    # Overlay observations
    if obs_nodes is not None:
        ox = [coords[node_names[ni]][0] for ni in obs_nodes
              if node_names[ni] in coords]
        oy = [coords[node_names[ni]][1] for ni in obs_nodes
              if node_names[ni] in coords]
        ax.scatter(ox, oy, c=obs_color, s=40, marker=obs_marker,
                   edgecolor="k", linewidth=0.5, zorder=5, label=obs_label)

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title(title)
    ax.set_aspect("equal")
    if obs_nodes is not None:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_network_correlation(
    node_names: Sequence[str],
    coords: Dict[str, Tuple[float, float]],
    edges: Sequence[Tuple[str, str, float]],
    net_cov,
    source_nodes: Sequence[int],
    source_labels: Optional[Sequence[str]] = None,
    *,
    cmap: str = "plasma",
    suptitle: str = "Network Correlation Structure",
    figsize: Tuple[float, float] = (18, 6),
) -> Tuple:
    """Multi-panel plot showing correlation footprint of selected nodes.

    Parameters
    ----------
    node_names    : ordered node names.
    coords        : dict node_name → (x, y).
    edges         : link list.
    net_cov       : ``NetworkCovariance`` instance (must have ``covariance_block``).
    source_nodes  : integer indices of nodes to visualise (one panel each).
    source_labels : optional panel titles; defaults to node names.

    Returns
    -------
    (fig, axes)
    """
    plt = _import_plt()
    n_panels = len(source_nodes)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize,
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    plot_idx = np.array([i for i in range(len(node_names))
                         if node_names[i] in coords])
    px = np.array([coords[node_names[i]][0] for i in plot_idx])
    py = np.array([coords[node_names[i]][1] for i in plot_idx])

    for k, (ax, ni) in enumerate(zip(axes, source_nodes)):
        corr_row = net_cov.covariance_block(np.array([ni]), plot_idx)[0, :]
        max_cov = max(abs(corr_row).max(), 1e-12)
        norm_corr = corr_row / max_cov

        _draw_edges(ax, edges, coords, linewidth=0.2, alpha=0.3)
        sc = ax.scatter(px, py, c=norm_corr, cmap=cmap, s=6,
                        edgecolor="none", zorder=3, vmin=0, vmax=1)

        nn = node_names[ni]
        if nn in coords:
            cx, cy = coords[nn]
            ax.scatter([cx], [cy], c="lime", s=100, marker="*",
                       edgecolor="k", linewidth=0.8, zorder=6)

        label = source_labels[k] if source_labels else nn
        ax.set_title(f"{label}\n({nn})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("Easting")
        if k == 0:
            ax.set_ylabel("Northing")

    plt.colorbar(sc, ax=list(axes), label="Normalised Correlation", shrink=0.7)
    fig.suptitle(suptitle, fontsize=12)
    return fig, axes


def plot_operator(
    net_cov,
    adjacency,
    kappa: float,
    *,
    obs_nodes: Optional[np.ndarray] = None,
    source_nodes: Optional[Sequence[int]] = None,
    source_labels: Optional[Sequence[str]] = None,
    suptitle: str = "Operator Visualisation — Graph-Laplacian Network Covariance",
    figsize: Tuple[float, float] = (14, 12),
    max_hops: int = 40,
) -> Tuple:
    """Four-panel operator visualisation.

    (a) Laplacian sparsity (RCM reordered)
    (b) Covariance matrix heatmap (RCM reordered)
    (c) Laplacian eigenvalue spectrum with κ² threshold
    (d) Covariance decay vs graph distance (hop count)

    Parameters
    ----------
    net_cov      : ``NetworkCovariance`` with ``.L``, ``.C_dense``.
    adjacency    : sparse adjacency matrix (used for hop-distance calculation).
    kappa        : spatial scale parameter (for annotation).
    obs_nodes    : integer indices of observed nodes (green lines on heatmap).
    source_nodes : nodes used for panel (d).  If None, uses *obs_nodes[:5]*.
    source_labels: legend labels for panel (d) lines.
    suptitle     : overall figure title.
    figsize      : figure size.
    max_hops     : x-axis limit for panel (d).

    Returns
    -------
    (fig, axes)  — axes is a 2×2 ndarray.
    """
    plt = _import_plt()
    from scipy.sparse.csgraph import shortest_path, reverse_cuthill_mckee

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    L_sparse = net_cov.L.tocsr()
    C_full = net_cov.C_dense

    # ── (a) Laplacian sparsity (RCM order) ──────────────────────
    ax_a = axes[0, 0]
    rcm = reverse_cuthill_mckee(L_sparse)
    L_rcm = L_sparse[rcm][:, rcm]
    ax_a.spy(L_rcm, markersize=0.3, color="navy", alpha=0.6)
    ax_a.set_title(f"(a) Graph Laplacian Sparsity Pattern\n"
                   f"(RCM reordered, {L_sparse.nnz} non-zeros)", fontsize=10)
    ax_a.set_xlabel("Node index (RCM order)")
    ax_a.set_ylabel("Node index (RCM order)")

    # ── (b) Covariance heatmap (RCM order) ──────────────────────
    ax_b = axes[0, 1]
    C_rcm = C_full[np.ix_(rcm, rcm)]
    vmax_c = np.percentile(np.abs(C_rcm), 95)
    im_b = ax_b.imshow(C_rcm, cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c,
                        aspect="equal", interpolation="nearest")
    plt.colorbar(im_b, ax=ax_b, shrink=0.7, label="Covariance")
    if obs_nodes is not None:
        obs_rcm = np.array([int(np.where(rcm == ni)[0][0]) for ni in obs_nodes])
        for p in obs_rcm:
            ax_b.axhline(p, color="lime", linewidth=0.3, alpha=0.5)
            ax_b.axvline(p, color="lime", linewidth=0.3, alpha=0.5)
    ax_b.set_title("(b) Covariance Matrix  C = σ²(κ²I + L)⁻¹\n"
                   "(RCM reordered; green = observed nodes)", fontsize=10)
    ax_b.set_xlabel("Node index (RCM order)")
    ax_b.set_ylabel("Node index (RCM order)")

    # ── (c) Eigenvalue spectrum ─────────────────────────────────
    ax_c = axes[1, 0]
    eigvals = np.linalg.eigvalsh(L_sparse.toarray())
    ax_c.semilogy(np.arange(1, len(eigvals) + 1),
                  np.maximum(eigvals, 1e-15), ".", markersize=3, color="darkblue")
    kappa2 = kappa ** 2
    ax_c.axhline(kappa2, color="red", linestyle="--", linewidth=1.2,
                 label=f"κ² = {kappa2:.4g}")
    ax_c.set_xlabel("Eigenvalue index")
    ax_c.set_ylabel("λ  (log scale)")
    ax_c.set_title("(c) Laplacian Eigenvalue Spectrum\n"
                   "Eigenvalues below κ² → long-range correlation", fontsize=10)
    ax_c.legend(fontsize=9)
    ax_c.set_xlim(0, len(eigvals))
    n_below = int(np.sum(eigvals < kappa2))
    ax_c.annotate(f"{n_below} eigenvalues < κ²",
                  xy=(n_below, kappa2), fontsize=8,
                  xytext=(n_below + 30, kappa2 * 5),
                  arrowprops=dict(arrowstyle="->", color="red"),
                  color="red")

    # ── (d) Covariance decay vs graph distance ──────────────────
    ax_d = axes[1, 1]
    dist_matrix = shortest_path(adjacency, directed=False, unweighted=True)

    if source_nodes is None:
        source_nodes = obs_nodes[:5] if obs_nodes is not None else []
    if source_labels is None:
        source_labels = [str(ni) for ni in source_nodes]

    n_lines = min(len(source_nodes), 8)
    colors_d = plt.cm.Set1(np.linspace(0, 0.8, max(n_lines, 1)))
    actual_max_hop = 0
    for j in range(n_lines):
        ni = source_nodes[j]
        dists = dist_matrix[ni, :]
        self_cov = C_full[ni, ni]
        if self_cov < 1e-15:
            continue
        corrs = C_full[ni, :] / self_cov
        finite_max = int(np.nanmax(dists[np.isfinite(dists)]))
        hop_vals, hop_means = [], []
        for h in range(finite_max + 1):
            mask = (dists == h) & np.isfinite(dists)
            if mask.any():
                hop_vals.append(h)
                hop_means.append(float(corrs[mask].mean()))
        ax_d.plot(hop_vals, hop_means, "o-", markersize=3, linewidth=1.2,
                  color=colors_d[j], label=source_labels[j], alpha=0.8)
        actual_max_hop = max(actual_max_hop, finite_max)

    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax_d.set_xlabel("Graph distance (hops)")
    ax_d.set_ylabel("Normalised correlation  C(i,j)/C(i,i)")
    ax_d.set_title(f"(d) Covariance Decay vs Graph Distance\n"
                   f"(Matérn ν=1 on graph, κ={kappa:.2f})", fontsize=10)
    ax_d.legend(fontsize=7, ncol=2, loc="upper right")
    ax_d.set_ylim(-0.1, 1.05)
    ax_d.set_xlim(-0.5, min(actual_max_hop, max_hops) + 0.5)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes
