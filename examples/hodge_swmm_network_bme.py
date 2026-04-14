#!/usr/bin/env python3
"""
Hodge-Laplacian BME on SWMM Network — Onondaga County
======================================================

Extends ``figure_network_bme_flowlines.py`` with a **time-varying
Hodge Laplacian** covariance driven by observed meter flows.  The
network is the Onondaga County SWMM model (~421 nodes) rendered in a
de-identified **tree layout** (BFS from the outfall).

The meter observations serve double duty:

* **Hard data** for the BME estimator (same as the existing examples).
* **Edge weight source** — at each time step the nearest observed flow
  at (or upstream of) each conduit sets the Hodge edge weight.  Edges
  with larger flow are more strongly correlated, so the graph operator
  adapts to wet- vs dry-weather conditions.

Comparison
----------
1. **Static baseline** — ``NetworkCovarianceST`` with frozen Laplacian.
2. **Hodge (time-varying)** — ``HodgeNetworkCovarianceST`` whose edge
   weights track the metered flows, producing a **non-separable**
   space-time covariance.
3. **Spectral Hodge** — ``SpectralHodgeNetworkCovarianceST`` with fixed
   eigenvectors from the reference Laplacian and Galerkin-projected
   time-varying eigenvalues.  Preserves persistent spectral similarity
   structure while adapting covariance to hydraulic state.

The script selects the 24 h window with the strongest meter coverage,
predicts at two representative timestamps (rising limb + peak), and
produces a 2×3 panel figure of de-identified tree-layout flowlines
(static vs Hodge × estimate / uncertainty / difference).

Data
----
Uses the private Onondaga SWMM model (not included in this repo).
Set environment variables or edit the paths below:

    PYBME_OC_DIR       base directory
    PYBME_OC_INP       .inp file
    PYBME_OC_OBS       ObservedTimeseries_converted_v2.csv
    PYBME_OC_METERS    wapug_inputs/MeterLocations.csv

Requirements: pybme (in project venv), numpy, scipy, matplotlib
"""

import csv
import os
import sys
import time as timer_mod
import warnings
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ── ensure pybme is importable ──────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path     = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.network import (
    NetworkCovariance, NetworkCovarianceST, adjacency_from_edges,
    build_mass_balance_operator, project_mass_balance,
)
from pybme.hodge import (
    build_oriented_incidence,
    hodge_decomposition,
    HodgeNetworkCovariance,
    HodgeNetworkCovarianceST,
    SpectralHodgeNetworkCovariance,
    SpectralHodgeNetworkCovarianceST,
)
from pybme.predict import bme_predict_network_st
from pybme.swmm import build_edge_array, parse_swmm_inp


# ═══════════════════════════════════════════════════════════════════════
# File paths  (same convention as figure_network_bme_flowlines.py)
# ═══════════════════════════════════════════════════════════════════════
OC_DIR = os.environ.get(
    "PYBME_OC_DIR",
    r"C:\Users\wiesnec\OneDrive - Jacobs Engineering Group Inc"
    r"\Documents\AutoCal_Projects\Onondaga",
)
INP_PATH = os.environ.get(
    "PYBME_OC_INP",
    os.path.join(OC_DIR,
                 "OC_2024-Conditions_5.1.010_V7-Calibrated_01282025.inp"),
)
OBS_PATH = os.environ.get(
    "PYBME_OC_OBS",
    os.path.join(OC_DIR, "ObservedTimeseries_converted_v2.csv"),
)
METER_LOC_PATH = os.environ.get(
    "PYBME_OC_METERS",
    os.path.join(OC_DIR, "wapug_inputs", "MeterLocations.csv"),
)

# ═══════════════════════════════════════════════════════════════════════
#  Tunable parameters
# ═══════════════════════════════════════════════════════════════════════
KAPPA = 0.1
ALPHA = 1.0
LAM   = 0.5
TEMPORAL_RANGE_HOURS = 6.0
WINDOW_HOURS   = 24.0
MAX_OBS        = 500
NHMAX          = 30
EDGE_FLOOR     = 0.05        # minimum normalised edge weight
SPECTRAL_ONLY  = True        # skip Static & Hodge; run Spectral + λ sweep


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_tree_layout(n_nodes, node_names, node_idx, edges, outfalls):
    """BFS tree layout for de-identified visualisation (root = outfall)."""
    adj = [[] for _ in range(n_nodes)]
    for fn, tn, *_ in edges:
        i, j = node_idx.get(fn, -1), node_idx.get(tn, -1)
        if i >= 0 and j >= 0:
            adj[i].append(j)
            adj[j].append(i)

    outfall_indices = [node_idx[n] for n in outfalls if n in node_idx]
    root = outfall_indices[0] if outfall_indices else max(
        range(n_nodes), key=lambda i: len(adj[i]))

    depth    = np.full(n_nodes, -1, dtype=int)
    parent   = np.full(n_nodes, -1, dtype=int)
    children = [[] for _ in range(n_nodes)]

    depth[root] = 0
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                parent[v] = u
                children[u].append(v)
                queue.append(v)

    # disconnected components
    for i in range(n_nodes):
        if depth[i] < 0:
            depth[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if depth[v] < 0:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        children[u].append(v)
                        queue.append(v)

    x_pos = np.zeros(n_nodes, dtype=float)
    leaf_counter = [0]

    def assign_x(node):
        if not children[node]:
            x_pos[node] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            for c in children[node]:
                assign_x(c)
            child_xs = [x_pos[c] for c in children[node]]
            x_pos[node] = 0.5 * (min(child_xs) + max(child_xs))

    roots = [i for i in range(n_nodes) if parent[i] < 0 or depth[i] == 0]
    for r in roots:
        if parent[r] < 0 or r == root:
            assign_x(r)

    y_pos = -depth.astype(float)

    tree_coords = {node_names[i]: (x_pos[i], y_pos[i]) for i in range(n_nodes)}
    tree_edges  = []
    for u in range(n_nodes):
        for c in children[u]:
            tree_edges.append((node_names[u], node_names[c]))

    return tree_coords, tree_edges, root, int(depth.max())


def _make_tree_flowlines(tree_edges, tree_coords, node_names, values,
                         cmap_name, vmin, vmax, linewidth=2.0):
    """LineCollection of tree edges coloured by endpoint-average value."""
    _idx = {n: i for i, n in enumerate(node_names)}
    segments, seg_vals, grey_segments = [], [], []

    for fn, tn in tree_edges:
        if fn not in tree_coords or tn not in tree_coords:
            continue
        x0, y0 = tree_coords[fn]
        x1, y1 = tree_coords[tn]
        i_from, i_to = _idx.get(fn, -1), _idx.get(tn, -1)
        if i_from < 0 or i_to < 0:
            grey_segments.append([(x0, y0), (x1, y1)])
            continue
        v_from, v_to = values[i_from], values[i_to]
        if np.isnan(v_from) or np.isnan(v_to):
            grey_segments.append([(x0, y0), (x1, y1)])
        else:
            segments.append([(x0, y0), (x1, y1)])
            seg_vals.append(0.5 * (v_from + v_to))

    norm = Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap_name, norm=norm,
                        linewidths=linewidth, capstyle="round", zorder=3)
    lc.set_array(np.array(seg_vals) if seg_vals else np.array([]))
    lc_grey = LineCollection(grey_segments, colors="lightgrey",
                             linewidths=0.4, alpha=0.35, zorder=1)
    return lc, lc_grey


def main() -> None:
    print("=" * 70)
    print("Hodge-Laplacian BME on SWMM Network — Onondaga County")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # 1. PARSE SWMM NETWORK
    # ═══════════════════════════════════════════════════════════════════
    print("\nParsing SWMM .inp ...")
    network    = parse_swmm_inp(INP_PATH)
    node_names = network.all_node_names
    n_nodes    = len(node_names)
    node_idx   = network.node_index
    edge_array = build_edge_array(node_names, network.edges)
    n_edges    = len(edge_array)
    print(f"  {n_nodes} nodes, {len(network.edges)} links "
          f"({n_edges} after dedup)")

    # ═══════════════════════════════════════════════════════════════════
    # 2. TREE LAYOUT FOR DE-IDENTIFICATION
    # ═══════════════════════════════════════════════════════════════════
    tree_coords, tree_edges, root, max_depth = _build_tree_layout(
        n_nodes, node_names, node_idx, network.edges, network.outfalls,
    )
    print(f"  Tree layout: root={node_names[root]}, max depth={max_depth}, "
          f"{len(tree_edges)} tree edges")

    # ═══════════════════════════════════════════════════════════════════
    # 3. BUILD ORIENTED INCIDENCE B₁
    # ═══════════════════════════════════════════════════════════════════
    B1 = build_oriented_incidence(n_nodes, edge_array)
    print(f"  Oriented incidence B₁: {B1.shape}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. READ METER LOCATIONS AND OBSERVATIONS
    # ═══════════════════════════════════════════════════════════════════
    meter_to_node = {}
    with open(METER_LOC_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meter_to_node[row["Meter"].strip()] = row["Node"].strip()

    print("Reading observations ...")
    with open(OBS_PATH, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    flow_col_info = []
    for ci, col_name in enumerate(header):
        if not col_name.endswith("_Flow"):
            continue
        meter_name = col_name.replace("_Flow", "")
        node_name  = meter_to_node.get(meter_name)
        if node_name and node_name in node_idx:
            flow_col_info.append((ci, meter_name, node_idx[node_name]))

    print(f"  {len(flow_col_info)} flow meters mapped to network nodes")

    datetimes = []
    all_obs_nodes, all_obs_times, all_obs_values = [], [], []
    t0_dt = None

    for row in rows:
        try:
            dt = datetime.strptime(row[0].strip(), "%m/%d/%Y %H:%M")
        except (ValueError, IndexError):
            continue
        if t0_dt is None:
            t0_dt = dt
        t_hours = (dt - t0_dt).total_seconds() / 3600.0
        datetimes.append(dt)

        for ci, meter_name, ni in flow_col_info:
            try:
                val = float(row[ci])
            except (ValueError, IndexError):
                continue
            if val < 0.001:
                continue
            all_obs_nodes.append(ni)
            all_obs_times.append(t_hours)
            all_obs_values.append(val)

    all_obs_nodes  = np.array(all_obs_nodes,  dtype=int)
    all_obs_times  = np.array(all_obs_times,  dtype=np.float64)
    all_obs_values = np.array(all_obs_values, dtype=np.float64)

    print(f"  {len(all_obs_values)} non-zero flow observations")
    print(f"  Time span: {datetimes[0]} to {datetimes[-1]}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. SELECT BEST 24 h WINDOW
    # ═══════════════════════════════════════════════════════════════════
    times_hours_all = np.array(
        [(dt - t0_dt).total_seconds() / 3600.0 for dt in datetimes])

    window_starts = np.arange(0, times_hours_all[-1] - WINDOW_HOURS, 1.0)
    best_score, best_start = 0.0, 0.0
    for ws in window_starts:
        mask = (all_obs_times >= ws) & (all_obs_times < ws + WINDOW_HOURS)
        if not mask.any():
            continue
        n_unique = len(np.unique(all_obs_nodes[mask]))
        vals = all_obs_values[mask]
        score = n_unique * 10 + np.ptp(vals) + np.max(vals) * 0.5
        if score > best_score:
            best_score, best_start = score, ws

    win_mask    = ((all_obs_times >= best_start) &
                   (all_obs_times < best_start + WINDOW_HOURS))
    ch_nodes_st = all_obs_nodes[win_mask]
    th_st       = all_obs_times[win_mask]
    zh_st       = all_obs_values[win_mask]

    # Thin if too many observations
    if len(zh_st) > MAX_OBS:
        step = len(zh_st) // MAX_OBS + 1
        keep = np.arange(0, len(zh_st), step)
        ch_nodes_st = ch_nodes_st[keep]
        th_st       = th_st[keep]
        zh_st       = zh_st[keep]

    start_dt = t0_dt + timedelta(hours=float(best_start))
    unique_obs_nodes = np.unique(ch_nodes_st)
    print(f"\n  Window: {start_dt} + {WINDOW_HOURS:.0f}h  |  "
          f"{len(zh_st)} obs, {len(unique_obs_nodes)} unique nodes")

    # ── Log-transform observations ───────────────────────────────────
    # BME in log-space: guarantees positive predictions and handles
    # the right-skewed flow distribution natural in combined sewers.
    LOG_FLOOR = 1e-4  # MGD – below instrument precision
    zh_log    = np.log(np.maximum(zh_st, LOG_FLOOR))
    data_var  = float(np.var(zh_log))
    data_mean = float(np.mean(zh_log))
    print(f"  Log-space:  mean(ln z)={data_mean:.3f}, var(ln z)={data_var:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 6. BUILD TIME-VARYING EDGE WEIGHTS FROM OBSERVED FLOWS
    # ═══════════════════════════════════════════════════════════════════
    # Strategy: at each observation time, compute a per-edge weight based
    # on the nearest-meter flow at the edge's upstream node.
    # Nodes without a nearby meter inherit a default (dry-weather) weight.
    #
    # Build a time grid at ~1 h resolution inside the window, snap
    # observed flows to nearest time step per node, then interpolate.

    t_grid = np.arange(best_start, best_start + WINDOW_HOURS + 0.01, 1.0)
    n_tgrid = len(t_grid)

    # Aggregate observed flow per node per grid hour (nearest-hour binning)
    node_flow_grid = np.full((n_nodes, n_tgrid), np.nan)
    for obs_i in range(len(zh_st)):
        ni = ch_nodes_st[obs_i]
        ti_rel = th_st[obs_i] - best_start
        ti_idx = int(np.clip(np.round(ti_rel), 0, n_tgrid - 1))
        # Keep the maximum if multiple observations fall in the same bin
        if np.isnan(node_flow_grid[ni, ti_idx]):
            node_flow_grid[ni, ti_idx] = zh_st[obs_i]
        else:
            node_flow_grid[ni, ti_idx] = max(
                node_flow_grid[ni, ti_idx], zh_st[obs_i])

    # For each edge (i→j), the edge weight = normalised upstream-node flow.
    # Where the upstream node has no observation, propagate the nearest
    # downstream/upstream meter value via BFS, then fall back to median.
    median_flow = float(np.nanmedian(zh_st)) if len(zh_st) else 1.0

    # Fill NaN by forward-/backward-filling in time for each metered node
    for ni in range(n_nodes):
        row = node_flow_grid[ni, :]
        if np.all(np.isnan(row)):
            continue
        # forward fill
        last = np.nan
        for k in range(n_tgrid):
            if np.isnan(row[k]):
                row[k] = last
            else:
                last = row[k]
        # backward fill remaining leading NaNs
        first = np.nan
        for k in range(n_tgrid - 1, -1, -1):
            if np.isnan(row[k]):
                row[k] = first
            else:
                first = row[k]

    # For entirely-unmetered nodes, use the median flow
    for ni in range(n_nodes):
        if np.all(np.isnan(node_flow_grid[ni, :])):
            node_flow_grid[ni, :] = median_flow

    # Edge weight = upstream-node flow, normalised per-edge to [FLOOR, 1]
    edge_flow_grid = node_flow_grid[edge_array[:, 0], :]  # (n_edges, n_tgrid)
    edge_max = np.nanmax(edge_flow_grid, axis=1, keepdims=True)
    edge_max = np.where(edge_max > 1e-12, edge_max, 1.0)
    edge_weight_grid = EDGE_FLOOR + (1.0 - EDGE_FLOOR) * edge_flow_grid / edge_max

    print(f"  Edge weight range: [{np.nanmin(edge_weight_grid):.3f}, "
          f"{np.nanmax(edge_weight_grid):.3f}]")

    def edge_weight_func(t: float) -> np.ndarray:
        """Interpolate edge weights from the metered-flow grid."""
        t = float(np.asarray(t).ravel()[0]) if np.ndim(t) > 0 else float(t)
        t_rel = np.clip(t - best_start, 0.0, WINDOW_HOURS)
        weights = np.empty(n_edges)
        for e in range(n_edges):
            weights[e] = np.interp(t_rel, t_grid - best_start,
                                   edge_weight_grid[e, :])
        return weights

    # ═══════════════════════════════════════════════════════════════════
    # 7. BUILD COVARIANCE MODELS
    # ═══════════════════════════════════════════════════════════════════
    W = adjacency_from_edges(n_nodes, edge_array)

    # Rescale sigma2 so that the marginal variance matches data variance
    net_cov_init = NetworkCovariance(
        W, kappa=KAPPA, sigma2=1.0, from_adjacency=True)
    diag_at_obs = net_cov_init.marginal_variance(unique_obs_nodes)
    scale = data_var / max(diag_at_obs.mean(), 1e-12) if data_var > 0 else 1.0

    TEMPORAL_MODEL  = "exponential"
    TEMPORAL_PARAMS = [1.0, TEMPORAL_RANGE_HOURS]

    if not SPECTRAL_ONLY:
        # ── 7a. Static baseline ──────────────────────────────────────
        net_cov = NetworkCovariance(
            W, kappa=KAPPA, sigma2=scale, from_adjacency=True)
        net_cov_st = NetworkCovarianceST(
            net_cov,
            model_t=TEMPORAL_MODEL,
            params_t=TEMPORAL_PARAMS,
            sigma2=scale,
        )

        # ── 7b. Hodge (time-varying) ─────────────────────────────────
        hodge_cov = HodgeNetworkCovariance(
            B1=B1,
            directed_edges=edge_array,
            edge_weight_func=edge_weight_func,
            kappa=KAPPA,
            sigma2=scale,
            alpha=ALPHA,
            lam=LAM,
        )
        hodge_cov_st = HodgeNetworkCovarianceST(
            hodge_cov,
            model_t=TEMPORAL_MODEL,
            params_t=TEMPORAL_PARAMS,
            sigma2=scale,
            blend="geometric",
        )

    # ── 7c. Spectral Hodge (fixed eigenvectors + time-varying λ) ─────
    spectral_cov = SpectralHodgeNetworkCovariance(
        B1=B1,
        directed_edges=edge_array,
        edge_weight_func=edge_weight_func,
        kappa=KAPPA,
        sigma2=scale,
        alpha=ALPHA,
        lam=LAM,
    )
    spectral_cov_st = SpectralHodgeNetworkCovarianceST(
        spectral_cov,
        model_t=TEMPORAL_MODEL,
        params_t=TEMPORAL_PARAMS,
        sigma2=scale,
        blend="geometric",
    )

    print(f"\n  sigma2={scale:.4f}  kappa={KAPPA}  alpha={ALPHA}  lam={LAM}")
    print(f"  Temporal: {TEMPORAL_MODEL}, range={TEMPORAL_RANGE_HOURS}h")

    # ═══════════════════════════════════════════════════════════════════
    # 8. SELECT TWO REPRESENTATIVE TIMESTAMPS AND PREDICT
    # ═══════════════════════════════════════════════════════════════════
    tk_all = np.arange(best_start, best_start + WINDOW_HOURS + 0.01, 1.0)

    obs_by_hour = {}
    for t_val, z_val in zip(th_st, zh_st):
        hr = int(round(t_val))
        obs_by_hour.setdefault(hr, []).append(z_val)

    peak_hour = max(obs_by_hour, key=lambda h: np.mean(obs_by_hour[h]))
    overall_median = float(np.median(zh_st))
    active_hours = sorted(
        h for h, vs in obs_by_hour.items()
        if np.mean(vs) > 2.0 * max(overall_median, 0.01))
    if active_hours and len(active_hours) >= 3:
        rising_hour = active_hours[max(1, len(active_hours) // 3)]
    else:
        rising_hour = peak_hour - 3
    if abs(rising_hour - peak_hour) < 2:
        rising_hour = peak_hour - 3

    snap = lambda target: tk_all[np.argmin(np.abs(tk_all - target))]
    t_rising = snap(rising_hour)
    t_peak   = snap(peak_hour)
    if t_rising == t_peak:
        t_rising = snap(peak_hour - 3)
    timestamps = np.sort(np.array([t_rising, t_peak]))

    pred_nodes = np.arange(n_nodes)
    n_pred     = n_nodes
    ck_flat = np.repeat(pred_nodes, 2)
    tk_flat = np.tile(timestamps, n_pred)

    time_labels = [(t0_dt + timedelta(hours=float(t))).strftime("%b %d %H:%M")
                   for t in timestamps]

    print(f"\n  Timestamps: rising={time_labels[0]}, peak={time_labels[1]}")
    print(f"  Predicting at {n_pred} nodes × 2 times = {n_pred * 2} points")

    all_unique_times = np.unique(np.concatenate([timestamps, th_st]))

    if not SPECTRAL_ONLY:
        # ── Static ────────────────────────────────────────────────────
        print("\n── Static (separable) prediction  [log-space] ──")
        t0 = timer_mod.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_static = bme_predict_network_st(
                ck_nodes=ck_flat, tk=tk_flat,
                ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
                net_cov_st=net_cov_st, nhmax=NHMAX,
                order=0, mean_prior=data_mean,
            )
        t_static = timer_mod.time() - t0
        print(f"  Done in {t_static:.1f}s")

        # ── Hodge ────────────────────────────────────────────────────
        print(f"\n  Pre-computing dense C(t) for {len(all_unique_times)} "
              f"unique time steps ...")
        t0 = timer_mod.time()
        hodge_cov.precompute_dense(all_unique_times)
        t_precompute = timer_mod.time() - t0
        print(f"  Dense precompute done in {t_precompute:.1f}s")

        print("\n── Hodge (non-separable) prediction  [log-space] ──")
        t0 = timer_mod.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_hodge = bme_predict_network_st(
                ck_nodes=ck_flat, tk=tk_flat,
                ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
                net_cov_st=hodge_cov_st, nhmax=NHMAX,
                order=0, mean_prior=data_mean,
            )
        t_hodge = timer_mod.time() - t0
        print(f"  Done in {t_hodge:.1f}s")

    # ── Spectral ─────────────────────────────────────────────────────
    print(f"\n  Pre-computing spectral dense C(t) for "
          f"{len(all_unique_times)} unique time steps ...")
    t0 = timer_mod.time()
    spectral_cov.precompute_dense(all_unique_times)
    t_precompute_sp = timer_mod.time() - t0
    print(f"  Spectral precompute done in {t_precompute_sp:.1f}s")

    print("\n── Spectral Hodge prediction  [log-space] ──")
    t0 = timer_mod.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_spectral = bme_predict_network_st(
            ck_nodes=ck_flat, tk=tk_flat,
            ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
            net_cov_st=spectral_cov_st, nhmax=NHMAX,
            order=0, mean_prior=data_mean,
        )
    t_spectral = timer_mod.time() - t0
    print(f"  Done in {t_spectral:.1f}s")

    # ── Back-transform from log-space to original MGD ────────────────
    # Log-normal: if ln(Z) ~ N(mu, s2) then
    #   E[Z] = exp(mu + s2/2)  (unbiased mean)
    #   Median[Z] = exp(mu)
    #   CI = [exp(mu - 1.96*s), exp(mu + 1.96*s)]
    log_mean_sp = np.array([r.mean for r in res_spectral]).reshape(n_pred, 2)
    log_var_sp  = np.array([r.variance for r in res_spectral]).reshape(n_pred, 2)
    spectral_mean = np.exp(log_mean_sp + log_var_sp / 2.0)
    spectral_std  = np.sqrt(log_var_sp)
    print(f"  Spectral mean range: [{spectral_mean.min():.4f}, "
          f"{spectral_mean.max():.4f}]  (back-transformed MGD)")

    if not SPECTRAL_ONLY:
        log_mean_s = np.array([r.mean for r in res_static]).reshape(n_pred, 2)
        log_var_s  = np.array([r.variance for r in res_static]).reshape(n_pred, 2)
        log_mean_h = np.array([r.mean for r in res_hodge]).reshape(n_pred, 2)
        log_var_h  = np.array([r.variance for r in res_hodge]).reshape(n_pred, 2)
        static_mean   = np.exp(log_mean_s + log_var_s / 2.0)
        hodge_mean    = np.exp(log_mean_h + log_var_h / 2.0)
        static_std   = np.sqrt(log_var_s)
        hodge_std    = np.sqrt(log_var_h)
        print(f"  Static   mean range: [{static_mean.min():.4f}, "
              f"{static_mean.max():.4f}]  (back-transformed MGD)")
        print(f"  Hodge    mean range: [{hodge_mean.min():.4f}, "
              f"{hodge_mean.max():.4f}]  (back-transformed MGD)")

    # ═══════════════════════════════════════════════════════════════════
    # 8b. TIME-SERIES PREDICTIONS AT METERED NODES (hourly)
    # ═══════════════════════════════════════════════════════════════════
    # Build reverse map: node index → meter name
    node_to_meter = {}
    for _, meter_name, ni in flow_col_info:
        node_to_meter[ni] = meter_name

    ts_nodes = unique_obs_nodes                        # (N_meters,)
    ts_times = tk_all                                  # hourly grid
    n_ts_nodes = len(ts_nodes)
    n_ts_times = len(ts_times)
    n_ts_pts   = n_ts_nodes * n_ts_times

    ck_ts = np.repeat(ts_nodes, n_ts_times)
    tk_ts = np.tile(ts_times, n_ts_nodes)

    print(f"\n── Time-series prediction at {n_ts_nodes} metered nodes "
          f"× {n_ts_times} hours = {n_ts_pts} points ──")

    if not SPECTRAL_ONLY:
        # Static  [log-space]
        t0 = timer_mod.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_ts_static = bme_predict_network_st(
                ck_nodes=ck_ts, tk=tk_ts,
                ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
                net_cov_st=net_cov_st, nhmax=NHMAX,
                order=0, mean_prior=data_mean,
            )
        print(f"  Static: {timer_mod.time() - t0:.1f}s")

        # Hodge  [log-space] (dense cache already pre-computed)
        t0 = timer_mod.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_ts_hodge = bme_predict_network_st(
                ck_nodes=ck_ts, tk=tk_ts,
                ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
                net_cov_st=hodge_cov_st, nhmax=NHMAX,
                order=0, mean_prior=data_mean,
            )
        print(f"  Hodge:    {timer_mod.time() - t0:.1f}s")

    # Spectral  [log-space] (dense cache already pre-computed)
    t0 = timer_mod.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_ts_spectral = bme_predict_network_st(
            ck_nodes=ck_ts, tk=tk_ts,
            ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
            net_cov_st=spectral_cov_st, nhmax=NHMAX,
            order=0, mean_prior=data_mean,
        )
    print(f"  Spectral: {timer_mod.time() - t0:.1f}s")

    # Back-transform spectral time series from log-space
    ts_log_mean_sp = np.array([r.mean for r in res_ts_spectral]).reshape(
        n_ts_nodes, n_ts_times)
    ts_log_var_sp  = np.array([r.variance for r in res_ts_spectral]).reshape(
        n_ts_nodes, n_ts_times)
    ts_log_std_sp = np.sqrt(np.maximum(ts_log_var_sp, 0.0))
    ts_spectral_mean = np.exp(ts_log_mean_sp + ts_log_var_sp / 2.0)
    ts_spectral_lo = np.exp(ts_log_mean_sp - 1.96 * ts_log_std_sp)
    ts_spectral_hi = np.exp(ts_log_mean_sp + 1.96 * ts_log_std_sp)
    ts_spectral_std = ts_spectral_mean * np.sqrt(np.exp(ts_log_var_sp) - 1.0)
    ts_datetimes = [t0_dt + timedelta(hours=float(t)) for t in ts_times]

    if not SPECTRAL_ONLY:
        ts_log_mean_s = np.array([r.mean for r in res_ts_static]).reshape(
            n_ts_nodes, n_ts_times)
        ts_log_var_s  = np.array([r.variance for r in res_ts_static]).reshape(
            n_ts_nodes, n_ts_times)
        ts_log_mean_h = np.array([r.mean for r in res_ts_hodge]).reshape(
            n_ts_nodes, n_ts_times)
        ts_log_var_h  = np.array([r.variance for r in res_ts_hodge]).reshape(
            n_ts_nodes, n_ts_times)
        ts_log_std_s  = np.sqrt(np.maximum(ts_log_var_s, 0.0))
        ts_log_std_h  = np.sqrt(np.maximum(ts_log_var_h, 0.0))
        ts_static_mean   = np.exp(ts_log_mean_s + ts_log_var_s / 2.0)
        ts_hodge_mean    = np.exp(ts_log_mean_h + ts_log_var_h / 2.0)
        ts_static_lo   = np.exp(ts_log_mean_s - 1.96 * ts_log_std_s)
        ts_static_hi   = np.exp(ts_log_mean_s + 1.96 * ts_log_std_s)
        ts_hodge_lo    = np.exp(ts_log_mean_h - 1.96 * ts_log_std_h)
        ts_hodge_hi    = np.exp(ts_log_mean_h + 1.96 * ts_log_std_h)
        ts_static_std   = ts_static_mean * np.sqrt(np.exp(ts_log_var_s) - 1.0)
        ts_hodge_std    = ts_hodge_mean  * np.sqrt(np.exp(ts_log_var_h) - 1.0)

    # ── Per-meter diagnostics ────────────────────────────────────────
    if not SPECTRAL_ONLY:
        print("\n── Per-meter diagnostics (log-space BME, back-transformed) ──")
        print(f"  {'Meter':<16s} {'RMSE_s':>8s} {'RMSE_h':>8s} {'RMSE_sp':>8s} "
              f"{'Δ_h':>8s} {'Δ_sp':>8s} {'Cov_s':>6s} {'Cov_h':>6s} {'Cov_sp':>6s}")
        print("  " + "-" * 100)
        for m_idx, ni in enumerate(ts_nodes):
            meter_label = node_to_meter.get(ni, node_names[ni])
            obs_mask = ch_nodes_st == ni
            obs_t_m = th_st[obs_mask]
            obs_z_m = zh_st[obs_mask]
            pred_s  = np.interp(obs_t_m, ts_times, ts_static_mean[m_idx])
            pred_h  = np.interp(obs_t_m, ts_times, ts_hodge_mean[m_idx])
            pred_sp = np.interp(obs_t_m, ts_times, ts_spectral_mean[m_idx])
            rmse_s  = float(np.sqrt(np.mean((pred_s - obs_z_m) ** 2)))
            rmse_h  = float(np.sqrt(np.mean((pred_h - obs_z_m) ** 2)))
            rmse_sp = float(np.sqrt(np.mean((pred_sp - obs_z_m) ** 2)))
            lo_s  = np.interp(obs_t_m, ts_times, ts_static_lo[m_idx])
            hi_s  = np.interp(obs_t_m, ts_times, ts_static_hi[m_idx])
            lo_h  = np.interp(obs_t_m, ts_times, ts_hodge_lo[m_idx])
            hi_h  = np.interp(obs_t_m, ts_times, ts_hodge_hi[m_idx])
            lo_sp = np.interp(obs_t_m, ts_times, ts_spectral_lo[m_idx])
            hi_sp = np.interp(obs_t_m, ts_times, ts_spectral_hi[m_idx])
            in_band_s  = (obs_z_m >= lo_s) & (obs_z_m <= hi_s)
            in_band_h  = (obs_z_m >= lo_h) & (obs_z_m <= hi_h)
            in_band_sp = (obs_z_m >= lo_sp) & (obs_z_m <= hi_sp)
            cov_s  = float(in_band_s.mean()) * 100
            cov_h  = float(in_band_h.mean()) * 100
            cov_sp = float(in_band_sp.mean()) * 100
            delta_h  = rmse_h - rmse_s
            delta_sp = rmse_sp - rmse_s
            print(f"  {meter_label:<16s} {rmse_s:8.3f} {rmse_h:8.3f} {rmse_sp:8.3f} "
                  f"{delta_h:+8.3f} {delta_sp:+8.3f} {cov_s:5.1f}% {cov_h:5.1f}% {cov_sp:5.1f}%")
        all_pred_s, all_pred_h, all_pred_sp, all_obs_z = [], [], [], []
        for m_idx, ni in enumerate(ts_nodes):
            obs_mask = ch_nodes_st == ni
            obs_t_m = th_st[obs_mask]
            obs_z_m = zh_st[obs_mask]
            all_pred_s.append(np.interp(obs_t_m, ts_times, ts_static_mean[m_idx]))
            all_pred_h.append(np.interp(obs_t_m, ts_times, ts_hodge_mean[m_idx]))
            all_pred_sp.append(np.interp(obs_t_m, ts_times, ts_spectral_mean[m_idx]))
            all_obs_z.append(obs_z_m)
        all_pred_s  = np.concatenate(all_pred_s)
        all_pred_h  = np.concatenate(all_pred_h)
        all_pred_sp = np.concatenate(all_pred_sp)
        all_obs_z   = np.concatenate(all_obs_z)
        rmse_all_s  = float(np.sqrt(np.mean((all_pred_s - all_obs_z) ** 2)))
        rmse_all_h  = float(np.sqrt(np.mean((all_pred_h - all_obs_z) ** 2)))
        rmse_all_sp = float(np.sqrt(np.mean((all_pred_sp - all_obs_z) ** 2)))
        print(f"\n  {'AGGREGATE':<16s} {rmse_all_s:8.3f} {rmse_all_h:8.3f} {rmse_all_sp:8.3f} "
              f"{rmse_all_h - rmse_all_s:+8.3f} {rmse_all_sp - rmse_all_s:+8.3f}")
        print(f"\n  Cov% = fraction of observations within log-normal 95% CI (target: 95%).")
        peak_col = 1
        cv_static   = float(np.std(static_std[:, peak_col])
                            / max(np.mean(static_std[:, peak_col]), 1e-12))
        cv_hodge    = float(np.std(hodge_std[:, peak_col])
                            / max(np.mean(hodge_std[:, peak_col]), 1e-12))
        cv_spectral = float(np.std(spectral_std[:, peak_col])
                            / max(np.mean(spectral_std[:, peak_col]), 1e-12))
        print(f"\n  Std-dev heterogeneity (CV of posterior σ at peak):")
        print(f"    Static:   {cv_static:.4f}")
        print(f"    Hodge:    {cv_hodge:.4f}")
        print(f"    Spectral: {cv_spectral:.4f}")
        print(f"  (Higher CV = more differentiated uncertainty across the network)")
    else:
        # Spectral-only per-meter diagnostics
        print("\n── Per-meter diagnostics (Spectral only, back-transformed) ──")
        print(f"  {'Meter':<16s} {'RMSE_sp':>8s} {'Cov_sp':>6s}")
        print("  " + "-" * 30)
        all_pred_sp, all_obs_z = [], []
        for m_idx, ni in enumerate(ts_nodes):
            meter_label = node_to_meter.get(ni, node_names[ni])
            obs_mask = ch_nodes_st == ni
            obs_t_m = th_st[obs_mask]
            obs_z_m = zh_st[obs_mask]
            pred_sp = np.interp(obs_t_m, ts_times, ts_spectral_mean[m_idx])
            rmse_sp = float(np.sqrt(np.mean((pred_sp - obs_z_m) ** 2)))
            lo_sp = np.interp(obs_t_m, ts_times, ts_spectral_lo[m_idx])
            hi_sp = np.interp(obs_t_m, ts_times, ts_spectral_hi[m_idx])
            cov_sp = float(np.mean((obs_z_m >= lo_sp) & (obs_z_m <= hi_sp))) * 100
            print(f"  {meter_label:<16s} {rmse_sp:8.3f} {cov_sp:5.1f}%")
            all_pred_sp.append(pred_sp)
            all_obs_z.append(obs_z_m)
        all_pred_sp = np.concatenate(all_pred_sp)
        all_obs_z   = np.concatenate(all_obs_z)
        rmse_all_sp = float(np.sqrt(np.mean((all_pred_sp - all_obs_z) ** 2)))
        print(f"\n  {'AGGREGATE':<16s} {rmse_all_sp:8.3f}")
        peak_col = 1
        cv_spectral = float(np.std(spectral_std[:, peak_col])
                            / max(np.mean(spectral_std[:, peak_col]), 1e-12))
        print(f"\n  Std-dev heterogeneity (CV of posterior σ at peak): {cv_spectral:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # 9. HODGE DECOMPOSITION OF OBSERVED EDGE FLOWS AT PEAK
    # ═══════════════════════════════════════════════════════════════════
    peak_weights = edge_weight_func(float(timestamps[1]))
    decomp = hodge_decomposition(B1, peak_weights)
    grad_energy  = float(np.sum(decomp["gradient"] ** 2))
    harm_energy  = float(np.sum(decomp["harmonic"] ** 2))
    total_energy = float(np.sum(peak_weights ** 2))
    print(f"\n  Hodge decomposition at peak:")
    print(f"    Total energy:    {total_energy:.4f}")
    print(f"    Gradient:        {grad_energy:.4f} "
          f"({100*grad_energy/max(total_energy,1e-12):.1f}%)")
    print(f"    Harmonic:        {harm_energy:.4f} "
          f"({100*harm_energy/max(total_energy,1e-12):.1f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # 10. FIGURES — de-identified tree-layout flowlines
    # ═══════════════════════════════════════════════════════════════════
    out_dir = os.path.join(script_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # ── Figures A, B, D require all 3 methods ────────────────────────
    if SPECTRAL_ONLY:
        print("\n  [SPECTRAL_ONLY] Skipping Figures A, B, D (3-way comparison)")

    if not SPECTRAL_ONLY:
        # ── Figure A: 3×3 panel (rows = static / Hodge / Spectral,
        #              cols = estimate, uncertainty, |difference from static|)
        col_idx = 1  # peak
        vmax_m = max(
            float(np.percentile(static_mean[:, col_idx][static_mean[:, col_idx] > 0], 95))
            if (static_mean[:, col_idx] > 0).any() else 1.0,
            float(np.percentile(hodge_mean[:, col_idx][hodge_mean[:, col_idx] > 0], 95))
            if (hodge_mean[:, col_idx] > 0).any() else 1.0,
            float(np.percentile(spectral_mean[:, col_idx][spectral_mean[:, col_idx] > 0], 95))
            if (spectral_mean[:, col_idx] > 0).any() else 1.0,
        )
        vmax_s = max(
            float(np.percentile(static_std[:, col_idx], 95)),
            float(np.percentile(hodge_std[:, col_idx], 95)),
            float(np.percentile(spectral_std[:, col_idx], 95)),
        )
        diff_hodge = np.abs(hodge_mean[:, col_idx] - static_mean[:, col_idx])
        diff_spectral = np.abs(spectral_mean[:, col_idx] - static_mean[:, col_idx])
        vmax_d = max(
            float(np.percentile(diff_hodge[diff_hodge > 0], 95)) if (diff_hodge > 0).any() else 0.5,
            float(np.percentile(diff_spectral[diff_spectral > 0], 95)) if (diff_spectral > 0).any() else 0.5,
        )
    
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        row_tags = ["Static", "Hodge", "Spectral"]
        col_tags = ["BME Estimate", "Posterior Std. Dev.", "|Method − Static|"]
        cmaps    = ["YlOrRd", "Blues", "PuRd"]
        vmaxs    = [vmax_m, vmax_s, vmax_d]
        panel_id = [["(a)", "(b)", "(c)"],
                    ["(d)", "(e)", "(f)"],
                    ["(g)", "(h)", "(i)"]]
    
        diff_by_row = [np.zeros(n_pred), diff_hodge, diff_spectral]
    
        for row, (row_tag, mean_arr, std_arr, diff_arr) in enumerate(zip(
            row_tags,
            [static_mean, hodge_mean, spectral_mean],
            [static_std, hodge_std, spectral_std],
            diff_by_row,
        )):
            for col, (col_tag, cmap, vm) in enumerate(zip(col_tags, cmaps, vmaxs)):
                ax = axes[row, col]
                if col == 0:
                    vals = mean_arr[:, col_idx]
                elif col == 1:
                    vals = std_arr[:, col_idx]
                else:
                    vals = diff_arr
    
                lc, lc_g = _make_tree_flowlines(
                    tree_edges, tree_coords, node_names,
                    vals, cmap, 0, vm, linewidth=2.2)
                ax.add_collection(lc_g)
                ax.add_collection(lc)
    
                # Overlay meter locations active near the peak
                obs_at_t = np.abs(th_st - timestamps[col_idx]) < 0.5
                if obs_at_t.any() and col < 2:
                    obs_ni = np.unique(ch_nodes_st[obs_at_t])
                    ox = [tree_coords[node_names[ni]][0] for ni in obs_ni
                          if node_names[ni] in tree_coords]
                    oy = [tree_coords[node_names[ni]][1] for ni in obs_ni
                          if node_names[ni] in tree_coords]
                    ax.scatter(ox, oy, marker="^", c="white", edgecolor="black",
                               s=40, linewidths=0.7, zorder=5)
    
                ax.set_aspect("auto")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_title(
                    f"{panel_id[row][col]}  {row_tag} — {col_tag}",
                    fontsize=10, fontweight="bold", loc="left")
                ax.autoscale_view()
    
        # Colour bars
        for col, (cmap, vm, label) in enumerate(zip(
            cmaps, vmaxs,
            ["Flow (MGD)", "Std. Dev. (MGD)", "|Δ| (MGD)"],
        )):
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, vm))
            sm.set_array([])
            for row in range(3):
                if col == 2 and row == 0:
                    continue  # static diff is zero — skip colourbar
                cb = fig.colorbar(sm, ax=axes[row, col], location="right",
                                  shrink=0.75, pad=0.02, aspect=25)
                cb.set_label(label, fontsize=8)
    
        fig.suptitle(
            f"Static vs Hodge vs Spectral BME at Peak ({time_labels[1]})",
            fontsize=14, fontweight="bold", y=0.99)
    
        fname_a = os.path.join(out_dir, "hodge_A_tree_comparison.png")
        fig.savefig(fname_a, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"\nSaved {os.path.basename(fname_a)}")
        plt.close(fig)
    
        # ── Figure B: 3×2 panel — rising + peak × static / Hodge / Spectral
        fig2, axes2 = plt.subplots(3, 2, figsize=(14, 14))
        for col_t in range(2):
            vmax_col = max(
                float(np.percentile(
                    static_mean[:, col_t][static_mean[:, col_t] > 0], 95))
                if (static_mean[:, col_t] > 0).any() else 1.0,
                float(np.percentile(
                    hodge_mean[:, col_t][hodge_mean[:, col_t] > 0], 95))
                if (hodge_mean[:, col_t] > 0).any() else 1.0,
                float(np.percentile(
                    spectral_mean[:, col_t][spectral_mean[:, col_t] > 0], 95))
                if (spectral_mean[:, col_t] > 0).any() else 1.0,
            )
    
            for row, (tag, arr) in enumerate(zip(
                ["Static", "Hodge", "Spectral"],
                [static_mean, hodge_mean, spectral_mean],
            )):
                ax = axes2[row, col_t]
                lc, lc_g = _make_tree_flowlines(
                    tree_edges, tree_coords, node_names,
                    arr[:, col_t], "YlOrRd", 0, vmax_col, linewidth=2.2)
                ax.add_collection(lc_g)
                ax.add_collection(lc)
    
                obs_at = np.abs(th_st - timestamps[col_t]) < 0.5
                if obs_at.any():
                    obs_ni = np.unique(ch_nodes_st[obs_at])
                    ox = [tree_coords[node_names[ni]][0] for ni in obs_ni
                          if node_names[ni] in tree_coords]
                    oy = [tree_coords[node_names[ni]][1] for ni in obs_ni
                          if node_names[ni] in tree_coords]
                    ax.scatter(ox, oy, marker="^", c="white", edgecolor="black",
                               s=40, linewidths=0.7, zorder=5)
    
                ax.set_aspect("auto")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_title(f"{tag} — {time_labels[col_t]}",
                             fontsize=10, fontweight="bold", loc="left")
                ax.autoscale_view()
    
            sm = plt.cm.ScalarMappable(
                cmap="YlOrRd", norm=Normalize(0, vmax_col))
            sm.set_array([])
            for row in range(3):
                cb = fig2.colorbar(sm, ax=axes2[row, col_t], location="right",
                                   shrink=0.75, pad=0.02, aspect=25)
                cb.set_label("Flow (MGD)", fontsize=8)
    
        fig2.suptitle(
            "Static vs Hodge vs Spectral BME — Rising Limb & Peak",
            fontsize=14, fontweight="bold", y=0.99)
        fname_b = os.path.join(out_dir, "hodge_B_tree_rising_peak.png")
        fig2.savefig(fname_b, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved {os.path.basename(fname_b)}")
        plt.close(fig2)
    
        # ── Figure D: Time series — Static vs Hodge vs Spectral ± 1.96σ ──
        n_cols_ts = min(3, n_ts_nodes)
        n_rows_ts = int(np.ceil(n_ts_nodes / n_cols_ts))
        fig4, axes4 = plt.subplots(
            n_rows_ts, n_cols_ts,
            figsize=(6 * n_cols_ts, 3.5 * n_rows_ts),
            sharex=True, squeeze=False,
        )
    
        for m_idx, ni in enumerate(ts_nodes):
            ax = axes4[m_idx // n_cols_ts, m_idx % n_cols_ts]
            meter_label = node_to_meter.get(ni, node_names[ni])
    
            # Observed data at this meter (within the window)
            obs_mask = ch_nodes_st == ni
            obs_t = [t0_dt + timedelta(hours=float(t)) for t in th_st[obs_mask]]
            obs_z = zh_st[obs_mask]
    
            # Static (log-normal CI — always positive)
            ax.plot(ts_datetimes, ts_static_mean[m_idx],
                    color="royalblue", linewidth=1.4, label="Static", zorder=3)
            ax.fill_between(
                ts_datetimes,
                ts_static_lo[m_idx],
                ts_static_hi[m_idx],
                color="royalblue", alpha=0.10, zorder=2,
            )
    
            # Hodge (log-normal CI — always positive)
            ax.plot(ts_datetimes, ts_hodge_mean[m_idx],
                    color="seagreen", linewidth=1.4, label="Hodge", zorder=4)
            ax.fill_between(
                ts_datetimes,
                ts_hodge_lo[m_idx],
                ts_hodge_hi[m_idx],
                color="seagreen", alpha=0.10, zorder=2,
            )
    
            # Spectral (log-normal CI — always positive)
            ax.plot(ts_datetimes, ts_spectral_mean[m_idx],
                    color="darkorange", linewidth=1.6, label="Spectral",
                    linestyle="--", zorder=5)
            ax.fill_between(
                ts_datetimes,
                ts_spectral_lo[m_idx],
                ts_spectral_hi[m_idx],
                color="darkorange", alpha=0.10, zorder=2,
            )
    
            # Observations
            if len(obs_z) > 0:
                ax.scatter(obs_t, obs_z, color="red", edgecolor="k",
                           linewidth=0.3, s=18, zorder=6, label="Observed")
    
            ax.set_ylabel("Flow (MGD)", fontsize=8)
            ax.set_title(f"Meter {meter_label}", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.25)
            if m_idx == 0:
                ax.legend(fontsize=7, loc="upper right")
    
        # Hide unused subplots
        for k in range(n_ts_nodes, n_rows_ts * n_cols_ts):
            axes4[k // n_cols_ts, k % n_cols_ts].set_visible(False)
    
        for ax in axes4[-1, :]:
            if ax.get_visible():
                ax.tick_params(axis="x", rotation=25, labelsize=7)
    
        fig4.suptitle(
            "Static vs Hodge vs Spectral BME Time Series (log-normal 95% CI)",
            fontsize=13, fontweight="bold", y=1.0)
        fig4.tight_layout()
        fname_d = os.path.join(out_dir, "hodge_D_timeseries.png")
        fig4.savefig(fname_d, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved {os.path.basename(fname_d)}")
        plt.close(fig4)

    # ── Figure C: Edge weight heatmap over the 24 h window ───────────
    fig3, ax3 = plt.subplots(figsize=(14, max(4, n_edges * 0.12)))
    im = ax3.imshow(edge_weight_grid, aspect="auto", cmap="viridis",
                    vmin=EDGE_FLOOR, vmax=1.0,
                    extent=[0, WINDOW_HOURS, n_edges, 0],
                    interpolation="nearest")
    ax3.set_xlabel("Hours into window")
    ax3.set_ylabel("Edge index")
    ax3.set_title("Time-Varying Hodge Edge Weights (normalised conduit flow)")
    plt.colorbar(im, ax=ax3, label="Normalised weight", shrink=0.7)
    fig3.tight_layout()
    fname_c = os.path.join(out_dir, "hodge_C_edge_weight_heatmap.png")
    fig3.savefig(fname_c, dpi=160, bbox_inches="tight")
    print(f"Saved {os.path.basename(fname_c)}")
    plt.close(fig3)

    # ═══════════════════════════════════════════════════════════════════
    # 10e. FIGURE E — Spectral BME: Estimate + Std Error through storm
    #      3 timestamps (rising / peak / receding) chosen from the
    #      metered node closest to the outlet.
    # ═══════════════════════════════════════════════════════════════════
    from matplotlib.colors import BoundaryNorm

    # Find the metered node closest to the outlet (smallest BFS depth).
    # tree_coords y = -depth, so largest y = smallest depth.
    outlet_meter_ni = max(
        ts_nodes,
        key=lambda ni: tree_coords.get(node_names[ni], (0, -9999))[1],
    )
    outlet_meter_label = node_to_meter.get(outlet_meter_ni,
                                           node_names[outlet_meter_ni])

    # Extract that meter's hourly hydrograph from the spectral prediction
    outlet_m_idx = list(ts_nodes).index(outlet_meter_ni)
    outlet_hydrograph = ts_spectral_mean[outlet_m_idx]  # (n_ts_times,)

    # Pick rising / peak / receding from the outlet meter's hydrograph
    i_peak_e = int(np.argmax(outlet_hydrograph))
    # Rising = steepest ascent before peak (max forward-diff)
    if i_peak_e > 1:
        diffs = np.diff(outlet_hydrograph[:i_peak_e])
        i_rising_e = int(np.argmax(diffs)) + 1       # midpoint of steepest rise
        i_rising_e = max(1, min(i_rising_e, i_peak_e - 1))
    else:
        i_rising_e = max(0, i_peak_e - 1)
    # Receding = ~symmetric time after peak, but stay within window
    lag_rise = i_peak_e - i_rising_e
    i_receding_e = min(i_peak_e + max(lag_rise, 2), n_ts_times - 1)

    t_stamps_e = np.array([ts_times[i_rising_e],
                           ts_times[i_peak_e],
                           ts_times[i_receding_e]])
    label_e = [(t0_dt + timedelta(hours=float(t))).strftime("%b %d %H:%M")
               for t in t_stamps_e]
    phase_tags = ["Rising", "Peak", "Receding"]

    print(f"\n── Figure E: Spectral BME storm snapshots (outlet meter: "
          f"{outlet_meter_label}) ──")
    for tag, lbl, ts in zip(phase_tags, label_e, t_stamps_e):
        print(f"    {tag:10s}  {lbl}  (t={ts:.1f}h)")

    # Run spectral prediction at 3 timestamps (all nodes)
    ck_e = np.repeat(pred_nodes, 3)
    tk_e = np.tile(t_stamps_e, n_pred)

    # Ensure dense cache includes these times
    spectral_cov.precompute_dense(t_stamps_e)

    print(f"  Predicting {n_pred} nodes × 3 times = {len(ck_e)} points ...")
    t0 = timer_mod.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_e = bme_predict_network_st(
            ck_nodes=ck_e, tk=tk_e,
            ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
            net_cov_st=spectral_cov_st, nhmax=NHMAX,
            order=0, mean_prior=data_mean,
        )
    print(f"  Done in {timer_mod.time() - t0:.1f}s")

    log_mean_e = np.array([r.mean for r in res_e]).reshape(n_pred, 3)
    log_var_e  = np.array([r.variance for r in res_e]).reshape(n_pred, 3)
    est_e = np.exp(log_mean_e + log_var_e / 2.0)
    std_e = np.sqrt(log_var_e)   # log-space posterior std

    # ── Build 2×3 figure ─────────────────────────────────────────────
    fig_e, axes_e = plt.subplots(
        2, 3, figsize=(22, 11),
        gridspec_kw={"right": 0.88, "wspace": 0.08, "hspace": 0.18},
    )

    # Estimate row: common linear scale
    vmax_est = float(np.percentile(
        est_e[est_e > 0], 97)) if (est_e > 0).any() else 1.0

    # Std error row: quantile-based BoundaryNorm for contrast
    all_std_vals = std_e.ravel()
    all_std_vals = all_std_vals[np.isfinite(all_std_vals) & (all_std_vals > 0)]
    if len(all_std_vals) > 10:
        quantiles = np.percentile(all_std_vals,
                                  [0, 5, 15, 30, 50, 70, 85, 95, 100])
        # Remove duplicates and ensure monotonic
        boundaries = np.unique(np.round(quantiles, 6))
        if len(boundaries) < 3:
            boundaries = np.linspace(all_std_vals.min(),
                                     all_std_vals.max(), 8)
    else:
        boundaries = np.linspace(0, float(std_e.max()) + 1e-6, 8)
    n_colors = len(boundaries) - 1
    cmap_std = matplotlib.colormaps.get_cmap("magma_r").resampled(n_colors)
    std_norm = BoundaryNorm(boundaries, ncolors=n_colors, clip=True)

    for col_i in range(3):
        # Row 0: Estimate
        ax = axes_e[0, col_i]
        lc, lc_g = _make_tree_flowlines(
            tree_edges, tree_coords, node_names,
            est_e[:, col_i], "YlOrRd", 0, vmax_est, linewidth=2.4)
        ax.add_collection(lc_g)
        ax.add_collection(lc)

        # Overlay active meters at this timestamp
        obs_at = np.abs(th_st - t_stamps_e[col_i]) < 0.5
        if obs_at.any():
            obs_ni = np.unique(ch_nodes_st[obs_at])
            ox = [tree_coords[node_names[ni]][0] for ni in obs_ni
                  if node_names[ni] in tree_coords]
            oy = [tree_coords[node_names[ni]][1] for ni in obs_ni
                  if node_names[ni] in tree_coords]
            ax.scatter(ox, oy, marker="^", c="white", edgecolor="black",
                       s=50, linewidths=0.8, zorder=5)

        ax.set_aspect("auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{phase_tags[col_i]} — {label_e[col_i]}\n"
                     f"Spectral BME Estimate",
                     fontsize=10, fontweight="bold", loc="left")
        ax.autoscale_view()

        # Row 1: Std Error (quantile colormap)
        ax2 = axes_e[1, col_i]
        _idx_map = {n: i for i, n in enumerate(node_names)}
        segments_s, seg_vals_s, grey_s = [], [], []
        for fn, tn in tree_edges:
            if fn not in tree_coords or tn not in tree_coords:
                continue
            x0, y0 = tree_coords[fn]
            x1, y1 = tree_coords[tn]
            i_from = _idx_map.get(fn, -1)
            i_to   = _idx_map.get(tn, -1)
            if i_from < 0 or i_to < 0:
                grey_s.append([(x0, y0), (x1, y1)])
                continue
            v_from = std_e[i_from, col_i]
            v_to   = std_e[i_to, col_i]
            if np.isnan(v_from) or np.isnan(v_to):
                grey_s.append([(x0, y0), (x1, y1)])
            else:
                segments_s.append([(x0, y0), (x1, y1)])
                seg_vals_s.append(0.5 * (v_from + v_to))

        lc_s = LineCollection(segments_s, cmap=cmap_std, norm=std_norm,
                              linewidths=2.4, capstyle="round", zorder=3)
        lc_s.set_array(np.array(seg_vals_s) if seg_vals_s else np.array([]))
        lc_g2 = LineCollection(grey_s, colors="lightgrey",
                               linewidths=0.4, alpha=0.35, zorder=1)
        ax2.add_collection(lc_g2)
        ax2.add_collection(lc_s)

        # Meters
        if obs_at.any():
            ax2.scatter(ox, oy, marker="^", c="white", edgecolor="black",
                        s=50, linewidths=0.8, zorder=5)

        ax2.set_aspect("auto")
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.set_title(f"{phase_tags[col_i]} — {label_e[col_i]}\n"
                      f"Posterior Std. Dev.",
                      fontsize=10, fontweight="bold", loc="left")
        ax2.autoscale_view()

    # Colorbars — placed in dedicated axes to avoid overlap
    # Estimate: standard linear (shared across top row)
    sm_est = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, vmax_est))
    sm_est.set_array([])
    cax_est = fig_e.add_axes([0.90, 0.53, 0.015, 0.38])   # [left, bottom, w, h]
    cb_est = fig_e.colorbar(sm_est, cax=cax_est)
    cb_est.set_label("Flow Estimate (MGD)", fontsize=10)

    # Std error: use the same cmap + norm that the LineCollections used
    sm_std = plt.cm.ScalarMappable(cmap=cmap_std, norm=std_norm)
    sm_std.set_array([])
    cax_std = fig_e.add_axes([0.90, 0.07, 0.015, 0.38])
    cb_std = fig_e.colorbar(sm_std, cax=cax_std, spacing="proportional")
    cb_std.set_label("Posterior Std. Dev. (log-space)", fontsize=10)
    # Label ticks at the boundary values
    tick_vals = boundaries
    tick_labels = [f"{v:.2f}" for v in tick_vals]
    cb_std.set_ticks(tick_vals)
    cb_std.set_ticklabels(tick_labels)

    fig_e.suptitle(
        f"Spectral BME — Storm Event at Outlet Meter "
        f"({outlet_meter_label})",
        fontsize=14, fontweight="bold", y=0.98)
    fname_e = os.path.join(out_dir, "hodge_E_spectral_storm.png")
    fig_e.savefig(fname_e, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved {os.path.basename(fname_e)}")
    plt.close(fig_e)

    # ═══════════════════════════════════════════════════════════════════
    # 10f. FIGURE F — Lambda Sensitivity: Mass-Balance Penalty Sweep
    #      Spectral BME only.  Varying λ controls how strongly the
    #      estimator penalises mass-balance violations.
    #
    #      In log-space the penalty M = HᵀH acts on log-flows:
    #        single-parent conduit → penalises log(x_j/x_parent) ≠ 0
    #            → pushes flow ratio toward 1  (sensible for pipes)
    #        multi-parent junction → penalises log(x_j) ≠ Σ log(x_pᵢ)
    #            → pushes x_j toward product(x_pᵢ), not sum  (approx.)
    #      Post-hoc mass-balance residuals are evaluated in original
    #      (MGD) space to show whether the soft penalty still helps.
    # ═══════════════════════════════════════════════════════════════════
    lam_values = [0.0, 0.5, 2.0, 5.0]
    n_lam = len(lam_values)

    # Build mass-balance operator for post-hoc diagnostics
    H_mb = build_mass_balance_operator(n_nodes, edge_array)

    # Use peak timestamp from Figure E
    t_peak_f = t_stamps_e[1]
    label_peak_f = label_e[1]

    print(f"\n{'=' * 70}")
    print("FIGURE F — Lambda Sensitivity Sweep (Spectral BME, log-space)")
    print(f"{'=' * 70}")
    print(f"  Lambda values: {lam_values}")
    print(f"  Timestamp: {label_peak_f} (peak)")
    print(f"\n  In log-space, the mass-balance operator penalises log-ratio")
    print(f"  deviations rather than additive flow conservation.")
    print(f"  For single-parent conduits:  penalises log(x_j / x_parent) != 0")
    print(f"  For multi-parent junctions:  penalises log(x_j) != sum(log(x_pi))")
    print(f"  Post-hoc residuals are computed in original (MGD) space.\n")

    sweep_est = np.zeros((n_lam, n_pred))
    sweep_std = np.zeros((n_lam, n_pred))
    sweep_rmse = np.zeros(n_lam)
    sweep_coverage = np.zeros(n_lam)
    sweep_mb_total = np.zeros(n_lam)   # sum |residual| (MGD)
    sweep_mb_rms = np.zeros(n_lam)     # RMS residual

    for li, lam_val in enumerate(lam_values):
        print(f"  λ = {lam_val} ...", end=" ", flush=True)
        t0 = timer_mod.time()

        sp_cov_i = SpectralHodgeNetworkCovariance(
            B1=B1,
            directed_edges=edge_array,
            edge_weight_func=edge_weight_func,
            kappa=KAPPA,
            sigma2=scale,
            alpha=ALPHA,
            lam=lam_val,
        )
        sp_cov_st_i = SpectralHodgeNetworkCovarianceST(
            sp_cov_i,
            model_t=TEMPORAL_MODEL,
            params_t=TEMPORAL_PARAMS,
            sigma2=scale,
            blend="geometric",
        )
        sp_cov_i.precompute_dense(all_unique_times)

        ck_f = pred_nodes
        tk_f = np.full(n_pred, t_peak_f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_f = bme_predict_network_st(
                ck_nodes=ck_f, tk=tk_f,
                ch_nodes=ch_nodes_st, th=th_st, zh=zh_log,
                net_cov_st=sp_cov_st_i, nhmax=NHMAX,
                order=0, mean_prior=data_mean,
            )

        log_mu_f = np.array([r.mean for r in res_f])
        log_v_f  = np.array([r.variance for r in res_f])
        log_s_f  = np.sqrt(np.maximum(log_v_f, 0.0))
        est_f = np.exp(log_mu_f + log_v_f / 2.0)

        sweep_est[li] = est_f
        sweep_std[li] = log_s_f

        # Mass-balance residual in original (MGD) space
        mb_residuals = H_mb @ est_f
        sweep_mb_total[li] = float(np.sum(np.abs(mb_residuals)))
        sweep_mb_rms[li] = float(np.sqrt(np.mean(mb_residuals ** 2)))

        # RMSE + coverage against observations near the peak
        obs_at_peak = np.abs(th_st - t_peak_f) < 0.5
        if obs_at_peak.any():
            obs_ni_pk = ch_nodes_st[obs_at_peak]
            obs_z_pk = zh_st[obs_at_peak]
            sweep_rmse[li] = float(np.sqrt(
                np.mean((est_f[obs_ni_pk] - obs_z_pk) ** 2)))
            lo_f = np.exp(log_mu_f[obs_ni_pk] - 1.96 * log_s_f[obs_ni_pk])
            hi_f = np.exp(log_mu_f[obs_ni_pk] + 1.96 * log_s_f[obs_ni_pk])
            sweep_coverage[li] = float(
                np.mean((obs_z_pk >= lo_f) & (obs_z_pk <= hi_f))) * 100

        elapsed = timer_mod.time() - t0
        print(f"done ({elapsed:.1f}s)")

    # ── Diagnostic table ─────────────────────────────────────────────
    print(f"\n  {'λ':>6s} {'RMSE':>8s} {'Cov%':>7s} "
          f"{'Σ|MB|':>10s} {'MB_rms':>8s} {'mean_σ':>8s}")
    print(f"  {'-' * 50}")
    for li, lam_val in enumerate(lam_values):
        print(f"  {lam_val:6.1f} {sweep_rmse[li]:8.3f} {sweep_coverage[li]:6.1f}% "
              f"{sweep_mb_total[li]:10.2f} {sweep_mb_rms[li]:8.4f} "
              f"{sweep_std[li].mean():8.4f}")

    # ── Build Figure F: 2 × n_lam panel ─────────────────────────────
    fig_f, axes_f = plt.subplots(
        2, n_lam, figsize=(6 * n_lam, 11),
        gridspec_kw={"right": 0.88, "wspace": 0.08, "hspace": 0.18},
    )

    vmax_f = float(np.percentile(
        sweep_est[sweep_est > 0], 97)) if (sweep_est > 0).any() else 1.0

    # Shared quantile-based std boundaries  across all λ
    all_std_f = sweep_std.ravel()
    all_std_f = all_std_f[np.isfinite(all_std_f) & (all_std_f > 0)]
    if len(all_std_f) > 10:
        q_f = np.percentile(all_std_f, [0, 5, 15, 30, 50, 70, 85, 95, 100])
        bounds_f = np.unique(np.round(q_f, 6))
        if len(bounds_f) < 3:
            bounds_f = np.linspace(all_std_f.min(), all_std_f.max(), 8)
    else:
        bounds_f = np.linspace(0, float(sweep_std.max()) + 1e-6, 8)
    n_colors_f = len(bounds_f) - 1
    cmap_std_f = matplotlib.colormaps.get_cmap("magma_r").resampled(n_colors_f)
    norm_std_f = BoundaryNorm(bounds_f, ncolors=n_colors_f, clip=True)

    _idx_map_f = {n: i for i, n in enumerate(node_names)}

    for li in range(n_lam):
        # Row 0: Estimate
        ax = axes_f[0, li]
        lc, lc_g = _make_tree_flowlines(
            tree_edges, tree_coords, node_names,
            sweep_est[li], "YlOrRd", 0, vmax_f, linewidth=2.4)
        ax.add_collection(lc_g)
        ax.add_collection(lc)
        obs_at = np.abs(th_st - t_peak_f) < 0.5
        if obs_at.any():
            obs_ni = np.unique(ch_nodes_st[obs_at])
            ox = [tree_coords[node_names[ni]][0] for ni in obs_ni
                  if node_names[ni] in tree_coords]
            oy = [tree_coords[node_names[ni]][1] for ni in obs_ni
                  if node_names[ni] in tree_coords]
            ax.scatter(ox, oy, marker="^", c="white", edgecolor="black",
                       s=50, linewidths=0.8, zorder=5)
        ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(
            f"λ = {lam_values[li]}\n"
            f"RMSE={sweep_rmse[li]:.3f}  Σ|MB|={sweep_mb_total[li]:.1f}",
            fontsize=10, fontweight="bold", loc="left")
        ax.autoscale_view()

        # Row 1: Std dev
        ax2 = axes_f[1, li]
        segments_f, seg_vals_f, grey_f = [], [], []
        for fn, tn in tree_edges:
            if fn not in tree_coords or tn not in tree_coords:
                continue
            x0, y0 = tree_coords[fn]
            x1, y1 = tree_coords[tn]
            i_from = _idx_map_f.get(fn, -1)
            i_to   = _idx_map_f.get(tn, -1)
            if i_from < 0 or i_to < 0:
                grey_f.append([(x0, y0), (x1, y1)])
                continue
            v_from = sweep_std[li, i_from]
            v_to   = sweep_std[li, i_to]
            if np.isnan(v_from) or np.isnan(v_to):
                grey_f.append([(x0, y0), (x1, y1)])
            else:
                segments_f.append([(x0, y0), (x1, y1)])
                seg_vals_f.append(0.5 * (v_from + v_to))

        lc_s = LineCollection(segments_f, cmap=cmap_std_f, norm=norm_std_f,
                              linewidths=2.4, capstyle="round", zorder=3)
        lc_s.set_array(np.array(seg_vals_f) if seg_vals_f else np.array([]))
        lc_g2 = LineCollection(grey_f, colors="lightgrey",
                               linewidths=0.4, alpha=0.35, zorder=1)
        ax2.add_collection(lc_g2)
        ax2.add_collection(lc_s)
        if obs_at.any():
            ax2.scatter(ox, oy, marker="^", c="white", edgecolor="black",
                        s=50, linewidths=0.8, zorder=5)
        ax2.set_aspect("auto"); ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.set_title(f"λ = {lam_values[li]}\nPosterior Std. Dev.",
                      fontsize=10, fontweight="bold", loc="left")
        ax2.autoscale_view()

    # Colorbars
    sm_est_f = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, vmax_f))
    sm_est_f.set_array([])
    cax_est_f = fig_f.add_axes([0.90, 0.53, 0.015, 0.38])
    cb_est_f = fig_f.colorbar(sm_est_f, cax=cax_est_f)
    cb_est_f.set_label("Flow Estimate (MGD)", fontsize=10)

    sm_std_f = plt.cm.ScalarMappable(cmap=cmap_std_f, norm=norm_std_f)
    sm_std_f.set_array([])
    cax_std_f = fig_f.add_axes([0.90, 0.07, 0.015, 0.38])
    cb_std_f = fig_f.colorbar(sm_std_f, cax=cax_std_f, spacing="proportional")
    cb_std_f.set_label("Posterior Std. Dev. (log-space)", fontsize=10)
    cb_std_f.set_ticks(bounds_f)
    cb_std_f.set_ticklabels([f"{v:.2f}" for v in bounds_f])

    fig_f.suptitle(
        f"Lambda Sensitivity — Spectral BME at Peak ({label_peak_f})\n"
        f"Mass-balance penalty: λ ∈ {lam_values}",
        fontsize=13, fontweight="bold", y=0.99)
    fname_f = os.path.join(out_dir, "hodge_F_lambda_sensitivity.png")
    fig_f.savefig(fname_f, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {os.path.basename(fname_f)}")
    plt.close(fig_f)

    # ═══════════════════════════════════════════════════════════════════
    # 10g. FIGURE G — Post-Estimation Mass-Balance Projection
    #      Apply the exact orthogonal projection x_c = x - H'(HH')^{-1}Hx
    #      in original (MGD) space to enforce conservation at every
    #      junction node.  Compare before / after on the best-λ result.
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("FIGURE G — Post-Estimation Mass-Balance Projection")
    print(f"{'=' * 70}")

    # Use baseline spectral estimate (λ = LAM = 0.5) at peak
    # spectral_mean / spectral_std are (n_pred, 2) — column 1 is peak
    est_peak = spectral_mean[:, 1]          # MGD, from main spectral prediction
    log_mu_peak = log_mean_sp[:, 1]
    log_s_peak = spectral_std[:, 1]         # log-space std dev

    # Back-transform std dev to MGD space for the projection
    # For log-normal: Var(X) = exp(2μ+σ²)(exp(σ²)-1)
    # std_mgd = sqrt(Var(X)) = E[X] * sqrt(exp(σ²) - 1)
    std_mgd_peak = est_peak * np.sqrt(np.expm1(log_s_peak**2))

    proj = project_mass_balance(est_peak, H_mb, sigma=std_mgd_peak)
    x_proj = proj["x_proj"]
    correction = proj["correction"]
    r_before = proj["residuals_before"]
    r_after = proj["residuals_after"]
    sigma_proj = proj["sigma_proj"]

    print(f"  Projection applied at peak ({label_peak_f})")
    print(f"  Before: Σ|r| = {np.sum(np.abs(r_before)):.2f} MGD,  "
          f"RMS = {np.sqrt(np.mean(r_before**2)):.4f} MGD")
    print(f"  After:  Σ|r| = {np.sum(np.abs(r_after)):.2f} MGD,  "
          f"RMS = {np.sqrt(np.mean(r_after**2)):.4f} MGD")
    print(f"  Clamped to zero: {proj['n_clamped']} / {n_pred} nodes")
    print(f"  Max |correction|: {np.max(np.abs(correction)):.4f} MGD")
    print(f"  Mean |correction|: {np.mean(np.abs(correction)):.4f} MGD")
    if sigma_proj is not None:
        print(f"  Mean uncertainty change: "
              f"{np.mean(std_mgd_peak):.4f} → {np.mean(sigma_proj):.4f} MGD")

    # RMSE against observations near the peak (same as sweep)
    obs_at_peak_g = np.abs(th_st - t_peak_f) < 0.5
    if obs_at_peak_g.any():
        obs_ni_g = ch_nodes_st[obs_at_peak_g]
        obs_z_g = zh_st[obs_at_peak_g]
        rmse_before = float(np.sqrt(np.mean((est_peak[obs_ni_g] - obs_z_g)**2)))
        rmse_after = float(np.sqrt(np.mean((x_proj[obs_ni_g] - obs_z_g)**2)))
        print(f"  RMSE vs obs:  before={rmse_before:.3f}  after={rmse_after:.3f}")

    # ── Build Figure G: 1 × 3 panel ─────────────────────────────────
    fig_g, axes_g = plt.subplots(
        1, 3, figsize=(18, 7),
        gridspec_kw={"right": 0.88, "wspace": 0.08},
    )
    vmax_g = float(np.percentile(
        est_peak[est_peak > 0], 97)) if (est_peak > 0).any() else 1.0

    # Panel 0: original estimate
    lc0, lc0_g = _make_tree_flowlines(
        tree_edges, tree_coords, node_names,
        est_peak, "YlOrRd", 0, vmax_g, linewidth=2.4)
    axes_g[0].add_collection(lc0_g); axes_g[0].add_collection(lc0)
    if obs_at_peak_g.any():
        obs_ni_u = np.unique(ch_nodes_st[obs_at_peak_g])
        ox = [tree_coords[node_names[ni]][0] for ni in obs_ni_u
              if node_names[ni] in tree_coords]
        oy = [tree_coords[node_names[ni]][1] for ni in obs_ni_u
              if node_names[ni] in tree_coords]
        axes_g[0].scatter(ox, oy, marker="^", c="white", edgecolor="black",
                          s=50, linewidths=0.8, zorder=5)
    axes_g[0].set_title(
        f"Before Projection\nΣ|r|={np.sum(np.abs(r_before)):.1f} MGD",
        fontsize=11, fontweight="bold", loc="left")

    # Panel 1: projected estimate
    lc1, lc1_g = _make_tree_flowlines(
        tree_edges, tree_coords, node_names,
        x_proj, "YlOrRd", 0, vmax_g, linewidth=2.4)
    axes_g[1].add_collection(lc1_g); axes_g[1].add_collection(lc1)
    if obs_at_peak_g.any():
        axes_g[1].scatter(ox, oy, marker="^", c="white", edgecolor="black",
                          s=50, linewidths=0.8, zorder=5)
    axes_g[1].set_title(
        f"After Projection\nΣ|r|={np.sum(np.abs(r_after)):.2g} MGD",
        fontsize=11, fontweight="bold", loc="left")

    # Panel 2: absolute correction magnitude
    abs_corr = np.abs(correction)
    vmax_corr = float(np.percentile(
        abs_corr[abs_corr > 0], 97)) if (abs_corr > 0).any() else 0.1
    lc2, lc2_g = _make_tree_flowlines(
        tree_edges, tree_coords, node_names,
        abs_corr, "PuRd", 0, vmax_corr, linewidth=2.4)
    axes_g[2].add_collection(lc2_g); axes_g[2].add_collection(lc2)
    if obs_at_peak_g.any():
        axes_g[2].scatter(ox, oy, marker="^", c="white", edgecolor="black",
                          s=50, linewidths=0.8, zorder=5)
    axes_g[2].set_title(
        f"|Correction|\nmax={np.max(abs_corr):.3f} MGD",
        fontsize=11, fontweight="bold", loc="left")

    for ax in axes_g:
        ax.set_aspect("auto"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.autoscale_view()

    # Colorbars
    sm_g0 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, vmax_g))
    sm_g0.set_array([])
    cax_g0 = fig_g.add_axes([0.90, 0.15, 0.015, 0.7])
    cb_g0 = fig_g.colorbar(sm_g0, cax=cax_g0)
    cb_g0.set_label("Flow (MGD)", fontsize=10)

    fig_g.suptitle(
        f"Post-Estimation Mass-Balance Projection — Peak ({label_peak_f})\n"
        f"Orthogonal projection onto ker(H): minimum-norm correction",
        fontsize=13, fontweight="bold", y=0.99)
    fname_g = os.path.join(out_dir, "hodge_G_mass_balance_projection.png")
    fig_g.savefig(fname_g, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved {os.path.basename(fname_g)}")
    plt.close(fig_g)

    # ═══════════════════════════════════════════════════════════════════
    # 11. SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY — Hodge-Laplacian BME on Onondaga SWMM Network")
    print("=" * 70)
    print(f"Network: {n_nodes} nodes, {n_edges} edges "
          f"(tree: {len(tree_edges)} edges, depth {max_depth})")
    print(f"Modeling: log-space BME with log-normal back-transform")
    print(f"Covariance:")
    if not SPECTRAL_ONLY:
        print(f"  Static:   graph-Laplacian, kappa={KAPPA}, separable S/T")
        print(f"  Hodge:    time-varying L₀(t), kappa={KAPPA}, alpha={ALPHA}, "
              f"lam={LAM}, geometric blend")
    print(f"  Spectral: fixed eigenvectors + Galerkin λ_eff(t), "
          f"kappa={KAPPA}, alpha={ALPHA}, lam={LAM}")
    print(f"  Temporal: {TEMPORAL_MODEL}, range={TEMPORAL_RANGE_HOURS}h")
    print(f"Window: {start_dt} + {WINDOW_HOURS:.0f}h  "
          f"({len(zh_st)} obs, {len(unique_obs_nodes)} meters)")
    print(f"Timestamps: {time_labels[0]} (rising), {time_labels[1]} (peak)")
    if not SPECTRAL_ONLY:
        print(f"Wall time: static={t_static:.1f}s, hodge={t_hodge:.1f}s, "
              f"spectral={t_spectral:.1f}s")
    else:
        print(f"Wall time: spectral={t_spectral:.1f}s  (SPECTRAL_ONLY mode)")
    print(f"Hodge decomposition: "
          f"gradient={100*grad_energy/max(total_energy,1e-12):.1f}%, "
          f"harmonic={100*harm_energy/max(total_energy,1e-12):.1f}%")
    print(f"\nLambda sweep results:")
    for li, lam_val in enumerate(lam_values):
        print(f"  λ={lam_val:4.1f}  RMSE={sweep_rmse[li]:.3f}  "
              f"Σ|MB|={sweep_mb_total[li]:.1f} MGD  "
              f"MB_rms={sweep_mb_rms[li]:.4f}")
    print(f"\nPost-estimation projection (λ={LAM}):")
    print(f"  Σ|MB| before: {np.sum(np.abs(r_before)):.2f} MGD  →  "
          f"after: {np.sum(np.abs(r_after)):.2g} MGD")
    print(f"  Clamped: {proj['n_clamped']} nodes")
    print("=" * 70)
    print(f"\nFigures written to: {out_dir}")

    plt.close("all")


if __name__ == "__main__":
    main()
