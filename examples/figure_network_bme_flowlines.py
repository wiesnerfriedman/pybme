#!/usr/bin/env python3
"""
Publication figure: de-identified network BME flowline map.

Produces a 2x2 panel (2 timestamps x [BME estimate, uncertainty])
with the network drawn as coloured flowlines.  The network is
rendered in a **tree layout** so that topology is visible but
real geographic coordinates are completely removed.

Data
----
Uses the private Onondaga SWMM model (not included in this repo).
Set these environment variables or edit the paths below:

    PYBME_OC_INP       .inp file
    PYBME_OC_OBS       ObservedTimeseries_converted_v2.csv
    PYBME_OC_METERS    wapug_inputs/MeterLocations.csv

Output
------
``examples/figures/fig_network_bme_flowlines.png``
"""

import csv
import os
import sys
import time
import warnings
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ── ensure pybme is importable ──────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path     = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.network import (
    NetworkCovariance, NetworkCovarianceST, adjacency_from_edges,
)
from pybme.predict import bme_predict_network_st
from pybme.swmm import build_edge_array, parse_swmm_inp

# ═══════════════════════════════════════════════════════════════════
# File paths
# ═══════════════════════════════════════════════════════════════════
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

KAPPA = 0.1
TEMPORAL_RANGE_HOURS = 6.0
WINDOW_HOURS = 24.0

# ═══════════════════════════════════════════════════════════════════
# 1. Parse network
# ═══════════════════════════════════════════════════════════════════
print("Parsing SWMM .inp ...")
network    = parse_swmm_inp(INP_PATH)
node_names = network.all_node_names
n_nodes    = len(node_names)
node_idx   = network.node_index
edge_array = build_edge_array(node_names, network.edges)
print(f"  {n_nodes} nodes, {len(network.edges)} links")

# ═══════════════════════════════════════════════════════════════════
# 2. Tree layout for de-identification
# ═══════════════════════════════════════════════════════════════════
# Build an adjacency list, find a root (outfall or max-degree node),
# and compute a layered tree layout using BFS.  Nodes that create
# cycles get attached to the tree at their first-visited position.

adj = [[] for _ in range(n_nodes)]
for fn, tn, *_ in network.edges:
    i, j = node_idx.get(fn, -1), node_idx.get(tn, -1)
    if i >= 0 and j >= 0:
        adj[i].append(j)
        adj[j].append(i)

# Prefer an outfall as root; otherwise use the highest-degree node
outfall_indices = [node_idx[n] for n in network.outfalls if n in node_idx]
if outfall_indices:
    root = outfall_indices[0]
else:
    root = max(range(n_nodes), key=lambda i: len(adj[i]))

# BFS to assign depth and parent
depth    = np.full(n_nodes, -1, dtype=int)
parent   = np.full(n_nodes, -1, dtype=int)
children = [[] for _ in range(n_nodes)]
order    = []

depth[root] = 0
queue = deque([root])
while queue:
    u = queue.popleft()
    order.append(u)
    for v in adj[u]:
        if depth[v] < 0:
            depth[v] = depth[u] + 1
            parent[v] = u
            children[u].append(v)
            queue.append(v)

# Handle disconnected components: BFS from each unvisited node
for i in range(n_nodes):
    if depth[i] < 0:
        depth[i] = 0
        queue = deque([i])
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in adj[u]:
                if depth[v] < 0:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    children[u].append(v)
                    queue.append(v)

max_depth = depth.max()

# Assign horizontal positions: leaf nodes get sequential x values,
# internal nodes are centred over their children.
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

# Process each tree in the forest
roots = [i for i in range(n_nodes) if parent[i] < 0 or depth[i] == 0]
for r in roots:
    if parent[r] < 0 or r == root:
        assign_x(r)

# y = negative depth so root is at the top
y_pos = -depth.astype(float)

# Build tree-layout coordinate dict
tree_coords = {}
for i, name in enumerate(node_names):
    tree_coords[name] = (x_pos[i], y_pos[i])

# Build tree edges (only BFS-tree edges, not back-edges)
tree_edges = []
for u in range(n_nodes):
    for c in children[u]:
        tree_edges.append((node_names[u], node_names[c]))

print(f"  Tree layout: root={node_names[root]}, max depth={max_depth}, "
      f"{len(tree_edges)} tree edges")

# ═══════════════════════════════════════════════════════════════════
# 3. Read meter locations
# ═══════════════════════════════════════════════════════════════════
meter_to_node = {}
with open(METER_LOC_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        meter_to_node[row["Meter"].strip()] = row["Node"].strip()

# ═══════════════════════════════════════════════════════════════════
# 4. Read observations (only _Flow columns)
# ═══════════════════════════════════════════════════════════════════
print("Reading observations ...")
with open(OBS_PATH, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

# Identify _Flow columns and map to node indices
flow_col_info = []  # (col_idx, meter_name, node_index)
for ci, col_name in enumerate(header):
    if not col_name.endswith("_Flow"):
        continue
    meter_name = col_name.replace("_Flow", "")
    node_name  = meter_to_node.get(meter_name)
    if node_name and node_name in node_idx:
        flow_col_info.append((ci, meter_name, node_idx[node_name]))

print(f"  {len(flow_col_info)} flow meters mapped to network nodes")

# Parse timestamps and build observation arrays
datetimes = []
all_obs_nodes, all_obs_times, all_obs_values = [], [], []
t0 = None

for row in rows:
    try:
        dt = datetime.strptime(row[0].strip(), "%m/%d/%Y %H:%M")
    except (ValueError, IndexError):
        continue
    if t0 is None:
        t0 = dt
    t_hours = (dt - t0).total_seconds() / 3600.0
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
times_hours    = np.array([(dt - t0).total_seconds() / 3600.0 for dt in datetimes])

print(f"  {len(all_obs_values)} non-zero flow observations")
print(f"  Time span: {datetimes[0]} to {datetimes[-1]}")

# ═══════════════════════════════════════════════════════════════════
# 5. Select the best 24h window
# ═══════════════════════════════════════════════════════════════════
window_starts = np.arange(0, times_hours[-1] - WINDOW_HOURS, 1.0)
best_score, best_start = 0, 0.0
for ws in window_starts:
    mask = (all_obs_times >= ws) & (all_obs_times < ws + WINDOW_HOURS)
    if not mask.any():
        continue
    n_unique = len(np.unique(all_obs_nodes[mask]))
    vals = all_obs_values[mask]
    # Prefer windows with many meters, high dynamic range, and high peak
    score = n_unique * 10 + np.ptp(vals) + np.max(vals) * 0.5
    if score > best_score:
        best_score, best_start = score, ws

win_mask    = ((all_obs_times >= best_start) &
               (all_obs_times < best_start + WINDOW_HOURS))
ch_nodes_st = all_obs_nodes[win_mask]
th_st       = all_obs_times[win_mask]
zh_st       = all_obs_values[win_mask]

# Thin if too many observations
MAX_OBS = 500
if len(zh_st) > MAX_OBS:
    step = len(zh_st) // MAX_OBS + 1
    keep = np.arange(0, len(zh_st), step)
    ch_nodes_st, th_st, zh_st = ch_nodes_st[keep], th_st[keep], zh_st[keep]

start_dt = t0 + timedelta(hours=float(best_start))
print(f"  Window: {start_dt} + 24h  |  {len(zh_st)} obs, "
      f"{len(np.unique(ch_nodes_st))} unique nodes")

# ═══════════════════════════════════════════════════════════════════
# 6. Build space-time covariance and predict at 2 timestamps
# ═══════════════════════════════════════════════════════════════════
W = adjacency_from_edges(n_nodes, edge_array)

data_var  = float(np.var(zh_st))
data_mean = float(np.mean(zh_st))

net_cov_init = NetworkCovariance(W, kappa=KAPPA, sigma2=1.0, from_adjacency=True)
unique_obs   = np.unique(ch_nodes_st)
diag_at_obs  = net_cov_init.marginal_variance(unique_obs)
scale = data_var / max(diag_at_obs.mean(), 1e-12) if data_var > 0 else 1.0

net_cov = NetworkCovariance(W, kappa=KAPPA, sigma2=scale, from_adjacency=True)
net_cov_st = NetworkCovarianceST(
    net_cov,
    model_t="exponential",
    params_t=[1.0, TEMPORAL_RANGE_HOURS],
    sigma2=scale,
)

# Pick two timestamps: rising limb + peak (both during active flow)
tk_all = np.arange(best_start, best_start + WINDOW_HOURS + 0.01, 1.0)

obs_by_hour = {}
for t_val, z_val in zip(th_st, zh_st):
    hr = int(round(t_val))
    obs_by_hour.setdefault(hr, []).append(z_val)

# Find the peak-flow hour
peak_hour = max(obs_by_hour, key=lambda h: np.mean(obs_by_hour[h]))
snap = lambda target: tk_all[np.argmin(np.abs(tk_all - target))]

# Find the rising-limb hour: ~halfway between the start of elevated
# flow and the peak.  "Elevated" = mean > 2x the overall median.
overall_median = float(np.median(zh_st))
active_hours = sorted(h for h, vs in obs_by_hour.items()
                      if np.mean(vs) > 2.0 * max(overall_median, 0.01))
if active_hours and len(active_hours) >= 3:
    # Pick the hour roughly 1/3 of the way through the active period
    rising_hour = active_hours[max(1, len(active_hours) // 3)]
else:
    # Fallback: 2-3 hours before the peak
    rising_hour = peak_hour - 3

# Ensure at least 2h separation
if abs(rising_hour - peak_hour) < 2:
    rising_hour = peak_hour - 3

t_rising = snap(rising_hour)
t_peak   = snap(peak_hour)
# Guarantee different columns
if t_rising == t_peak:
    t_rising = snap(peak_hour - 3)
# Always order chronologically: left column = earlier time
timestamps = np.sort(np.array([t_rising, t_peak]))

pred_nodes = np.arange(n_nodes)
n_pred     = n_nodes

print(f"  Timestamps: rising={t_rising:.0f}h, peak={t_peak:.0f}h")
print(f"  Predicting at {n_pred} nodes x 2 times = {n_pred * 2} points ...")

ck_flat = np.repeat(pred_nodes, 2)
tk_flat = np.tile(timestamps, n_pred)

t_wall = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = bme_predict_network_st(
        ck_nodes   = ck_flat,
        tk         = tk_flat,
        ch_nodes   = ch_nodes_st,
        th         = th_st,
        zh         = zh_st,
        net_cov_st = net_cov_st,
        nhmax      = 30,
        order      = 0,
        mean_prior = data_mean,
    )
print(f"  Done in {time.time() - t_wall:.1f}s")

bme_mean = np.array([r.mean for r in results]).reshape(n_pred, 2)
bme_std  = np.array([np.sqrt(r.variance) for r in results]).reshape(n_pred, 2)

print(f"  BME mean range: [{bme_mean.min():.4f}, {bme_mean.max():.4f}]")
print(f"  BME std  range: [{bme_std.min():.4f}, {bme_std.max():.4f}]")

# ═══════════════════════════════════════════════════════════════════
# 7. Build flowline segments for tree layout
# ═══════════════════════════════════════════════════════════════════

def make_tree_flowlines(tree_edges, tree_coords, node_names, values,
                        cmap_name, vmin, vmax, linewidth=2.0):
    """LineCollection of tree edges coloured by endpoint-average value."""
    _idx = {n: i for i, n in enumerate(node_names)}
    segments, seg_vals = [], []
    grey_segments = []

    for fn, tn in tree_edges:
        if fn not in tree_coords or tn not in tree_coords:
            continue
        x0, y0 = tree_coords[fn]
        x1, y1 = tree_coords[tn]
        i_from = _idx.get(fn, -1)
        i_to   = _idx.get(tn, -1)
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

# ═══════════════════════════════════════════════════════════════════
# 8. Draw the 2x2 panel
# ═══════════════════════════════════════════════════════════════════
print("Drawing figure ...")

# Per-column normalization so both timestamps show contrast
vmin_m, vmax_m_col = [0, 0], [1.0, 1.0]
vmin_s, vmax_s_col = [0, 0], [1.0, 1.0]
for col in range(2):
    m_vals = bme_mean[:, col]
    m_pos = m_vals[(np.isfinite(m_vals)) & (m_vals > 0)]
    vmax_m_col[col] = float(np.percentile(m_pos, 95)) if len(m_pos) else 1.0
    s_vals = bme_std[:, col]
    s_fin = s_vals[np.isfinite(s_vals)]
    vmax_s_col[col] = float(np.percentile(s_fin, 95)) if len(s_fin) else 1.0

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

time_labels = []
for t_val in timestamps:
    dt = t0 + timedelta(hours=float(t_val))
    time_labels.append(dt.strftime("%b %d %H:%M"))

panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

for col in range(2):
    # ── Row 0: BME estimate ──────────────────────────────────
    ax_est = axes[0, col]
    lc_e, lc_eg = make_tree_flowlines(
        tree_edges, tree_coords, node_names,
        bme_mean[:, col], "YlOrRd", 0, vmax_m_col[col], linewidth=2.2)
    ax_est.add_collection(lc_eg)
    ax_est.add_collection(lc_e)

    # Overlay meter locations active near this timestamp
    obs_at_t = np.abs(th_st - timestamps[col]) < 0.5
    if obs_at_t.any():
        obs_ni = np.unique(ch_nodes_st[obs_at_t])
        ox = [tree_coords[node_names[ni]][0] for ni in obs_ni]
        oy = [tree_coords[node_names[ni]][1] for ni in obs_ni]
        ax_est.scatter(ox, oy, marker="^", c="white", edgecolor="black",
                       s=40, linewidths=0.7, zorder=5, label="Meter")

    ax_est.set_aspect("auto")
    ax_est.set_xticks([]); ax_est.set_yticks([])
    for sp in ax_est.spines.values():
        sp.set_visible(False)
    ax_est.set_title(
        f"{panel_labels[0][col]}  BME Estimate, t = {time_labels[col]}",
        fontsize=11, fontweight="bold", loc="left")
    if col == 0 and obs_at_t.any():
        ax_est.legend(fontsize=8, loc="lower right", framealpha=0.8)
    ax_est.autoscale_view()

    # ── Row 1: uncertainty ───────────────────────────────────
    ax_unc = axes[1, col]
    lc_u, lc_ug = make_tree_flowlines(
        tree_edges, tree_coords, node_names,
        bme_std[:, col], "Blues", 0, vmax_s_col[col], linewidth=2.2)
    ax_unc.add_collection(lc_ug)
    ax_unc.add_collection(lc_u)

    ax_unc.set_aspect("auto")
    ax_unc.set_xticks([]); ax_unc.set_yticks([])
    for sp in ax_unc.spines.values():
        sp.set_visible(False)
    ax_unc.set_title(
        f"{panel_labels[1][col]}  Posterior Std. Dev., t = {time_labels[col]}",
        fontsize=11, fontweight="bold", loc="left")
    ax_unc.autoscale_view()

# Per-panel colour bars (each column has its own scale)
for col in range(2):
    sm_e = plt.cm.ScalarMappable(cmap="YlOrRd",
                                 norm=Normalize(0, vmax_m_col[col]))
    sm_e.set_array([])
    cb_e = fig.colorbar(sm_e, ax=axes[0, col], location="right",
                        shrink=0.75, pad=0.02, aspect=25)
    cb_e.set_label("Flow (MGD)", fontsize=9)

    sm_u = plt.cm.ScalarMappable(cmap="Blues",
                                 norm=Normalize(0, vmax_s_col[col]))
    sm_u.set_array([])
    cb_u = fig.colorbar(sm_u, ax=axes[1, col], location="right",
                        shrink=0.75, pad=0.02, aspect=25)
    cb_u.set_label("Std. Dev. (MGD)", fontsize=9)

fig.suptitle("Network-Domain BME: Space-Time Flow Estimation",
             fontsize=14, fontweight="bold", y=0.99)

out_dir = os.path.join(script_dir, "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig_network_bme_flowlines.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
plt.close(fig)
