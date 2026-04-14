#!/usr/bin/env python3
"""Animated GIF: BME tree plot over a wet-weather event.

Three side-by-side tree-layout panels at each timestep:
  1. SWMM soft-routed prior
  2. Hard-data-only conditional (kriging)
  3. BME estimate (hard + soft)

A rain hyetograph bar runs along the top with a moving time cursor.
Frames are stitched into a GIF using Pillow.
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import deque

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ── ensure pybme is importable ──────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.network import NetworkCovariance, NetworkCovarianceST, adjacency_from_edges
from pybme.predict import bme_predict_network_st
from pybme.synthetic_network import (
    build_soft_pdf_inputs,
    prediction_grid,
    simulate_synthetic_routed_network,
)

# ═══════════════════════════════════════════════════════════════════
# 1.  Generate the synthetic dataset and run BME
# ═══════════════════════════════════════════════════════════════════
print("Generating synthetic dataset ...")
dataset = simulate_synthetic_routed_network(seed=42)
ck_nodes, tk = prediction_grid(dataset)
cs_nodes, ts, soft_pdfs, _ = build_soft_pdf_inputs(dataset)

n_nodes, n_times = dataset.truth_matrix.shape
node_names = dataset.node_names
coords = dataset.coords
edges = dataset.edges
sensor_idx = [node_names.index(n) for n in dataset.sensor_nodes]

W = adjacency_from_edges(n_nodes, dataset.edge_array)
sigma2 = max(float(np.var(dataset.soft_matrix)), 0.02)
net_cov = NetworkCovariance(W, kappa=0.55, sigma2=sigma2, from_adjacency=True)
net_cov_st = NetworkCovarianceST(
    net_cov,
    model_t="exponential",
    params_t=[1.0, 3.0],
    sigma2=sigma2,
)

mean_prior = float(np.mean(dataset.hard_values))

print("Running BME hard-only ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hard_only = bme_predict_network_st(
        ck_nodes=ck_nodes, tk=tk,
        ch_nodes=dataset.hard_nodes, th=dataset.hard_times,
        zh=dataset.hard_values,
        net_cov_st=net_cov_st,
        nhmax=28, nsmax=0, order=0, n_grid=101,
        mean_prior=mean_prior,
    )

print("Running BME hard+soft ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hard_soft = bme_predict_network_st(
        ck_nodes=ck_nodes, tk=tk,
        ch_nodes=dataset.hard_nodes, th=dataset.hard_times,
        zh=dataset.hard_values,
        cs_nodes=cs_nodes, ts=ts, soft_pdfs=soft_pdfs,
        net_cov_st=net_cov_st,
        nhmax=28, nsmax=14, order=0, n_grid=101,
        mean_prior=mean_prior,
    )

hard_only_mean = np.array([r.mean for r in hard_only]).reshape(n_nodes, n_times)
hard_soft_mean = np.array([r.mean for r in hard_soft]).reshape(n_nodes, n_times)

print("BME estimation complete.")

# ═══════════════════════════════════════════════════════════════════
# 2.  Build tree layout (BFS from OUTFALL)
# ═══════════════════════════════════════════════════════════════════
name_to_idx = {n: i for i, n in enumerate(node_names)}
adj = [[] for _ in range(n_nodes)]
for fn, tn, *_ in edges:
    i, j = name_to_idx.get(fn, -1), name_to_idx.get(tn, -1)
    if i >= 0 and j >= 0:
        adj[i].append(j)
        adj[j].append(i)

root = name_to_idx["OUTFALL"]
depth = np.full(n_nodes, -1, dtype=int)
parent = np.full(n_nodes, -1, dtype=int)
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

# Handle disconnected components
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

assign_x(root)
y_pos = -depth.astype(float)

tree_coords = {name: (x_pos[i], y_pos[i]) for i, name in enumerate(node_names)}
tree_edges = []
for u in range(n_nodes):
    for c in children[u]:
        tree_edges.append((node_names[u], node_names[c]))

print(f"Tree layout: root=OUTFALL, {len(tree_edges)} edges, max depth={depth.max()}")

# ═══════════════════════════════════════════════════════════════════
# 3.  Rendering helpers
# ═══════════════════════════════════════════════════════════════════

def make_tree_lc(values, cmap_name, vmin, vmax, lw=3.0):
    """Return (coloured LineCollection, grey LineCollection)."""
    segments, seg_vals, grey = [], [], []
    for fn, tn in tree_edges:
        x0, y0 = tree_coords[fn]
        x1, y1 = tree_coords[tn]
        i_f, i_t = name_to_idx[fn], name_to_idx[tn]
        vf, vt = values[i_f], values[i_t]
        if np.isnan(vf) or np.isnan(vt):
            grey.append([(x0, y0), (x1, y1)])
        else:
            segments.append([(x0, y0), (x1, y1)])
            seg_vals.append(0.5 * (vf + vt))
    norm = Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap_name, norm=norm,
                        linewidths=lw, capstyle="round", zorder=3)
    lc.set_array(np.array(seg_vals) if seg_vals else np.array([]))
    lc_g = LineCollection(grey, colors="lightgrey", linewidths=0.6,
                          alpha=0.35, zorder=1)
    return lc, lc_g


def draw_tree_panel(ax, values, title, cmap, vmin, vmax, t_idx):
    """Draw a single tree panel with node markers."""
    lc, lc_g = make_tree_lc(values, cmap, vmin, vmax)
    ax.add_collection(lc_g)
    ax.add_collection(lc)

    # Sensor nodes with hard observations at this timestep
    for si in sensor_idx:
        sx, sy = tree_coords[node_names[si]]
        has_obs = not np.isnan(dataset.hard_observation_matrix[si, t_idx])
        marker = "^" if has_obs else "o"
        color = "red" if has_obs else "white"
        edge = "black"
        ax.scatter([sx], [sy], marker=marker, c=color, edgecolor=edge,
                   s=50, linewidths=0.8, zorder=6)

    ax.set_xlim(x_pos.min() - 0.5, x_pos.max() + 0.5)
    ax.set_ylim(y_pos.min() - 0.5, y_pos.max() + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)


# ═══════════════════════════════════════════════════════════════════
# 4.  Render frames
# ═══════════════════════════════════════════════════════════════════
# Global colour scale across all timesteps
all_vals = np.concatenate([
    dataset.soft_matrix.ravel(),
    hard_only_mean.ravel(),
    hard_soft_mean.ravel(),
])
vmin_g = 0.0
vmax_g = float(np.percentile(all_vals[np.isfinite(all_vals)], 97))

out_dir = os.path.join(script_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

frames = []
print("Rendering frames ...")
for t_idx in range(n_times):
    dt = dataset.datetimes[t_idx]
    time_label = dt.strftime("%b %d  %H:%M")

    fig = plt.figure(figsize=(13, 6.5), facecolor="white")

    # ── Rain bar at top ──────────────────────────────────────
    ax_rain = fig.add_axes([0.06, 0.82, 0.88, 0.14])
    bars = ax_rain.bar(
        np.arange(n_times), dataset.rain, width=0.85,
        color="steelblue", edgecolor="none", alpha=0.7,
    )
    # Highlight current bar
    if dataset.rain[t_idx] > 0:
        bars[t_idx].set_color("navy")
        bars[t_idx].set_alpha(1.0)
    ax_rain.axvline(t_idx, color="red", linewidth=1.5, linestyle="--", zorder=5)
    ax_rain.set_xlim(-0.5, n_times - 0.5)
    ax_rain.set_ylabel("Rain\n(in/hr)", fontsize=8)
    ax_rain.set_xticks([])
    ax_rain.set_title(
        f"Wet-Weather Event  —  {time_label}",
        fontsize=12, fontweight="bold", loc="center",
    )
    for sp in ["top", "right"]:
        ax_rain.spines[sp].set_visible(False)

    # ── Three tree panels ────────────────────────────────────
    panel_w, gap = 0.27, 0.035
    left_start = 0.04
    bottom, height = 0.08, 0.68

    panels = [
        (dataset.soft_matrix[:, t_idx], "SWMM Routed Prior", "YlOrRd"),
        (hard_only_mean[:, t_idx], "Hard-Data Conditional", "YlOrRd"),
        (hard_soft_mean[:, t_idx], "BME Estimate (Hard+Soft)", "YlOrRd"),
    ]

    for p_idx, (vals, title, cmap) in enumerate(panels):
        left = left_start + p_idx * (panel_w + gap)
        ax = fig.add_axes([left, bottom, panel_w, height])
        draw_tree_panel(ax, vals, title, cmap, vmin_g, vmax_g, t_idx)

    # ── Shared colour bar ────────────────────────────────────
    cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.68])
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(vmin_g, vmax_g))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Flow (MGD)", fontsize=9)

    # ── Legend ────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markeredgecolor="k", markersize=8, label="Hard obs (this step)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="k", markersize=7, label="Sensor (no obs)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    # Save frame as image buffer
    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    frames.append(buf.copy())
    plt.close(fig)

    if (t_idx + 1) % 6 == 0 or t_idx == n_times - 1:
        print(f"  frame {t_idx + 1}/{n_times}")

# ═══════════════════════════════════════════════════════════════════
# 5.  Stitch into GIF
# ═══════════════════════════════════════════════════════════════════
from PIL import Image

pil_frames = [Image.fromarray(f) for f in frames]
gif_path = os.path.join(out_dir, "bme_tree_event.gif")

# 350ms per frame, with a 1.2s pause on the peak-rain frame
peak_frame = int(np.argmax(dataset.rain))
durations = [350] * len(pil_frames)
durations[peak_frame] = 1200

pil_frames[0].save(
    gif_path,
    save_all=True,
    append_images=pil_frames[1:],
    duration=durations,
    loop=0,
    optimize=True,
)

print(f"\nGIF saved: {gif_path}")
print(f"  {len(pil_frames)} frames, {n_times} timesteps")
