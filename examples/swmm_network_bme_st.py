#!/usr/bin/env python3
"""
Space-Time Network-BME on a SWMM Sewer Model -- Onondaga County
================================================================

Extends the spatial-only example (``swmm_network_bme.py``) to the
**separable space-time** setting:

    C((i,t), (j,t')) = sigma2 * rho_s(i,j) * rho_t(|t - t'|)

where rho_s is the graph-Laplacian (Matern nu=1) network correlation
and rho_t is an exponential temporal covariance.

Key steps
---------
1.  Parse the SWMM .inp for the full network topology (421 nodes).
2.  Build ``NetworkCovariance`` (graph-Laplacian, kappa=0.1, unit
    weights) -- identical to the spatial-only example.
3.  Read the *full* meter time series (not just one snapshot) and
    build a (node, time) observation dataset.
4.  Wrap the spatial covariance in ``NetworkCovarianceST`` with an
    exponential temporal model.
5.  Predict at every node for a sequence of target times using
    ``bme_predict_network_st``.
6.  Produce figures: spatial maps at selected times, time series
    at selected nodes, and a spatio-temporal Hovmoller-style heatmap.

Data
----
Same files as the spatial example:
    SWMM model  : OC_2024-Conditions_5.1.010_V7-Calibrated_01282025.inp
    Observations: ObservedData.csv  (17 flow meters, 5-min, May 2024-Jun 2025)
    Meter map   : MeterLocations.csv

Requirements: pybme (in project venv), numpy, scipy, matplotlib
"""

import os, sys, warnings, time
from datetime import datetime, timedelta
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ensure pybme is importable ──────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path     = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.network import (
    NetworkCovariance, NetworkCovarianceST,
    adjacency_from_edges,
)
from pybme.predict import bme_predict_network_st
from pybme.network_plots import (
    plot_network_field,
    plot_network_observations,
)
from pybme.swmm import (
    build_edge_array,
    parse_swmm_inp,
    read_meter_node_map,
    read_observation_csv,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. PARSE THE SWMM .INP FILE  (identical to spatial example)
# ═══════════════════════════════════════════════════════════════════════════

PRIVATE_SWMM_DIR = os.path.join(script_dir, "private_swmm")
INP_PATH = os.environ.get("PYBME_SWMM_INP", os.path.join(PRIVATE_SWMM_DIR, "model.inp"))
OBS_PATH = os.environ.get("PYBME_SWMM_OBS", os.path.join(PRIVATE_SWMM_DIR, "ObservedData.csv"))
METER_LOC_PATH = os.environ.get("PYBME_SWMM_METER_MAP", os.path.join(PRIVATE_SWMM_DIR, "MeterLocations.csv"))


print("Parsing SWMM .inp file ...")
network = parse_swmm_inp(INP_PATH)
node_names = network.all_node_names
n_nodes = len(node_names)
node_idx = {name: i for i, name in enumerate(node_names)}
print(f"  {n_nodes} nodes, {len(network.edges)} links")

# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD GRAPH ADJACENCY AND SPATIAL NETWORK COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════
edge_array = build_edge_array(node_names, network.edges)

W = adjacency_from_edges(n_nodes, edge_array)  # unit weights
KAPPA = 0.1
SIGMA2_INIT = 1.0
net_cov = NetworkCovariance(W, kappa=KAPPA, sigma2=SIGMA2_INIT, from_adjacency=True)
print(f"  Spatial NetworkCovariance: kappa={KAPPA}, {n_nodes} nodes")

# ═══════════════════════════════════════════════════════════════════════════
# 3. READ THE FULL METER TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════

# ── 3a. Meter-to-node mapping ────────────────────────────────────────────
meter_node_map = read_meter_node_map(METER_LOC_PATH)

# ── 3b. Read full observation CSV ─────────────────────────────────────────
obs_table = read_observation_csv(OBS_PATH, value_type="flow")
rows = obs_table.rows

meter_names_hdr = obs_table.meter_names
flow_cols = obs_table.value_cols
flow_meter_names = obs_table.value_names
print(f"  {len(flow_cols)} flow meters: {flow_meter_names}")

# Build per-meter node index
meter_to_node_idx = {}
for meter_name in flow_meter_names:
    node_name = meter_node_map.get(meter_name)
    if node_name and node_name in node_idx:
        meter_to_node_idx[meter_name] = node_idx[node_name]

# Parse all timestamps and flow values
# Data row format: "M/D/YYYY H:MM", val1, val2, ...
# Time will be represented as hours from the first timestamp.
datetimes = obs_table.datetimes
valid_row_indices = obs_table.valid_row_indices

if not datetimes:
    print("ERROR: No valid timestamps found in observation data.")
    sys.exit(1)

t0 = datetimes[0]
times_hours = np.array([(dt - t0).total_seconds() / 3600.0 for dt in datetimes])
n_times_total = len(datetimes)
print(f"  {n_times_total} time steps from {datetimes[0]} to {datetimes[-1]}")
print(f"  Time range: {times_hours[-1]:.1f} hours "
      f"({times_hours[-1]/24:.1f} days)")

# Build full observation arrays: for each (meter, timestep) with valid data
# we get one (node_idx, time_hours, flow_value) triple.
all_obs_nodes = []
all_obs_times = []
all_obs_values = []
meter_labels = []  # unique meter labels (for dedup'd nodes)
dedup_nodes = {}   # node_idx -> position in hard_nodes list (for averaging)

for row_pos, row_idx in enumerate(valid_row_indices):
    for col_idx in flow_cols:
        meter_name = meter_names_hdr[col_idx]
        ni = meter_to_node_idx.get(meter_name)
        if ni is None:
            continue
        try:
            val = float(rows[row_idx][col_idx])
        except (ValueError, IndexError):
            continue
        if abs(val) < 0.001:
            continue  # skip inactive readings
        all_obs_nodes.append(ni)
        all_obs_times.append(times_hours[row_pos])
        all_obs_values.append(val)

all_obs_nodes = np.array(all_obs_nodes, dtype=int)
all_obs_times = np.array(all_obs_times, dtype=np.float64)
all_obs_values = np.array(all_obs_values, dtype=np.float64)

print(f"\n  Total space-time observations: {len(all_obs_values)}")
print(f"  Unique observed nodes: {len(np.unique(all_obs_nodes))}")
print(f"  Flow range: [{all_obs_values.min():.4f}, {all_obs_values.max():.4f}] MGD")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SELECT A TIME WINDOW AND TARGET TIMES
# ═══════════════════════════════════════════════════════════════════════════
# Using the full year of data is too large for dense BME.
# Select a 24-hour window with good meter coverage, then predict at
# hourly intervals.

# Find the 24-hour window with the most active observations
WINDOW_HOURS = 24.0
PRED_INTERVAL_HOURS = 1.0  # predict every hour

# Count observations in sliding 24h windows (1-hour steps)
window_starts = np.arange(0, times_hours[-1] - WINDOW_HOURS,
                          PRED_INTERVAL_HOURS)
best_window_count = 0
best_window_start = 0.0
for ws in window_starts:
    mask = (all_obs_times >= ws) & (all_obs_times < ws + WINDOW_HOURS)
    n_unique_nodes = len(np.unique(all_obs_nodes[mask])) if mask.any() else 0
    n_obs = mask.sum()
    # Score: prefer many unique nodes AND many observations
    score = n_unique_nodes * 10 + n_obs
    if score > best_window_count:
        best_window_count = score
        best_window_start = ws

t_start = best_window_start
t_end = t_start + WINDOW_HOURS

# Extract observations in this window
win_mask = (all_obs_times >= t_start) & (all_obs_times < t_end)
ch_nodes_st = all_obs_nodes[win_mask]
th_st       = all_obs_times[win_mask]
zh_st       = all_obs_values[win_mask]

# Determine the actual datetime for this window
start_dt = t0 + timedelta(hours=float(t_start))
end_dt = t0 + timedelta(hours=float(t_end))
unique_obs_nodes = np.unique(ch_nodes_st)

print(f"\nSelected 24h window: {start_dt} to {end_dt}")
print(f"  Observations in window: {len(zh_st)}")
print(f"  Unique observed nodes: {len(unique_obs_nodes)}")
print(f"  Flow stats: mean={zh_st.mean():.4f}, std={zh_st.std():.4f} MGD")

# Thin observations: for dense BME, limit to ~500 observations max.
# Sub-sample if needed (keep every Nth observation).
MAX_OBS = 500
if len(zh_st) > MAX_OBS:
    step = len(zh_st) // MAX_OBS + 1
    keep = np.arange(0, len(zh_st), step)
    ch_nodes_st = ch_nodes_st[keep]
    th_st = th_st[keep]
    zh_st = zh_st[keep]
    print(f"  Thinned to {len(zh_st)} observations (every {step}th)")

# Target prediction times: hourly within the window
tk_times = np.arange(t_start, t_end + 0.01, PRED_INTERVAL_HOURS)
n_pred_times = len(tk_times)
print(f"  Prediction times: {n_pred_times} (hourly from {t_start:.0f}h to {t_end:.0f}h)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. RESCALE COVARIANCE AND BUILD SPACE-TIME MODEL
# ═══════════════════════════════════════════════════════════════════════════

# Rescale spatial sigma2 to match data variance
data_var = float(np.var(zh_st))
data_mean = float(np.mean(zh_st))
diag_at_obs = net_cov.marginal_variance(unique_obs_nodes)
scale = data_var / max(diag_at_obs.mean(), 1e-12) if data_var > 0 else 1.0

net_cov_scaled = NetworkCovariance(
    W, kappa=KAPPA, sigma2=SIGMA2_INIT * scale, from_adjacency=True
)

# Temporal covariance: exponential with range ~ 6 hours
# This means observations are strongly correlated within ~2h and
# weakly correlated beyond 6h.  The "range" in our exponential model
# is defined so C(h) = sill * exp(-3h/range), i.e. correlation drops
# to ~5% at h = range.
TEMPORAL_MODEL = "exponential"
TEMPORAL_RANGE_HOURS = 6.0
TEMPORAL_PARAMS = [1.0, TEMPORAL_RANGE_HOURS]  # [sill=1, range]

net_cov_st = NetworkCovarianceST(
    net_cov_scaled,
    model_t=TEMPORAL_MODEL,
    params_t=TEMPORAL_PARAMS,
    sigma2=SIGMA2_INIT * scale,
)

print(f"\nData mean = {data_mean:.4f} MGD, data var = {data_var:.4f}")
print(f"Rescaled sigma2 = {SIGMA2_INIT * scale:.4f}")
print(f"Temporal model: {TEMPORAL_MODEL}, range = {TEMPORAL_RANGE_HOURS}h")

# ═══════════════════════════════════════════════════════════════════════════
# 6. SPACE-TIME BME PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
# Predict at ALL nodes x ALL target times.
# Build flattened arrays: (n_nodes * n_pred_times,) for ck_nodes and tk.

# For efficiency, predict at a subset of nodes (observed nodes + a sample
# of unobserved nodes) rather than all 421 x 25 = 10,525 points.
# This keeps the covariance blocks manageable.

# Select nodes to predict at: all observed + sample of unobserved
pred_nodes_set = set(unique_obs_nodes.tolist())
unobs_nodes = [i for i in range(n_nodes) if i not in pred_nodes_set]
N_UNOBS_SAMPLE = min(50, len(unobs_nodes))
rng = np.random.default_rng(42)
sampled_unobs = rng.choice(unobs_nodes, size=N_UNOBS_SAMPLE, replace=False)
pred_nodes = np.sort(np.concatenate([unique_obs_nodes, sampled_unobs]))
n_pred_nodes = len(pred_nodes)

print(f"\nPrediction grid: {n_pred_nodes} nodes x {n_pred_times} times "
      f"= {n_pred_nodes * n_pred_times} points")

# Build flattened estimation arrays
ck_flat = np.repeat(pred_nodes, n_pred_times)
tk_flat = np.tile(tk_times, n_pred_nodes)

print(f"Running space-time BME prediction ...")
t_wall_start = time.time()
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
t_wall = time.time() - t_wall_start
print(f"  Done in {t_wall:.1f}s ({len(results)} predictions)")

# Reshape results into (n_pred_nodes, n_pred_times) grids
bme_mean_st = np.array([r.mean for r in results]).reshape(n_pred_nodes, n_pred_times)
bme_std_st  = np.array([np.sqrt(r.variance) for r in results]).reshape(n_pred_nodes, n_pred_times)

print(f"  BME mean range: [{bme_mean_st.min():.4f}, {bme_mean_st.max():.4f}] MGD")
print(f"  BME std  range: [{bme_std_st.min():.4f}, {bme_std_st.max():.4f}] MGD")

# ═══════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ═══════════════════════════════════════════════════════════════════════════
out_dir = os.path.join(script_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# Helper: map pred_nodes index to node name
pred_node_names = [node_names[ni] for ni in pred_nodes]

# ── Figure A: Spatial map at 3 time slices ─────────────────────────────────
time_slices = [0, n_pred_times // 2, n_pred_times - 1]
fig_a, axes_a = plt.subplots(1, len(time_slices), figsize=(18, 6),
                              constrained_layout=True)

for ax, ti in zip(axes_a, time_slices):
    # Build full-node value array (NaN for unpredicted nodes)
    vals = np.full(n_nodes, np.nan)
    vals[pred_nodes] = bme_mean_st[:, ti]

    # Only plot nodes that were predicted
    has_val = np.isfinite(vals)
    plot_x = [network["coords"][node_names[i]][0] for i in range(n_nodes)
              if has_val[i] and node_names[i] in network["coords"]]
    plot_y = [network["coords"][node_names[i]][1] for i in range(n_nodes)
              if has_val[i] and node_names[i] in network["coords"]]
    plot_v = [vals[i] for i in range(n_nodes)
              if has_val[i] and node_names[i] in network["coords"]]

    vmax = np.percentile([v for v in plot_v if v > 0], 95) if plot_v else 1.0
    sc = ax.scatter(plot_x, plot_y, c=plot_v, cmap="YlOrRd", s=15,
                    edgecolor="none", vmin=0, vmax=vmax, zorder=3)

    # Draw edges
    for fn, tn, _ in network["edges"]:
        if fn in network["coords"] and tn in network["coords"]:
            x0, y0 = network["coords"][fn]
            x1, y1 = network["coords"][tn]
            ax.plot([x0, x1], [y0, y1], "grey", linewidth=0.2, alpha=0.3)

    t_hr = tk_times[ti]
    dt_label = (t0 + timedelta(hours=float(t_hr))).strftime("%m/%d %H:%M")
    ax.set_title(f"t = {dt_label}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (ft)")
    if ti == time_slices[0]:
        ax.set_ylabel("Northing (ft)")

plt.colorbar(sc, ax=list(axes_a), label="BME Mean Flow (MGD)", shrink=0.7)
fig_a.suptitle("Space-Time BME: Posterior Mean at 3 Time Slices", fontsize=13)
fig_a.savefig(os.path.join(out_dir, "fig_swmm_ST_A_spatial_snapshots.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_swmm_ST_A_spatial_snapshots.png")

# ── Figure B: Time series at selected nodes ────────────────────────────────
# Pick 4 observed nodes to show BME posterior mean vs actual observations
obs_node_list = sorted(set(ch_nodes_st.tolist()))
show_nodes = obs_node_list[:min(4, len(obs_node_list))]

fig_b, axes_b = plt.subplots(len(show_nodes), 1, figsize=(12, 3 * len(show_nodes)),
                              sharex=True, constrained_layout=True)
if len(show_nodes) == 1:
    axes_b = [axes_b]

for ax, ni in zip(axes_b, show_nodes):
    # Find this node in pred_nodes
    pred_pos = np.where(pred_nodes == ni)[0]
    if len(pred_pos) == 0:
        continue
    pred_pos = pred_pos[0]

    # BME time series
    bme_ts = bme_mean_st[pred_pos, :]
    bme_std_ts = bme_std_st[pred_pos, :]

    # Convert tk_times to datetime labels
    tk_dt = [t0 + timedelta(hours=float(t)) for t in tk_times]

    ax.fill_between(tk_dt, bme_ts - 2 * bme_std_ts, bme_ts + 2 * bme_std_ts,
                    alpha=0.2, color="steelblue", label="95% CI")
    ax.plot(tk_dt, bme_ts, "b-", linewidth=1.5, label="BME mean")

    # Overlay actual observations at this node
    obs_mask = ch_nodes_st == ni
    if obs_mask.any():
        obs_dt = [t0 + timedelta(hours=float(t)) for t in th_st[obs_mask]]
        ax.scatter(obs_dt, zh_st[obs_mask], c="red", s=15, zorder=5,
                   label="Observed", edgecolor="k", linewidth=0.3)

    ax.set_ylabel("Flow (MGD)")
    ax.set_title(f"Node: {node_names[ni]}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

axes_b[-1].set_xlabel("Time")
fig_b.suptitle("Space-Time BME: Time Series at Observed Nodes", fontsize=13)
fig_b.savefig(os.path.join(out_dir, "fig_swmm_ST_B_timeseries.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_swmm_ST_B_timeseries.png")

# ── Figure C: Hovmoller-style heatmap (node vs time) ──────────────────────
fig_c, axes_c = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

# Panel 1: BME mean
vmax_mean = np.percentile(bme_mean_st[bme_mean_st > 0], 95) if (bme_mean_st > 0).any() else 1.0
im1 = axes_c[0].imshow(bme_mean_st, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=vmax_mean,
                        extent=[tk_times[0], tk_times[-1], n_pred_nodes, 0],
                        interpolation="nearest")
plt.colorbar(im1, ax=axes_c[0], label="BME Mean Flow (MGD)", shrink=0.8)
axes_c[0].set_ylabel("Prediction node index")
axes_c[0].set_title("(a) BME Posterior Mean Flow")

# Panel 2: BME std
vmax_std = np.percentile(bme_std_st, 95)
im2 = axes_c[1].imshow(bme_std_st, aspect="auto", cmap="Blues",
                        vmin=0, vmax=vmax_std,
                        extent=[tk_times[0], tk_times[-1], n_pred_nodes, 0],
                        interpolation="nearest")
plt.colorbar(im2, ax=axes_c[1], label="BME Std. Dev. (MGD)", shrink=0.8)
axes_c[1].set_ylabel("Prediction node index")
axes_c[1].set_xlabel(f"Time (hours from {datetimes[0].strftime('%m/%d/%Y')})")
axes_c[1].set_title("(b) BME Posterior Std. Dev.")

fig_c.suptitle("Space-Time BME: Hovmoller Diagram (Node vs Time)",
               fontsize=13, fontweight="bold")
fig_c.savefig(os.path.join(out_dir, "fig_swmm_ST_C_hovmoller.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_swmm_ST_C_hovmoller.png")

# ── Figure D: Uncertainty reduction over time ─────────────────────────────
fig_d, ax_d = plt.subplots(figsize=(10, 5))

# Mean std across all prediction nodes at each time step
mean_std_by_time = bme_std_st.mean(axis=0)
median_std_by_time = np.median(bme_std_st, axis=0)
p25 = np.percentile(bme_std_st, 25, axis=0)
p75 = np.percentile(bme_std_st, 75, axis=0)

tk_dt = [t0 + timedelta(hours=float(t)) for t in tk_times]
ax_d.fill_between(tk_dt, p25, p75, alpha=0.3, color="steelblue",
                  label="IQR (25th-75th percentile)")
ax_d.plot(tk_dt, mean_std_by_time, "b-", linewidth=1.5, label="Mean std")
ax_d.plot(tk_dt, median_std_by_time, "b--", linewidth=1, label="Median std")

# Mark times when observations are available
obs_times_unique = np.unique(th_st)
for ot in obs_times_unique:
    ot_dt = t0 + timedelta(hours=float(ot))
    ax_d.axvline(ot_dt, color="red", alpha=0.05, linewidth=0.5)

ax_d.set_xlabel("Time")
ax_d.set_ylabel("BME Posterior Std. Dev. (MGD)")
ax_d.set_title("Uncertainty Across Network Over Time")
ax_d.legend(fontsize=9)
ax_d.grid(True, alpha=0.3)
fig_d.tight_layout()
fig_d.savefig(os.path.join(out_dir, "fig_swmm_ST_D_uncertainty.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_swmm_ST_D_uncertainty.png")

# ═══════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY -- Space-Time Network-BME on SWMM Sewer Model")
print("=" * 70)
print(f"Network: {n_nodes} nodes, {len(network['edges'])} links")
print(f"Spatial covariance: graph-Laplacian, kappa={KAPPA}")
print(f"Temporal covariance: {TEMPORAL_MODEL}, range={TEMPORAL_RANGE_HOURS}h")
print(f"Rescaled sigma2: {SIGMA2_INIT * scale:.4f}")
print(f"Time window: {start_dt} to {end_dt} ({WINDOW_HOURS}h)")
print(f"Observations in window: {len(zh_st)} "
      f"({len(unique_obs_nodes)} unique nodes)")
print(f"Prediction grid: {n_pred_nodes} nodes x {n_pred_times} times "
      f"= {n_pred_nodes * n_pred_times} points")
print(f"Wall time: {t_wall:.1f}s")
print(f"BME mean range: [{bme_mean_st.min():.4f}, {bme_mean_st.max():.4f}] MGD")
print(f"BME std  range: [{bme_std_st.min():.4f}, {bme_std_st.max():.4f}] MGD")
print("=" * 70)

plt.show()
