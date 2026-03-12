#!/usr/bin/env python3
"""
SWMM-as-Soft-Data: Network BME with Physics-Based Prior
========================================================

Fuses a calibrated SWMM model (physics-based prior) with field meter
observations (hard data) using BME on the network graph.

Concept
-------
At each node and time step the SWMM simulation provides a modelled
flow estimate.  At meter locations we also have real observations.
The residual between SWMM and the meters defines how much to trust
the model at each location:

    soft PDF at node i, time t  ~  N(Q_swmm(i,t),  sigma_i^2)

where sigma_i is estimated from the SWMM-vs-observed residual
variance at each meter.  For unobserved nodes we use the network-wide
median residual variance.

Hard data are the meter readings themselves.

Steps
-----
1.  Parse SWMM .inp for the network graph (reused from other examples).
2.  Read SWMM .out via ``swmm_bridge`` to get modelled flow at every
    node and every 15-min time step.
3.  Read observed meter data and compute per-meter residual statistics.
4.  Build ``SoftPDF`` objects (Gaussian) at all nodes from SWMM output,
    with variance set by the local residual variance.
5.  Run ``bme_predict_network_st`` with hard data (meters) + soft data
    (SWMM) to obtain the posterior field.
6.  Figures comparing SWMM-only vs BME-corrected fields.

Data
----
SWMM .inp + .out :  OC_2024-Conditions_5.1.010_V7-Calibrated_01282025
Observations     :  ObservedData.csv  (17 flow meters, 5-min)
Meter map        :  MeterLocations.csv

Requirements: pybme, swmm_bridge (SWMMpy_RWR), numpy, scipy, matplotlib
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
    NetworkCovariance, NetworkCovarianceST, adjacency_from_edges,
)
from pybme.predict import bme_predict_network_st
from pybme.soft_data import SoftPDF
from pybme.network_plots import plot_network_field
from pybme.swmm import (
    build_edge_array,
    nearest_timeseries_value,
    parse_swmm_inp,
    read_meter_node_map,
    read_observation_csv,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════
PRIVATE_SWMM_DIR = os.path.join(script_dir, "private_swmm")
INP_PATH = os.environ.get("PYBME_SWMM_INP", os.path.join(PRIVATE_SWMM_DIR, "model.inp"))
OUT_PATH = os.environ.get("PYBME_SWMM_OUT", os.path.join(PRIVATE_SWMM_DIR, "model.out"))
OBS_PATH = os.environ.get("PYBME_SWMM_OBS", os.path.join(PRIVATE_SWMM_DIR, "ObservedData.csv"))
METER_LOC_PATH = os.environ.get("PYBME_SWMM_METER_MAP", os.path.join(PRIVATE_SWMM_DIR, "MeterLocations.csv"))

# ═══════════════════════════════════════════════════════════════════════════
# 2. PARSE SWMM .INP FOR GRAPH TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SWMM-as-Soft-Data: Network BME with Physics-Based Prior")
print("=" * 70)

print("\n[1] Parsing SWMM .inp ...")
network = parse_swmm_inp(INP_PATH)
node_names = network.all_node_names
n_nodes = len(node_names)
node_idx = {name: i for i, name in enumerate(node_names)}
print(f"    {n_nodes} nodes, {len(network.edges)} links")

# ═══════════════════════════════════════════════════════════════════════════
# 3. BUILD GRAPH ADJACENCY AND COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════
edge_array = build_edge_array(node_names, network.edges)

W = adjacency_from_edges(n_nodes, edge_array)
KAPPA = 0.1

# Initial sigma2=1; will be rescaled after computing residuals
net_cov_base = NetworkCovariance(W, kappa=KAPPA, sigma2=1.0, from_adjacency=True)
print(f"    Spatial covariance: graph-Laplacian, kappa={KAPPA}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. READ SWMM OUTPUT (modelled flow at every node)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Reading SWMM .out ...")
from swmm_bridge import get_output_reader
from epaswmm import output as epa_out

reader = get_output_reader(OUT_PATH)
swmm_times = reader.times  # list of datetime
swmm_t0 = swmm_times[0]
n_swmm_steps = len(swmm_times)
print(f"    {n_swmm_steps} time steps, "
      f"{swmm_times[0]} to {swmm_times[-1]}")
print(f"    Flow units: {reader.flow_units}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. READ METER LOCATIONS AND OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Reading observations ...")

meter_node_map = read_meter_node_map(METER_LOC_PATH)

obs_table = read_observation_csv(OBS_PATH, value_type="flow")
obs_rows = obs_table.rows

meter_names_hdr = obs_table.meter_names
flow_cols = obs_table.value_cols
flow_meter_names = obs_table.value_names
print(f"    {len(flow_cols)} flow meters")

# Parse all observation timestamps
obs_datetimes = obs_table.datetimes
obs_valid_rows = obs_table.valid_row_indices

obs_t0 = obs_datetimes[0]
print(f"    Obs range: {obs_datetimes[0]} to {obs_datetimes[-1]}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. SELECT TIME WINDOW  (overlap between SWMM and observations)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] Selecting time window ...")

# Overlap period
overlap_start = max(swmm_times[0], obs_datetimes[0])
overlap_end   = min(swmm_times[-1], obs_datetimes[-1])
print(f"    Overlap: {overlap_start} to {overlap_end}")

# Use a 24-hour window centred on Mar 13 2025 (best meter coverage from
# the spatial example).  Fall back to the middle of the overlap if that
# date is outside the range.
TARGET_CENTRE = datetime(2025, 3, 13, 13, 0)
WINDOW_HOURS = 24.0

if TARGET_CENTRE < overlap_start or TARGET_CENTRE > overlap_end:
    TARGET_CENTRE = overlap_start + (overlap_end - overlap_start) / 2

win_start = TARGET_CENTRE - timedelta(hours=WINDOW_HOURS / 2)
win_end   = TARGET_CENTRE + timedelta(hours=WINDOW_HOURS / 2)
print(f"    Analysis window: {win_start} to {win_end}")

# Common t0 for this analysis (hours from win_start)
def dt_to_hours(dt):
    return (dt - win_start).total_seconds() / 3600.0

# ═══════════════════════════════════════════════════════════════════════════
# 7. BUILD PER-METER RESIDUAL STATISTICS
# ═══════════════════════════════════════════════════════════════════════════
# For each meter, compare SWMM TOTAL_INFLOW at the meter's node against
# the observed flow.  The residual variance tells us how much to trust
# SWMM at that location.
print("\n[5] Computing per-meter residual statistics ...")

# Build observed time series per meter (only in overlap period)
# obs_by_meter[meter_name] = {datetime: flow_mgd}
obs_by_meter = {}
for row_pos, row_idx in enumerate(obs_valid_rows):
    dt = obs_datetimes[row_pos]
    if dt < overlap_start or dt > overlap_end:
        continue
    for col_idx in flow_cols:
        meter_name = meter_names_hdr[col_idx]
        try:
            val = float(obs_rows[row_idx][col_idx])
        except (ValueError, IndexError):
            continue
        if abs(val) < 0.001:
            continue
        obs_by_meter.setdefault(meter_name, {})[dt] = val

# Get SWMM timeseries at each meter node and compute residuals
residual_stats = {}  # meter -> {mean_resid, var_resid, n_matched}
swmm_by_meter = {}

for meter_name in flow_meter_names:
    node_name = meter_node_map.get(meter_name)
    if node_name is None or node_name not in node_idx:
        continue

    # Get SWMM output for this node
    try:
        swmm_ts = reader.get_node_timeseries(
            node_name, epa_out.NodeAttribute.TOTAL_INFLOW)
    except Exception:
        continue

    swmm_by_meter[meter_name] = swmm_ts

    if meter_name not in obs_by_meter:
        continue

    # Match timestamps: find closest SWMM timestep for each obs
    swmm_dt_list = list(swmm_ts.keys())
    swmm_dt_arr = np.array([(dt - overlap_start).total_seconds()
                             for dt in swmm_dt_list])

    residuals = []
    for obs_dt, obs_val in obs_by_meter[meter_name].items():
        obs_sec = (obs_dt - overlap_start).total_seconds()
        idx_nearest = np.argmin(np.abs(swmm_dt_arr - obs_sec))
        # Only match if within 10 minutes
        if abs(swmm_dt_arr[idx_nearest] - obs_sec) < 600:
            swmm_val = swmm_ts[swmm_dt_list[idx_nearest]]
            residuals.append(obs_val - swmm_val)

    if len(residuals) >= 10:
        residuals = np.array(residuals)
        residual_stats[meter_name] = {
            "mean": float(residuals.mean()),
            "var": float(residuals.var()),
            "std": float(residuals.std()),
            "n": len(residuals),
            "node": node_name,
        }

print(f"    Residual stats computed for {len(residual_stats)} meters:")
for m, s in residual_stats.items():
    nn = s["node"]
    print(f"      {m:35s} node={nn:20s}  "
          f"bias={s['mean']:+.4f}  std={s['std']:.4f} MGD  (n={s['n']})")

# Median residual variance across all meters (for unobserved nodes)
all_resid_vars = [s["var"] for s in residual_stats.values()]
median_resid_var = float(np.median(all_resid_vars)) if all_resid_vars else 1.0
print(f"    Median residual variance: {median_resid_var:.4f} MGD^2")

# ═══════════════════════════════════════════════════════════════════════════
# 8. BUILD HARD AND SOFT DATA FOR BME
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] Building hard + soft data arrays ...")

# Target prediction times: every hour within the window
PRED_INTERVAL = 1.0  # hours
pred_times_h = np.arange(0, WINDOW_HOURS + 0.01, PRED_INTERVAL)
n_pred_times = len(pred_times_h)

# Prediction datetimes
pred_datetimes = [win_start + timedelta(hours=float(t)) for t in pred_times_h]

# ── HARD DATA: meter observations in the window ──────────────────────────
hard_nodes_list = []
hard_times_list = []
hard_values_list = []
hard_labels = []

# Map meter to node index, handling duplicates (same as spatial example)
meter_to_node = {}
for meter_name in flow_meter_names:
    nn = meter_node_map.get(meter_name)
    if nn and nn in node_idx:
        meter_to_node[meter_name] = node_idx[nn]

# Collect hard data within the window, sub-sampling to keep manageable
for row_pos, row_idx in enumerate(obs_valid_rows):
    dt = obs_datetimes[row_pos]
    if dt < win_start or dt >= win_end:
        continue
    t_h = dt_to_hours(dt)
    for col_idx in flow_cols:
        meter_name = meter_names_hdr[col_idx]
        ni = meter_to_node.get(meter_name)
        if ni is None:
            continue
        try:
            val = float(obs_rows[row_idx][col_idx])
        except (ValueError, IndexError):
            continue
        if abs(val) < 0.001:
            continue
        hard_nodes_list.append(ni)
        hard_times_list.append(t_h)
        hard_values_list.append(val)

ch_nodes = np.array(hard_nodes_list, dtype=int)
th_hard  = np.array(hard_times_list, dtype=np.float64)
zh_hard  = np.array(hard_values_list, dtype=np.float64)

# Thin hard data if needed
MAX_HARD = 300
if len(zh_hard) > MAX_HARD:
    step = len(zh_hard) // MAX_HARD + 1
    keep = np.arange(0, len(zh_hard), step)
    ch_nodes = ch_nodes[keep]
    th_hard  = th_hard[keep]
    zh_hard  = zh_hard[keep]

print(f"    Hard data: {len(zh_hard)} observations "
      f"({len(np.unique(ch_nodes))} unique nodes)")

# ── SOFT DATA: SWMM model output at ALL nodes x pred times ──────────────
# For each (node, pred_time) we create a Gaussian SoftPDF centred at the
# SWMM modelled flow, with variance from the residual analysis.
#
# At meter nodes: use that meter's residual variance.
# At other nodes: use the median residual variance.

# Build a lookup: node_name -> residual variance
node_resid_var = {}
for m, s in residual_stats.items():
    nn = s["node"]
    # If multiple meters share a node, average their variances
    if nn in node_resid_var:
        node_resid_var[nn] = 0.5 * (node_resid_var[nn] + s["var"])
    else:
        node_resid_var[nn] = s["var"]

# Limit soft data to a manageable subset of nodes
# Use all observed nodes + a random sample of unobserved nodes
obs_node_set = set(ch_nodes.tolist())
unobs_list = [i for i in range(n_nodes) if i not in obs_node_set]
N_SOFT_SAMPLE = min(40, len(unobs_list))
rng = np.random.default_rng(42)
soft_sample_nodes = sorted(
    list(obs_node_set) + rng.choice(unobs_list, N_SOFT_SAMPLE, replace=False).tolist()
)

# Pre-load SWMM timeseries for all soft nodes (dict: node_name -> {dt: flow})
swmm_cache = {}
for ni in soft_sample_nodes:
    nn = node_names[ni]
    if nn in swmm_cache:
        continue
    try:
        swmm_cache[nn] = reader.get_node_timeseries(
            nn, epa_out.NodeAttribute.TOTAL_INFLOW)
    except Exception:
        pass

def get_swmm_flow(node_name, target_dt):
    """Get SWMM flow at the nearest timestep to target_dt."""
    ts = swmm_cache.get(node_name)
    return nearest_timeseries_value(ts, target_dt, max_diff_seconds=900.0)


# Build soft data arrays
cs_nodes_list = []
ts_soft_list  = []
soft_pdfs_list = []

MIN_VAR = 0.01  # floor on soft-data variance to avoid degenerate PDFs

for ni in soft_sample_nodes:
    nn = node_names[ni]
    local_var = node_resid_var.get(nn, median_resid_var)
    local_var = max(local_var, MIN_VAR)

    for ti, pred_dt in enumerate(pred_datetimes):
        swmm_flow = get_swmm_flow(nn, pred_dt)
        if swmm_flow is None:
            continue
        # Ensure the Gaussian is truncated at 0 (flow can't be negative)
        # Use from_truncnorm with a=0 for physical consistency
        soft_pdf = SoftPDF.from_truncnorm(
            mu=float(swmm_flow),
            sigma=float(np.sqrt(local_var)),
            a=0.0,    # lower bound = 0 (non-negative flow)
            b=None,   # no upper bound
        )
        cs_nodes_list.append(ni)
        ts_soft_list.append(pred_times_h[ti])
        soft_pdfs_list.append(soft_pdf)

cs_nodes = np.array(cs_nodes_list, dtype=int)
ts_soft  = np.array(ts_soft_list, dtype=np.float64)

print(f"    Soft data: {len(soft_pdfs_list)} SWMM-based PDFs "
      f"({len(soft_sample_nodes)} nodes x {n_pred_times} times)")

# ═══════════════════════════════════════════════════════════════════════════
# 9. RESCALE COVARIANCE AND BUILD SPACE-TIME MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7] Building space-time covariance ...")

# sigma2 based on data variance
data_var = float(np.var(zh_hard)) if len(zh_hard) > 1 else 1.0
data_mean = float(np.mean(zh_hard)) if len(zh_hard) > 0 else 0.0
diag_obs = net_cov_base.marginal_variance(np.unique(ch_nodes))
scale = data_var / max(diag_obs.mean(), 1e-12) if data_var > 0 else 1.0

net_cov_scaled = NetworkCovariance(
    W, kappa=KAPPA, sigma2=scale, from_adjacency=True)

TEMPORAL_MODEL = "exponential"
TEMPORAL_RANGE = 6.0   # hours
TEMPORAL_PARAMS = [1.0, TEMPORAL_RANGE]

net_cov_st = NetworkCovarianceST(
    net_cov_scaled,
    model_t=TEMPORAL_MODEL,
    params_t=TEMPORAL_PARAMS,
    sigma2=scale,
)
print(f"    sigma2={scale:.4f}, temporal range={TEMPORAL_RANGE}h")

# ═══════════════════════════════════════════════════════════════════════════
# 10. BME PREDICTION: HARD + SOFT
# ═══════════════════════════════════════════════════════════════════════════
# Predict at the soft-data nodes x prediction times

pred_nodes = np.array(sorted(soft_sample_nodes), dtype=int)
n_pred_nodes = len(pred_nodes)

ck_flat = np.repeat(pred_nodes, n_pred_times)
tk_flat = np.tile(pred_times_h, n_pred_nodes)

# Limit soft data per prediction point to keep integration tractable
NSMAX = 6
N_GRID = 100   # z-grid resolution (100 is plenty for truncated normals)

print(f"\n[8] Running BME (hard + SWMM soft data) ...")
print(f"    {n_pred_nodes} nodes x {n_pred_times} times = "
      f"{len(ck_flat)} prediction points")
print(f"    Hard: {len(zh_hard)}, Soft: {len(soft_pdfs_list)}")

t_wall = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results_bme = bme_predict_network_st(
        ck_nodes   = ck_flat,
        tk         = tk_flat,
        ch_nodes   = ch_nodes,
        th         = th_hard,
        zh         = zh_hard,
        cs_nodes   = cs_nodes,
        ts         = ts_soft,
        soft_pdfs  = soft_pdfs_list,
        net_cov_st = net_cov_st,
        nhmax      = 30,
        nsmax      = NSMAX,
        n_grid     = N_GRID,
        order      = 0,
        mean_prior = data_mean,
        method     = "laplace",
    )
t_wall = time.time() - t_wall

bme_mean = np.array([r.mean for r in results_bme]).reshape(n_pred_nodes, n_pred_times)
bme_std  = np.array([np.sqrt(r.variance) for r in results_bme]).reshape(n_pred_nodes, n_pred_times)
print(f"    Done in {t_wall:.1f}s")
print(f"    BME mean: [{bme_mean.min():.4f}, {bme_mean.max():.4f}] MGD")
print(f"    BME std:  [{bme_std.min():.4f}, {bme_std.max():.4f}] MGD")

# ═══════════════════════════════════════════════════════════════════════════
# 11. ALSO RUN HARD-ONLY BME FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n[9] Running BME (hard only, no SWMM) for comparison ...")
t_wall2 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results_hard_only = bme_predict_network_st(
        ck_nodes   = ck_flat,
        tk         = tk_flat,
        ch_nodes   = ch_nodes,
        th         = th_hard,
        zh         = zh_hard,
        net_cov_st = net_cov_st,
        nhmax      = 30,
        order      = 0,
        mean_prior = data_mean,
    )
t_wall2 = time.time() - t_wall2

bme_mean_ho = np.array([r.mean for r in results_hard_only]).reshape(n_pred_nodes, n_pred_times)
bme_std_ho  = np.array([np.sqrt(r.variance) for r in results_hard_only]).reshape(n_pred_nodes, n_pred_times)
print(f"    Done in {t_wall2:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# 12. FIGURES
# ═══════════════════════════════════════════════════════════════════════════
out_dir = os.path.join(script_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# ── Figure A: Time series comparison at 4 observed nodes ──────────────────
obs_in_pred = [ni for ni in np.unique(ch_nodes) if ni in set(pred_nodes)]
show_nodes = obs_in_pred[:min(4, len(obs_in_pred))]

fig_a, axes_a = plt.subplots(len(show_nodes), 1,
                              figsize=(14, 3.5 * len(show_nodes)),
                              sharex=True, constrained_layout=True)
if len(show_nodes) == 1:
    axes_a = [axes_a]

for ax, ni in zip(axes_a, show_nodes):
    nn = node_names[ni]
    pi = np.where(pred_nodes == ni)[0]
    if len(pi) == 0:
        continue
    pi = pi[0]

    tk_dt = [win_start + timedelta(hours=float(t)) for t in pred_times_h]

    # BME with soft data
    ax.fill_between(tk_dt,
                    bme_mean[pi] - 2 * bme_std[pi],
                    bme_mean[pi] + 2 * bme_std[pi],
                    alpha=0.15, color="steelblue")
    ax.plot(tk_dt, bme_mean[pi], "b-", linewidth=1.5,
            label="BME (hard + SWMM soft)")

    # BME hard-only
    ax.plot(tk_dt, bme_mean_ho[pi], "g--", linewidth=1,
            label="BME (hard only)")

    # SWMM model output
    swmm_ts = swmm_cache.get(nn, {})
    swmm_dt_win = [(dt, v) for dt, v in swmm_ts.items()
                   if win_start <= dt <= win_end]
    if swmm_dt_win:
        sx, sy = zip(*swmm_dt_win)
        ax.plot(sx, sy, "m:", linewidth=1, alpha=0.8, label="SWMM model")

    # Observed hard data
    obs_mask = ch_nodes == ni
    if obs_mask.any():
        obs_dt = [win_start + timedelta(hours=float(t)) for t in th_hard[obs_mask]]
        ax.scatter(obs_dt, zh_hard[obs_mask], c="red", s=12, zorder=5,
                   label="Observed", edgecolor="k", linewidth=0.3)

    ax.set_ylabel("Flow (MGD)")
    ax.set_title(f"Node: {nn}", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

axes_a[-1].set_xlabel("Time")
fig_a.suptitle("SWMM-as-Soft-Data: BME Time Series vs SWMM vs Observations",
               fontsize=13, fontweight="bold")
fig_a.savefig(os.path.join(out_dir, "fig_swmm_soft_A_timeseries.png"),
              dpi=200, bbox_inches="tight")
print("\nSaved fig_swmm_soft_A_timeseries.png")

# ── Figure B: Std reduction (soft data vs hard-only) ─────────────────────
fig_b, ax_b = plt.subplots(figsize=(10, 5))

mean_std_bme = bme_std.mean(axis=0)
mean_std_ho  = bme_std_ho.mean(axis=0)
tk_dt = [win_start + timedelta(hours=float(t)) for t in pred_times_h]

ax_b.plot(tk_dt, mean_std_ho, "g-", linewidth=1.5, label="Hard only")
ax_b.plot(tk_dt, mean_std_bme, "b-", linewidth=1.5, label="Hard + SWMM soft")
ax_b.fill_between(tk_dt, mean_std_bme, mean_std_ho, alpha=0.2, color="orange",
                  label="Uncertainty reduction from SWMM")
ax_b.set_ylabel("Mean Posterior Std. Dev. (MGD)")
ax_b.set_xlabel("Time")
ax_b.set_title("Network-Average Uncertainty: SWMM Soft Data vs Hard Only")
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3)
fig_b.tight_layout()
fig_b.savefig(os.path.join(out_dir, "fig_swmm_soft_B_uncertainty_reduction.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_swmm_soft_B_uncertainty_reduction.png")

# ── Figure C: Spatial snapshot comparison (mid-window) ────────────────────
mid_ti = n_pred_times // 2
mid_dt = pred_datetimes[mid_ti]

fig_c, axes_c = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

# Helper to scatter pred nodes on map
def scatter_pred(ax, values, title, cmap="YlOrRd"):
    for fn, tn, _ in network.edges:
        if fn in network.coords and tn in network.coords:
            x0, y0 = network.coords[fn]
            x1, y1 = network.coords[tn]
            ax.plot([x0, x1], [y0, y1], "grey", lw=0.2, alpha=0.3)
    px = [network.coords[node_names[ni]][0] for ni in pred_nodes
          if node_names[ni] in network.coords]
    py = [network.coords[node_names[ni]][1] for ni in pred_nodes
          if node_names[ni] in network.coords]
    pv = [values[j] for j, ni in enumerate(pred_nodes)
          if node_names[ni] in network.coords]
    vmax = np.percentile([v for v in pv if v > 0], 95) if pv else 1.0
    sc = ax.scatter(px, py, c=pv, cmap=cmap, s=20, edgecolor="none",
                    vmin=0, vmax=vmax, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (ft)")
    return sc

# Panel 1: SWMM model at mid-window
swmm_vals = np.zeros(n_pred_nodes)
for j, ni in enumerate(pred_nodes):
    nn = node_names[ni]
    v = get_swmm_flow(nn, mid_dt)
    swmm_vals[j] = v if v is not None else np.nan
sc1 = scatter_pred(axes_c[0], swmm_vals, f"SWMM Model\n{mid_dt}")
axes_c[0].set_ylabel("Northing (ft)")

# Panel 2: BME hard-only
sc2 = scatter_pred(axes_c[1], bme_mean_ho[:, mid_ti], f"BME Hard Only\n{mid_dt}")

# Panel 3: BME hard + SWMM soft
sc3 = scatter_pred(axes_c[2], bme_mean[:, mid_ti],
                   f"BME Hard + SWMM Soft\n{mid_dt}")

plt.colorbar(sc3, ax=list(axes_c), label="Flow (MGD)", shrink=0.7)
fig_c.suptitle("Spatial Comparison: SWMM vs Hard-Only vs Hard+Soft BME",
               fontsize=13, fontweight="bold")
fig_c.savefig(os.path.join(out_dir, "fig_swmm_soft_C_spatial_comparison.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_swmm_soft_C_spatial_comparison.png")

# ── Figure D: Residual analysis (SWMM model error at meters) ─────────────
fig_d, axes_d = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Panel 1: residual std by meter
meters_sorted = sorted(residual_stats.keys(),
                       key=lambda m: residual_stats[m]["std"])
ax_d1 = axes_d[0]
y_pos = np.arange(len(meters_sorted))
stds = [residual_stats[m]["std"] for m in meters_sorted]
biases = [residual_stats[m]["mean"] for m in meters_sorted]
ax_d1.barh(y_pos, stds, color="steelblue", alpha=0.7, label="Residual Std")
ax_d1.barh(y_pos, [abs(b) for b in biases], color="orange", alpha=0.7,
           label="|Bias|")
ax_d1.set_yticks(y_pos)
ax_d1.set_yticklabels([m[:25] for m in meters_sorted], fontsize=7)
ax_d1.set_xlabel("MGD")
ax_d1.set_title("SWMM Model Error by Meter")
ax_d1.legend(fontsize=8)

# Panel 2: histogram of all residuals
ax_d2 = axes_d[1]
all_resid = []
for m, s in residual_stats.items():
    all_resid.extend([s["mean"]] * s["n"])  # approximate
ax_d2.hist([s["mean"] for s in residual_stats.values()],
           bins=15, color="steelblue", alpha=0.7, edgecolor="k")
ax_d2.axvline(0, color="red", linestyle="--")
ax_d2.set_xlabel("Mean Residual (Obs - SWMM) [MGD]")
ax_d2.set_ylabel("Count (meters)")
ax_d2.set_title("Distribution of SWMM-vs-Observed Bias")

fig_d.suptitle("SWMM Model Error Analysis (basis for soft-data variance)",
               fontsize=12)
fig_d.savefig(os.path.join(out_dir, "fig_swmm_soft_D_residuals.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_swmm_soft_D_residuals.png")

# ═══════════════════════════════════════════════════════════════════════════
# 13. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
pct_reduction = 100 * (1 - bme_std.mean() / max(bme_std_ho.mean(), 1e-12))
print("\n" + "=" * 70)
print("SUMMARY -- SWMM-as-Soft-Data BME")
print("=" * 70)
print(f"Network: {n_nodes} nodes, {len(network.edges)} links")
print(f"Window: {win_start} to {win_end} ({WINDOW_HOURS}h)")
print(f"Hard data: {len(zh_hard)} observations ({len(np.unique(ch_nodes))} nodes)")
print(f"Soft data: {len(soft_pdfs_list)} SWMM-based PDFs "
      f"({len(soft_sample_nodes)} nodes)")
print(f"Median SWMM residual std: {np.sqrt(median_resid_var):.4f} MGD")
print(f"BME (hard+soft) mean: [{bme_mean.min():.4f}, {bme_mean.max():.4f}] MGD")
print(f"BME (hard+soft) std:  [{bme_std.min():.4f}, {bme_std.max():.4f}] MGD")
print(f"BME (hard only) std:  [{bme_std_ho.min():.4f}, {bme_std_ho.max():.4f}] MGD")
print(f"Uncertainty reduction from SWMM soft data: {pct_reduction:.1f}%")
print(f"Wall time: BME+soft={t_wall:.1f}s, hard-only={t_wall2:.1f}s")
print("=" * 70)

plt.show()
