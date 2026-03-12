#!/usr/bin/env python3
"""
Network-BME on a SWMM Sewer Model — Onondaga County
====================================================

Demonstrates pybme's network-domain BME (graph-Laplacian covariance)
applied to a real SWMM sewer model.

1. Parses the SWMM .inp file directly (no external parser needed) to
   extract the full network topology: junctions, storage nodes, outfalls,
   conduits, pumps, orifices, weirs, outlets, and node coordinates.
2. Builds the network graph adjacency matrix (weighted by inverse conduit
   length) and constructs a ``NetworkCovariance`` object.
3. Reads meter observations (flow, MGD) and meter-to-node mapping.
4. Applies BME network prediction to estimate flow at every unobserved
   node — the first moment (posterior mean) and second moment (posterior
   variance) — using the graph-Laplacian covariance.

All distances are *network* distances derived from the graph topology —
no geographic projection is needed.

Data
----
SWMM model  : OC_2024-Conditions_5.1.010_V7-Calibrated_01282025.inp
Observations: ObservedData.csv  (17 flow meters, 5-minute interval,
              May 2024 – June 2025)
Meter map   : MeterLocations.csv  (meter → link → node)

Requirements: pybme (in project venv), numpy, scipy, matplotlib
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── ensure pybme is importable ──────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path     = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.network import (
    NetworkCovariance, adjacency_from_edges, build_graph_laplacian,
)
from pybme.predict import bme_predict_network
from pybme.network_plots import (
    plot_network_observations,
    plot_network_field,
    plot_network_correlation,
    plot_operator,
)
from pybme.swmm import (
    build_edge_array,
    parse_swmm_inp,
    read_meter_node_map,
    read_observation_csv,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. PARSE THE SWMM .INP FILE
# ═══════════════════════════════════════════════════════════════════════════
# We extract only what's needed for the graph: node names, conduit/link
# connectivity (from_node → to_node + length), and COORDINATES.

PRIVATE_SWMM_DIR = os.path.join(script_dir, "private_swmm")
INP_PATH = os.environ.get("PYBME_SWMM_INP", os.path.join(PRIVATE_SWMM_DIR, "model.inp"))
OBS_PATH = os.environ.get("PYBME_SWMM_OBS", os.path.join(PRIVATE_SWMM_DIR, "ObservedData.csv"))
METER_LOC_PATH = os.environ.get("PYBME_SWMM_METER_MAP", os.path.join(PRIVATE_SWMM_DIR, "MeterLocations.csv"))


print("Parsing SWMM .inp file ...")
network = parse_swmm_inp(INP_PATH)

node_names = network.all_node_names
n_nodes = len(node_names)
node_idx = {name: i for i, name in enumerate(node_names)}

print(f"  {len(network.junctions)} junctions, "
    f"{len(network.storages)} storage nodes, "
    f"{len(network.outfalls)} outfalls")
print(f"  {len(network.edges)} links (conduits + pumps + orifices + weirs + outlets)")
print(f"  {n_nodes} total nodes, {len(network.coords)} with coordinates")


# ═══════════════════════════════════════════════════════════════════════════
# 2. BUILD GRAPH ADJACENCY AND NETWORK COVARIANCE
# ═══════════════════════════════════════════════════════════════════════════
# Use UNIT edge weights (topology only) so the graph Laplacian reflects
# network connectivity rather than pipe length.  With inverse-length
# weights the typical values (~0.001) are dwarfed by κ² and the
# covariance collapses to a near-diagonal matrix — every unobserved node
# reverts to the prior mean.  Unit weights keep the mean node degree ~2
# and produce a meaningful correlation range across the network.
#
# The SWMM model is a directed graph (water flows downhill) but for the
# covariance we treat it as undirected — correlation is symmetric.

edge_array = build_edge_array(node_names, network.edges)

# Unit weights — all edges have the same coupling strength
W = adjacency_from_edges(n_nodes, edge_array)  # default weights = 1.0

# Build the graph Laplacian and NetworkCovariance
# κ controls the correlation range on the graph:
#   large κ  → covariance ≈ (1/κ²)I, nearly uncorrelated (too local)
#   small κ  → long-range correlation, smoother field
# With unit weights and κ=0.1, each meter influences ~130 nodes (31%
# of the 421-node network), giving good overlap between the 16 meters.
KAPPA = 0.1
SIGMA2 = 1.0   # will be scaled by data variance later

net_cov = NetworkCovariance(W, kappa=KAPPA, sigma2=SIGMA2, from_adjacency=True)

print(f"\nNetworkCovariance built: {net_cov.n_nodes} nodes, "
      f"method=regularised, kappa={KAPPA}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. READ METER LOCATIONS AND OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════

# ── 3a. Meter locations ────────────────────────────────────────────────────
meter_node_map = read_meter_node_map(METER_LOC_PATH)

print(f"\nLoaded {len(meter_node_map)} meter locations")

# ── 3b. Observation data ──────────────────────────────────────────────────
obs_table = read_observation_csv(OBS_PATH, value_type="flow")
rows = obs_table.rows

# Header rows: 0=Meter, 1=Location/Node, 2=Type, 3=Unit
# row[0][col] → meter name;  row[2][col] → 'flow' or 'depth'
meter_names_hdr = obs_table.meter_names
node_names_hdr  = obs_table.node_names
types_hdr       = obs_table.types
units_hdr       = obs_table.units

# Identify flow columns (odd indices: 1,3,5,...,33)
flow_cols = obs_table.value_cols
flow_meter_names = obs_table.value_names
flow_node_names  = [node_names_hdr[i] for i in flow_cols]

print(f"  {len(flow_cols)} flow meters: {flow_meter_names}")

# ── 3c. Pick a representative time snapshot ────────────────────────────────
# Find the row with the most active (nonzero) flow meters.
best_count, best_row_idx = 0, 4
for i in range(4, len(rows)):
    cnt = sum(1 for c in flow_cols if abs(float(rows[i][c])) > 0.001)
    if cnt > best_count:
        best_count = cnt
        best_row_idx = i

best_ts = rows[best_row_idx][0]
print(f"\nBest snapshot: {best_ts} ({best_count}/{len(flow_cols)} active meters)")

# Extract flow values (MGD) for the best time step
obs_values = {}   # meter_name → flow in MGD
for col_idx in flow_cols:
    meter = meter_names_hdr[col_idx]
    val = float(rows[best_row_idx][col_idx])
    obs_values[meter] = val

# ── 3d. Map meters to graph node indices ───────────────────────────────────
# For flow meters, the observation is on the link (conduit between two nodes).
# MeterLocations.csv gives a "Node" which is the downstream node of the link.
# We'll assign the flow observation to that node.

hard_nodes = []   # graph node indices
hard_values = []  # flow values (MGD)
meter_labels = [] # for plotting
skipped = []

for meter_name, flow_val in obs_values.items():
    if abs(flow_val) < 0.001:
        continue  # skip inactive meters
    node_name = meter_node_map.get(meter_name)
    if node_name is None:
        skipped.append(f"{meter_name}: no node mapping")
        continue
    idx = node_idx.get(node_name)
    if idx is None:
        skipped.append(f"{meter_name} → {node_name}: node not in SWMM model")
        continue
    # CSO-007_IntercepUS and IntercepDS both map to Reg-007.
    # If already observed, take the average.
    if idx in hard_nodes:
        pos = hard_nodes.index(idx)
        hard_values[pos] = 0.5 * (hard_values[pos] + flow_val)
        meter_labels[pos] += f"+{meter_name}"
        continue
    hard_nodes.append(idx)
    hard_values.append(flow_val)
    meter_labels.append(meter_name)

ch_nodes = np.array(hard_nodes, dtype=int)
zh = np.array(hard_values, dtype=np.float64)

if skipped:
    print(f"  Skipped: {skipped}")
print(f"  Hard data: {len(ch_nodes)} observed nodes")
for lbl, v, ni in zip(meter_labels, hard_values, hard_nodes):
    print(f"    {lbl:30s}  node {node_names[ni]:25s}  flow = {v:.4f} MGD")


# ═══════════════════════════════════════════════════════════════════════════
# 4. RESCALE COVARIANCE TO MATCH DATA VARIANCE
# ═══════════════════════════════════════════════════════════════════════════
# We want the marginal variance of the covariance model to approximately
# match the variance of the observed data.  The simplest approach: set
# σ² so that the diagonal of C equals the sample variance of the flows.

data_var = float(np.var(zh))
data_mean = float(np.mean(zh))

# The diagonal of C = σ²(κ²I + L)⁻¹ varies by node.  Rescale σ² so that
# the average diagonal entry at observed nodes ≈ data variance.
diag_at_obs = net_cov.marginal_variance(ch_nodes)
scale = data_var / max(diag_at_obs.mean(), 1e-12) if data_var > 0 else 1.0

# Rebuild with scaled σ²
net_cov_scaled = NetworkCovariance(
    W, kappa=KAPPA, sigma2=SIGMA2 * scale, from_adjacency=True
)

print(f"\nData mean = {data_mean:.4f} MGD, data var = {data_var:.4f}")
print(f"Rescaled sigma2 = {SIGMA2 * scale:.4f} (scale factor = {scale:.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# 5. BME NETWORK PREDICTION — AT ALL NODES
# ═══════════════════════════════════════════════════════════════════════════
# Predict at every node in the graph using the observed flow data.
# This is purely spatial — one snapshot — so bme_predict_network is used.

# Select nodes to predict: all nodes (or a subset for speed)
# For a large graph we estimate at all nodes.
ck_nodes = np.arange(n_nodes, dtype=int)

print(f"\nRunning BME network prediction at {n_nodes} nodes ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = bme_predict_network(
        ck_nodes   = ck_nodes,
        ch_nodes   = ch_nodes,
        zh         = zh,
        net_cov    = net_cov_scaled,
        order      = float("nan"),    # simple kriging
        mean_prior = data_mean,
        n_grid     = 150,
        ci_prob    = 0.95,
    )

bme_mean = np.array([r.mean for r in results])
bme_var  = np.array([r.variance for r in results])
bme_std  = np.sqrt(np.clip(bme_var, 0, None))

print(f"  BME mean range:  [{bme_mean.min():.4f}, {bme_mean.max():.4f}] MGD")
print(f"  BME std  range:  [{bme_std.min():.4f}, {bme_std.max():.4f}] MGD")

# Verify interpolation: at observed nodes, mean should match observations
for lbl, true_val, ni in zip(meter_labels, hard_values, hard_nodes):
    est = bme_mean[ni]
    err = abs(est - true_val)
    print(f"    {lbl:30s}  obs={true_val:.4f}  est={est:.4f}  |err|={err:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. FIGURES  (using pybme.network_plots)
# ═══════════════════════════════════════════════════════════════════════════
out_dir = os.path.join(script_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# ── Figure A: Network map with observation stations ────────────────────────
fig1, _ = plot_network_observations(
    node_names, network["coords"], network["edges"],
    obs_nodes=hard_nodes, obs_values=zh, obs_labels=meter_labels,
    title=f"SWMM Network — Onondaga County\nFlow Meter Observations, {best_ts}",
    units="MGD",
)
fig1.savefig(os.path.join(out_dir, "fig_swmm_A_network_map.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig_swmm_A_network_map.png")

# ── Figure B: BME 1st moment (posterior mean flow) ─────────────────────────
fig2, _ = plot_network_field(
    node_names, network["coords"], network["edges"], bme_mean,
    obs_nodes=hard_nodes,
    title=f"1st Moment — BME Posterior Mean Flow\nNetwork-Laplacian Covariance, κ={KAPPA}",
    cmap="YlOrRd", units="BME Mean Flow (MGD)",
    vmax_percentile=95,
)
fig2.savefig(os.path.join(out_dir, "fig_swmm_B_bme_mean.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig_swmm_B_bme_mean.png")

# ── Figure C: BME 2nd moment (posterior std. dev.) ─────────────────────────
fig3, _ = plot_network_field(
    node_names, network["coords"], network["edges"], bme_std,
    obs_nodes=hard_nodes,
    title=f"2nd Moment — BME Posterior Std. Dev.\nNetwork-Laplacian Covariance, κ={KAPPA}",
    cmap="Blues", units="BME Std. Dev. (MGD)",
    vmax_percentile=95, marker_size=12,
    obs_color="red", obs_label="Meters",
)
fig3.savefig(os.path.join(out_dir, "fig_swmm_C_bme_std.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig_swmm_C_bme_std.png")

# ── Figure D: Network correlation structure ────────────────────────────────
show_nodes = [hard_nodes[0], hard_nodes[len(hard_nodes)//2], hard_nodes[-1]]
show_labels = [meter_labels[0], meter_labels[len(hard_nodes)//2], meter_labels[-1]]
fig4, _ = plot_network_correlation(
    node_names, network["coords"], network["edges"],
    net_cov_scaled, show_nodes, show_labels,
    suptitle="Network Correlation Structure (graph-Laplacian covariance)",
)
fig4.savefig(os.path.join(out_dir, "fig_swmm_D_correlation.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig_swmm_D_correlation.png")

# ── Figure E: Operator visualisation ──────────────────────────────────────
fig5, _ = plot_operator(
    net_cov_scaled, W, KAPPA,
    obs_nodes=ch_nodes,
    source_nodes=[hard_nodes[i] for i in range(min(5, len(hard_nodes)))],
    source_labels=[meter_labels[i] for i in range(min(5, len(hard_nodes)))],
)
fig5.savefig(os.path.join(out_dir, "fig_swmm_E_operator.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig_swmm_E_operator.png")


# ═══════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY -- Network-BME on SWMM Sewer Model (Onondaga County)")
print("=" * 65)
print(f"Network: {n_nodes} nodes, {len(network['edges'])} links")
print(f"Snap time: {best_ts}")
print(f"Observed flow meters: {len(ch_nodes)} (of {len(flow_cols)} total)")
print(f"Covariance: regularised graph-Laplacian, kappa={KAPPA}, "
      f"sigma2={SIGMA2*scale:.4f}")
print(f"Prior mean: {data_mean:.4f} MGD (sample mean of observations)")
print(f"BME posterior mean range: [{bme_mean.min():.4f}, "
      f"{bme_mean.max():.4f}] MGD")
print(f"BME posterior std range:  [{bme_std.min():.4f}, "
      f"{bme_std.max():.4f}] MGD")
print("=" * 65)

plt.show()
