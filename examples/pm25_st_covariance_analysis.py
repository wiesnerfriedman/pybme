#!/usr/bin/env python3
"""
Space–Time Covariance Analysis of PM2.5 across California (1997–2016)
=====================================================================

This script:
1. Loads the annual-average PM2.5 data (115 monitoring stations, 20 years).
2. Treats -9999 values as missing (NaN).
3. Estimates the spatial and temporal covariance margins SEPARATELY:
   • Spatial margin Ĉ_s(r): subtract each year's cross-station mean,
     compute station-pair products binned by distance, average over years.
   • Temporal margin Ĉ_t(τ): subtract each station's across-year mean,
     compute year-pair products binned by temporal lag, average over stations.
   This removes the large temporal trend from the spatial margin and
   the spatial pattern from the temporal margin, yielding physically
   sensible (non-negative) experimental covariance curves.
4. Fits a separable exponential + nugget model to both margins jointly
   via weighted least squares.
5. Produces six publication-quality figures.

Requirements: numpy, scipy, matplotlib  (all in the project venv)
"""

import os, warnings
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")          # non-interactive backend → safe for scripts
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, "pm2p5_CA_1997-2016.txt")

# File format:
#   Line 1  – description
#   Line 2  – number of columns (25)
#   Lines 3-27 – one column name per line (State, County, Site, Lon, Lat,
#                PM25_1997, …, PM25_2016)  (PM25 names have trailing commas)
#   Lines 28+ – whitespace-delimited data rows

with open(data_path, "r") as f:
    lines = [l.rstrip("\n") for l in f.readlines()]

n_cols   = int(lines[1].strip())
col_names = [lines[2 + i].strip().rstrip(",") for i in range(n_cols)]

year_cols = [c for c in col_names if c.startswith("PM25_")]
years     = np.array([int(c.split("_")[1]) for c in year_cols])
n_years   = len(years)

# Column indices
lon_idx = col_names.index("Longitute")   # sic – typo in data file
lat_idx = col_names.index("Latitude")
pm_start = col_names.index(year_cols[0])

# Read station data from remaining lines
lats, lons, pm_data = [], [], []
for line in lines[2 + n_cols:]:
    parts = line.split()
    if len(parts) < n_cols:
        continue
    lons.append(float(parts[lon_idx]))
    lats.append(float(parts[lat_idx]))
    vals = [float(v) for v in parts[pm_start:pm_start + n_years]]
    pm_data.append(vals)

lon = np.array(lons)
lat = np.array(lats)
Z   = np.array(pm_data)        # shape (n_stations, n_years)

# -9999 → NaN
Z[Z <= -9998] = np.nan

n_stations, _ = Z.shape
n_total       = n_stations * n_years
valid_mask    = ~np.isnan(Z)
n_obs         = int(valid_mask.sum())

print(f"Loaded {n_stations} stations × {n_years} years "
      f"({n_obs}/{n_total} valid, {100*n_obs/n_total:.1f}%)")

# ---------------------------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------
overall_mean = np.nanmean(Z)
overall_std  = np.nanstd(Z)

station_means = np.nanmean(Z, axis=1)          # length n_stations
year_means    = np.nanmean(Z, axis=0)           # length n_years

# IQR per year (for Figure 2)
year_q25 = np.nanpercentile(Z, 25, axis=0)
year_q75 = np.nanpercentile(Z, 75, axis=0)

print(f"Overall mean = {overall_mean:.2f} µg/m³,  std = {overall_std:.2f}")

# ---------------------------------------------------------------------------
# 3. MARGINAL COVARIANCE ESTIMATION
# ---------------------------------------------------------------------------
# For a separable model  C(r,τ) = σ²·C_s(r)·C_t(τ) + c₀·δ
# we estimate C_s and C_t separately:
#
#   Spatial margin:  for each year j, compute residuals
#       e_ij = Z_ij − mean_j   (removes temporal trend)
#     then form station-pair products e_ij·e_kj, bin by dist(i,k).
#
#   Temporal margin: for each station i, compute residuals
#       f_ij = Z_ij − mean_i   (removes spatial pattern)
#     then form year-pair products f_ij·f_il, bin by |j−l|.
#
# This avoids cross-contamination between spatial and temporal structure.

# -- Spatial residuals (subtract year means) --------------------------------
Z_spatial = Z - year_means[None, :]          # shape (n_stations, n_years)
Z_spatial[~valid_mask] = np.nan

# -- Temporal residuals (subtract station means) ----------------------------
Z_temporal = Z - station_means[:, None]      # shape (n_stations, n_years)
Z_temporal[~valid_mask] = np.nan

# Pre-compute pairwise station distances
dist_matrix = np.sqrt((lon[:, None] - lon[None, :]) ** 2 +
                       (lat[:, None] - lat[None, :]) ** 2)

# ── 3a. Experimental SPATIAL covariance ────────────────────────────────────
max_spatial  = 8.0    # degrees (~890 km)
n_sbins      = 16
spatial_bin_edges   = np.linspace(0, max_spatial, n_sbins + 1)
spatial_bin_centres = 0.5 * (spatial_bin_edges[:-1] + spatial_bin_edges[1:])

spatial_cov_sum   = np.zeros(n_sbins)
spatial_cov_count = np.zeros(n_sbins, dtype=int)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for yr_idx in range(n_years):
        col = Z_spatial[:, yr_idx]             # residuals for this year
        prod = col[:, None] * col[None, :]     # station-pair products
        for sb in range(n_sbins):
            lo, hi = spatial_bin_edges[sb], spatial_bin_edges[sb + 1]
            # First bin: include distance ≥ 0 (includes self-pairs at r=0)
            if sb == 0:
                mask = (dist_matrix >= 0) & (dist_matrix < hi)
            else:
                mask = (dist_matrix >= lo) & (dist_matrix < hi)
            vals = prod[mask]
            good = vals[~np.isnan(vals)]
            if len(good) > 0:
                spatial_cov_sum[sb]   += good.sum()
                spatial_cov_count[sb] += len(good)

exp_cov_spatial = np.where(spatial_cov_count > 0,
                           spatial_cov_sum / spatial_cov_count, np.nan)

print(f"\nSpatial margin: C_s(0) = {exp_cov_spatial[0]:.4f}")
print(f"  min = {np.nanmin(exp_cov_spatial):.4f}, "
      f"max = {np.nanmax(exp_cov_spatial):.4f}")

# ── 3b. Experimental TEMPORAL covariance ───────────────────────────────────
max_temporal = n_years // 2     # only use lags 0 … 10 yr (half the span)
temporal_lags = np.arange(max_temporal + 1)

temporal_cov_sum   = np.zeros(max_temporal + 1)
temporal_cov_count = np.zeros(max_temporal + 1, dtype=int)

# Only use stations with ≥ 5 valid observations to ensure stable mean est.
min_obs_station = 5
station_valid_count = valid_mask.sum(axis=1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for stn_idx in range(n_stations):
        if station_valid_count[stn_idx] < min_obs_station:
            continue
        row = Z_temporal[stn_idx, :]           # residuals for this station
        for t_lag in range(max_temporal + 1):
            for yr1 in range(n_years - t_lag):
                yr2 = yr1 + t_lag
                v1, v2 = row[yr1], row[yr2]
                if np.isnan(v1) or np.isnan(v2):
                    continue
                temporal_cov_sum[t_lag]   += v1 * v2
                temporal_cov_count[t_lag] += 1

exp_cov_temporal = np.where(temporal_cov_count > 0,
                            temporal_cov_sum / temporal_cov_count, np.nan)

print(f"Temporal margin: C_t(0) = {exp_cov_temporal[0]:.4f}")
print(f"  min = {np.nanmin(exp_cov_temporal):.4f}, "
      f"max = {np.nanmax(exp_cov_temporal):.4f}")

# ---------------------------------------------------------------------------
# 4. FIT SEPARABLE EXPONENTIAL + NUGGET MODEL
# ---------------------------------------------------------------------------
# The full model is:
#   C(r, τ) = σ² · exp(−3r/a_s) · exp(−3τ/a_t) + c₀ · δ(r=0,τ=0)
#
# The spatial margin at τ=0:  C_s(r) = σ² · exp(−3r/a_s) + c₀·δ(r=0)
# The temporal margin at r=0: C_t(τ) = σ² · exp(−3τ/a_t) + c₀·δ(τ=0)
#
# We fit all four parameters (σ², a_s, a_t, c₀) jointly to both margins.

def model_spatial(r, sill, range_s, nugget):
    """Spatial margin: C_s(r) = σ² exp(−3r/a_s) + c₀ δ(r=0)."""
    C = sill * np.exp(-3.0 * np.abs(r) / range_s)
    C = np.where(np.abs(r) < 1e-12, C + nugget, C)
    return C

def model_temporal(tau, sill, range_t, nugget):
    """Temporal margin: C_t(τ) = σ² exp(−3τ/a_t) + c₀ δ(τ=0)."""
    C = sill * np.exp(-3.0 * np.abs(tau) / range_t)
    C = np.where(np.abs(tau) < 1e-12, C + nugget, C)
    return C

def model_cov(r, tau, sill, range_s, range_t, nugget):
    """Full separable exponential covariance with nugget."""
    C = sill * np.exp(-3.0 * np.abs(r) / range_s) * \
             np.exp(-3.0 * np.abs(tau) / range_t)
    C = np.where((np.abs(r) < 1e-12) & (np.abs(tau) < 1e-12),
                 C + nugget, C)
    return C

# Prepare data for joint fit
keep_s = ~np.isnan(exp_cov_spatial) & (spatial_cov_count > 20)
r_fit  = spatial_bin_centres[keep_s]
Cs_fit = exp_cov_spatial[keep_s]
Ws_fit = spatial_cov_count[keep_s].astype(float)

keep_t = ~np.isnan(exp_cov_temporal) & (temporal_cov_count > 20)
tau_fit  = temporal_lags[keep_t]
Ct_fit   = exp_cov_temporal[keep_t]
Wt_fit   = temporal_cov_count[keep_t].astype(float)

# Normalise weights so spatial and temporal margins have equal influence
Ws_norm = Ws_fit / Ws_fit.sum()
Wt_norm = Wt_fit / Wt_fit.sum()

def objective(log_params):
    sill, range_s, range_t, nugget = np.exp(log_params)
    pred_s = model_spatial(r_fit, sill, range_s, nugget)
    pred_t = model_temporal(tau_fit, sill, range_t, nugget)
    cost_s = np.sum(Ws_norm * (Cs_fit - pred_s) ** 2)
    cost_t = np.sum(Wt_norm * (Ct_fit - pred_t) ** 2)
    return cost_s + cost_t

# Initial guesses from the data
var0 = max(exp_cov_spatial[0], exp_cov_temporal[0])
x0 = np.log([var0 * 0.8, 2.0, 3.0, var0 * 0.1])
result = minimize(objective, x0, method="Nelder-Mead",
                  options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10})

sill_fit, range_s_fit, range_t_fit, nugget_fit = np.exp(result.x)

print(f"\nFitted covariance model parameters:")
print(f"  σ²  (sill)           = {sill_fit:.4f}")
print(f"  a_s (spatial range)  = {range_s_fit:.4f} deg  ≈ {range_s_fit*111:.0f} km")
print(f"  a_t (temporal range) = {range_t_fit:.4f} yr")
print(f"  c₀  (nugget)         = {nugget_fit:.4f}")
print(f"  Total sill (σ² + c₀) = {sill_fit + nugget_fit:.4f}")
print(f"  Optimisation success: {result.success}")

# ---------------------------------------------------------------------------
# 5. GENERATE FIGURES
# ---------------------------------------------------------------------------

out_dir = os.path.join(script_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

# ── Figure 1: Station map ─────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 8))
sc = ax1.scatter(lon, lat, c=station_means, cmap="YlOrRd", s=40,
                 edgecolor="k", linewidth=0.3, vmin=4, vmax=25)
ax1.set_xlabel("Longitude (°W)")
ax1.set_ylabel("Latitude (°N)")
ax1.set_title("PM$_{2.5}$ Monitoring Stations — California\n"
              "(colour = station mean, µg/m³)")
plt.colorbar(sc, ax=ax1, label="Mean PM$_{2.5}$ (µg/m³)", shrink=0.7)
ax1.set_aspect("equal")
fig1.tight_layout()
fig1.savefig(os.path.join(out_dir, "fig1_station_map.png"), dpi=200)
print(f"\nSaved fig1_station_map.png")

# ── Figure 2: Statewide annual mean time series ──────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(years, year_means, "ko-", markersize=5, linewidth=1.5)
ax2.fill_between(years, year_q25, year_q75,
                 alpha=0.25, color="steelblue", label="IQR")
ax2.set_xlabel("Year")
ax2.set_ylabel("PM$_{2.5}$ (µg/m³)")
ax2.set_title("Statewide Annual Mean PM$_{2.5}$ (1997–2016)")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, "fig2_timeseries.png"), dpi=200)
print("Saved fig2_timeseries.png")

# ── Figure 3: Marginal spatial covariance ────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 4.5))
ax3.plot(spatial_bin_centres, exp_cov_spatial, "ks", markersize=8,
         label="Experimental")
r_fine = np.linspace(0, spatial_bin_edges[-1], 200)
C_model_s = model_spatial(r_fine, sill_fit, range_s_fit, nugget_fit)
ax3.plot(r_fine, C_model_s, "r-", linewidth=2, label="Model")
ax3.axhline(sill_fit + nugget_fit, color="gray", linestyle="--", linewidth=0.8,
            label=f"Sill ($\\sigma^2$+$c_0$) = {sill_fit+nugget_fit:.2f}")
ax3.set_xlabel("Spatial lag r (degrees)")
ax3.set_ylabel("Covariance $\\hat{C}_s(r)$")
ax3.set_title("Marginal Spatial Covariance")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=0)
fig3.tight_layout()
fig3.savefig(os.path.join(out_dir, "fig3_spatial_covariance.png"), dpi=200)
print("Saved fig3_spatial_covariance.png")

# ── Figure 4: Marginal temporal covariance ───────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 4.5))
ax4.plot(temporal_lags, exp_cov_temporal, "bs", markersize=8,
         label="Experimental")
tau_fine = np.linspace(0, temporal_lags[-1], 200)
C_model_t = model_temporal(tau_fine, sill_fit, range_t_fit, nugget_fit)
ax4.plot(tau_fine, C_model_t, "r-", linewidth=2, label="Model")
ax4.axhline(sill_fit + nugget_fit, color="gray", linestyle="--", linewidth=0.8,
            label=f"Sill ($\\sigma^2$+$c_0$) = {sill_fit+nugget_fit:.2f}")
ax4.set_xlabel("Temporal lag τ (years)")
ax4.set_ylabel("Covariance $\\hat{C}_t(\\tau)$")
ax4.set_title("Marginal Temporal Covariance")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(left=0)
fig4.tight_layout()
fig4.savefig(os.path.join(out_dir, "fig4_temporal_covariance.png"), dpi=200)
print("Saved fig4_temporal_covariance.png")

# ── Figure 5: 2-D covariance surface (model, reconstructed from margins) ─
fig5, ax5 = plt.subplots(figsize=(7, 5))
R_mesh, T_mesh = np.meshgrid(spatial_bin_centres, temporal_lags, indexing="ij")
C_model_2d = model_cov(R_mesh, T_mesh, sill_fit, range_s_fit,
                        range_t_fit, nugget_fit)
im = ax5.pcolormesh(R_mesh, T_mesh, C_model_2d,
                     cmap="viridis", shading="auto")
ax5.set_xlabel("Spatial lag r (degrees)")
ax5.set_ylabel("Temporal lag τ (years)")
ax5.set_title("Fitted Space-Time Covariance Model — PM$_{2.5}$ California")
plt.colorbar(im, ax=ax5, label="C(r, τ)")
fig5.tight_layout()
fig5.savefig(os.path.join(out_dir, "fig5_st_covariance_2d.png"), dpi=200)
print("Saved fig5_st_covariance_2d.png")

# ── Figure 6: Both margins on one summary figure ────────────────────────
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 4.5))

# Spatial
ax6a.plot(spatial_bin_centres, exp_cov_spatial, "ks", markersize=7,
          label="Experimental", zorder=3)
ax6a.plot(r_fine, C_model_s, "r-", linewidth=2, label="Fitted model")
ax6a.set_xlabel("Spatial lag r (degrees)")
ax6a.set_ylabel("$\\hat{C}_s(r)$")
ax6a.set_title("(a) Spatial Covariance Margin")
ax6a.legend(fontsize=9)
ax6a.grid(True, alpha=0.3)
ax6a.set_xlim(left=0)

# Temporal
ax6b.plot(temporal_lags, exp_cov_temporal, "bs", markersize=7,
          label="Experimental", zorder=3)
ax6b.plot(tau_fine, C_model_t, "r-", linewidth=2, label="Fitted model")
ax6b.set_xlabel("Temporal lag τ (years)")
ax6b.set_ylabel("$\\hat{C}_t(\\tau)$")
ax6b.set_title("(b) Temporal Covariance Margin")
ax6b.legend(fontsize=9)
ax6b.grid(True, alpha=0.3)
ax6b.set_xlim(left=0)

fig6.suptitle("Experimental and Fitted Covariance Margins", fontsize=13)
fig6.tight_layout()
fig6.savefig(os.path.join(out_dir, "fig6_model_vs_experimental.png"), dpi=200)
print("Saved fig6_model_vs_experimental.png")

# ---------------------------------------------------------------------------
# 6. ANNUAL SPATIAL MAPPING — Simple Kriging
# ---------------------------------------------------------------------------
# For each selected year we perform simple kriging on a regular grid:
#
#   ẑ(x₀, tⱼ) = μⱼ + Σᵢ λᵢ [Z(sᵢ, tⱼ) − μⱼ]
#
# where μⱼ is the year mean (the spatially-varying trend estimate),
# λ = K⁻¹ k(x₀), K is the n × n spatial covariance matrix among the
# n valid stations for year j, and k(x₀) is the n-vector of covariances
# between x₀ and those stations (all at τ = 0).
#
# First moment  = ẑ(x₀)               [kriging mean estimate]
# Second moment = σ²(x₀) = C(0,0) − k(x₀)ᵀ λ   [kriging variance]
#
# Grid points farther than 1.5 × range_s (~340 km) from every station
# are masked out, naturally hiding ocean/Nevada/Arizona regions.

# ── 6a. Build estimation grid ─────────────────────────────────────────────
# California approximate bounding box
lon_min, lon_max = -124.6, -114.1
lat_min, lat_max =   32.4,   42.1
grid_spacing = 0.15                          # degrees  ≈ 17 km

lon_grid_1d = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)
lat_grid_1d = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
LON_g, LAT_g = np.meshgrid(lon_grid_1d, lat_grid_1d)   # (ny, nx)
grid_lon = LON_g.ravel()
grid_lat = LAT_g.ravel()
n_grid = len(grid_lon)

# Distance from every grid point to the nearest station
dist_grid_sta = np.sqrt((grid_lon[:, None] - lon[None, :]) ** 2 +
                         (grid_lat[:, None] - lat[None, :]) ** 2)   # (n_grid, n_sta)
nearest_dist  = dist_grid_sta.min(axis=1)
mask_ca = nearest_dist <= 1.5 * range_s_fit    # keep only "data-rich" zone

# ── 6b. Simple-kriging helper ─────────────────────────────────────────────
total_var = sill_fit + nugget_fit             # C(0,0)

def simple_krige_year(yr_idx):
    """Return (mean_map, var_map) arrays of shape LON_g.shape for one year."""
    valid_stn = np.where(valid_mask[:, yr_idx])[0]
    n_d = len(valid_stn)
    if n_d < 3:
        return np.full(n_grid, np.nan), np.full(n_grid, total_var)

    # Covariance matrix among data points (τ = 0)
    d_dd = dist_matrix[np.ix_(valid_stn, valid_stn)]     # (n_d, n_d)
    K    = model_spatial(d_dd, sill_fit, range_s_fit, nugget_fit)

    # Add small nugget regularisation for numerical stability
    K   += 1e-8 * np.eye(n_d)
    L    = np.linalg.cholesky(K)                         # stable solve

    residuals = Z[valid_stn, yr_idx] - year_means[yr_idx]  # (n_d,)
    K_inv_r   = np.linalg.solve(L.T, np.linalg.solve(L, residuals))  # K⁻¹ r

    # Cross-covariance grid → data  (n_grid, n_d)
    d_gd = dist_grid_sta[:, valid_stn]                   # (n_grid, n_d)
    k_gd = model_spatial(d_gd, sill_fit, range_s_fit, 0.0)  # no nugget for cross-cov

    # First moment: ẑ = μⱼ + k K⁻¹ r
    mean_map = year_means[yr_idx] + k_gd @ K_inv_r       # (n_grid,)

    # Second moment: σ² = C(0,0) − k K⁻¹ kᵀ  (row-wise)
    K_inv_k  = np.linalg.solve(L.T, np.linalg.solve(L, k_gd.T))  # (n_d, n_grid)
    var_map  = total_var - np.sum(k_gd * K_inv_k.T, axis=1)
    var_map  = np.clip(var_map, 0, None)                 # numerical safety

    # Mask far-from-data grid points
    mean_map[~mask_ca] = np.nan
    var_map[~mask_ca]  = np.nan

    return mean_map.reshape(LON_g.shape), var_map.reshape(LON_g.shape)

# ── 6c. Select years to map ───────────────────────────────────────────────
# Pick four years that span the period and have good coverage
map_years   = [2000, 2005, 2010, 2016]
map_indices = [np.where(years == y)[0][0] for y in map_years]

# Check actual station counts (print for info)
for y, yi in zip(map_years, map_indices):
    n_valid = int(valid_mask[:, yi].sum())
    print(f"  Year {y}: {n_valid} valid stations")

# ── 6d. Run kriging ───────────────────────────────────────────────────────
print("\nRunning simple kriging for selected years …")
krige_results = {}
for y, yi in zip(map_years, map_indices):
    mean_g, var_g = simple_krige_year(yi)
    krige_results[y] = (mean_g, var_g)
    print(f"  {y} done — est. range [{np.nanmin(mean_g):.1f}, "
          f"{np.nanmax(mean_g):.1f}] µg/m³, "
          f"max σ = {np.sqrt(np.nanmax(var_g)):.1f} µg/m³")

# ── 6e. Fig 7: 1st-moment maps (kriging estimate) ────────────────────────
vmin_mean, vmax_mean = 4.0, 22.0       # shared colour scale
fig7, axes7 = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)

for ax, y in zip(axes7, map_years):
    mean_g, _ = krige_results[y]
    im = ax.pcolormesh(LON_g, LAT_g, mean_g, cmap="YlOrRd",
                       vmin=vmin_mean, vmax=vmax_mean, shading="auto")
    # Overlay station locations
    valid_stn = np.where(valid_mask[:, map_indices[map_years.index(y)]])[0]
    ax.scatter(lon[valid_stn], lat[valid_stn],
               s=12, c="k", marker="^", zorder=4, label="Stations")
    ax.set_title(f"{y}")
    ax.set_xlabel("Longitude (°)")
    ax.set_aspect("equal")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

axes7[0].set_ylabel("Latitude (°)")
plt.colorbar(im, ax=axes7, label="PM$_{2.5}$ estimate (µg/m³)", shrink=0.6,
             pad=0.02)
fig7.suptitle("1st Moment — Kriging Estimate of Annual PM$_{2.5}$ (µg/m³)\n"
              "California", fontsize=13)
fig7.savefig(os.path.join(out_dir, "fig7_kriging_mean.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig7_kriging_mean.png")

# ── 6f. Fig 8: 2nd-moment maps (kriging std. dev.) ────────────────────────
fig8, axes8 = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)

# Shared colour scale: 0 → sqrt(total_var)
vmax_std = np.sqrt(total_var)

for ax, y in zip(axes8, map_years):
    _, var_g = krige_results[y]
    std_g = np.sqrt(var_g)
    im2 = ax.pcolormesh(LON_g, LAT_g, std_g, cmap="Blues",
                        vmin=0, vmax=vmax_std, shading="auto")
    valid_stn = np.where(valid_mask[:, map_indices[map_years.index(y)]])[0]
    ax.scatter(lon[valid_stn], lat[valid_stn],
               s=12, c="k", marker="^", zorder=4)
    ax.set_title(f"{y}")
    ax.set_xlabel("Longitude (°)")
    ax.set_aspect("equal")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

axes8[0].set_ylabel("Latitude (°)")
plt.colorbar(im2, ax=axes8, label="Kriging std. dev. σ (µg/m³)", shrink=0.6,
             pad=0.02)
fig8.suptitle("2nd Moment — Kriging Std. Dev. of Annual PM$_{2.5}$ (µg/m³)\n"
              "California", fontsize=13)
fig8.savefig(os.path.join(out_dir, "fig8_kriging_std.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig8_kriging_std.png")

# ---------------------------------------------------------------------------
# 7. PRINT SUMMARY FOR REPORT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY — Space-Time Covariance Model for CA PM2.5")
print("=" * 60)
print(f"Data: {n_stations} stations, {n_years} years (1997–2016)")
print(f"Valid observations: {n_obs} ({100*n_obs/n_total:.1f}%)")
print(f"Overall mean: {overall_mean:.2f} µg/m³")
print(f"Overall std:  {overall_std:.2f} µg/m³")
print()
print("Fitted separable covariance model:")
print("  C(r,τ) = σ² · exp(−3r/a_s) · exp(−3τ/a_t) + c₀·δ(r,τ)")
print(f"  σ²  = {sill_fit:.4f}  (sill)")
print(f"  a_s = {range_s_fit:.4f} deg  ≈ {range_s_fit*111:.0f} km  (spatial range)")
print(f"  a_t = {range_t_fit:.4f} yr  (temporal range)")
print(f"  c₀  = {nugget_fit:.4f}  (nugget)")
print(f"  Total variance at origin = σ²+c₀ = {sill_fit+nugget_fit:.4f}")
print("=" * 60)

plt.show()
