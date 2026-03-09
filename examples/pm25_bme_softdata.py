#!/usr/bin/env python3
"""
BME Estimation of PM2.5 across California with Soft Data (2005)
================================================================

This example extends the space-time covariance analysis by performing
Bayesian Maximum Entropy (BME) spatial estimation for the year 2005,
distinguishing two classes of monitoring stations based on record length:

  Hard data  (exact)  — stations with ≥ 12 valid annual averages out of 20.
                         These are established, long-running EPA monitors
                         whose annual means are treated as known values.

  Soft data  (uncertain) — stations with 3–11 valid years.
                         These are newer or sporadically operating sites.
                         Their annual average is real, but the underlying
                         uncertainty is represented by a Gaussian PDF:

                           σ²_soft = σ²_noise + σ²_sampling

                         where:
                           σ_noise    = 0.12 × Z   (12 % relative precision,
                                        consistent with EPA FRM network audits)
                           σ_sampling = σ_station / √n_valid
                                        (standard error of the annual mean)

                         Stations with < 3 valid years are excluded.

The fitted covariance model from the companion script is reused:
  C(r) = σ² exp(−3r / a_s) + c₀ δ(r)
  σ² = 16.17,  a_s ≈ 200 km,  c₀ ≈ 0

Outputs (figures/):
  fig_soft_A_station_types.png  — map: hard / soft stations, σ_soft colour scale
  fig_soft_B_bme_mean.png       — 1st moment: BME posterior mean
  fig_soft_C_bme_std.png        — 2nd moment: BME posterior std. dev.
  fig_soft_D_bme_vs_krige.png   — difference: BME mean − kriging mean
  fig_soft_E_posterior_pdfs.png — posterior PDFs at selected grid points

Requirements: pybme (in project venv), numpy, scipy, matplotlib
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── ensure pybme is importable ──────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path     = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.predict  import bme_predict
from pybme.soft_data import SoftPDF

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
data_path = os.path.join(script_dir, "pm2p5_CA_1997-2016.txt")

with open(data_path, "r") as f:
    lines = [l.rstrip("\n") for l in f.readlines()]

n_cols    = int(lines[1].strip())
col_names = [lines[2 + i].strip().rstrip(",") for i in range(n_cols)]
year_cols = [c for c in col_names if c.startswith("PM25_")]
years     = np.array([int(c.split("_")[1]) for c in year_cols])
n_years   = len(years)

lon_idx  = col_names.index("Longitute")
lat_idx  = col_names.index("Latitude")
pm_start = col_names.index(year_cols[0])

lons, lats, pm_data = [], [], []
for line in lines[2 + n_cols:]:
    parts = line.split()
    if len(parts) < n_cols:
        continue
    lons.append(float(parts[lon_idx]))
    lats.append(float(parts[lat_idx]))
    pm_data.append([float(v) for v in parts[pm_start:pm_start + n_years]])

lon = np.array(lons)
lat = np.array(lats)
Z   = np.array(pm_data)      # (n_stations, n_years)
Z[Z <= -9998] = np.nan

n_stations = len(lon)
valid_mask = ~np.isnan(Z)

# ---------------------------------------------------------------------------
# 1b. PROJECT LON/LAT TO KILOMETRES
# ---------------------------------------------------------------------------
# Equirectangular projection centred on the station centroid — same as the
# companion covariance-analysis script.  All distances are now in km.

lat_mean_rad = np.radians(lat.mean())
KM_PER_DEG_LON = np.cos(lat_mean_rad) * 111.32   # ≈ 88 km at ~37.5°N
KM_PER_DEG_LAT = 111.32

lon_ref = lon.mean()
lat_ref = lat.mean()
x_km = (lon - lon_ref) * KM_PER_DEG_LON   # easting  (km)
y_km = (lat - lat_ref) * KM_PER_DEG_LAT   # northing (km)

# ---------------------------------------------------------------------------
# 2. COVARIANCE MODEL PARAMETERS (from companion script)
# ---------------------------------------------------------------------------
# Fitted separable exponential model:
#   C(r, τ) = σ² exp(−3r/a_s) exp(−3τ/a_t)
# For spatial-only estimation (single year) we use the τ=0 slice:
#   C_s(r) = σ² exp(−3r/a_s)   plus nugget c₀ ≈ 0

SILL    = 16.17    # σ²   (µg/m³)²
RANGE_S = 200.0    # a_s  km  (re-fitted with projected coordinates)
NUGGET  = 0.0      # c₀
TOTAL_VAR = SILL + NUGGET

# pybme uses exponential model: C(r) = sill * exp(-3r/range)
COV_MODEL  = "exponential"
COV_PARAMS = [TOTAL_VAR, RANGE_S]

# ---------------------------------------------------------------------------
# 3. CHOOSE ANALYSIS YEAR AND PARTITION STATIONS
# ---------------------------------------------------------------------------
YEAR_TARGET  = 2005
HARD_THRESH  = 12    # ≥ HARD_THRESH valid years → hard data
SOFT_THRESH  = 3     # SOFT_THRESH to HARD_THRESH-1 → soft data
               # < SOFT_THRESH valid years in total → excluded

yr_idx = int(np.where(years == YEAR_TARGET)[0][0])
year_mean = float(np.nanmean(Z[:, yr_idx]))   # used as prior mean

n_valid_per_station = int(valid_mask.sum(axis=1)[0])  # re‑compute below
n_valid_stn = valid_mask.sum(axis=1)          # (n_stations,) — total valid years

has_data_this_year = valid_mask[:, yr_idx]

# Classify stations
is_hard = (n_valid_stn >= HARD_THRESH) & has_data_this_year
is_soft = (n_valid_stn >= SOFT_THRESH) & (n_valid_stn < HARD_THRESH) & has_data_this_year

print(f"Year: {YEAR_TARGET}  (year mean = {year_mean:.2f} µg/m³)")
print(f"  Hard stations (≥{HARD_THRESH} valid yrs, has {YEAR_TARGET} data): {is_hard.sum()}")
print(f"  Soft stations ({SOFT_THRESH}–{HARD_THRESH-1} valid yrs, has {YEAR_TARGET} data): {is_soft.sum()}")
print(f"  Excluded (< {SOFT_THRESH} valid yrs or no {YEAR_TARGET} data): "
      f"{n_stations - is_hard.sum() - is_soft.sum()}")

# ---------------------------------------------------------------------------
# 4. BUILD SOFT-DATA UNCERTAINTY MODEL
# ---------------------------------------------------------------------------
# For each soft station i:
#   σ_noise    = 0.12 × Z_i       (12 % relative precision, EPA FRM typical)
#   σ_sampling = σ_station_i / √n_valid_i  (std error of annual mean)
#   σ_total    = sqrt(σ_noise² + σ_sampling²)
#
# σ_station is the across-year std (after subtracting station mean); if only
# 1–2 years available we use the overall residual std as a conservative proxy.

station_std = np.nanstd(Z, axis=1)                 # interannual std, each station
global_residual_std = float(np.nanstd(Z - np.nanmean(Z, axis=0)[None, :]))
# Replace zero or very small std with global proxy
station_std = np.where(station_std < 0.5, global_residual_std, station_std)

soft_idx  = np.where(is_soft)[0]
hard_idx  = np.where(is_hard)[0]

# Compute σ_total for every soft station
Z_soft    = Z[soft_idx, yr_idx]           # measured values (for centring the PDF)
n_s_stn   = n_valid_stn[soft_idx].astype(float)
sigma_noise    = 0.12 * Z_soft
sigma_sampling = station_std[soft_idx] / np.sqrt(np.maximum(n_s_stn, 1.0))
sigma_total    = np.sqrt(sigma_noise**2 + sigma_sampling**2)

print(f"\nSoft-data σ_total: "
      f"min={sigma_total.min():.2f}, mean={sigma_total.mean():.2f}, "
      f"max={sigma_total.max():.2f} µg/m³")

# Build SoftPDF objects (Gaussian, truncated at 0 from below)
soft_pdfs = [
    SoftPDF.from_truncnorm(mu=float(Z_soft[i]),
                           sigma=float(sigma_total[i]),
                           a=0.0, b=None)
    for i in range(len(soft_idx))
]

# Hard-data arrays
zh = Z[hard_idx, yr_idx]
ch = np.column_stack([x_km[hard_idx], y_km[hard_idx]])   # (n_hard, 2) — km
cs = np.column_stack([x_km[soft_idx], y_km[soft_idx]])   # (n_soft, 2) — km

# ---------------------------------------------------------------------------
# 5. BUILD ESTIMATION GRID AND RUN BME
# ---------------------------------------------------------------------------
lon_min, lon_max = -124.6, -114.1
lat_min, lat_max =   32.4,   42.1
GRID_SPACING = 0.25     # degrees  (~28 km) — coarser than kriging example for speed

lon1d = np.arange(lon_min, lon_max + GRID_SPACING, GRID_SPACING)
lat1d = np.arange(lat_min, lat_max + GRID_SPACING, GRID_SPACING)
LON_g, LAT_g = np.meshgrid(lon1d, lat1d)
grid_lon = LON_g.ravel()
grid_lat = LAT_g.ravel()
n_grid = len(grid_lon)

# Project grid to km (same projection as station coordinates)
grid_x = (grid_lon - lon_ref) * KM_PER_DEG_LON
grid_y = (grid_lat - lat_ref) * KM_PER_DEG_LAT

# Mask: only estimate where at least one station is within 1.5 × range (km)
all_sta_x = np.concatenate([x_km[hard_idx], x_km[soft_idx]])
all_sta_y = np.concatenate([y_km[hard_idx], y_km[soft_idx]])
dist_to_nearest = np.sqrt(
    (grid_x[:, None] - all_sta_x[None, :]) ** 2 +
    (grid_y[:, None] - all_sta_y[None, :]) ** 2
).min(axis=1)
mask_active = dist_to_nearest <= 1.5 * RANGE_S

ck_active = np.column_stack([grid_x[mask_active], grid_y[mask_active]])
n_active  = ck_active.shape[0]
print(f"\nEstimation grid: {n_active} active points out of {n_grid}")

print("Running BME (this may take a minute) …")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = bme_predict(
        ck         = ck_active,
        ch         = ch,
        zh         = zh,
        cs         = cs,
        soft_pdfs  = soft_pdfs,
        model      = COV_MODEL,
        params     = COV_PARAMS,
        nhmax      = 15,
        nsmax      = 4,          # ≤ 4 soft neighbours (for speed + stability)
        dmax       = 2.5 * RANGE_S,
        order      = float("nan"),   # simple kriging with constant mean
        mean_prior = year_mean,
        ci_prob    = 0.95,
        n_grid     = 150,
    )

# Unpack results
bme_mean  = np.array([r.mean          for r in results])
bme_var   = np.array([r.variance      for r in results])
krige_mean = np.array([r.kriging_mean for r in results])
krige_var  = np.array([r.kriging_var  for r in results])
n_h_used  = np.array([r.n_hard        for r in results])
n_s_used  = np.array([r.n_soft        for r in results])

print(f"BME mean range:  [{np.nanmin(bme_mean):.1f}, {np.nanmax(bme_mean):.1f}] µg/m³")
print(f"BME std   range: [{np.nanmin(np.sqrt(bme_var)):.2f}, "
      f"{np.nanmax(np.sqrt(bme_var)):.2f}] µg/m³")

# Rebuild full-grid arrays (NaN outside active mask)
def to_map(vals_active):
    out = np.full(n_grid, np.nan)
    out[mask_active] = vals_active
    return out.reshape(LON_g.shape)

bme_mean_map  = to_map(bme_mean)
bme_std_map   = to_map(np.sqrt(np.clip(bme_var, 0, None)))
krige_mean_map = to_map(krige_mean)
diff_map      = to_map(bme_mean - krige_mean)
n_soft_map    = to_map(n_s_used.astype(float))

# ---------------------------------------------------------------------------
# 6. FIGURES
# ---------------------------------------------------------------------------
out_dir = os.path.join(script_dir, "figures")
os.makedirs(out_dir, exist_ok=True)

def _ca_axes(ax, title):
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title, fontsize=11)

# ── Figure A: Station classification and soft-data uncertainty map ─────────
fig_a, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

# Left: station map coloured by type
ax_a1.scatter(lon[hard_idx], lat[hard_idx],
              c="steelblue", s=35, marker="o", edgecolor="k",
              linewidth=0.4, label=f"Hard ({is_hard.sum()}): ≥{HARD_THRESH} yrs",
              zorder=4)
sc_soft = ax_a1.scatter(lon[soft_idx], lat[soft_idx],
                         c=sigma_total, cmap="Oranges", s=55, marker="s",
                         edgecolor="k", linewidth=0.4,
                         label=f"Soft ({is_soft.sum()}): {SOFT_THRESH}–{HARD_THRESH-1} yrs",
                         zorder=5, vmin=0, vmax=sigma_total.max())
excluded_mask = ~has_data_this_year | (n_valid_stn < SOFT_THRESH)
ax_a1.scatter(lon[excluded_mask], lat[excluded_mask],
              c="lightgray", s=18, marker="x", linewidth=0.8,
              label="Excluded", zorder=3)
plt.colorbar(sc_soft, ax=ax_a1, label="σ_soft (µg/m³)", shrink=0.7)
ax_a1.legend(fontsize=8, loc="lower left")
_ca_axes(ax_a1, f"Station Classification — {YEAR_TARGET}")

# Right: example PDFs for 5 soft stations spanning the σ_total range
n_show = min(5, len(soft_idx))
order_sigma = np.argsort(sigma_total)
show_idx = order_sigma[np.linspace(0, len(soft_idx)-1, n_show, dtype=int)]
cmap_pdf = plt.cm.plasma(np.linspace(0.1, 0.85, n_show))
for k, si in enumerate(show_idx):
    sp   = soft_pdfs[si]
    z_ev = np.linspace(*sp.support, 200)
    p_ev = sp.evaluate(z_ev)
    ax_a2.plot(z_ev, p_ev, color=cmap_pdf[k], linewidth=1.8,
               label=f"Z={Z_soft[si]:.1f}, σ={sigma_total[si]:.2f}")
    ax_a2.axvline(Z_soft[si], color=cmap_pdf[k], linewidth=0.8, linestyle="--")
ax_a2.set_xlabel("PM$_{2.5}$ (µg/m³)")
ax_a2.set_ylabel("Probability density")
ax_a2.set_title(f"Example soft-data PDFs — {YEAR_TARGET}")
ax_a2.legend(fontsize=8, title="value, σ_soft")
ax_a2.grid(True, alpha=0.3)

fig_a.suptitle(f"Hard and Soft PM$_{{2.5}}$ Data — California {YEAR_TARGET}", fontsize=13)
fig_a.savefig(os.path.join(out_dir, "fig_soft_A_station_types.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_soft_A_station_types.png")

# ── Figure B: BME 1st moment (posterior mean) ─────────────────────────────
fig_b, ax_b = plt.subplots(figsize=(7, 7), constrained_layout=True)
vmin_m, vmax_m = 4.0, 22.0
im_b = ax_b.pcolormesh(LON_g, LAT_g, bme_mean_map, cmap="YlOrRd",
                        vmin=vmin_m, vmax=vmax_m, shading="auto")
ax_b.scatter(lon[hard_idx], lat[hard_idx],
             c="steelblue", s=30, marker="o", edgecolor="k",
             linewidth=0.4, zorder=4, label="Hard")
ax_b.scatter(lon[soft_idx], lat[soft_idx],
             c="orange", s=40, marker="s", edgecolor="k",
             linewidth=0.4, zorder=5, label="Soft")
plt.colorbar(im_b, ax=ax_b, label="PM$_{2.5}$ BME mean (µg/m³)", shrink=0.75)
ax_b.legend(fontsize=9, loc="lower left")
_ca_axes(ax_b, f"1st Moment — BME Posterior Mean, {YEAR_TARGET}")
fig_b.savefig(os.path.join(out_dir, "fig_soft_B_bme_mean.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_soft_B_bme_mean.png")

# ── Figure C: BME 2nd moment (posterior std dev) ──────────────────────────
fig_c, ax_c = plt.subplots(figsize=(7, 7), constrained_layout=True)
vmax_s = np.sqrt(TOTAL_VAR)
im_c = ax_c.pcolormesh(LON_g, LAT_g, bme_std_map, cmap="Blues",
                        vmin=0, vmax=vmax_s, shading="auto")
ax_c.scatter(lon[hard_idx], lat[hard_idx],
             c="steelblue", s=30, marker="o", edgecolor="k",
             linewidth=0.4, zorder=4, label="Hard")
ax_c.scatter(lon[soft_idx], lat[soft_idx],
             c="orange", s=40, marker="s", edgecolor="k",
             linewidth=0.4, zorder=5, label="Soft")
plt.colorbar(im_c, ax=ax_c, label="PM$_{2.5}$ BME std. dev. (µg/m³)", shrink=0.75)
ax_c.legend(fontsize=9, loc="lower left")
_ca_axes(ax_c, f"2nd Moment — BME Posterior Std. Dev., {YEAR_TARGET}")
fig_c.savefig(os.path.join(out_dir, "fig_soft_C_bme_std.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_soft_C_bme_std.png")

# ── Figure D: BME mean − Kriging mean (effect of soft data) ───────────────
fig_d, ax_d = plt.subplots(figsize=(7, 7), constrained_layout=True)
diff_abs = np.abs(diff_map)
v_diff = np.nanpercentile(diff_abs, 98)    # 98th percentile for colour scale
im_d = ax_d.pcolormesh(LON_g, LAT_g, diff_map, cmap="RdBu_r",
                        vmin=-v_diff, vmax=v_diff, shading="auto")
ax_d.scatter(lon[hard_idx], lat[hard_idx],
             c="steelblue", s=30, marker="o", edgecolor="k",
             linewidth=0.4, zorder=4, label="Hard")
ax_d.scatter(lon[soft_idx], lat[soft_idx],
             c="orange", s=40, marker="s", edgecolor="k",
             linewidth=0.4, zorder=5, label="Soft")
plt.colorbar(im_d, ax=ax_d, label="BME mean − Kriging mean (µg/m³)", shrink=0.75)
ax_d.legend(fontsize=9, loc="lower left")
_ca_axes(ax_d, f"Soft-Data Impact: BME − Kriging Mean, {YEAR_TARGET}")
fig_d.savefig(os.path.join(out_dir, "fig_soft_D_bme_vs_krige.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_soft_D_bme_vs_krige.png")

# ── Figure E: Posterior PDFs at selected grid points ──────────────────────
# Pick 4 interesting active grid points: near hard stn, near soft stn,
# between stations, far from all stations.

def nearest_station_dist(gx, gy, sx, sy):
    return np.sqrt((gx - sx)**2 + (gy - sy)**2).min()

# Candidates among active points
cands = ck_active

# (i) Near a soft station
d_soft = np.sqrt((cands[:, 0:1] - x_km[soft_idx][None, :]) ** 2 +
                  (cands[:, 1:2] - y_km[soft_idx][None, :]) ** 2).min(axis=1)
pt_near_soft = int(np.argmin(d_soft))

# (ii) Near a hard station
d_hard = np.sqrt((cands[:, 0:1] - x_km[hard_idx][None, :]) ** 2 +
                  (cands[:, 1:2] - y_km[hard_idx][None, :]) ** 2).min(axis=1)
pt_near_hard = int(np.argmin(d_hard))

# (iii) Midpoint between stations — largest minimum distance
pt_far = int(np.argmax(d_soft + d_hard))

# (iv) Somewhere in the San Joaquin Valley (high concentration area)
sjv_x_min = (-121.5 - lon_ref) * KM_PER_DEG_LON
sjv_x_max = (-119.0 - lon_ref) * KM_PER_DEG_LON
sjv_y_min = (35.5 - lat_ref) * KM_PER_DEG_LAT
sjv_y_max = (37.5 - lat_ref) * KM_PER_DEG_LAT
sjv_mask = ((cands[:, 0] > sjv_x_min) & (cands[:, 0] < sjv_x_max) &
             (cands[:, 1] > sjv_y_min)  & (cands[:, 1] < sjv_y_max))
pt_sjv = int(np.argmax(sjv_mask.astype(float) * bme_mean))
if not sjv_mask.any():
    pt_sjv = pt_near_hard   # fallback

point_indices = [pt_near_hard, pt_near_soft, pt_sjv, pt_far]
point_labels  = ["Near hard station", "Near soft station",
                 "San Joaquin Valley", "Between stations"]
point_colours = ["steelblue", "darkorange", "firebrick", "gray"]

fig_e, axes_e = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes_e = axes_e.ravel()

for ax, pi, lbl, col in zip(axes_e, point_indices, point_labels, point_colours):
    r = results[pi]
    if r.z_grid is not None and r.pdf is not None:
        ax.plot(r.z_grid, r.pdf, color=col, linewidth=2)
        ax.fill_between(r.z_grid, r.pdf, alpha=0.20, color=col)
        ax.axvline(r.mean,  color="k",   linewidth=1.5, linestyle="-",
                   label=f"BME mean = {r.mean:.1f}")
        ax.axvline(r.mode,  color="k",   linewidth=1.0, linestyle="--",
                   label=f"BME mode = {r.mode:.1f}")
        if not np.isnan(r.ci_lower):
            ax.axvspan(r.ci_lower, r.ci_upper, alpha=0.08, color="k",
                       label=f"95% CI [{r.ci_lower:.1f}, {r.ci_upper:.1f}]")
    else:
        # Gaussian approximation fallback (kriging)
        from scipy.stats import norm as _norm
        z_plt = np.linspace(r.kriging_mean - 4*np.sqrt(r.kriging_var),
                            r.kriging_mean + 4*np.sqrt(r.kriging_var), 200)
        ax.plot(z_plt, _norm.pdf(z_plt, r.kriging_mean, np.sqrt(r.kriging_var)),
                color=col, linewidth=2)
        ax.axvline(r.kriging_mean, color="k", linewidth=1.5,
                   label=f"Mean = {r.kriging_mean:.1f}")
    gx, gy = cands[pi, 0], cands[pi, 1]
    glon = gx / KM_PER_DEG_LON + lon_ref
    glat = gy / KM_PER_DEG_LAT + lat_ref
    ax.set_title(f"{lbl}\n({glon:.2f}°, {glat:.2f}°)  nh={r.n_hard}  ns={r.n_soft}",
                 fontsize=9)
    ax.set_xlabel("PM$_{2.5}$ (µg/m³)")
    ax.set_ylabel("Posterior PDF")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

fig_e.suptitle(f"BME Posterior PDFs at Selected Grid Points — {YEAR_TARGET}",
               fontsize=12)
fig_e.savefig(os.path.join(out_dir, "fig_soft_E_posterior_pdfs.png"),
              dpi=200, bbox_inches="tight")
print("Saved fig_soft_E_posterior_pdfs.png")

# ---------------------------------------------------------------------------
# 7. PRINT SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"BME Soft-Data Estimation Summary — {YEAR_TARGET}")
print("=" * 60)
print(f"Hard data:  {is_hard.sum()} stations (>= {HARD_THRESH} valid yrs)")
print(f"Soft data:  {is_soft.sum()} stations ({SOFT_THRESH}-{HARD_THRESH-1} valid yrs)")
print(f"  σ_soft range: {sigma_total.min():.2f} – {sigma_total.max():.2f} µg/m³")
print(f"  σ_soft mean:  {sigma_total.mean():.2f} µg/m³")
print(f"  Covariance range: {RANGE_S:.0f} km")
print()
print(f"BME posterior mean:     [{np.nanmin(bme_mean):.1f}, {np.nanmax(bme_mean):.1f}] µg/m³")
print(f"BME posterior std dev:  [{np.nanmin(bme_std_map):.2f}, "
      f"{np.nanmax(bme_std_map):.2f}] µg/m³")
print(f"  (prior std = sqrt({TOTAL_VAR:.1f}) = {np.sqrt(TOTAL_VAR):.2f} µg/m³)")
print()
diff_at_soft = diff_map.ravel()[mask_active][n_s_used > 0]
if len(diff_at_soft):
    print(f"BME - Kriging mean near soft stations:")
    print(f"  mean shift = {np.nanmean(diff_at_soft):.3f} µg/m³")
    print(f"  max  shift = {np.nanmax(np.abs(diff_at_soft)):.3f} µg/m³")
print("=" * 60)

plt.show()
