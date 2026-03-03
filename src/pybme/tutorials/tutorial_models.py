"""Tutorial: Covariance & Variogram Models
==========================================
Corresponds to MATLAB ``MODELSLIBtutorial.m``.

Demonstrates:
  1. All available covariance models plotted as C(h) vs h
  2. Corresponding variograms  γ(h) = C(0) − C(h)
  3. Nested (additive) models

Run::

    python -m pybme.tutorials.tutorial_models
"""

from __future__ import annotations
import numpy as np

from pybme import (
    exponential_cov, gaussian_cov, spherical_cov,
    matern_cov, nugget_cov, hole_cos_cov, eval_cov,
    eval_cov_st,
)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    h = np.linspace(0, 1.5, 300)

    # ── model definitions ──
    models = [
        ("Nugget",      "nugget",      [1.0]),
        ("Exponential", "exponential", [1.0, 1.0]),
        ("Spherical",   "spherical",   [1.0, 1.0]),
        ("Gaussian",    "gaussian",    [1.0, 1.0]),
        ("Matérn ν=1.5","matern",      [1.0, 1.0, 1.5]),
        ("Hole-Cosine", "hole_cos",    [1.0, 0.5]),
    ]

    print("=" * 60)
    print("PyBME Tutorial — Covariance Models")
    print("=" * 60)
    for name, model, params in models:
        c0 = eval_cov(0.0, model, params)
        c1 = eval_cov(1.0, model, params)
        print(f"  {name:20s}  C(0) = {c0:.4f}   C(1.0) = {c1:.4f}")

    # ── nested model ──
    nest_model = ["nugget", "spherical", "spherical"]
    nest_params = [[0.2], [0.4, 0.3], [0.4, 1.2]]
    c_nest = eval_cov(h, nest_model, nest_params)
    print(f"\n  Nested (nug + 2×sph)  C(0) = {eval_cov(0.0, nest_model, nest_params):.4f}")

    if not HAS_MPL:
        print("\n  (matplotlib not available — skipping plots)")
        return

    # ── covariance plot ──
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, model, params) in zip(axes.ravel(), models):
        c = eval_cov(h, model, params)
        ax.plot(h, c, "b-", lw=1.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("h")
        ax.set_ylabel("C(h)")
        ax.set_ylim(bottom=-0.2)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Covariance Models  C(h)", fontsize=13)
    fig.tight_layout()
    fig.savefig("tutorial_models_covariance.png", dpi=120)
    print("\n  Saved tutorial_models_covariance.png")

    # ── variogram plot:  γ(h) = C(0) − C(h)  ──
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, model, params) in zip(axes2.ravel(), models):
        c = eval_cov(h, model, params)
        gamma = eval_cov(0.0, model, params) - c
        ax.plot(h, gamma, "r-", lw=1.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("h")
        ax.set_ylabel("γ(h)")
        ax.grid(True, alpha=0.3)
    fig2.suptitle("Variogram Models  γ(h) = C(0) − C(h)", fontsize=13)
    fig2.tight_layout()
    fig2.savefig("tutorial_models_variogram.png", dpi=120)
    print("  Saved tutorial_models_variogram.png")

    # ── nested model plot ──
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    gamma_nest = eval_cov(0.0, nest_model, nest_params) - c_nest
    ax3.plot(h, gamma_nest, "k-", lw=2, label="Nested: nug(0.2) + sph(0.4,0.3) + sph(0.4,1.2)")
    ax3.set_xlabel("h")
    ax3.set_ylabel("γ(h)")
    ax3.set_title("Nested Variogram Model")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("tutorial_models_nested.png", dpi=120)
    print("  Saved tutorial_models_nested.png")

    # ── space-time covariance ──
    r = np.linspace(0, 3.0, 100)
    t = np.linspace(0, 6.0, 100)
    R, T = np.meshgrid(r, t)

    # Separable: exp(space) / exp(time)
    sill, range_s, range_t = 1.0, 1.0, 2.0
    C_sep = eval_cov_st(R, T, "exponential", [1.0, range_s],
                        "exponential", [1.0, range_t], sill=sill)

    # Non-separable: gaussian ST with metric d = r + k*t
    sill_ns, range_st, k_st = 1.0, 2.0, 0.5
    C_nonsep = eval_cov_st(R, T, "gaussian_st", [sill_ns, range_st, k_st])

    # 1D slices
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4.5))
    C_r_sep = eval_cov_st(r, 0.0, "exponential", [1.0, range_s],
                          "exponential", [1.0, range_t], sill=sill)
    C_t_sep = eval_cov_st(0.0, t, "exponential", [1.0, range_s],
                          "exponential", [1.0, range_t], sill=sill)
    axes4[0].plot(r, C_r_sep, "b-", lw=2)
    axes4[0].axhline(sill, color="gray", ls="--", lw=0.8, label=f"sill = {sill}")
    axes4[0].set_xlabel("Spatial lag r")
    axes4[0].set_ylabel("C(r, t=0)")
    axes4[0].set_title("Separable S/T — Spatial Slice")
    axes4[0].legend(fontsize=9)
    axes4[0].grid(True, alpha=0.3)

    axes4[1].plot(t, C_t_sep, "r-", lw=2)
    axes4[1].axhline(sill, color="gray", ls="--", lw=0.8, label=f"sill = {sill}")
    axes4[1].set_xlabel("Temporal lag t")
    axes4[1].set_ylabel("C(r=0, t)")
    axes4[1].set_title("Separable S/T — Temporal Slice")
    axes4[1].legend(fontsize=9)
    axes4[1].grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig("tutorial_models_st_slices.png", dpi=120)
    print("  Saved tutorial_models_st_slices.png")

    # 2D surface comparison
    fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    im1 = ax1.pcolormesh(R, T, C_sep, cmap="viridis", shading="auto", vmin=0, vmax=1)
    cs1 = ax1.contour(R, T, C_sep, levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
                       colors="w", linewidths=0.8)
    ax1.clabel(cs1, fontsize=8, fmt="%.2f")
    ax1.plot(0, 0, "w*", ms=14, zorder=5)
    ax1.set_xlabel("Spatial lag r")
    ax1.set_ylabel("Temporal lag t")
    ax1.set_title("Separable: exp/exp")
    plt.colorbar(im1, ax=ax1, label="C(r,t)")

    im2 = ax2.pcolormesh(R, T, C_nonsep, cmap="viridis", shading="auto", vmin=0, vmax=1)
    cs2 = ax2.contour(R, T, C_nonsep, levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
                       colors="w", linewidths=0.8)
    ax2.clabel(cs2, fontsize=8, fmt="%.2f")
    ax2.plot(0, 0, "w*", ms=14, zorder=5)
    ax2.set_xlabel("Spatial lag r")
    ax2.set_ylabel("Temporal lag t")
    ax2.set_title(f"Non-separable: gaussian_st (k={k_st})")
    plt.colorbar(im2, ax=ax2, label="C(r,t)")

    fig5.suptitle("S/T Covariance C(r,t) — sill at (0,0)", fontsize=13)
    fig5.tight_layout()
    fig5.savefig("tutorial_models_st_surface.png", dpi=120)
    print("  Saved tutorial_models_st_surface.png")

    plt.show()


if __name__ == "__main__":
    main()
