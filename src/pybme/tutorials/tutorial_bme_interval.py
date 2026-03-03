"""Tutorial: BME with Interval Data
=====================================
Corresponds to MATLAB ``BMEINTLIBtutorial.m``.

Demonstrates:
  1. Set up hard data + interval (uniform) soft data
  2. BME estimation with intervals versus kriging
  3. Effect of interval width on the posterior

Run::

    python -m pybme.tutorials.tutorial_bme_interval
"""

from __future__ import annotations
import numpy as np

from pybme import bme_predict, SoftPDF

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    rng = np.random.default_rng(7)

    print("=" * 60)
    print("PyBME Tutorial — BME with Interval Data")
    print("=" * 60)

    # ── synthetic data ──
    # True field: sin surface
    def true_field(x, y):
        return 3.0 + np.sin(x / 3.0) + 0.5 * np.cos(y / 2.5)

    # hard data
    n_hard = 15
    ch = rng.uniform([0, 0], [10, 10], (n_hard, 2))
    zh = true_field(ch[:, 0], ch[:, 1]) + rng.normal(0, 0.2, n_hard)

    # interval soft data — lower/upper bounds around truth
    n_soft = 8
    cs = rng.uniform([0, 0], [10, 10], (n_soft, 2))
    z_true_soft = true_field(cs[:, 0], cs[:, 1])
    widths = rng.uniform(0.3, 1.5, n_soft)
    soft_pdfs = [
        SoftPDF.from_interval(z_true_soft[i] - widths[i],
                              z_true_soft[i] + widths[i])
        for i in range(n_soft)
    ]

    print(f"  Hard data:     {n_hard} points")
    print(f"  Interval data: {n_soft} points (widths {widths.min():.2f}–{widths.max():.2f})")

    # ── estimation grid ──
    gx, gy = np.meshgrid(np.linspace(0, 10, 25), np.linspace(0, 10, 25))
    ck = np.column_stack([gx.ravel(), gy.ravel()])

    model, params = "exponential", [1.0, 5.0]

    # kriging (hard only)
    res_krig = bme_predict(ck, ch, zh, model=model, params=params,
                           nhmax=10, dmax=20.0, order=0)
    # BME with intervals
    res_bme = bme_predict(ck, ch, zh, cs, soft_pdfs,
                          model=model, params=params,
                          nhmax=10, nsmax=3, dmax=20.0, order=0)

    z_krig = np.array([r.mean for r in res_krig])
    z_bme  = np.array([r.mean for r in res_bme])
    v_krig = np.array([r.variance for r in res_krig])
    v_bme  = np.array([r.variance for r in res_bme])

    z_ref = true_field(ck[:, 0], ck[:, 1])
    rmse_k = np.sqrt(np.mean((z_krig - z_ref) ** 2))
    rmse_b = np.sqrt(np.mean((z_bme  - z_ref) ** 2))
    print(f"\n  RMSE  kriging = {rmse_k:.4f}   BME+interval = {rmse_b:.4f}")
    print(f"  Mean variance: kriging = {v_krig.mean():.4f}   BME = {v_bme.mean():.4f}")

    # ── effect of interval width at one point ──
    ck_pt = np.array([[5.0, 5.0]])
    zt_pt = true_field(5.0, 5.0)
    print(f"\n  Effect of interval width at (5,5), true value = {zt_pt:.3f}:")
    for w in [0.1, 0.5, 1.0, 2.0, 5.0, 50.0]:
        sp = [SoftPDF.from_interval(zt_pt - w, zt_pt + w)]
        cs_pt = np.array([[5.5, 5.0]])
        r = bme_predict(ck_pt, ch, zh, cs_pt, sp,
                        model=model, params=params,
                        nhmax=10, nsmax=2, dmax=20.0, order=0)[0]
        print(f"    width ±{w:5.1f}  →  mean = {r.mean:.3f}  var = {r.variance:.4f}")

    if not HAS_MPL:
        print("\n  (matplotlib not available — skipping plots)")
        return

    nx, ny = gx.shape[1], gx.shape[0]

    # ── Plot 1: Data locations ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ch[:, 0], ch[:, 1], c=zh, cmap="hot", marker="s", s=80,
               edgecolors="k", zorder=5, label="hard data")
    for i in range(n_soft):
        lo, hi = z_true_soft[i] - widths[i], z_true_soft[i] + widths[i]
        ax.scatter(cs[i, 0], cs[i, 1], c="steelblue", marker="o",
                   s=30 + 50 * widths[i], edgecolors="b", zorder=4)
    ax.scatter([], [], c="steelblue", marker="o", s=60, edgecolors="b",
               label="interval data")
    ax.legend()
    ax.set_title("Hard & Interval Data")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("tutorial_interval_data.png", dpi=120)
    print("\n  Saved tutorial_interval_data.png")

    # ── Plot 2: BME vs kriging maps ──
    fig2, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, Z, title in [
        (axes[0, 0], z_bme, "BME Mean (hard + interval)"),
        (axes[0, 1], z_krig, "Kriging Mean (hard only)"),
        (axes[1, 0], v_bme, "BME Variance"),
        (axes[1, 1], v_krig, "Kriging Variance"),
    ]:
        im = ax.pcolormesh(gx, gy, Z.reshape(ny, nx),
                           cmap="hot" if "Mean" in title else "YlOrRd",
                           shading="auto")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)
    fig2.tight_layout()
    fig2.savefig("tutorial_interval_maps.png", dpi=120)
    print("  Saved tutorial_interval_maps.png")
    plt.show()


if __name__ == "__main__":
    main()
