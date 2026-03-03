"""Tutorial: Kriging (Hard Data Only)
======================================
Corresponds to MATLAB ``BMEHRLIBtutorial.m``.

Demonstrates:
  1. Simple / Ordinary kriging on a 2D grid
  2. Effect of covariance model choice
  3. Kriging variance map

Run::

    python -m pybme.tutorials.tutorial_kriging
"""

from __future__ import annotations
import numpy as np

from pybme import bme_predict, build_cov_matrix

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    rng = np.random.default_rng(12)

    print("=" * 60)
    print("PyBME Tutorial — Kriging (Hard Data Only)")
    print("=" * 60)

    # ── synthetic data from a known field ──
    def true_field(x, y):
        return 50 + 20 * np.sin(x / 3000) + 10 * np.cos(y / 2000)

    # hard data (simulating soil sampling)
    n_hard = 25
    ch = np.column_stack([
        rng.uniform(178000, 195000, n_hard),
        rng.uniform(90000, 108000, n_hard),
    ])
    zh = true_field(ch[:, 0], ch[:, 1]) + rng.normal(0, 5.0, n_hard)

    # estimation grid (matching MATLAB tutorial: 17×19 grid)
    gx, gy = np.meshgrid(
        np.linspace(178000, 194000, 17),
        np.linspace(90000, 108000, 19),
    )
    ck = np.column_stack([gx.ravel(), gy.ravel()])
    z_ref = true_field(ck[:, 0], ck[:, 1])

    print(f"  Hard data:      {n_hard} points")
    print(f"  Estimation grid: {ck.shape[0]} points (17×19)")

    # ── Kriging with different covariance models ──
    cov_configs = [
        ("Exponential",     "exponential", [100.0, 5000.0]),
        ("Gaussian (RBF)",  "gaussian",    [100.0, 5000.0]),
        ("Spherical",       "spherical",   [100.0, 8000.0]),
        ("Nested nug+exp",  ["nugget", "exponential"], [[10.0], [90.0, 5000.0]]),
    ]

    for name, model, params in cov_configs:
        results = bme_predict(ck, ch, zh, model=model, params=params,
                              nhmax=10, dmax=15000.0, order=0)
        z_est = np.array([r.mean for r in results])
        v_est = np.array([r.variance for r in results])
        rmse = np.sqrt(np.mean((z_est - z_ref) ** 2))
        print(f"\n  {name:20s}  RMSE = {rmse:.3f}  mean var = {v_est.mean():.2f}")

    # ── simple kriging (order=NaN, fixed mean) ──
    results_sk = bme_predict(ck, ch, zh, model="exponential",
                             params=[100.0, 5000.0],
                             nhmax=10, dmax=15000.0,
                             order=float("nan"), mean_prior=50.0)
    z_sk = np.array([r.mean for r in results_sk])
    rmse_sk = np.sqrt(np.mean((z_sk - z_ref) ** 2))
    print(f"\n  Simple Kriging       RMSE = {rmse_sk:.3f}")

    # ── ordinary kriging (order=0) ──
    results_ok = bme_predict(ck, ch, zh, model="exponential",
                             params=[100.0, 5000.0],
                             nhmax=10, dmax=15000.0, order=0)
    z_ok = np.array([r.mean for r in results_ok])
    v_ok = np.array([r.variance for r in results_ok])
    rmse_ok = np.sqrt(np.mean((z_ok - z_ref) ** 2))
    print(f"  Ordinary Kriging     RMSE = {rmse_ok:.3f}")

    if not HAS_MPL:
        print("\n  (matplotlib not available — skipping plots)")
        return

    # ── Plots ──
    nx, ny = 17, 19

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    # True field
    im0 = axes[0].pcolormesh(gx, gy, z_ref.reshape(ny, nx),
                              cmap="hot", shading="auto")
    axes[0].scatter(ch[:, 0], ch[:, 1], c="cyan", marker="v", s=20, zorder=5)
    axes[0].set_title("True Field")
    plt.colorbar(im0, ax=axes[0])

    # Ordinary kriging
    im1 = axes[1].pcolormesh(gx, gy, z_ok.reshape(ny, nx),
                              cmap="hot", shading="auto")
    axes[1].scatter(ch[:, 0], ch[:, 1], c="cyan", marker="v", s=20, zorder=5)
    axes[1].set_title(f"Ordinary Kriging (RMSE={rmse_ok:.2f})")
    plt.colorbar(im1, ax=axes[1])

    # Kriging variance
    im2 = axes[2].pcolormesh(gx, gy, v_ok.reshape(ny, nx),
                              cmap="YlOrRd", shading="auto")
    axes[2].scatter(ch[:, 0], ch[:, 1], c="cyan", marker="v", s=20, zorder=5)
    axes[2].set_title("Kriging Variance")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_aspect("equal")
    fig.suptitle("PyBME: Kriging Tutorial", fontsize=13)
    fig.tight_layout()
    fig.savefig("tutorial_kriging.png", dpi=120)
    print("\n  Saved tutorial_kriging.png")
    plt.show()


if __name__ == "__main__":
    main()
