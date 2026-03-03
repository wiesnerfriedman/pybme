"""Tutorial: General Utilities — Grid, NN, Kernel Smoothing
=============================================================
Corresponds to MATLAB ``GENLIBtutorial.m``.

Demonstrates:
  1. Creating estimation grids
  2. Nearest-neighbour estimation (inverse distance weighting)
  3. Kernel smoothing
  4. Comparing all three estimation approaches

Run::

    python -m pybme.tutorials.tutorial_genlib
"""

from __future__ import annotations
import numpy as np

from pybme import coord2dist

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def create_grid(origin, spacing, n_pts):
    """Create a regular estimation grid (≈ MATLAB ``creategrid``).

    Parameters
    ----------
    origin  : (d,) origin coordinates
    spacing : (d,) spacing in each dimension
    n_pts   : (d,) number of points per dimension

    Returns
    -------
    (N, d) array of grid coordinates
    """
    axes = [origin[i] + np.arange(n_pts[i]) * spacing[i] for i in range(len(origin))]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def inverse_distance(ck, ch, zh, power=2, nhmax=20, dmax=np.inf):
    """Inverse-distance-weighted interpolation.

    ``power=np.inf`` gives nearest-neighbour estimation.
    """
    ck, ch = np.atleast_2d(ck), np.atleast_2d(ch)
    D = coord2dist(ck, ch)
    z_est = np.zeros(len(ck))
    for i in range(len(ck)):
        d = D[i]
        idx = np.where(d <= dmax)[0]
        if len(idx) == 0:
            z_est[i] = np.nan
            continue
        idx = idx[np.argsort(d[idx])][:nhmax]
        di = d[idx]
        if np.min(di) < 1e-10:
            z_est[i] = zh[idx[np.argmin(di)]]
        elif np.isinf(power):
            z_est[i] = zh[idx[0]]  # nearest neighbour
        else:
            w = 1.0 / di ** power
            z_est[i] = np.sum(w * zh[idx]) / np.sum(w)
    return z_est


def kernel_smoothing(ck, ch, zh, bandwidth, nhmax=20, dmax=np.inf):
    """Gaussian kernel smoothing.

    ``bandwidth`` is the Gaussian kernel variance σ².
    """
    ck, ch = np.atleast_2d(ck), np.atleast_2d(ch)
    D = coord2dist(ck, ch)
    z_est = np.zeros(len(ck))
    for i in range(len(ck)):
        d = D[i]
        idx = np.where(d <= dmax)[0]
        if len(idx) == 0:
            z_est[i] = np.nan
            continue
        idx = idx[np.argsort(d[idx])][:nhmax]
        w = np.exp(-0.5 * d[idx] ** 2 / bandwidth)
        z_est[i] = np.sum(w * zh[idx]) / np.sum(w)
    return z_est


def main():
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("PyBME Tutorial — General Utilities")
    print("=" * 60)

    # ── synthetic data ──
    def true_field(x, y):
        return 50 + 15 * np.sin(x / 3) + 10 * np.cos(y / 2.5)

    n = 30
    ch = np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n)])
    zh = true_field(ch[:, 0], ch[:, 1]) + rng.normal(0, 3, n)

    # ── create estimation grid ──
    grid = create_grid(origin=[0, 0], spacing=[1, 1], n_pts=[21, 21])
    print(f"  Data points: {n}")
    print(f"  Grid: 21×21 = {len(grid)} points")

    z_true = true_field(grid[:, 0], grid[:, 1])

    # ── nearest-neighbour ──
    z_nn = inverse_distance(grid, ch, zh, power=np.inf, nhmax=20, dmax=50)
    rmse_nn = np.sqrt(np.nanmean((z_nn - z_true) ** 2))
    print(f"\n  Nearest-neighbour  RMSE = {rmse_nn:.3f}")

    # ── IDW power=2 ──
    z_idw = inverse_distance(grid, ch, zh, power=2, nhmax=20, dmax=50)
    rmse_idw = np.sqrt(np.nanmean((z_idw - z_true) ** 2))
    print(f"  IDW (power=2)      RMSE = {rmse_idw:.3f}")

    # ── kernel smoothing ──
    z_ks = kernel_smoothing(grid, ch, zh, bandwidth=5.0, nhmax=20, dmax=50)
    rmse_ks = np.sqrt(np.nanmean((z_ks - z_true) ** 2))
    print(f"  Kernel smoothing   RMSE = {rmse_ks:.3f}")

    if not HAS_MPL:
        print("\n  (matplotlib not available — skipping plots)")
        return

    gx = grid[:, 0].reshape(21, 21)
    gy = grid[:, 1].reshape(21, 21)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    titles = ["True Field", "Nearest Neighbour", "IDW (power=2)", "Kernel Smoothing"]
    data = [z_true, z_nn, z_idw, z_ks]
    for ax, Z, title in zip(axes.ravel(), data, titles):
        im = ax.pcolormesh(gx, gy, Z.reshape(21, 21), cmap="hot", shading="auto")
        ax.scatter(ch[:, 0], ch[:, 1], c="cyan", marker="v", s=15, zorder=5)
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)
    fig.suptitle("Estimation Methods Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig("tutorial_genlib.png", dpi=120)
    print("\n  Saved tutorial_genlib.png")
    plt.show()


if __name__ == "__main__":
    main()
