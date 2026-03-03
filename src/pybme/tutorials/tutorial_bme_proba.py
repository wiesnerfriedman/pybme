"""Tutorial: BME with Soft Probabilistic Data
==============================================
Corresponds to MATLAB ``BMEPROBALIBtutorial.m``.

Demonstrates the full BME workflow:
  1. Generate synthetic hard + soft data on a 2D grid
  2. Compute BME moments (mean, variance) at estimation grid
  3. Extract full posterior PDF + confidence intervals at a point
  4. Compare BME vs kriging maps

Run::

    python -m pybme.tutorials.tutorial_bme_proba
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm

from pybme import bme_predict, SoftPDF, eval_cov, build_cov_matrix

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _simulate(coords, model, params, mean, seed=42):
    """Simulate from a Gaussian random field via Cholesky."""
    K = build_cov_matrix(coords, coords, model, params)
    K += np.eye(len(coords)) * 1e-10
    L = np.linalg.cholesky(K)
    rng = np.random.default_rng(seed)
    return mean + L @ rng.standard_normal(len(coords))


def main():
    rng = np.random.default_rng(42)

    # ── Part A: Data generation ──
    print("=" * 60)
    print("PyBME Tutorial — BME with Soft Probabilistic Data")
    print("=" * 60)

    # candidate locations on a 5×6 grid
    gx, gy = np.meshgrid(np.arange(0, 9, 2), np.arange(0, 11, 2))
    candidates = np.column_stack([gx.ravel(), gy.ravel()])  # 30 pts
    idx = rng.permutation(len(candidates))
    n_hard, n_soft = 10, 11
    ch = candidates[idx[:n_hard]]
    cs = candidates[idx[n_hard:n_hard + n_soft]]

    model, params, mean_val = "exponential", [1.0, 5.0], 3.0
    z_all = _simulate(candidates, model, params, mean_val, seed=42)
    zh = z_all[idx[:n_hard]]
    z_soft_true = z_all[idx[n_hard:n_hard + n_soft]]

    # construct trapezoidal soft PDFs around true values
    soft_pdfs = []
    for zt in z_soft_true:
        width = rng.uniform(0.5, 1.5)
        lo, hi = zt - width, zt + width
        z_grid = np.linspace(lo - 0.3, hi + 0.3, 25)
        pdf_vals = np.where((z_grid >= lo) & (z_grid <= hi), 1.0, 0.0)
        soft_pdfs.append(SoftPDF.from_linear(z_grid, pdf_vals))

    print(f"  Hard data:  {n_hard} points")
    print(f"  Soft data:  {n_soft} points (trapezoidal PDFs)")
    print(f"  Covariance: exponential, sill=1, range=5")
    print(f"  Mean:       {mean_val}")

    # ── Part B: BME estimation on a grid ──
    gx_e, gy_e = np.meshgrid(np.arange(0, 9, 1), np.arange(0, 11, 1))
    ck = np.column_stack([gx_e.ravel(), gy_e.ravel()])
    print(f"\n  Estimation grid: {ck.shape[0]} points ({gx_e.shape[1]}×{gx_e.shape[0]})")

    results = bme_predict(
        ck, ch, zh, cs, soft_pdfs,
        model=model, params=params,
        nhmax=10, nsmax=2, dmax=100.0,
        order=0, n_grid=150, ci_prob=0.95,
    )

    zk = np.array([r.mean for r in results])
    vk = np.array([r.variance for r in results])
    zk_krig = np.array([r.kriging_mean for r in results])
    vk_krig = np.array([r.kriging_var for r in results])

    print(f"  BME  — mean of estimates: {zk.mean():.3f},  mean variance: {vk.mean():.3f}")
    print(f"  Krig — mean of estimates: {zk_krig.mean():.3f},  mean variance: {vk_krig.mean():.3f}")

    # ── Part B2: Posterior PDF at a specific point ──
    ck_pt = np.array([[5.0, 5.0]])
    res_pt = bme_predict(
        ck_pt, ch, zh, cs, soft_pdfs,
        model=model, params=params,
        nhmax=10, nsmax=2, dmax=100.0,
        order=0, n_grid=200, ci_prob=0.68,
    )[0]
    # also get 90% and 99% CIs
    res_90 = bme_predict(ck_pt, ch, zh, cs, soft_pdfs,
                         model=model, params=params,
                         nhmax=10, nsmax=2, dmax=100.0,
                         order=0, n_grid=200, ci_prob=0.90)[0]
    res_99 = bme_predict(ck_pt, ch, zh, cs, soft_pdfs,
                         model=model, params=params,
                         nhmax=10, nsmax=2, dmax=100.0,
                         order=0, n_grid=200, ci_prob=0.99)[0]

    print(f"\n  Posterior at (5,5):")
    print(f"    mode  = {res_pt.mode:.3f}")
    print(f"    mean  = {res_pt.mean:.3f}")
    print(f"    var   = {res_pt.variance:.3f}")
    print(f"    68% CI = [{res_pt.ci_lower:.2f}, {res_pt.ci_upper:.2f}]")
    print(f"    90% CI = [{res_90.ci_lower:.2f}, {res_90.ci_upper:.2f}]")
    print(f"    99% CI = [{res_99.ci_lower:.2f}, {res_99.ci_upper:.2f}]")

    if not HAS_MPL:
        print("\n  (matplotlib not available — skipping plots)")
        return

    # ── Plot 1: Hard + soft data map ──
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.scatter(ch[:, 0], ch[:, 1], c=zh, cmap="hot", marker="s", s=80,
               edgecolors="k", zorder=5, label="hard")
    soft_means = [sp.moments()[0] for sp in soft_pdfs]
    ax.scatter(cs[:, 0], cs[:, 1], c=soft_means, cmap="hot", marker="o",
               s=60, edgecolors="b", linewidths=1.5, zorder=4, label="soft")
    ax.set_title("Hard & Soft Data Locations")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("tutorial_proba_data.png", dpi=120)
    print("\n  Saved tutorial_proba_data.png")

    # ── Plot 2: BME mean map ──
    nx, ny = gx_e.shape[1], gx_e.shape[0]
    Zk = zk.reshape(ny, nx)
    Vk = vk.reshape(ny, nx)
    Zk_kr = zk_krig.reshape(ny, nx)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.pcolormesh(gx_e, gy_e, Zk, cmap="hot", shading="auto")
    ax1.set_title("BME Mean Estimate")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.pcolormesh(gx_e, gy_e, Zk_kr, cmap="hot", shading="auto")
    ax2.set_title("Kriging Mean Estimate")
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    fig2.savefig("tutorial_proba_mean_map.png", dpi=120)
    print("  Saved tutorial_proba_mean_map.png")

    # ── Plot 3: Error variance map ──
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.pcolormesh(gx_e, gy_e, Vk, cmap="YlOrRd", shading="auto")
    ax1.set_title("BME Error Variance")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.pcolormesh(gx_e, gy_e, vk_krig.reshape(ny, nx), cmap="YlOrRd", shading="auto")
    ax2.set_title("Kriging Error Variance")
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2)
    fig3.tight_layout()
    fig3.savefig("tutorial_proba_variance_map.png", dpi=120)
    print("  Saved tutorial_proba_variance_map.png")

    # ── Plot 4: Posterior PDF at (5,5) with CIs ──
    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    ax4.plot(res_pt.z_grid, res_pt.pdf, "b-", lw=2, label="posterior PDF")
    # CI shading
    for r, alpha, lab in [(res_99, 0.08, "99%"), (res_90, 0.15, "90%"),
                          (res_pt, 0.25, "68%")]:
        mask = (res_pt.z_grid >= r.ci_lower) & (res_pt.z_grid <= r.ci_upper)
        ax4.fill_between(res_pt.z_grid, 0, res_pt.pdf, where=mask,
                         alpha=alpha, color="blue", label=f"{lab} CI")
    ax4.axvline(res_pt.mode, color="r", ls="--", lw=1, label=f"mode = {res_pt.mode:.2f}")
    ax4.set_xlabel("z")
    ax4.set_ylabel("f(z)")
    ax4.set_title("BME Posterior PDF at (5, 5)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig("tutorial_proba_posterior.png", dpi=120)
    print("  Saved tutorial_proba_posterior.png")

    # ── Plot 5: Soft PDFs at 4 selected points ──
    fig5, axes5 = plt.subplots(2, 2, figsize=(9, 6))
    for ax, i in zip(axes5.ravel(), range(min(4, n_soft))):
        sp = soft_pdfs[i]
        z = np.linspace(*sp.support, 200)
        ax.plot(z, sp.evaluate(z), "b-", lw=1.5)
        ax.fill_between(z, sp.evaluate(z), alpha=0.2)
        ax.set_title(f"Soft point {i+1} at ({cs[i,0]:.0f}, {cs[i,1]:.0f})")
        ax.set_xlabel("z")
        ax.set_ylabel("f(z)")
        ax.grid(True, alpha=0.3)
    fig5.suptitle("Soft PDFs at Selected Points")
    fig5.tight_layout()
    fig5.savefig("tutorial_proba_soft_pdfs.png", dpi=120)
    print("  Saved tutorial_proba_soft_pdfs.png")
    plt.show()


if __name__ == "__main__":
    main()
