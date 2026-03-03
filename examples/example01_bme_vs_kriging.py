"""End-to-end example — 1-D BME vs kriging with diverse soft data.

Replicates the demo from bme_core.py and MATLAB example01.m.
"""

from __future__ import annotations
import numpy as np
from pybme import bme_predict, SoftPDF, fit_covariance

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    rng = np.random.default_rng(42)

    # ── true field ──
    x = np.linspace(0, 30, 500)
    z_true = np.sin(x / 3) + 0.5 * np.cos(x / 1.5)

    # ── hard data ──
    idx_h = rng.choice(len(x), 15, replace=False)
    ch = x[idx_h].reshape(-1, 1)
    zh = z_true[idx_h] + rng.normal(0, 0.15, 15)

    # ── soft data (5 types) ──
    idx_s = rng.choice(np.setdiff1d(range(len(x)), idx_h), 5, replace=False)
    cs = x[idx_s].reshape(-1, 1)
    soft_pdfs = [
        SoftPDF.from_gaussian(z_true[idx_s[0]] + 0.1, 0.3),              # Gaussian
        SoftPDF.from_uniform(z_true[idx_s[1]] - 0.5, z_true[idx_s[1]] + 0.5),  # Uniform
        SoftPDF.from_truncnorm(z_true[idx_s[2]], 0.4, a=0),              # Truncated-normal
        SoftPDF.from_lognormal(np.log(max(z_true[idx_s[3]], 0.5)), 0.3), # Lognormal
        SoftPDF.from_triangular(z_true[idx_s[4]] - 0.8,
                                z_true[idx_s[4]],
                                z_true[idx_s[4]] + 0.8),                  # Triangular
    ]

    # ── fit covariance ──
    fit = fit_covariance(ch, zh, model="exponential", order=0)
    print(f"Fitted covariance: sill={fit['sill']:.3f}  range={fit['range']:.3f}  "
          f"nugget={fit['nugget']:.4f}")

    # ── predict ──
    n_pred = 150
    ck = np.linspace(0, 30, n_pred).reshape(-1, 1)
    results = bme_predict(
        ck, ch, zh, cs, soft_pdfs,
        model="exponential", params=[fit["sill"], fit["range"]],
        nhmax=20, nsmax=4, dmax=15.0,
        order=0, n_grid=200, n_quad=15,
    )

    bme_mean = np.array([r.mean for r in results])
    bme_var  = np.array([r.variance for r in results])
    krig_mean = np.array([r.kriging_mean for r in results])
    krig_var  = np.array([r.kriging_var for r in results])

    # ── RMSE comparison ──
    z_ref = np.interp(ck.ravel(), x, z_true)
    rmse_k = np.sqrt(np.mean((krig_mean - z_ref) ** 2))
    rmse_b = np.sqrt(np.mean((bme_mean - z_ref) ** 2))
    print(f"RMSE  kriging = {rmse_k:.4f}    BME = {rmse_b:.4f}   "
          f"improvement = {100 * (rmse_k - rmse_b) / rmse_k:.1f}%")

    # ── optional plot ──
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, z_true, "k-", lw=1, alpha=0.4, label="true")
        ax.plot(ch, zh, "ko", ms=4, label="hard")
        ax.plot(ck, krig_mean, "b--", lw=1, label="kriging")
        ax.fill_between(ck.ravel(),
                        krig_mean - 1.96 * np.sqrt(krig_var),
                        krig_mean + 1.96 * np.sqrt(krig_var),
                        color="b", alpha=0.08)
        ax.plot(ck, bme_mean, "r-", lw=1.5, label="BME")
        ax.fill_between(ck.ravel(),
                        bme_mean - 1.96 * np.sqrt(bme_var),
                        bme_mean + 1.96 * np.sqrt(bme_var),
                        color="r", alpha=0.12)
        ax.legend(fontsize=8)
        ax.set_title("PyBME: BME vs Kriging (1-D example)")
        fig.tight_layout()
        fig.savefig("pybme_example01.png", dpi=120)
        print("Saved pybme_example01.png")


if __name__ == "__main__":
    main()
