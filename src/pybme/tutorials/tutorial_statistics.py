"""Tutorial: Statistics & Variogram Estimation
================================================
Corresponds to MATLAB ``STATLIBtutorial.m``.

Demonstrates:
  1. Descriptive statistics (mean, variance, skewness)
  2. Histograms and kernel density estimation
  3. Empirical variogram computation
  4. Variogram model fitting via REML
  5. Gaussian anamorphosis (normal-score transform)

Run::

    python -m pybme.tutorials.tutorial_statistics
"""

from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde, skew, kurtosis

from pybme import coord2dist, eval_cov, fit_covariance

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def empirical_variogram(coords, values, n_bins=15, max_dist=None):
    """Compute the omnidirectional empirical variogram.

    Returns
    -------
    bin_centers  : (n_bins,) distance bin centres
    gamma        : (n_bins,) semivariance in each bin
    counts       : (n_bins,) number of pairs per bin
    """
    D = coord2dist(coords, coords)
    n = len(values)
    if max_dist is None:
        max_dist = D.max() / 2
    edges = np.linspace(0, max_dist, n_bins + 1)
    gamma = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            b = int(np.searchsorted(edges, d, side="right")) - 1
            if 0 <= b < n_bins:
                gamma[b] += (values[i] - values[j]) ** 2
                counts[b] += 1
    mask = counts > 0
    gamma[mask] /= 2.0 * counts[mask]
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, gamma, counts


def normal_score_transform(x):
    """Gaussian anamorphosis: rank-based normal score transform."""
    from scipy.stats import norm as _norm
    n = len(x)
    ranks = np.argsort(np.argsort(x)).astype(float)
    # Blom formula: (rank - 3/8) / (n + 1/4)
    p = (ranks + 0.625) / (n + 0.25)
    return _norm.ppf(p)


def main():
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("PyBME Tutorial — Statistics & Variogram Estimation")
    print("=" * 60)

    # ── synthetic soil data (3 variables) ──
    n = 60
    coords = np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n)])
    # correlated variables
    sand = 40 + 15 * np.sin(coords[:, 0] / 4) + rng.normal(0, 5, n)
    silt = 35 - 10 * np.sin(coords[:, 0] / 4) + rng.normal(0, 4, n)
    clay = 100 - sand - silt + rng.normal(0, 2, n)

    # ── 1. Descriptive statistics ──
    print("\n  Descriptive Statistics:")
    for name, vals in [("Sand", sand), ("Silt", silt), ("Clay", clay)]:
        print(f"    {name:5s}  mean={vals.mean():.2f}  var={vals.var():.2f}  "
              f"skew={skew(vals):.3f}  kurtosis={kurtosis(vals):.3f}")

    # ── 2. Empirical variogram ──
    d_sand, g_sand, c_sand = empirical_variogram(coords, sand, n_bins=12, max_dist=15)
    print(f"\n  Empirical variogram for sand: {sum(c_sand > 0)}/{len(c_sand)} non-empty bins")

    # ── 3. Fit covariance model ──
    fit = fit_covariance(coords, sand, model="exponential", order=0)
    print(f"\n  REML fit (exponential):  sill={fit['sill']:.2f}  "
          f"range={fit['range']:.2f}  nugget={fit['nugget']:.3f}")

    # ── 4. Normal-score transform ──
    sand_ns = normal_score_transform(sand)
    print(f"\n  Normal-score transform:")
    print(f"    Original sand: mean={sand.mean():.2f}  var={sand.var():.2f}  skew={skew(sand):.3f}")
    print(f"    Transformed:   mean={sand_ns.mean():.4f}  var={sand_ns.var():.4f}  "
          f"skew={skew(sand_ns):.3f}")

    if not HAS_MPL:
        print("\n  (matplotlib not available — skipping plots)")
        return

    # ── Plot 1: Histograms + kernel density ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, name in zip(axes, [sand, silt, clay], ["Sand", "Silt", "Clay"]):
        ax.hist(vals, bins=15, density=True, alpha=0.5, color="steelblue",
                edgecolor="k", linewidth=0.5)
        kde = gaussian_kde(vals)
        x = np.linspace(vals.min() - 5, vals.max() + 5, 200)
        ax.plot(x, kde(x), "r-", lw=1.5, label="KDE")
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("Density-Scaled Histograms", fontsize=12)
    fig.tight_layout()
    fig.savefig("tutorial_stats_histograms.png", dpi=120)
    print("\n  Saved tutorial_stats_histograms.png")

    # ── Plot 2: Empirical variogram + fitted model ──
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    mask = c_sand > 0
    ax2.plot(d_sand[mask], g_sand[mask], "ko", ms=6, label="empirical")
    h_fit = np.linspace(0, d_sand[mask].max(), 200)
    gamma_fit = fit["sill"] - eval_cov(h_fit, "exponential", [fit["sill"], fit["range"]])
    gamma_fit += fit["nugget"] * (h_fit > 0)
    ax2.plot(h_fit, gamma_fit, "r-", lw=2,
             label=f"exponential (sill={fit['sill']:.1f}, range={fit['range']:.1f})")
    ax2.set_xlabel("Distance")
    ax2.set_ylabel("γ(h)")
    ax2.set_title("Empirical Variogram & Fitted Model (Sand)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("tutorial_stats_variogram.png", dpi=120)
    print("  Saved tutorial_stats_variogram.png")

    # ── Plot 3: Normal-score histograms ──
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(sand, bins=15, density=True, alpha=0.5, color="steelblue",
             edgecolor="k", linewidth=0.5)
    ax1.set_title("Sand — Original")
    ax2.hist(sand_ns, bins=15, density=True, alpha=0.5, color="orange",
             edgecolor="k", linewidth=0.5)
    from scipy.stats import norm
    x_ns = np.linspace(-3, 3, 200)
    ax2.plot(x_ns, norm.pdf(x_ns), "k--", lw=1, label="N(0,1)")
    ax2.set_title("Sand — Normal Score")
    ax2.legend()
    fig3.suptitle("Gaussian Anamorphosis", fontsize=12)
    fig3.tight_layout()
    fig3.savefig("tutorial_stats_anamorphosis.png", dpi=120)
    print("  Saved tutorial_stats_anamorphosis.png")

    # ── Plot 4: Scatter matrix (like histscatterplot) ──
    data = np.column_stack([sand, silt, clay])
    labels = ["Sand", "Silt", "Clay"]
    fig4, axes4 = plt.subplots(3, 3, figsize=(9, 9))
    for i in range(3):
        for j in range(3):
            ax = axes4[i, j]
            if i == j:
                ax.hist(data[:, i], bins=12, density=True, alpha=0.5,
                        color="steelblue", edgecolor="k", linewidth=0.5)
            else:
                ax.scatter(data[:, j], data[:, i], s=8, alpha=0.6)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=9)
            if i == 2:
                ax.set_xlabel(labels[j], fontsize=9)
    fig4.suptitle("Histogram-Scatter Plot", fontsize=12)
    fig4.tight_layout()
    fig4.savefig("tutorial_stats_scatter.png", dpi=120)
    print("  Saved tutorial_stats_scatter.png")
    plt.show()


if __name__ == "__main__":
    main()
