#!/usr/bin/env python3
"""Synthetic routed-network BME example without external SWMM files.

This example builds a small sewer-like network with dry-weather inflow,
hour-of-day and day-of-week patterns, and a single RTK triangular response.
The nominal routed run is treated as soft data. A perturbed routed run is
treated as latent truth, from which sparse hard observations are sampled.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.network import NetworkCovariance, NetworkCovarianceST, adjacency_from_edges
from pybme.network_plots import plot_network_field
from pybme.predict import bme_predict_network_st
from pybme.synthetic_network import (
    build_soft_pdf_inputs,
    prediction_grid,
    simulate_synthetic_routed_network,
)


def _reshape_result_field(results, shape):
    return np.array([result.mean for result in results], dtype=float).reshape(shape)


def _reshape_result_std(results, shape):
    return np.array([np.sqrt(result.variance) for result in results], dtype=float).reshape(shape)


def main() -> None:
    dataset = simulate_synthetic_routed_network(seed=42)
    ck_nodes, tk = prediction_grid(dataset)
    cs_nodes, ts, soft_pdfs, soft_std = build_soft_pdf_inputs(dataset)

    n_nodes, n_times = dataset.truth_matrix.shape
    sensor_idx = [dataset.node_names.index(name) for name in dataset.sensor_nodes]

    W = adjacency_from_edges(n_nodes, dataset.edge_array)
    sigma2 = max(float(np.var(dataset.soft_matrix)), 0.02)
    net_cov = NetworkCovariance(W, kappa=0.55, sigma2=sigma2, from_adjacency=True)
    net_cov_st = NetworkCovarianceST(
        net_cov,
        model_t="exponential",
        params_t=[1.0, 3.0],
        sigma2=sigma2,
    )

    print("Synthetic routed network example")
    print(f"  nodes={n_nodes}, links={len(dataset.edges)}, hours={n_times}")
    print(
        "  soft RTK (R, T, K)="
        f"{dataset.rtk_params_soft}, truth RTK={dataset.rtk_params_truth}"
    )
    print(
        "  routing velocity soft/truth="
        f"{dataset.velocity_soft_fps:.2f}/{dataset.velocity_truth_fps:.2f} ft/s"
    )
    print(f"  hard observations={len(dataset.hard_values)}, soft PDFs={len(soft_pdfs)}")

    hard_only = bme_predict_network_st(
        ck_nodes=ck_nodes,
        tk=tk,
        ch_nodes=dataset.hard_nodes,
        th=dataset.hard_times,
        zh=dataset.hard_values,
        net_cov_st=net_cov_st,
        nhmax=28,
        nsmax=0,
        order=0,
        n_grid=101,
        mean_prior=float(np.mean(dataset.hard_values)),
    )
    hard_soft = bme_predict_network_st(
        ck_nodes=ck_nodes,
        tk=tk,
        ch_nodes=dataset.hard_nodes,
        th=dataset.hard_times,
        zh=dataset.hard_values,
        cs_nodes=cs_nodes,
        ts=ts,
        soft_pdfs=soft_pdfs,
        net_cov_st=net_cov_st,
        nhmax=28,
        nsmax=14,
        order=0,
        n_grid=101,
        mean_prior=float(np.mean(dataset.hard_values)),
    )

    hard_only_mean = _reshape_result_field(hard_only, (n_nodes, n_times))
    hard_only_std = _reshape_result_std(hard_only, (n_nodes, n_times))
    hard_soft_mean = _reshape_result_field(hard_soft, (n_nodes, n_times))
    hard_soft_std = _reshape_result_std(hard_soft, (n_nodes, n_times))

    rmse_hard_only = float(np.sqrt(np.mean((hard_only_mean - dataset.truth_matrix) ** 2)))
    rmse_hard_soft = float(np.sqrt(np.mean((hard_soft_mean - dataset.truth_matrix) ** 2)))
    avg_std_hard_only = float(np.mean(hard_only_std))
    avg_std_hard_soft = float(np.mean(hard_soft_std))
    peak_ti = int(np.argmax(dataset.truth_matrix[dataset.node_names.index("OUTFALL")]))

    print(f"  RMSE hard only:   {rmse_hard_only:.4f}")
    print(f"  RMSE hard + soft: {rmse_hard_soft:.4f}")
    print(f"  Avg posterior std hard only:   {avg_std_hard_only:.4f}")
    print(f"  Avg posterior std hard + soft: {avg_std_hard_soft:.4f}")

    out_dir = os.path.join(script_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    fig_a, axes_a = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    axes_a[0].bar(dataset.datetimes, dataset.rain, width=0.03, color="steelblue", edgecolor="none")
    axes_a[0].set_ylabel("Rain\n(in/hr)")
    axes_a[0].set_title("Synthetic routed network forcing and outlet response")

    source_name = "NORTH_A"
    axes_a[1].plot(
        dataset.datetimes,
        dataset.dry_weather_soft[source_name],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="DWF component",
    )
    axes_a[1].plot(
        dataset.datetimes,
        dataset.local_soft[source_name],
        color="darkorange",
        linewidth=2.0,
        label="Local inflow with RTK",
    )
    axes_a[1].plot(
        dataset.datetimes,
        dataset.local_truth[source_name],
        color="firebrick",
        linewidth=1.2,
        alpha=0.8,
        label="Perturbed truth inflow",
    )
    axes_a[1].set_ylabel("Source flow\n(MGD)")
    axes_a[1].legend(loc="upper right")

    outfall_idx = dataset.node_names.index("OUTFALL")
    axes_a[2].plot(dataset.datetimes, dataset.soft_matrix[outfall_idx], color="darkorange", linewidth=2.0, label="Soft routed run")
    axes_a[2].plot(dataset.datetimes, dataset.truth_matrix[outfall_idx], color="black", linewidth=1.6, label="Truth")
    obs_mask = ~np.isnan(dataset.hard_observation_matrix[outfall_idx])
    if obs_mask.any():
        axes_a[2].scatter(
            np.array(dataset.datetimes)[obs_mask],
            dataset.hard_observation_matrix[outfall_idx][obs_mask],
            color="red",
            edgecolor="k",
            linewidth=0.3,
            s=22,
            zorder=4,
            label="Hard observations",
        )
    axes_a[2].set_ylabel("Outlet flow\n(MGD)")
    axes_a[2].legend(loc="upper right")
    axes_a[2].set_xlabel("Time")
    fig_a.savefig(os.path.join(out_dir, "synthetic_routed_network_forcing.png"), dpi=160)

    vmax = float(
        np.percentile(
            np.concatenate(
                [
                    dataset.soft_matrix[:, peak_ti],
                    dataset.truth_matrix[:, peak_ti],
                    hard_only_mean[:, peak_ti],
                    hard_soft_mean[:, peak_ti],
                ]
            ),
            97,
        )
    )
    fig_b, axes_b = plt.subplots(1, 4, figsize=(22, 5.6))
    field_specs = [
        (dataset.soft_matrix[:, peak_ti], "Soft routed prior"),
        (dataset.truth_matrix[:, peak_ti], "Synthetic truth"),
        (hard_only_mean[:, peak_ti], "BME hard only"),
        (hard_soft_mean[:, peak_ti], "BME hard + soft"),
    ]
    for ax, (values, title) in zip(axes_b, field_specs):
        plot_network_field(
            dataset.node_names,
            dataset.coords,
            dataset.edges,
            values,
            obs_nodes=sensor_idx,
            title=f"{title}\n@ t={dataset.times_hours[peak_ti]:.0f} h",
            units="Flow (MGD)",
            vmin=0.0,
            vmax=vmax,
            ax=ax,
        )
    fig_b.savefig(os.path.join(out_dir, "synthetic_routed_network_snapshot.png"), dpi=160)

    fig_c, axes_c = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for ax, node_name in zip(axes_c, ["JUNC_C", "TRUNK", "OUTFALL"]):
        node_idx = dataset.node_names.index(node_name)
        ax.plot(dataset.datetimes, dataset.truth_matrix[node_idx], color="black", linewidth=1.8, label="Truth")
        ax.plot(dataset.datetimes, dataset.soft_matrix[node_idx], color="darkorange", linewidth=1.4, alpha=0.85, label="Soft routed run")
        ax.plot(dataset.datetimes, hard_only_mean[node_idx], color="royalblue", linewidth=1.5, label="BME hard only")
        ax.plot(dataset.datetimes, hard_soft_mean[node_idx], color="seagreen", linewidth=1.8, label="BME hard + soft")
        obs_mask = ~np.isnan(dataset.hard_observation_matrix[node_idx])
        if obs_mask.any():
            ax.scatter(
                np.array(dataset.datetimes)[obs_mask],
                dataset.hard_observation_matrix[node_idx][obs_mask],
                color="red",
                edgecolor="k",
                linewidth=0.3,
                s=18,
                zorder=5,
                label="Hard observations",
            )
        ax.fill_between(
            dataset.datetimes,
            hard_soft_mean[node_idx] - 1.96 * hard_soft_std[node_idx],
            hard_soft_mean[node_idx] + 1.96 * hard_soft_std[node_idx],
            color="seagreen",
            alpha=0.12,
        )
        ax.set_ylabel(f"{node_name}\n(MGD)")
    axes_c[0].legend(loc="upper right", ncol=3)
    axes_c[-1].set_xlabel("Time")
    fig_c.savefig(os.path.join(out_dir, "synthetic_routed_network_timeseries.png"), dpi=160)

    fig_d, ax_d = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    improvement = hard_only_std.mean(axis=0) - hard_soft_std.mean(axis=0)
    ax_d.plot(dataset.datetimes, improvement, color="seagreen", linewidth=2.0)
    ax_d.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax_d.set_title("Average uncertainty reduction from routed soft data")
    ax_d.set_ylabel("Std reduction (MGD)")
    ax_d.set_xlabel("Time")
    fig_d.savefig(os.path.join(out_dir, "synthetic_routed_network_uncertainty.png"), dpi=160)

    print(f"  Figures written to: {out_dir}")
    print("  Files:")
    print("    synthetic_routed_network_forcing.png")
    print("    synthetic_routed_network_snapshot.png")
    print("    synthetic_routed_network_timeseries.png")
    print("    synthetic_routed_network_uncertainty.png")


if __name__ == "__main__":
    main()