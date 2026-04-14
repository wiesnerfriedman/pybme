#!/usr/bin/env python3
"""Animated GIF: routed-network flowlines over an event hydrograph.

This example reuses the synthetic routed-network event so the result is
fully shareable, but it renders the network in its actual coordinates
instead of the de-identified tree layout used by
``examples/gif_bme_tree_event.py``.

At each timestep the GIF shows:
  1. an outfall hydrograph with rainfall bars and a moving time cursor,
  2. the routed soft prior on the actual network,
  3. the hard-data-only conditional field,
  4. the BME estimate using hard + soft data.

Output
------
``examples/figures/bme_network_event_flowlines.gif``
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pybme.network import NetworkCovariance, NetworkCovarianceST, adjacency_from_edges
from pybme.network_plots import plot_network_flowlines
from pybme.predict import bme_predict_network_st
from pybme.synthetic_network import (
    build_soft_pdf_inputs,
    prediction_grid,
    simulate_synthetic_routed_network,
)


def _reshape_result_field(results, shape):
    return np.array([result.mean for result in results], dtype=float).reshape(shape)


def _sensor_overlay(ax, dataset, t_idx: int) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(dataset.node_names)}
    for node_name in dataset.sensor_nodes:
        node_idx = name_to_idx[node_name]
        x_coord, y_coord = dataset.coords[node_name]
        has_obs = not np.isnan(dataset.hard_observation_matrix[node_idx, t_idx])
        marker = "^" if has_obs else "o"
        facecolor = "red" if has_obs else "white"
        ax.scatter(
            [x_coord],
            [y_coord],
            marker=marker,
            c=facecolor,
            edgecolor="black",
            s=52,
            linewidths=0.8,
            zorder=6,
        )


def _style_network_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_hydrograph_panel(fig, dataset, hard_soft_mean, t_idx: int):
    outfall_idx = dataset.node_names.index("OUTFALL")
    current_dt = dataset.datetimes[t_idx]

    ax = fig.add_axes([0.06, 0.80, 0.88, 0.16])
    rain_ax = ax.twinx()

    rain_ax.bar(
        dataset.datetimes,
        dataset.rain,
        width=0.035,
        color="steelblue",
        edgecolor="none",
        alpha=0.25,
        zorder=1,
        label="Rain",
    )
    rain_ax.set_ylabel("Rain (in/hr)", fontsize=8, color="steelblue")
    rain_ax.tick_params(axis="y", labelsize=7, colors="steelblue")
    rain_ax.spines["top"].set_visible(False)

    ax.plot(
        dataset.datetimes,
        dataset.truth_matrix[outfall_idx],
        color="black",
        linewidth=1.8,
        label="Truth",
        zorder=4,
    )
    ax.plot(
        dataset.datetimes,
        dataset.soft_matrix[outfall_idx],
        color="darkorange",
        linewidth=1.8,
        linestyle=":",
        label="Routed prior",
        zorder=4,
    )
    ax.plot(
        dataset.datetimes,
        hard_soft_mean[outfall_idx],
        color="seagreen",
        linewidth=2.0,
        label="BME",
        zorder=5,
    )

    obs_mask = ~np.isnan(dataset.hard_observation_matrix[outfall_idx])
    if obs_mask.any():
        obs_times = np.array(dataset.datetimes)[obs_mask]
        obs_values = dataset.hard_observation_matrix[outfall_idx][obs_mask]
        ax.scatter(
            obs_times,
            obs_values,
            color="red",
            edgecolor="black",
            linewidth=0.4,
            s=20,
            zorder=6,
            label="Hard obs",
        )

    ax.axvline(current_dt, color="crimson", linewidth=1.6, linestyle="--", zorder=7)
    ax.set_ylabel("Outfall flow (MGD)", fontsize=9)
    ax.set_xlim(dataset.datetimes[0], dataset.datetimes[-1])
    ax.set_title(
        f"Wet-weather event on actual network coordinates  |  {current_dt:%b %d %H:%M}",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )
    ax.tick_params(axis="x", labelbottom=False)
    ax.spines["top"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    rain_handles, rain_labels = rain_ax.get_legend_handles_labels()
    ax.legend(
        handles + rain_handles,
        labels + rain_labels,
        loc="upper left",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    return ax


def main() -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to write GIF output. Install with: pip install Pillow") from exc

    print("Generating synthetic routed event ...")
    dataset = simulate_synthetic_routed_network(seed=42)
    ck_nodes, tk = prediction_grid(dataset)
    cs_nodes, ts, soft_pdfs, _ = build_soft_pdf_inputs(dataset)

    n_nodes, n_times = dataset.truth_matrix.shape
    W = adjacency_from_edges(n_nodes, dataset.edge_array)
    sigma2 = max(float(np.var(dataset.soft_matrix)), 0.02)
    net_cov = NetworkCovariance(W, kappa=0.55, sigma2=sigma2, from_adjacency=True)
    net_cov_st = NetworkCovarianceST(
        net_cov,
        model_t="exponential",
        params_t=[1.0, 3.0],
        sigma2=sigma2,
    )
    mean_prior = float(np.mean(dataset.hard_values))

    print("Running hard-data conditional field ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
            mean_prior=mean_prior,
        )

    print("Running BME hard + soft field ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
            mean_prior=mean_prior,
        )

    hard_only_mean = _reshape_result_field(hard_only, (n_nodes, n_times))
    hard_soft_mean = _reshape_result_field(hard_soft, (n_nodes, n_times))

    all_values = np.concatenate(
        [
            dataset.soft_matrix.ravel(),
            dataset.truth_matrix.ravel(),
            hard_only_mean.ravel(),
            hard_soft_mean.ravel(),
        ]
    )
    vmax_g = float(np.percentile(all_values[np.isfinite(all_values)], 97))

    out_dir = os.path.join(script_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    gif_path = os.path.join(out_dir, "bme_network_event_flowlines.gif")

    frames = []
    print("Rendering frames ...")
    panel_specs = [
        (dataset.soft_matrix, "Routed prior"),
        (hard_only_mean, "Hard-data conditional"),
        (hard_soft_mean, "BME estimate"),
    ]

    for t_idx in range(n_times):
        fig = plt.figure(figsize=(13.5, 7.2), facecolor="white")
        _draw_hydrograph_panel(fig, dataset, hard_soft_mean, t_idx)

        panel_width = 0.27
        gap = 0.035
        left0 = 0.045
        bottom = 0.08
        height = 0.63

        for panel_idx, (matrix, title) in enumerate(panel_specs):
            left = left0 + panel_idx * (panel_width + gap)
            ax = fig.add_axes([left, bottom, panel_width, height])
            plot_network_flowlines(
                dataset.node_names,
                dataset.coords,
                dataset.edges,
                matrix[:, t_idx],
                title=title,
                cmap="YlOrRd",
                vmin=0.0,
                vmax=vmax_g,
                linewidth=5.0,
                base_edge_width=1.0,
                base_edge_alpha=0.35,
                n_edge_subsegments=20,
                draw_colorbar=False,
                show_nodes=False,
                ax=ax,
            )
            _sensor_overlay(ax, dataset, t_idx)
            _style_network_axis(ax, title)

        cbar_ax = fig.add_axes([0.935, 0.08, 0.015, 0.63])
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0.0, vmax_g))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label("Flow (MGD)", fontsize=9)

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="^", color="w", markerfacecolor="red", markeredgecolor="k", markersize=8, label="Hard obs (this step)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="k", markersize=7, label="Sensor (no obs)"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )

        fig.canvas.draw()
        frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(frame.copy()))
        plt.close(fig)

        if (t_idx + 1) % 6 == 0 or t_idx == n_times - 1:
            print(f"  frame {t_idx + 1}/{n_times}")

    peak_frame = int(np.argmax(dataset.rain))
    durations = [300] * len(frames)
    durations[peak_frame] = 1000
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    print(f"GIF saved: {gif_path}")
    print(f"  frames={len(frames)}, peak hour={dataset.peak_hour:.1f}")


if __name__ == "__main__":
    main()