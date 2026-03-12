"""Synthetic routed network example data for shareable BME tutorials.

The generator creates a small sewer-like directed network with:

* dry-weather inflows modulated by hour-of-day and day-of-week patterns,
* a single SWMM-style RTK triangular response,
* downstream routing by travel time on each link,
* minimal edge attenuation, and
* sparse simulated hard observations derived from a perturbed truth run.

It is intended for tutorials and example scripts that need a routed network
without depending on external SWMM files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SyntheticRoutedNetwork:
    """Container for the synthetic routed-network example."""

    node_names: List[str]
    coords: Dict[str, Tuple[float, float]]
    edges: List[Tuple[str, str, float]]
    edge_array: np.ndarray
    node_depth: np.ndarray
    times_hours: np.ndarray
    datetimes: List[datetime]
    rain: np.ndarray
    soft_matrix: np.ndarray
    truth_matrix: np.ndarray
    local_soft: Dict[str, np.ndarray]
    local_truth: Dict[str, np.ndarray]
    dry_weather_soft: Dict[str, np.ndarray]
    dry_weather_truth: Dict[str, np.ndarray]
    rdii_soft: Dict[str, np.ndarray]
    rdii_truth: Dict[str, np.ndarray]
    sensor_nodes: List[str]
    hard_mask: np.ndarray
    hard_nodes: np.ndarray
    hard_times: np.ndarray
    hard_values: np.ndarray
    hard_observation_matrix: np.ndarray
    peak_hour: float
    velocity_soft_fps: float
    velocity_truth_fps: float
    rtk_params_soft: Tuple[float, float, float]
    rtk_params_truth: Tuple[float, float, float]


def build_synthetic_network_topology() -> Tuple[
    List[str], Dict[str, Tuple[float, float]], List[Tuple[str, str, float]], np.ndarray, np.ndarray
]:
    """Return the shareable routed-network topology used by the examples."""

    node_names = [
        "NORTH_A",
        "NORTH_B",
        "EAST_A",
        "WEST_A",
        "JUNC_N",
        "JUNC_E",
        "JUNC_C",
        "CORE",
        "SOUTH_A",
        "TRUNK",
        "OUTFALL",
    ]
    coords = {
        "NORTH_A": (0.0, 4200.0),
        "NORTH_B": (1200.0, 4050.0),
        "EAST_A": (2700.0, 3950.0),
        "WEST_A": (-300.0, 2200.0),
        "JUNC_N": (650.0, 2950.0),
        "JUNC_E": (1900.0, 2850.0),
        "JUNC_C": (1250.0, 1900.0),
        "CORE": (900.0, 1100.0),
        "SOUTH_A": (2250.0, 1450.0),
        "TRUNK": (1200.0, 450.0),
        "OUTFALL": (1200.0, 0.0),
    }
    edges = [
        ("NORTH_A", "JUNC_N", 1450.0),
        ("NORTH_B", "JUNC_N", 1325.0),
        ("EAST_A", "JUNC_E", 1500.0),
        ("JUNC_N", "JUNC_C", 1225.0),
        ("JUNC_E", "JUNC_C", 1180.0),
        ("WEST_A", "CORE", 1360.0),
        ("JUNC_C", "CORE", 1120.0),
        ("SOUTH_A", "TRUNK", 1280.0),
        ("CORE", "TRUNK", 1080.0),
        ("TRUNK", "OUTFALL", 860.0),
    ]
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    edge_array = np.array(
        [[name_to_idx[from_node], name_to_idx[to_node]] for from_node, to_node, _ in edges],
        dtype=int,
    )
    node_depth = np.array([0, 0, 0, 0, 1, 1, 2, 3, 1, 4, 5], dtype=float)
    return node_names, coords, edges, edge_array, node_depth


def hour_of_day_multiplier(times_hours: np.ndarray) -> np.ndarray:
    """Return a weekday-style dry-weather diurnal multiplier."""

    profile = np.array(
        [
            0.69, 0.65, 0.62, 0.60, 0.63, 0.75,
            0.91, 1.06, 1.17, 1.15, 1.09, 1.02,
            0.98, 0.99, 1.03, 1.10, 1.20, 1.28,
            1.24, 1.14, 1.03, 0.93, 0.83, 0.75,
        ],
        dtype=float,
    )
    hours = np.floor(times_hours % 24.0).astype(int)
    return profile[hours]


def day_of_week_multiplier(times_hours: np.ndarray) -> np.ndarray:
    """Return a simple Monday-to-Sunday dry-weather multiplier."""

    profile = np.array([1.00, 1.02, 1.04, 1.03, 1.06, 0.93, 0.88], dtype=float)
    day_idx = (np.floor(times_hours / 24.0).astype(int)) % 7
    return profile[day_idx]


def make_rain_hyetograph(times_hours: np.ndarray) -> np.ndarray:
    """Create two synthetic storm pulses in inches per hour."""

    rain = (
        0.58 * np.exp(-0.5 * ((times_hours - 45.0) / 2.6) ** 2)
        + 0.42 * np.exp(-0.5 * ((times_hours - 48.2) / 1.4) ** 2)
        + 0.24 * np.exp(-0.5 * ((times_hours - 96.5) / 3.4) ** 2)
    )
    rain += 0.02 * np.maximum(np.sin(2.0 * np.pi * (times_hours - 6.0) / 24.0), 0.0)
    return rain.astype(float)


def triangular_unit_hydrograph(
    dt_hours: float,
    t_peak_hours: float,
    recession_ratio: float,
) -> np.ndarray:
    """Return a unit-area SWMM-style triangular RTK response kernel."""

    t_base = t_peak_hours * (1.0 + recession_ratio)
    kernel_t = np.arange(0.0, t_base + dt_hours, dt_hours)
    kernel = np.zeros_like(kernel_t)

    rising = kernel_t <= t_peak_hours
    kernel[rising] = kernel_t[rising] / max(t_peak_hours, 1e-12)

    falling = (kernel_t > t_peak_hours) & (kernel_t <= t_base)
    kernel[falling] = (t_base - kernel_t[falling]) / max(t_base - t_peak_hours, 1e-12)

    area = np.trapz(kernel, kernel_t)
    if area <= 0.0:
        raise ValueError("Triangular RTK kernel has zero area.")
    return kernel / area


def _delay_signal(values: np.ndarray, delay_hours: float, dt_hours: float) -> np.ndarray:
    """Shift a causal signal forward in time using linear interpolation."""

    grid = np.arange(values.size, dtype=float) * dt_hours
    return np.interp(grid - delay_hours, grid, values, left=0.0, right=values[-1])


def _aggregate_hourly(values: np.ndarray, steps_per_hour: int) -> np.ndarray:
    """Average a time series or matrix to hourly values."""

    n_complete = values.shape[-1] // steps_per_hour
    trimmed = values[..., : n_complete * steps_per_hour]
    new_shape = trimmed.shape[:-1] + (n_complete, steps_per_hour)
    return trimmed.reshape(new_shape).mean(axis=-1)


def _build_local_components(
    node_names: List[str],
    times_hours: np.ndarray,
    rain: np.ndarray,
    dt_hours: float,
    base_dwf: Dict[str, float],
    rdii_scale: Dict[str, float],
    rtk_params: Tuple[float, float, float],
    phase_offsets: Dict[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build dry-weather and RTK inflow components at source nodes."""

    hod = hour_of_day_multiplier(times_hours)
    dow = day_of_week_multiplier(times_hours)
    effective_rain = np.maximum(rain - 0.04, 0.0)
    kernel = triangular_unit_hydrograph(dt_hours, rtk_params[1], rtk_params[2])
    rtk_base = np.convolve(effective_rain, kernel, mode="full")[: times_hours.size] * dt_hours

    dwf = {}
    rdii = {}
    for node_name in node_names:
        baseline = float(base_dwf.get(node_name, 0.0))
        phase = float(phase_offsets.get(node_name, 0.0))
        subdaily = 1.0 + 0.05 * np.sin(2.0 * np.pi * (times_hours / 24.0 + phase))
        subdaily += 0.03 * np.cos(4.0 * np.pi * (times_hours / 24.0 + phase))
        subdaily = np.clip(subdaily, 0.84, 1.16)
        dwf[node_name] = np.maximum(baseline * hod * dow * subdaily, 0.0)

        scale = float(rdii_scale.get(node_name, 0.0))
        rdii[node_name] = np.maximum(scale * rtk_params[0] * rtk_base, 0.0)
    return dwf, rdii


def _combine_components(
    node_names: List[str],
    dwf: Dict[str, np.ndarray],
    rdii: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Sum dry-weather and RTK components at each node."""

    combined = {}
    for node_name in node_names:
        combined[node_name] = dwf[node_name] + rdii[node_name]
    return combined


def _route_local_inflows(
    node_names: List[str],
    edges: List[Tuple[str, str, float]],
    local_inflows: Dict[str, np.ndarray],
    dt_hours: float,
    velocity_fps: float,
    loss_per_1000ft: float,
) -> np.ndarray:
    """Route local inflows downstream with travel time and small attenuation."""

    incoming = {name: [] for name in node_names}
    for from_node, to_node, length_ft in edges:
        travel_hours = length_ft / max(velocity_fps, 1e-12) / 3600.0
        attenuation = float(np.exp(-loss_per_1000ft * length_ft / 1000.0))
        incoming[to_node].append((from_node, travel_hours, attenuation))

    routed = {}
    for node_name in node_names:
        total = np.array(local_inflows[node_name], dtype=float, copy=True)
        for upstream, travel_hours, attenuation in incoming[node_name]:
            total += attenuation * _delay_signal(routed[upstream], travel_hours, dt_hours)
        routed[node_name] = np.maximum(total, 0.0)

    return np.vstack([routed[node_name] for node_name in node_names])


def _add_process_noise(rng: np.random.Generator, values: np.ndarray) -> np.ndarray:
    """Add smooth process deviations so truth is not identical to the soft run."""

    raw = rng.normal(0.0, 1.0, size=values.shape)
    smooth = np.empty_like(raw)
    smooth[:, 0] = raw[:, 0]
    for col in range(1, values.shape[1]):
        smooth[:, col] = 0.84 * smooth[:, col - 1] + 0.32 * raw[:, col]

    scale = 0.010 + 0.035 * np.sqrt(np.maximum(values, 0.0))
    return np.clip(values + scale * smooth, 0.0, None)


def prediction_grid(dataset: SyntheticRoutedNetwork) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened node/time prediction coordinates for the dataset."""

    n_nodes, n_times = dataset.truth_matrix.shape
    ck_nodes = np.repeat(np.arange(n_nodes, dtype=int), n_times)
    tk = np.tile(dataset.times_hours.astype(float), n_nodes)
    return ck_nodes, tk


def build_soft_pdf_inputs(
    dataset: SyntheticRoutedNetwork,
    *,
    exclude_hard: bool = True,
):
    """Build truncated-normal soft PDFs from the routed soft signal."""

    from .soft_data import SoftPDF

    n_nodes, n_times = dataset.soft_matrix.shape
    nodes_flat = np.repeat(np.arange(n_nodes, dtype=int), n_times)
    times_flat = np.tile(dataset.times_hours.astype(float), n_nodes)
    means_flat = dataset.soft_matrix.reshape(-1)

    depth_norm = dataset.node_depth / max(float(dataset.node_depth.max()), 1.0)
    rain_scale = dataset.rain / max(float(dataset.rain.max()), 1e-12)
    std_matrix = 0.035 + 0.055 * np.sqrt(np.maximum(dataset.soft_matrix, 0.0))
    std_matrix += 0.020 * depth_norm[:, None] + 0.015 * rain_scale[None, :]

    keep = np.ones(nodes_flat.size, dtype=bool)
    if exclude_hard:
        keep &= ~dataset.hard_mask.reshape(-1)

    cs_nodes = nodes_flat[keep]
    ts = times_flat[keep]
    std_flat = std_matrix.reshape(-1)[keep]
    mean_flat = means_flat[keep]

    soft_pdfs = []
    for mean_value, std_value in zip(mean_flat, std_flat):
        upper = max(float(mean_value + 5.0 * std_value), 0.30)
        soft_pdfs.append(
            SoftPDF.from_truncnorm(
                mu=float(mean_value),
                sigma=float(std_value),
                a=0.0,
                b=upper,
                n_pts=41,
            )
        )

    return cs_nodes, ts, soft_pdfs, std_matrix


def simulate_synthetic_routed_network(
    *,
    seed: int = 42,
    dt_minutes: int = 15,
    duration_days: int = 6,
    window_hours: int = 36,
    pre_peak_hours: int = 12,
    hard_stride_hours: int = 2,
    start_datetime: datetime | None = None,
) -> SyntheticRoutedNetwork:
    """Generate the routed-network example used by the shareable BME script.

    The nominal run is used as soft data. A perturbed run with slightly
    different DWF, RTK, routing velocity, and smooth process deviations is
    treated as the latent truth from which sparse hard observations are drawn.
    """

    rng = np.random.default_rng(seed)
    node_names, coords, edges, edge_array, node_depth = build_synthetic_network_topology()
    dt_hours = dt_minutes / 60.0
    total_hours = int(duration_days * 24)
    n_steps = int(round(total_hours / dt_hours))
    times_hi = np.arange(n_steps, dtype=float) * dt_hours
    start_dt = start_datetime or datetime(2024, 4, 1, 0, 0, 0)
    rain_hi = make_rain_hyetograph(times_hi)

    base_dwf_soft = {
        "NORTH_A": 0.36,
        "NORTH_B": 0.27,
        "EAST_A": 0.31,
        "WEST_A": 0.23,
        "SOUTH_A": 0.18,
        "CORE": 0.12,
    }
    rdii_scale_soft = {
        "NORTH_A": 2.2,
        "NORTH_B": 1.6,
        "EAST_A": 1.9,
        "WEST_A": 1.4,
        "SOUTH_A": 1.1,
    }
    phase_soft = {
        "NORTH_A": 0.02,
        "NORTH_B": 0.08,
        "EAST_A": 0.15,
        "WEST_A": -0.06,
        "SOUTH_A": 0.21,
        "CORE": -0.02,
    }

    base_dwf_truth = {
        key: value
        for key, value in zip(
            base_dwf_soft.keys(),
            [0.378, 0.258, 0.326, 0.214, 0.192, 0.128],
        )
    }
    rdii_scale_truth = {
        key: value
        for key, value in zip(
            rdii_scale_soft.keys(),
            [2.38, 1.48, 2.05, 1.30, 1.18],
        )
    }
    phase_truth = {
        key: value
        for key, value in zip(
            phase_soft.keys(),
            [0.06, 0.12, 0.20, -0.01, 0.27, 0.02],
        )
    }

    rtk_params_soft = (0.085, 2.0, 3.0)
    rtk_params_truth = (0.095, 2.35, 3.4)
    velocity_soft_fps = 0.50
    velocity_truth_fps = 0.47

    dwf_soft_hi, rdii_soft_hi = _build_local_components(
        node_names,
        times_hi,
        rain_hi,
        dt_hours,
        base_dwf_soft,
        rdii_scale_soft,
        rtk_params_soft,
        phase_soft,
    )
    dwf_truth_hi, rdii_truth_hi = _build_local_components(
        node_names,
        times_hi,
        rain_hi,
        dt_hours,
        base_dwf_truth,
        rdii_scale_truth,
        rtk_params_truth,
        phase_truth,
    )

    local_soft_hi = _combine_components(node_names, dwf_soft_hi, rdii_soft_hi)
    local_truth_hi = _combine_components(node_names, dwf_truth_hi, rdii_truth_hi)

    soft_hi = _route_local_inflows(
        node_names,
        edges,
        local_soft_hi,
        dt_hours,
        velocity_soft_fps,
        loss_per_1000ft=0.003,
    )
    truth_hi = _route_local_inflows(
        node_names,
        edges,
        local_truth_hi,
        dt_hours,
        velocity_truth_fps,
        loss_per_1000ft=0.002,
    )
    truth_hi = _add_process_noise(rng, truth_hi)

    steps_per_hour = int(round(1.0 / dt_hours))
    soft_hourly = _aggregate_hourly(soft_hi, steps_per_hour)
    truth_hourly = _aggregate_hourly(truth_hi, steps_per_hour)
    rain_hourly = _aggregate_hourly(rain_hi[None, :], steps_per_hour)[0]

    dwf_soft_hourly = {name: _aggregate_hourly(values[None, :], steps_per_hour)[0] for name, values in dwf_soft_hi.items()}
    dwf_truth_hourly = {name: _aggregate_hourly(values[None, :], steps_per_hour)[0] for name, values in dwf_truth_hi.items()}
    rdii_soft_hourly = {name: _aggregate_hourly(values[None, :], steps_per_hour)[0] for name, values in rdii_soft_hi.items()}
    rdii_truth_hourly = {name: _aggregate_hourly(values[None, :], steps_per_hour)[0] for name, values in rdii_truth_hi.items()}
    local_soft_hourly = {name: dwf_soft_hourly[name] + rdii_soft_hourly[name] for name in node_names}
    local_truth_hourly = {name: dwf_truth_hourly[name] + rdii_truth_hourly[name] for name in node_names}

    peak_idx = int(np.argmax(rain_hourly))
    max_start = max(soft_hourly.shape[1] - window_hours, 0)
    window_start = int(np.clip(peak_idx - pre_peak_hours, 0, max_start))
    window_end = window_start + window_hours

    times_window = np.arange(window_start, window_end, dtype=float)
    datetimes_window = [start_dt + timedelta(hours=float(hour)) for hour in times_window]
    rain_window = rain_hourly[window_start:window_end]
    soft_window = soft_hourly[:, window_start:window_end]
    truth_window = truth_hourly[:, window_start:window_end]

    local_soft_window = {name: values[window_start:window_end] for name, values in local_soft_hourly.items()}
    local_truth_window = {name: values[window_start:window_end] for name, values in local_truth_hourly.items()}
    dwf_soft_window = {name: values[window_start:window_end] for name, values in dwf_soft_hourly.items()}
    dwf_truth_window = {name: values[window_start:window_end] for name, values in dwf_truth_hourly.items()}
    rdii_soft_window = {name: values[window_start:window_end] for name, values in rdii_soft_hourly.items()}
    rdii_truth_window = {name: values[window_start:window_end] for name, values in rdii_truth_hourly.items()}

    sensor_nodes = ["JUNC_N", "CORE", "TRUNK", "OUTFALL"]
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    sensor_idx = [name_to_idx[name] for name in sensor_nodes]

    n_nodes, n_times = truth_window.shape
    hard_mask = np.zeros((n_nodes, n_times), dtype=bool)
    storm_time = int(np.argmax(rain_window))
    for time_idx in range(n_times):
        if time_idx % hard_stride_hours != 0:
            continue
        for node_idx in sensor_idx:
            if rng.random() > 0.12:
                hard_mask[node_idx, time_idx] = True
    hard_mask[sensor_idx, 0] = True
    hard_mask[sensor_idx, storm_time] = True
    hard_mask[sensor_idx, -1] = True

    nodes_flat = np.repeat(np.arange(n_nodes, dtype=int), n_times)
    times_flat = np.tile(times_window, n_nodes)
    truth_flat = truth_window.reshape(-1)
    hard_mask_flat = hard_mask.reshape(-1)

    hard_std = 0.012 + 0.018 * np.sqrt(np.maximum(truth_flat[hard_mask_flat], 0.0))
    hard_values = np.clip(
        truth_flat[hard_mask_flat] + rng.normal(0.0, hard_std),
        0.0,
        None,
    )
    hard_nodes = nodes_flat[hard_mask_flat]
    hard_times = times_flat[hard_mask_flat]
    hard_observation_matrix = np.full_like(truth_window, np.nan)
    hard_observation_matrix.reshape(-1)[hard_mask_flat] = hard_values

    return SyntheticRoutedNetwork(
        node_names=node_names,
        coords=coords,
        edges=edges,
        edge_array=edge_array,
        node_depth=node_depth,
        times_hours=times_window,
        datetimes=datetimes_window,
        rain=rain_window,
        soft_matrix=soft_window,
        truth_matrix=truth_window,
        local_soft=local_soft_window,
        local_truth=local_truth_window,
        dry_weather_soft=dwf_soft_window,
        dry_weather_truth=dwf_truth_window,
        rdii_soft=rdii_soft_window,
        rdii_truth=rdii_truth_window,
        sensor_nodes=sensor_nodes,
        hard_mask=hard_mask,
        hard_nodes=hard_nodes,
        hard_times=hard_times,
        hard_values=hard_values,
        hard_observation_matrix=hard_observation_matrix,
        peak_hour=float(times_window[storm_time]),
        velocity_soft_fps=velocity_soft_fps,
        velocity_truth_fps=velocity_truth_fps,
        rtk_params_soft=rtk_params_soft,
        rtk_params_truth=rtk_params_truth,
    )