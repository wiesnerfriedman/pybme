"""Reusable SWMM utilities for network-domain example workflows.

This module intentionally keeps dependencies light so that the public
package can expose the reusable parsing and CSV-loading logic without
depending on external SWMM reader libraries.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SwmmNetwork:
    """Parsed SWMM network topology from an ``.inp`` file."""

    junctions: dict
    storages: dict
    outfalls: dict
    all_node_names: list[str]
    edges: list[tuple[str, str, float]]
    coords: dict[str, tuple[float, float]]

    @property
    def node_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.all_node_names)}


@dataclass(frozen=True)
class ObservationTable:
    """Parsed observation CSV with header metadata and valid timestamps."""

    rows: list[list[str]]
    meter_names: list[str]
    node_names: list[str]
    types: list[str]
    units: list[str]
    value_cols: list[int]
    value_names: list[str]
    datetimes: list[datetime]
    valid_row_indices: list[int]


def _read_section(lines: Iterable[str], section_name: str):
    """Yield non-comment, non-blank lines from a SWMM ``[SECTION]``."""
    in_section = False
    target = f"[{section_name.upper()}]"
    for line in lines:
        stripped = line.rstrip("\n")
        if stripped.startswith("["):
            in_section = stripped.strip().upper() == target
            continue
        if not in_section:
            continue
        text = stripped.strip()
        if not text or text.startswith(";;") or text.startswith(";"):
            continue
        yield text


def parse_swmm_inp(inp_path: str | Path,
                   default_link_length: float = 50.0) -> SwmmNetwork:
    """Parse a SWMM ``.inp`` file into nodes, edges, and coordinates."""
    with open(inp_path, "r", errors="replace") as handle:
        lines = handle.readlines()

    junctions = {}
    for line in _read_section(lines, "JUNCTIONS"):
        parts = line.split()
        junctions[parts[0]] = {"elev": float(parts[1])}

    storages = {}
    for line in _read_section(lines, "STORAGE"):
        parts = line.split()
        storages[parts[0]] = {"elev": float(parts[1])}

    outfalls = {}
    for line in _read_section(lines, "OUTFALLS"):
        parts = line.split()
        outfalls[parts[0]] = {"elev": float(parts[1])}

    all_node_names = set(junctions) | set(storages) | set(outfalls)

    edges: list[tuple[str, str, float]] = []
    for line in _read_section(lines, "CONDUITS"):
        parts = line.split()
        from_node, to_node = parts[1], parts[2]
        edges.append((from_node, to_node, float(parts[3])))
        all_node_names.update([from_node, to_node])

    for section in ["PUMPS", "ORIFICES", "WEIRS", "OUTLETS"]:
        for line in _read_section(lines, section):
            parts = line.split()
            if len(parts) >= 3:
                from_node, to_node = parts[1], parts[2]
                edges.append((from_node, to_node, float(default_link_length)))
                all_node_names.update([from_node, to_node])

    coords = {}
    for line in _read_section(lines, "COORDINATES"):
        parts = line.split()
        if len(parts) >= 3:
            coords[parts[0]] = (float(parts[1]), float(parts[2]))

    return SwmmNetwork(
        junctions=junctions,
        storages=storages,
        outfalls=outfalls,
        all_node_names=sorted(all_node_names),
        edges=edges,
        coords=coords,
    )


def build_edge_array(node_names: Sequence[str],
                     edges: Sequence[tuple[str, str, float]]) -> np.ndarray:
    """Build a 2-column integer edge array from parsed SWMM edges."""
    node_idx = {name: i for i, name in enumerate(node_names)}
    edge_array = []
    for from_node, to_node, _ in edges:
        i = node_idx.get(from_node)
        j = node_idx.get(to_node)
        if i is not None and j is not None and i != j:
            edge_array.append([i, j])
    return np.asarray(edge_array, dtype=int)


def read_meter_node_map(csv_path: str | Path,
                        meter_column: str = "Meter",
                        node_column: str = "Node") -> dict[str, str]:
    """Read the meter-to-node mapping CSV used by SWMM examples."""
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row[meter_column]: row[node_column]
        for row in rows
        if row.get(meter_column) and row.get(node_column)
    }


def read_observation_csv(csv_path: str | Path,
                         value_type: str = "flow",
                         datetime_format: str = "%m/%d/%Y %H:%M",
                         data_start_row: int = 4) -> ObservationTable:
    """Read the example observation CSV and expose common metadata."""
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))

    meter_names = rows[0]
    node_names = rows[1] if len(rows) > 1 else []
    types = rows[2] if len(rows) > 2 else []
    units = rows[3] if len(rows) > 3 else []

    value_cols = [
        i for i in range(1, len(types))
        if types[i].strip().lower() == value_type.lower()
    ]
    value_names = [meter_names[i] for i in value_cols]

    datetimes = []
    valid_row_indices = []
    for row_idx in range(data_start_row, len(rows)):
        try:
            dt = datetime.strptime(rows[row_idx][0].strip(), datetime_format)
        except (IndexError, ValueError):
            continue
        datetimes.append(dt)
        valid_row_indices.append(row_idx)

    return ObservationTable(
        rows=rows,
        meter_names=meter_names,
        node_names=node_names,
        types=types,
        units=units,
        value_cols=value_cols,
        value_names=value_names,
        datetimes=datetimes,
        valid_row_indices=valid_row_indices,
    )


def nearest_timeseries_value(timeseries: Mapping[datetime, float],
                             target_dt: datetime,
                             max_diff_seconds: float = 900.0) -> Optional[float]:
    """Return the nearest time-series value within ``max_diff_seconds``."""
    if not timeseries:
        return None

    best_dt = None
    best_diff = None
    for sample_dt in timeseries:
        diff = abs((sample_dt - target_dt).total_seconds())
        if best_diff is None or diff < best_diff:
            best_dt = sample_dt
            best_diff = diff

    if best_dt is None or best_diff is None or best_diff > max_diff_seconds:
        return None
    return float(timeseries[best_dt])