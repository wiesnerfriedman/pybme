"""Tests for reusable SWMM utility helpers."""

from datetime import datetime, timedelta

import numpy as np

from pybme import (
    build_edge_array,
    nearest_timeseries_value,
    parse_swmm_inp,
    read_meter_node_map,
    read_observation_csv,
)


def test_parse_swmm_inp_extracts_nodes_edges_and_coords(tmp_path):
    inp = tmp_path / "mini.inp"
    inp.write_text(
        """
[JUNCTIONS]
J1 100 0 0 0 0
J2 101 0 0 0 0

[STORAGE]
S1 95 0 0 0 0

[OUTFALLS]
O1 90 FREE

[CONDUITS]
C1 J1 J2 120 0 0 0 0

[PUMPS]
P1 J2 O1

[COORDINATES]
J1 0 0
J2 10 0
O1 20 0
""".strip(),
        encoding="utf-8",
    )

    network = parse_swmm_inp(inp)

    assert network.all_node_names == ["J1", "J2", "O1", "S1"]
    assert network.edges == [("J1", "J2", 120.0), ("J2", "O1", 50.0)]
    assert network.coords["J2"] == (10.0, 0.0)

    edge_array = build_edge_array(network.all_node_names, network.edges)
    np.testing.assert_array_equal(edge_array, np.array([[0, 1], [1, 2]]))


def test_read_meter_and_observation_csv_helpers(tmp_path):
    meter_csv = tmp_path / "MeterLocations.csv"
    meter_csv.write_text(
        "Meter,Node\nM1,N1\nM2,N2\n",
        encoding="utf-8",
    )

    obs_csv = tmp_path / "ObservedData.csv"
    obs_csv.write_text(
        "Date,M1,M2\n"
        "Node,Link1,Link2\n"
        "Type,flow,depth\n"
        "Unit,MGD,ft\n"
        "03/10/2025 00:00,1.0,2.0\n"
        "bad-date,9.0,9.0\n"
        "03/10/2025 01:00,1.5,2.5\n",
        encoding="utf-8",
    )

    meter_map = read_meter_node_map(meter_csv)
    obs = read_observation_csv(obs_csv, value_type="flow")

    assert meter_map == {"M1": "N1", "M2": "N2"}
    assert obs.value_cols == [1]
    assert obs.value_names == ["M1"]
    assert obs.valid_row_indices == [4, 6]
    assert obs.datetimes[0] == datetime(2025, 3, 10, 0, 0)


def test_nearest_timeseries_value_respects_tolerance():
    start = datetime(2025, 3, 10, 0, 0)
    series = {
        start: 1.0,
        start + timedelta(minutes=15): 2.0,
    }

    assert nearest_timeseries_value(series, start + timedelta(minutes=10), 600.0) == 2.0
    assert nearest_timeseries_value(series, start + timedelta(hours=2), 600.0) is None