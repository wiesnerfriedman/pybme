import numpy as np

from pybme.network_plots import _build_flowline_segment_data


def test_build_flowline_segment_data_interpolates_edge_values():
    node_names = ["A", "B"]
    coords = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
    edges = [("A", "B", 10.0)]
    values = np.array([2.0, 6.0])

    segments, segment_values, grey_segments = _build_flowline_segment_data(
        node_names,
        coords,
        edges,
        values,
        n_edge_subsegments=4,
    )

    assert len(segments) == 4
    assert len(grey_segments) == 0
    np.testing.assert_allclose(segment_values, [2.5, 3.5, 4.5, 5.5])
    np.testing.assert_allclose(segments[0][0], [0.0, 0.0])
    np.testing.assert_allclose(segments[-1][1], [10.0, 0.0])


def test_build_flowline_segment_data_uses_grey_edge_for_missing_values():
    node_names = ["A", "B"]
    coords = {"A": (0.0, 0.0), "B": (1.0, 1.0)}
    edges = [("A", "B", 1.0)]
    values = np.array([np.nan, 3.0])

    segments, segment_values, grey_segments = _build_flowline_segment_data(
        node_names,
        coords,
        edges,
        values,
        n_edge_subsegments=3,
    )

    assert len(segments) == 0
    assert len(segment_values) == 0
    assert len(grey_segments) == 1