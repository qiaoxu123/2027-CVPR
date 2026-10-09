import numpy as np
import pytest
import torch

from graphnav.geometry import (
    candidate_polygon_geometry, grid_centers, node_edge_geometry, normalize_xy,
    resample_contour,
)


def test_world_normalization_and_contour_perimeter():
    bounds = (100, 300, 500, 700)
    xy = normalize_xy([[100, 300], [300, 500], [500, 700]], bounds)
    assert np.allclose(xy, [[0, 0], [0.5, 0.5], [1, 1]])
    polygon = resample_contour([[0, 0], [1, 0], [1, 1], [0, 1]], 16)
    assert polygon.shape == (16, 2)
    assert np.allclose(polygon[0], [0, 0])


def test_signed_polygon_distance_and_absolute_directions():
    polygon = torch.tensor(
        [[[[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]]]
    )
    cand = torch.tensor([[[0.5, 0.5], [0.8, 0.5], [0.2, 0.5]]])
    f = candidate_polygon_geometry(cand, polygon, chunk_size=2)
    assert f.shape == (1, 3, 1, 9)
    assert f[0, 0, 0, 5] < 0  # inside polygon
    assert f[0, 1, 0, 5] > 0  # outside
    assert f[0, 1, 0, 0] > 0  # east of reference
    assert f[0, 2, 0, 0] < 0  # west of reference


def test_graph_edges_world_and_uav_heading():
    nodes = torch.tensor([[[0.3, 0.3], [0.5, 0.3]]])
    zero = node_edge_geometry(nodes, torch.tensor([0.0]))
    north = node_edge_geometry(nodes, torch.tensor([torch.pi / 2]))
    assert zero.shape == (1, 2, 2, 8)
    assert zero[0, 0, 1, 5].abs() < 1e-5
    assert north[0, 0, 1, 5] < -0.99


def test_grid_is_cartesian_xy():
    grid = grid_centers(2)
    assert torch.allclose(grid, torch.tensor(
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
    ))


def test_invalid_polygon_rejected():
    with pytest.raises(ValueError):
        resample_contour([[0, 0], [1, 1]])
