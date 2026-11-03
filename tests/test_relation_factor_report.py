import numpy as np
from scripts.report_world_relation_factor import world_grid_extent


def test_heatmap_pixel_centers_align_with_world_grid_centers():
    world = np.array([[[1., 2.], [3., 2.]], [[1., 6.], [3., 6.]]])
    assert world_grid_extent(world) == [0., 4., 0., 8.]
