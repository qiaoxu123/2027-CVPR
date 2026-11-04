"""Public map neighborhood is for AFTER-ARRIVAL verification, never a policy cue."""
from shapely.geometry import Polygon

from graphnav.landmark_neighborhood import nearby_public_geometry


def test_named_landmark_neighborhood_has_metric_direction_and_topology():
    ref = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    objects = {
        "target": dict(name="The Food Warehouse", object_type="Building",
                       contour=[[0, 0], [10, 0], [10, 10], [0, 10]]),
        "east": dict(name="Parking Lot", object_type="Parking",
                     contour=[[15, 0], [25, 0], [25, 10], [15, 10]]),
        "west": dict(name="Road", object_type="TrafficRoad",
                     contour=[[-15, 0], [-5, 0], [-5, 10], [-15, 10]]),
        "inside": dict(name="Courtyard", object_type="Ground",
                       contour=[[2, 2], [8, 2], [8, 8], [2, 8]]),
        "far": dict(name="Far Town", object_type="Building",
                    contour=[[200, 200], [220, 200], [220, 220], [200, 220]]),
    }
    result = nearby_public_geometry(objects, ref, ["target"], radius_m=50)
    assert [r["public_id"] for r in result] == ["inside", "west", "east"]
    neighbors = {r["public_id"]: r for r in result}
    assert neighbors["east"]["relative_centroid_dx_m"] > 0
    assert neighbors["west"]["relative_centroid_dx_m"] < 0
    assert neighbors["east"]["polygon_boundary_distance_m"] == 5.
    assert neighbors["inside"]["contains_or_within"] is True
    assert neighbors["inside"]["intersects_public_polygon"] is True
    assert "far" not in neighbors
    assert "target" not in neighbors


def test_invalid_and_distant_public_objects_are_rejected():
    ref = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    candidates = {
        "bad": dict(name="Invalid", contour=[[1, 1], [1, 1], [1, 1]]),
        "nan": dict(name="NaN", contour=[[float("nan"), 0], [2, 0], [3, 3]]),
        "far": dict(name="Far", contour=[[90, 90], [100, 90], [100, 100], [90, 100]]),
    }
    assert nearby_public_geometry(candidates, ref, [], radius_m=20) == []
