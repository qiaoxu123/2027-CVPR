"""A full-Polygon RGB probe may NEVER masquerade as a causal UAV image."""
import numpy as np
import pytest

from graphnav.privileged_landmark_rgb import (
    whole_landmark_view, privileged_whole_landmark_rgb,
)


def _rect(xmin, ymin, xmax, ymax):
    return [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]


def test_complete_view_includes_disconnected_public_polygon_parts():
    polygons = [_rect(20, 50, 40, 70), _rect(90, 60, 100, 80)]
    output = whole_landmark_view(polygons, margin_fraction=.2)
    assert output["polygon_fraction"] >= .999
    assert output["side_m"] > 80
    assert output["nominal_patch_m"] == pytest.approx(output["side_m"]/24)


def test_no_goal_or_future_pose_allowed_in_crop_api():
    with pytest.raises(TypeError):
        whole_landmark_view([_rect(20,20,40,40)], target_xy=[30,30])


def test_invalid_view_parameters_rejected():
    with pytest.raises(ValueError):
        whole_landmark_view([_rect(20,20,40,40)], margin_fraction=-.2)
    with pytest.raises(ValueError):
        whole_landmark_view([_rect(20,20,40,40)], min_extent_m=0)


class _Raster:
    count = 1
    def index(self, x, y):
        return (int(y), int(x))
    def read(self, band):
        assert band == 1
        return np.ones((192,192),np.float32)
    def read_masks(self, band):
        assert band == 1
        return np.full((192,192),255,np.uint8)


class _Renderer:
    def __init__(self):
        self._rgb=np.zeros((192,192,3),dtype=np.uint8)
        self._rgb[...,1]=150
        self._raster=_Raster()
        self._height_valid=None
        self.open_count=0
    def _open(self,name):
        assert name=="test_scene"
        self.open_count+=1


def test_privileged_crop_is_native384_and_has_provenance():
    render=_Renderer()
    polygon=[_rect(40,45,90,75)]
    image,diagnostics=privileged_whole_landmark_rgb(
        render,"test_scene",polygon
    )
    assert image["bgr"].shape==(384,384,3)
    assert image["depth_valid"].shape==(384,384)
    assert image["depth_m"].shape==(384,384)
    assert np.isnan(image["depth_m"]).all()
    assert image["depth_valid"].mean() > .99
    assert np.count_nonzero(image["bgr"][...,1]) > 100000
    assert diagnostics["polygon_fraction"] >= .999
    assert image["pseudo_view"]
    assert image["provenance"]=="PRIVILEGED_SOURCE_ORTHOPHOTO_NOT_UAV_OBSERVATION"
    # Repeated same map reuses source mask, not an expensive full raster read.
    before=render._height_valid
    privileged_whole_landmark_rgb(render,"test_scene",polygon)
    assert render._height_valid is before
