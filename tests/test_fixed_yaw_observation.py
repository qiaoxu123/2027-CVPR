"""Camera sampling changes must not mutate agent pose or desynchronize footprint."""
import numpy as np
import pytest

from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.landmark_projection import world_to_frame, frame_to_world
from graphnav.memory import ObservedRGBMemory


class RecordingRenderer:
    def __init__(self):
        self.received = []

    def current_rgbd(self, map_name, pose, ground_level):
        self.received.append(np.array(pose, copy=True))
        return dict(bgr=np.zeros((384, 384, 3), np.uint8),
                    footprint_world=ObservedRGBMemory.view_corners(pose[:2], pose[3], pose[2]-ground_level),
                    pose=np.array(pose, copy=True))


def test_camera_fixed_agent_untouched_and_world_roundtrip():
    renderer = RecordingRenderer()
    camera = FixedYawObservation(renderer, yaw_rad=0.)
    pose = np.array([124., 87., 70., 1.8])
    saved = pose.copy()
    result = camera.current_rgbd('public_map', pose, 10.)
    np.testing.assert_array_equal(pose, saved)
    np.testing.assert_array_equal(result['agent_pose'], saved)
    np.testing.assert_array_equal(result['pose'], [124., 87., 70., 0.])
    np.testing.assert_array_equal(renderer.received[0], result['pose'])
    assert result['camera_yaw_rad'] == 0.
    assert result['agent_yaw_rad'] == saved[3]
    pixels = np.array([[0., 0.], [383., 383.], [100., 250.]])
    world = frame_to_world(pixels, result['footprint_world'], result['bgr'].shape)
    np.testing.assert_allclose(world_to_frame(world, result['footprint_world'], result['bgr'].shape), pixels, atol=1e-3)


def test_identical_xyz_gives_identical_footprint_regardless_of_agent_yaw():
    camera = FixedYawObservation(RecordingRenderer(), yaw_rad=.3)
    a = camera.current_rgbd('map', [12., 19., 45., 0.], 5.)
    b = camera.current_rgbd('map', [12., 19., 45., 2.6], 5.)
    np.testing.assert_array_equal(a['footprint_world'], b['footprint_world'])
    assert a['agent_yaw_rad'] != b['agent_yaw_rad']


@pytest.mark.parametrize('yaw', [float('nan'), float('inf')])
def test_nonfinite_camera_yaw_rejected(yaw):
    with pytest.raises(ValueError):
        FixedYawObservation(RecordingRenderer(), yaw_rad=yaw)


@pytest.mark.parametrize('pose', [[1,2,3], [1,2,3,float('nan')]])
def test_invalid_agent_pose_rejected(pose):
    with pytest.raises(ValueError):
        FixedYawObservation(RecordingRenderer()).current_rgbd('map', pose, 0.)


def test_low_altitude_roi_support_is_geometry_not_visibility():
    from graphnav.observed_patch_scales import observed_region_masks
    observation = dict(bgr=np.full((384,384,3),128,np.uint8),
                       depth_valid=np.ones((384,384),bool),
                       footprint_world=ObservedRGBMemory.view_corners([0,0],0.,10.))
    center = np.array([[0.,0.]],np.float32)
    legacy, _ = observed_region_masks(observation, center, extents=(32.,64.,128.))
    local, _ = observed_region_masks(observation, center, extents=(8.,16.,32.))
    assert not legacy.any()
    assert local[0,0].any() and local[0,1].any()
    assert not local[0,2].any()
