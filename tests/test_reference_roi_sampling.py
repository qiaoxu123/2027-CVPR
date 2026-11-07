import numpy as np
import pytest
from scripts.audit_reference_roi_supervision import reference_indices, select_groups, reference_observation


def test_fixed_reference_samples_deduplicate_identical_poses():
    trajectory = np.array([[0,0,30,0],[0,0,30,0],[0,0,30,0],
                           [5,0,30,0],[10,0,30,0]], dtype=float)
    assert reference_indices(trajectory) == [('start',0), ('end',4)]
    assert reference_indices(trajectory[:1]) == [('start',0)]
    assert reference_indices([]) == []


def test_episode_selection_ignores_gt_and_uses_unique_physical_groups():
    labels = [dict(index=i, group=str(i//2), split='fit', target_world=[i,0])
              for i in range(12)]
    original = select_groups(labels, 'fit', 4)
    changed = [{**r, 'target_world':[100000,-100000]} for r in labels]
    assert original == select_groups(changed, 'fit', 4)
    assert len({labels[i]['group'] for i in original}) == 4


def test_degenerate_reference_frame_is_unknown_without_pose_repair():
    class Renderer:
        def current(self, map_name, pose, ground, centers):
            assert pose[2] == 38.4937311609
            raise ValueError('degenerate camera footprint')
    pose = np.array([737.95,1489.77,38.4937311609,.175])
    before = pose.copy()
    assert reference_observation(Renderer(), 'map', pose, 38.49224945510375,
                                 np.zeros((2,2)))[2] == 'degenerate_camera_footprint'
    np.testing.assert_array_equal(pose, before)


def test_unrelated_rendering_failure_is_not_silenced():
    class Renderer:
        def current(self, *args):
            raise ValueError('RGB/height grid mismatch')
    with pytest.raises(ValueError, match='grid mismatch'):
        reference_observation(Renderer(), 'map', np.array([0,0,30,0]),
                              0, np.zeros((2,2)))
