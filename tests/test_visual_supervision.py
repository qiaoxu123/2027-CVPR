import cv2
import numpy as np
import torch

from graphnav.foundations import CandidateVisionCache
from graphnav.memory import ObservedRGBMemory
from graphnav.visual_supervision import visual_candidate_batch


class DummyEncoder:
    def encode(self, images):
        return torch.stack([
            torch.tensor([float(img[..., 0].mean() / 255), 1.0]) for img in images
        ])


class FakeOutput:
    candidate_xy = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]], dtype=torch.float32)


def test_visual_supervision_never_crops_unseen_future_area(tmp_path):
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 240
    path = tmp_path / "observed.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # In world coordinates, first target is visible, second is not.
    corners = [[40, 120], [120, 120], [120, 40], [40, 40]]
    visual, coverage = visual_candidate_batch(
        FakeOutput(),
        [[{"path": str(path), "footprint_world": corners}]],
        [[0, 0, 400, 400]],
        DummyEncoder(),
        vision_dim=2, min_coverage=0.1, crop_extent_m=40,
        memory_size=256, cache=CandidateVisionCache(),
    )
    assert coverage[0, 0] > 0.1
    assert coverage[0, 1] == 0
    assert visual[0, 0, 0] > 0
    assert visual[0, 1].abs().sum() == 0


def test_empty_center_screen_preserves_exact_coverage_on_rotated_view(tmp_path):
    bounds = [0, 0, 80, 400]
    image = np.full((64, 64, 3), 240, np.uint8)
    path = tmp_path / "rotated.png"
    cv2.imwrite(str(path), image)
    corners = ObservedRGBMemory.view_corners([40, 200], .6, 35)
    memory = ObservedRGBMemory(bounds, size=128)
    memory.observe(image, corners)
    points = np.random.default_rng(7).uniform(0, 1, (100, 2)).astype(np.float32)
    output = type("Output", (), {"candidate_xy": torch.from_numpy(points[None])})()
    _, coverage = visual_candidate_batch(output,
        [[{"path": str(path), "footprint_world": corners.tolist()}]], [bounds],
        DummyEncoder(), vision_dim=2, min_coverage=.1, crop_extent_m=40, memory_size=128)
    expected = []
    for point in points:
        crop = memory.crop(np.asarray(bounds[:2]) + point * (np.asarray(bounds[2:]) - bounds[:2]), 40)
        value = min(crop.coverage, crop.center_coverage)
        expected.append(value if value >= .1 else 0)
    np.testing.assert_allclose(coverage.numpy()[0], expected, atol=1e-7)
