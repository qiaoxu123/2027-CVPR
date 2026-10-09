import cv2
import numpy as np
import torch

from graphnav.foundations import CandidateVisionCache
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
