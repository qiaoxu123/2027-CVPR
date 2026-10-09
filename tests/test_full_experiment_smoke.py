"""End-to-end CPU training smoke with fake pretrained encoders (no downloads)."""

from argparse import Namespace
import json

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_experiment import train_one, evaluate
from graphnav.foundations import CandidateVisionCache


class FakeFrozenBERT:
    def encode_tokens(self, texts):
        b = len(texts)
        return torch.zeros(b, 5, 768), torch.ones(b, 5, dtype=torch.bool)

    def encode(self, texts):
        return torch.zeros(len(texts), 768)


class FakeFrozenVision:
    def encode(self, images):
        assert all(img.dtype == np.uint8 and img.shape == (224, 224, 3) for img in images)
        return torch.ones(len(images), 6)


def test_gat_checkpoint_to_rgb_visual_only_finetuning(tmp_path):
    torch.manual_seed(1)
    torch.set_num_threads(2)
    image = np.zeros((224, 224, 3), np.uint8)
    image[..., 1] = 255
    path = tmp_path / "view.png"
    cv2.imwrite(str(path), image)
    record = {
        "map_name": "mini_map", "episode_id": "mini_map:1:0:0",
        "dataset_version": "revised", "match_mode": "fuzzy",
        "instruction": "east of Church",
        "world_bounds": [0, 0, 400, 400],
        "physical_map_sizes": [400, 400],
        "target_xy": [.7, .5],
        "landmarks": [{
            "id": "1", "name": "Church", "object_type": "Building",
            "contour": [[.1,.1],[.2,.1],[.2,.2],[.1,.2]],
        }],
        "rgb_observations": [{
            "path": str(path),
            "footprint_world": [[0, 400], [400, 400], [400, 0], [0, 0]],
            "step": 0,
        }],
        "observation_cutoff_step": 0,
    }
    filename = tmp_path / "samples.jsonl"
    filename.write_text(json.dumps(record) + "\n", encoding="utf-8")
    ds = FullCandidateDataset(
        filename, grid_size=4, max_landmarks=2,
        max_roads=0, max_relations=2, visual_required=True,
    )
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_full)
    loaders = {"train": dl, "val_seen": dl, "val_unseen": dl}
    args = Namespace(
        hidden=16, heads=4, max_relations=2,
        vision_dim=6, top_k=4, rule_weight=1.0, grid_size=4,
        min_visual_coverage=.1, crop_extent_m=40.,
        lr=1e-3, rgb_lr=1e-3, epochs=1, rgb_epochs=1,
        max_train_batches=1,
        dataset_version="revised", match_mode="fuzzy",
    )
    cache = CandidateVisionCache()
    model, gat_logs = train_one(
        "gat", loaders, FakeFrozenBERT(), None, cache,
        args, torch.device("cpu"), tmp_path
    )
    assert (tmp_path / "gat_best.pt").exists()
    assert len(gat_logs) == 1
    rgb, rgb_logs = train_one(
        "rgb", loaders, FakeFrozenBERT(), FakeFrozenVision(),
        cache, args, torch.device("cpu"), tmp_path,
    )
    assert rgb_logs[0]["valid_visual_candidates_train"] > 0
    metrics = evaluate(
        rgb, dl, FakeFrozenBERT(), FakeFrozenVision(), cache,
        args, "rgb", torch.device("cpu"),
    )
    assert metrics["samples"] == 1
    assert metrics["visual_proposals_used"] > 0
    assert (tmp_path / "rgb_best.pt").exists()
