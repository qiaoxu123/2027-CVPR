"""CityNav-derived independent JSONL dataset with deterministic landmark binding.

Only model-facing observation fields enter collate(). Ground-truth target and
human future trajectory are returned SEPARATELY as training labels.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import GraphNavConfig
from .geometry import normalize_xy, resample_contour


def normalized_name(s: str) -> str:
    return " ".join(str(s).casefold().replace("-", " ").split())


def bind_reference_landmarks(reference_names, objects: dict, max_landmarks: int):
    """Exact normalized textual match; return ALL ambiguous matches, never GT-nearest.

    If a name corresponds to two polygons we preserve both distinct IDs. No
    target information or human trajectory is consulted to break ties.
    """
    matches = []
    seen = set()
    searchable = sorted(objects.items(), key=lambda pair: str(pair[0]))
    for wanted in reference_names:
        norm = normalized_name(wanted)
        for object_id, entry in searchable:
            name = str(entry.get("name", ""))
            if not name or normalized_name(name) != norm:
                continue
            if len(entry.get("contour", [])) < 3:
                continue
            if object_id not in seen:
                matches.append({"id": str(object_id), "name": name, "contour": entry["contour"]})
                seen.add(object_id)
            if len(matches) >= max_landmarks:
                return matches
    return matches


def resample_teacher_path(path_xy, future_start, steps):
    """Time-parameterized HUMAN path, not a target-generated straight line."""
    a = np.asarray(path_xy, dtype=np.float32)
    if a.ndim != 2 or a.shape[-1] != 2 or not len(a):
        raise ValueError("no human teacher trajectory")
    tail = a[future_start:]
    if len(tail) == 0:
        tail = a[-1:]
    # Use uniform time interpolation; this is intentionally not a shortest path.
    indices = np.linspace(0, len(tail) - 1, steps)
    return np.stack([
        np.interp(indices, np.arange(len(tail)), tail[:, axis]) for axis in range(2)
    ], axis=-1).astype(np.float32)


class CityNavJSONL(Dataset):
    def __init__(self, path, config: GraphNavConfig):
        self.records = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        self.config = config
        if not self.records:
            raise ValueError("no records in JSONL")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        c = self.config
        names = [""] * c.max_landmarks
        contours = np.zeros((c.max_landmarks, c.contour_points, 2), np.float32)
        validity = np.zeros(c.max_landmarks, bool)
        for i, landmark in enumerate(record["landmarks"][:c.max_landmarks]):
            names[i] = landmark["name"]
            contours[i] = resample_contour(landmark["contour"], c.contour_points)
            validity[i] = True
        maps = np.asarray(record.get(
            "spatial_maps", np.zeros((2, c.grid_size, c.grid_size)).tolist()
        ), dtype=np.float32)
        if maps.shape != (2, c.grid_size, c.grid_size):
            raise ValueError("spatial map grid size incompatible with config")
        future = np.asarray(record["teacher_future"], dtype=np.float32)
        if future.shape != (c.trajectory_waypoints, 2):
            raise ValueError("human teacher waypoint length mismatch")
        return {
            "instruction_text": record["instruction"],
            "landmark_text": names,
            "landmark_contours": torch.from_numpy(contours),
            "landmark_valid": torch.from_numpy(validity),
            "uav_pose": torch.tensor(record["uav_pose"], dtype=torch.float32),
            "spatial_maps": torch.from_numpy(maps),
            "target_xy": torch.tensor(record["target_xy"], dtype=torch.float32),
            "teacher_future": torch.from_numpy(future),
            "physical_map_sizes": torch.tensor(record["physical_map_sizes"], dtype=torch.float32),
            "episode_id": record.get("episode_id", ""),
            "rgb_observations": record.get("rgb_observations", []),
            "world_bounds": record.get("world_bounds", None),
        }


def collate_episodes(items):
    """Keep text raw until the frozen BERT encoder runs once per minibatch."""
    return {
        key: (
            [item[key] for item in items] if key in ("instruction_text", "landmark_text", "episode_id",
                                    "rgb_observations", "world_bounds")
            else torch.stack([item[key] for item in items], dim=0)
        )
        for key in items[0]
    }


def encode_batch_text(collated, frozen_bert, device):
    """Text embeddings are static and cacheable; BERT remains frozen."""
    b = len(collated["instruction_text"])
    m = len(collated["landmark_text"][0])
    instruction = frozen_bert.encode(collated["instruction_text"]).to(device)
    flat_names = [n for group in collated["landmark_text"] for n in group]
    landmark_vectors = frozen_bert.encode(flat_names).reshape(b, m, -1).to(device)
    inputs = {
        "instruction": instruction,
        "landmark_names": landmark_vectors,
        "landmark_contours": collated["landmark_contours"].to(device),
        "landmark_valid": collated["landmark_valid"].to(device),
        "uav_pose": collated["uav_pose"].to(device),
        "spatial_maps": collated["spatial_maps"].to(device),
    }
    labels = {
        key: collated[key].to(device)
        for key in ("target_xy", "teacher_future", "physical_map_sizes")
    }
    return inputs, labels
