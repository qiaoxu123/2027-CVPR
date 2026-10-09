"""Per-candidate RELATIVE geometry to each whole source landmark contour.

Features are computed for EVERY 28x28 candidate before top-K, not only for
graph neighbors. 2D CityRefer polygons are region nodes, NOT polyline roads.
Coordinates are normalized for network stability but distances are computed
from known physical CityNav public map bounds. No goal/GT is used.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
import torch

from .semantic_geometry import geometry_distance, point_inside_polygon
from .static_localization import StaticLocalizationDataset


def relative_entity_features(world_xy, entities, capacity, map_size_scale=400.):
    """[Q,N,10] positive EAST/NORTH, bearing, signed boundary, extent, line bit."""
    xy = np.asarray(world_xy, dtype=np.float32).reshape(-1, 2)
    output = np.zeros((len(xy), capacity, 10), np.float32)
    if len(entities) > capacity:
        raise ValueError("more map entities than packed graph nodes")
    if map_size_scale <= 0:
        raise ValueError("map_size_scale must be positive")
    for i, e in enumerate(entities):
        poly = np.asarray(e["geometry_m"], np.float32)
        center = poly.mean(axis=0)
        delta = xy - center
        dist = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(dist[:, None], 1e-5)
        line = e["kind"] == "road"
        # geometry_distance deliberately returns zero inside a surface. That
        # is appropriate for a region prior but loses signed boundary depth.
        closed_line = poly if line else np.vstack((poly, poly[0]))
        boundary = geometry_distance(xy, closed_line, is_road=True)
        inside = np.zeros(len(xy), np.float32) if line else point_inside_polygon(xy, poly).astype(np.float32)
        boundary = np.where(inside > 0, -boundary, boundary)
        width_height = poly.max(axis=0) - poly.min(axis=0)
        output[:, i, :] = np.column_stack((
            delta[:, 0] / map_size_scale,
            delta[:, 1] / map_size_scale,
            dist / map_size_scale,
            unit[:, 0], unit[:, 1],
            boundary / map_size_scale,
            inside,
            np.broadcast_to(width_height[0] / map_size_scale, len(xy)),
            np.broadcast_to(width_height[1] / map_size_scale, len(xy)),
            np.full(len(xy), float(line), np.float32),
        ))
    return output


class FullCandidateDataset(StaticLocalizationDataset):
    """Adds graph-aligned pairwise geometry and optional ACTUAL observation prefixes."""

    def __init__(self, *args, visual_required=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_required = visual_required
        self._pair_cache = lru_cache(maxsize=None)(self._pair_features)

    def _pair_features(self, idx):
        record = self.records[idx]
        graph = self._prepared(idx)[0]
        bounds = np.asarray(record["world_bounds"], np.float32)
        span = bounds[2:] - bounds[:2]
        world_xy = bounds[:2] + self.grid_xy * span
        # Save half precision on CPU: 784xN geometry/query could otherwise
        # retain several GB across 5-epoch CityNav runs. Cast to fp32 in batch.
        return relative_entity_features(
            world_xy, graph["entities"], self.capacity,
        ).astype(np.float16)

    def __getitem__(self, idx):
        output = super().__getitem__(idx)
        record = self.records[idx]
        source = record.get("rgb_observations", [])
        if self.visual_required:
            unavailable = record.get("observation_unavailable_reason") == "nonpositive_camera_altitude"
            if not source and not unavailable:
                raise ValueError(
                    "RGB variant requested but record has no observed prefix. "
                    "Prepare episode observations first; never use a full map."
                )
            cutoff = record.get("observation_cutoff_step")
            if not isinstance(cutoff, int) or cutoff < 0:
                raise ValueError("RGB records require explicit nonnegative observation_cutoff_step")
            steps = [observation.get("step") for observation in source]
            if any(not isinstance(step, int) for step in steps):
                raise ValueError("RGB observations require explicit per-frame causal step")
            if steps != sorted(steps) or any(step < 0 or step > cutoff for step in steps):
                raise ValueError("RGB observation contains a future/out-of-order frame")
            if source and source[-1]["step"] != cutoff:
                raise ValueError("RGB prefix does not include its declared cutoff frame")
        output["pair_geometry"] = torch.from_numpy(self._pair_cache(idx))
        count = len(self._prepared(idx)[0]["entities"])
        entity_mask = torch.zeros_like(output["node_valid"])
        entity_mask[:count] = True
        output["entity_valid"] = entity_mask
        output["rgb_observations"] = source
        output["world_bounds"] = record["world_bounds"]
        output["dataset_version"] = record.get("dataset_version", "unspecified")
        return output


def collate_full(items):
    return {
        key: (
            [item[key] for item in items]
            if key in ("text", "node_names", "episode_id",
                       "rgb_observations", "world_bounds", "dataset_version")
            else torch.stack([item[key] for item in items])
            if isinstance(items[0][key], torch.Tensor)
            else torch.tensor([item[key] for item in items])
        ) for key in items[0]
    }
