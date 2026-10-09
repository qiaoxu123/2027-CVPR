"""Online GraphNav runner for an external UAV simulator.

The simulator supplies ONLY the current top-down RGB observation, UAV pose,
world-coordinate footprint, and static named landmark polygons.
Future RGB and target ground truth must NEVER enter this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .config import GraphNavConfig
from .foundations import CandidateVisionCache
from .geometry import normalize_xy, resample_contour
from .memory import ObservedRGBMemory
from .model import GraphNav, bounded_next_step


@dataclass(frozen=True)
class NavigationDecision:
    target_world: tuple[float, float]
    next_world: tuple[float, float]
    stop_probability: float
    visual_candidates_used: int
    observed_area_fraction: float


class GraphNavRuntime:
    def __init__(
        self, network: GraphNav, text_encoder, image_encoder,
        map_bounds, instruction: str, reference_landmarks,
        *, device="cuda", max_step_m=50.0,
        stop_threshold=0.95, crop_extent_m=100.0,
    ):
        self.network = network.to(device).eval()
        self.text = text_encoder
        self.vision = image_encoder  # may be None for graph-only ablations
        self.bounds = np.asarray(map_bounds, dtype=np.float32)
        if self.bounds.shape != (4,) or np.any(self.bounds[2:] <= self.bounds[:2]):
            raise ValueError("invalid public map bounds")
        self.device = torch.device(device)
        self.c = network.config
        self.max_step_m = max_step_m
        self.stop_threshold = stop_threshold
        self.crop_extent_m = crop_extent_m
        self.memory = ObservedRGBMemory(map_bounds)
        self.cache = CandidateVisionCache()
        names = [""] * self.c.max_landmarks
        polygon = np.zeros(
            (self.c.max_landmarks, self.c.contour_points, 2), dtype=np.float32
        )
        valid = np.zeros((self.c.max_landmarks,), dtype=bool)
        for j, item in enumerate(reference_landmarks[:self.c.max_landmarks]):
            # item must ALREADY be a name-polygon pair resolved against map data.
            names[j] = item["name"]
            contour_xy = normalize_xy(item["contour"], self.bounds)
            polygon[j] = resample_contour(contour_xy, self.c.contour_points)
            valid[j] = True
        self.landmark_names = self.text.encode(names).unsqueeze(0).to(self.device)
        self.instruction = self.text.encode([instruction]).to(self.device)
        self.polygons = torch.from_numpy(polygon).unsqueeze(0).to(self.device)
        self.valid = torch.from_numpy(valid).unsqueeze(0).to(self.device)

    def _world(self, normalized_xy):
        return self.bounds[:2] + np.asarray(normalized_xy) * (
            self.bounds[2:] - self.bounds[:2]
        )

    @torch.inference_mode()
    def step(self, rgb, footprint_world, world_xy, yaw):
        """Perform one control decision; no GT goal or future observation input."""
        self.memory.observe(rgb, footprint_world)
        uav_xy = normalize_xy(world_xy, self.bounds)
        uav_pose = torch.tensor(
            [[uav_xy[0], uav_xy[1], np.sin(yaw), np.cos(yaw)]],
            dtype=torch.float32, device=self.device,
        )
        observed = torch.from_numpy(
            self.memory.masks(self.c.grid_size)
        ).unsqueeze(0).to(self.device)
        out = self.network(
            instruction=self.instruction,
            landmark_names=self.landmark_names,
            landmark_contours=self.polygons,
            landmark_valid=self.valid,
            uav_pose=uav_pose,
            spatial_maps=observed,
            physical_map_sizes=torch.as_tensor(
                self.bounds[2:] - self.bounds[:2],
                device=self.device, dtype=torch.float32
            ).unsqueeze(0),
        )

        used = 0
        if self.vision is not None:
            # Evidence index is EXACTLY the order of the model top-K proposals.
            patches, ids, coverages, vectors = [], [], [], [
                None
            ] * out.candidate_xy.shape[1]
            for j, candidate in enumerate(out.candidate_xy[0].cpu().numpy()):
                world = self._world(candidate)
                crop = self.memory.crop(world, extent_m=self.crop_extent_m)
                coverages.append(crop.coverage)
                if (
                    crop.coverage < self.c.min_visual_coverage
                    or crop.center_coverage < self.c.min_visual_coverage
                ):
                    continue
                cached = self.cache.get(crop.image, crop.valid)
                if cached is not None:
                    vectors[j] = cached.to(self.device)
                else:
                    patches.append(crop.image)
                    ids.append((j, crop))
                used += 1
            if patches:
                embeddings = self.vision.encode(patches).to(self.device)
                for (j, crop), vec in zip(ids, embeddings):
                    vectors[j] = vec
                    self.cache.put(crop.image, crop.valid, vec)
            visual = torch.stack([
                torch.zeros(self.c.vision_dim, device=self.device)
                if vec is None else vec.to(self.device)
                for vec in vectors
            ]).unsqueeze(0)
            cover = torch.tensor(
                [coverage if vectors[i] is not None else 0.0
                 for i, coverage in enumerate(coverages)],
                device=self.device, dtype=torch.float32,
            ).unsqueeze(0)
            out = self.network.apply_visual_evidence(out, visual, cover)

        # The model selects goal and mode jointly. Last-mile stop stays
        # conservatively disabled unless explicitly calibrated by user.
        winner = (
            out.candidate_logits[:, :, None] + out.path_logits
        ).reshape(1, -1).argmax(-1)
        candidate_id = torch.div(
            winner, self.c.trajectory_modes, rounding_mode="floor"
        )
        stop_probability = torch.sigmoid(
            out.stop_logits[0, candidate_id.item()]
        ).item()
        target_world = self._world(out.selected_goal_xy[0].cpu().numpy())
        next_world = self._world(out.next_waypoint_xy[0].cpu().numpy())
        stepped = bounded_next_step(
            torch.as_tensor(world_xy, device=self.device, dtype=torch.float32)[None],
            torch.as_tensor(next_world, device=self.device, dtype=torch.float32)[None],
            self.max_step_m,
        )[0].cpu().numpy()
        return NavigationDecision(
            target_world=tuple(float(i) for i in target_world),
            next_world=tuple(float(i) for i in stepped),
            stop_probability=float(stop_probability),
            visual_candidates_used=used,
            observed_area_fraction=self.memory.explored_fraction,
        )
