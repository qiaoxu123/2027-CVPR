"""Observe-only query-conditioned GoalField refinement for dynamic closed-loop.

Opt-in adapter: existing checkpoints, rollout modes and A3 predictions are left
unchanged. This path requires an independently trained target-evidence checkpoint;
an untrained adapter must not be represented as a navigation improvement.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .dynamic_runtime import DynamicWaypointPolicy
from .evidence_grounding import (
    SpatialEvidenceMemory, TargetEvidenceRefiner, appearance_query,
)
from .landmark_projection import frame_to_world


def native_patch_world_xy(observation, patches: int, bounds, *, device):
    """Project pixel patch centers from the *current* image to public map XY.

    The patch validity criterion uses current RGB+depth source support only.
    It is not target segmentation or visibility ground truth.
    """
    bgr = np.asarray(observation["bgr"])
    depth_valid = np.asarray(observation["depth_valid"], dtype=bool)
    h, w = bgr.shape[:2]
    if bgr.ndim != 3 or bgr.shape[2] != 3 or depth_valid.shape != (h, w):
        raise ValueError("invalid observed RGB/Depth frame")
    side = int(round(patches ** .5))
    if side * side != patches:
        raise ValueError("expected square SigLIP patch sequence")
    if h % side or w % side:
        raise ValueError("patch grid must divide native RGB dimensions")
    yy, xx = np.meshgrid(
        (np.arange(side, dtype=np.float32) + .5) * h / side - .5,
        (np.arange(side, dtype=np.float32) + .5) * w / side - .5,
        indexing="ij",
    )
    pixels = np.stack((xx, yy), -1).reshape(-1, 2)
    world = frame_to_world(pixels, observation["footprint_world"], bgr.shape)
    bounds = np.asarray(bounds, dtype=np.float32)
    span = bounds[2:] - bounds[:2]
    if np.any(span <= 0) or not np.isfinite(world).all():
        raise ValueError("invalid public map bounds or camera projection")
    xy = (world - bounds[:2]) / span
    # A partially observed/invalid patch is unknown, rather than evidence.
    support = ((bgr.max(-1) > 0) & depth_valid).astype(np.float32)
    valid = cv2.resize(support, (side, side), interpolation=cv2.INTER_AREA).reshape(-1) > .9
    valid &= (xy >= 0).all(-1) & (xy <= 1).all(-1)
    return (
        torch.as_tensor(xy.copy(), dtype=torch.float32, device=device)[None],
        torch.as_tensor(valid.copy(), dtype=torch.bool, device=device)[None],
    )


class TargetEvidenceWaypointPolicy(DynamicWaypointPolicy):
    """A3 -> existing optional graph -> trainable relation/appearance adapter.

    Constructed exactly like DynamicWaypointPolicy with extra checkpoint and
    frozen vision model. No additional private/evaluator state is accessed.
    """

    def __init__(self, *args, evidence_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        from .landmark_visual_pretrain import FrozenSigLIP2Patches
        state = torch.load(evidence_checkpoint, map_location="cpu", weights_only=True)
        if state.get("protocol") != "graphnav-target-evidence-v1":
            raise ValueError("target evidence checkpoint protocol mismatch")
        from .checkpointing import file_sha256
        if state.get("a3_sha256") != file_sha256(args[0]):
            raise ValueError("target evidence and A3 checkpoints differ")
        config = state["config"]
        self.target = TargetEvidenceRefiner(**config).to(self.device).eval()
        self.target.load_state_dict(state["model"], strict=True)
        self.target.requires_grad_(False)
        self.target_vision = FrozenSigLIP2Patches(
            state["vision_backbone"], device=str(self.device),
            microbatch=1, cache_limit=4,
        )
        if self.target_vision.hidden_size != config["vision_dim"]:
            raise ValueError("target evidence vision backbone dimension mismatch")
        self.target_text = self.encoder
        self.target_history = None
        self.target_text_cache = None

    def set_episode(self, episode, transform, catalog):
        super().set_episode(episode, transform, catalog)
        self.target_history = None
        self.target_text_cache = None

    @torch.no_grad()
    def execute_field(self, state, field, inputs, evidence_diagnostics):
        if self.target_history is None:
            self.target_history = SpatialEvidenceMemory(
                1, field.belief_logits.shape[1], self.device
            )
        if self.target_text_cache is None:
            instruction = appearance_query(self.record["instruction"])
            self.target_text_cache = self.target_text.encode_tokens([instruction])
        tokens, mask = self.target_text_cache
        observation = state.current_observation
        # A real currently rendered crop, never an unvisited orthomosaic.
        with TemporaryDirectory(prefix="graphnav-target-") as directory:
            path = Path(directory) / "current.png"
            if not cv2.imwrite(str(path), observation["bgr"]):
                raise IOError("could not prepare observed RGB")
            patch_features, patch_mask = self.target_vision.encode([str(path)])
        patch_xy, pixel_valid = native_patch_world_xy(
            observation, patch_features.shape[1],
            self.record["world_bounds"], device=self.device,
        )
        patch_valid = patch_mask.bool() & pixel_valid
        public_pose = torch.as_tensor(
            [state.pose[0], state.pose[1], state.pose[3]],
            dtype=torch.float32, device=self.device,
        )[None]
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            refined = self.target(
                field,
                grid_xy=inputs["grid_xy"].float(),
                relation_fields=inputs["relation_fields"].float(),
                text_tokens=tokens.float(),
                text_mask=mask.bool(),
                patch_features=patch_features.float(),
                patch_xy=patch_xy,
                patch_valid=patch_valid,
                memory=self.target_history,
                step=state.step,
                pose_xyyaw=public_pose,
            )
        diagnostics = dict(
            evidence_diagnostics,
            target_gate_visual=float(self.target.visual_gate.tanh()),
            target_gate_relation=float(self.target.relation_gate.tanh()),
            target_observed_patch_count=int(patch_valid.sum().item()),
            target_observed_cell_count=int((self.target_history.weights > 0).sum()),
        )
        return super().execute_field(state, refined, inputs, diagnostics)
