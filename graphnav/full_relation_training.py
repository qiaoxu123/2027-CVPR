"""Checkpoint-safe full-instruction relation model loader and TRAIN batches."""
from pathlib import Path
from types import SimpleNamespace

import torch

from .full_experiment import new_model
from .unified_pgvb import UnifiedPGVB


FORMAT = "unified-pgvb-full-relation-v1"


def load_full_relation_model(bundle, device="cuda"):
    if bundle.get("format") not in ("unified-pgvb-v2", FORMAT):
        raise ValueError("Unsupported source UnifiedPGVB format")
    args = SimpleNamespace(**bundle["config"]["geometry_args"])
    if int(args.grid_size) != 28 or int(args.hidden) != 128:
        raise ValueError("Requires the verified CityNav 28x28, 128-d controller")
    device = torch.device(device)
    geometry = new_model(args, device)
    del geometry.vision_proj
    del geometry.visual_score
    model = UnifiedPGVB(geometry=geometry, full_relation=True).to(device)
    load = model.load_state_dict(bundle["model"], strict=False)
    if load.unexpected_keys:
        raise ValueError("Unexpected model weights: " + str(load.unexpected_keys))
    if bundle["format"] == "unified-pgvb-v2":
        if not load.missing_keys or not all(
                name.startswith("full_relation.") for name in load.missing_keys):
            raise ValueError("Original v2 must initialize only the new relation weights")
    elif load.missing_keys:
        raise ValueError("Trained relation checkpoint is incomplete: " + str(load.missing_keys))
    model.full_relation_ready = bundle["format"] == FORMAT and bundle.get("completed_epochs", 0) >= 1
    model.direction_ready = False  # Geographic CRS not independently validated.
    model.target_ready = False
    model.joint_ready = False
    return model


def full_relation_inputs(batch, bert, device):
    """Forward whitelist excludes target_xy, episode IDs, teacher poses and GT."""
    from .full_experiment import _batch_inputs
    static, tokens, mask, names = _batch_inputs(batch, bert, device, "geometry")
    return dict(static_inputs=static, tokens=tokens.float(),
                token_mask=mask, node_emb=names.float())
