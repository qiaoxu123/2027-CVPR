"""Bundle Stage1 A3 geometry + PGVB + direction into ONE UnifiedPGVB-v2 model.

The only remaining independent trained module at runtime is the frozen
Trajectory/Stop controller. Existing source checkpoint files are read-only.
This is weight migration, NOT joint training and NOT an SR result.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from graphnav.checkpointing import atomic_save, file_sha256
from graphnav.full_experiment import new_model
from graphnav.unified_pgvb import UnifiedPGVB


def build_unified_checkpoint(stage1_path, pgvb_path, output, direction_path=None):
    stage1_path, pgvb_path, output = map(Path, (stage1_path, pgvb_path, output))
    if output.exists():
        raise FileExistsError(output)
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=True)
    if stage1.get("protocol", {}).get("name") != "graphnav-v1-stage1":
        raise ValueError("Require frozen original GraphNav Stage1 checkpoint")
    args = SimpleNamespace(**stage1["a3_args"])
    if int(args.grid_size) != 28 or int(args.hidden) != 128:
        raise ValueError("First integrated checkpoint requires 28x28/128-d controller")
    geometry = new_model(args, torch.device("cpu"))
    geo_weights = {
        name[3:]: value
        for name, value in stage1["model"].items()
        if name.startswith("a3.")
    }
    geometry.load_state_dict(geo_weights, strict=True)
    # These legacy A3 RGB layers are inactive in geometry mode and must NOT
    # appear as a duplicate head in the new unified model.
    del geometry.vision_proj
    del geometry.visual_score
    unified = UnifiedPGVB(geometry=geometry)
    visual = torch.load(pgvb_path, map_location="cpu", weights_only=True)
    if visual.get("config", {}).get("no_geometry_belief") is not True:
        raise ValueError("Require standalone trained local Anchor PGVB weights")
    unified.pgvb.load_state_dict(visual["model"], strict=True)
    fingerprints = {
        "stage1_sha256": file_sha256(stage1_path),
        "pgvb_sha256": file_sha256(pgvb_path),
    }
    relation_ready = False
    if direction_path is not None:
        direction_path = Path(direction_path)
        relation = torch.load(
            direction_path, map_location="cpu", weights_only=True,
        )
        if (
            relation.get("ignore_direction") is not False
            or relation.get("phase") != "real"
        ):
            raise ValueError("Expected trained word-conditioned real direction head")
        unified.direction.load_state_dict(relation["model"], strict=True)
        relation_ready = True
        fingerprints["direction_sha256"] = file_sha256(direction_path)

    unified.direction_ready = relation_ready
    unified.target_ready = False
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(
        format="unified-pgvb-v2",
        model=unified.state_dict(),
        config=dict(
            source=fingerprints,
            geometry_integrated=True,
            single_rgb_encoder=True,
            geometry_args=stage1["a3_args"],
            legacy_visual_residual_disabled=True,
            geometry_frozen_at_inference=True,
            joint_fusion_ready=False,
            direction_ready=relation_ready,
            target_ready=False,
            jointly_trained=False,
            controller_source="stage1_checkpoint:trajectory",
        ),
    ), output)
    return dict(
        path=str(output), sha256=file_sha256(output),
        trainable_if_unfrozen=sum(p.numel() for p in unified.parameters()),
        geometry_parameters=sum(p.numel() for p in unified.geometry.parameters()),
        pgvb_parameters=sum(p.numel() for p in unified.pgvb.parameters()),
        direction_ready=relation_ready, target_ready=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--pgvb", type=Path, required=True)
    parser.add_argument("--direction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_unified_checkpoint(
        args.stage1, args.pgvb, args.output, args.direction,
    ))


if __name__ == "__main__":
    main()
