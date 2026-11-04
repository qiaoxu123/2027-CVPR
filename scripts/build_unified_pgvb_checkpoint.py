"""Bundle existing PGVB/direction weights into one UnifiedPGVB checkpoint.

No training is performed. The target head is untrained and disabled; original
model files are never overwritten.
"""
import argparse
from pathlib import Path

import torch

from graphnav.checkpointing import atomic_save, file_sha256
from graphnav.unified_pgvb import UnifiedPGVB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgvb", type=Path, required=True)
    parser.add_argument("--direction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = torch.load(args.pgvb, map_location="cpu", weights_only=True)
    if source.get("config", {}).get("no_geometry_belief") is not True:
        raise ValueError("Expected frozen local PGVB checkpoint")
    network = UnifiedPGVB()
    network.pgvb.load_state_dict(source["model"], strict=True)
    fingerprints = {"pgvb_sha256": file_sha256(args.pgvb)}
    relation_ready = False
    if args.direction is not None:
        relation = torch.load(
            args.direction, map_location="cpu", weights_only=True,
        )
        if (relation.get("ignore_direction") is not False
                or relation.get("phase") != "real"):
            raise ValueError("Expected learned word-conditioned real direction head")
        network.direction.load_state_dict(relation["model"], strict=True)
        relation_ready = True
        fingerprints["direction_sha256"] = file_sha256(args.direction)

    network.direction_ready = relation_ready
    network.target_ready = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(
        format="unified-pgvb-v1",
        model=network.state_dict(),
        config=dict(
            single_rgb_encoder=True,
            legacy_visual_residual_disabled=True,
            direction_ready=relation_ready,
            target_ready=False,
            jointly_trained=False,
            source=fingerprints,
        ),
    ), args.output)
    print(dict(path=str(args.output), sha256=file_sha256(args.output),
               direction_ready=relation_ready, target_ready=False))


if __name__ == "__main__":
    main()
