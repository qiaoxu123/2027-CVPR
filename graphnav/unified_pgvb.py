"""One PGVB-centric network for all visual evidence and future target reasoning.

SigLIP2 is a single frozen vision/text backbone owned by the runtime.
This module reuses one visual projection. The relation module is not
an additional RGB encoder; the target appearance head is disabled until trained.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .direction_belief import DirectionBelief
from .polygon_guided_visual import PolygonGuidedVisualBelief


class UnifiedPGVB(nn.Module):
    def __init__(self, dimension=1024, hidden=64, heads=4):
        super().__init__()
        self.pgvb = PolygonGuidedVisualBelief(dimension, hidden, heads)
        self.direction = DirectionBelief()
        self.target_text = nn.Linear(dimension, hidden)
        self.target_score = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.target_gain = nn.Parameter(torch.zeros(()))
        self.direction_ready = False
        self.target_ready = False

    def forward_structured(self, instruction, **encoded_inputs):
        """Take the validated language adapter output without a second network.

        Complex targets are explicitly not enabled until the target head and
        language/spatial reference frame are independently validated.
        """
        if (instruction.get("intent") != "navigate_to_landmark"
                or instruction.get("approved_for_inference") is not True
                or instruction.get("target_GT_used") is not False
                or instruction.get("relations")):
            raise ValueError("Only validated named-landmark tasks are supported")
        return self(**encoded_inputs)

    def forward(
        self, *, image_features, name_text, type_text, polygon_tokens, valid,
        anchor_world=None, direction_id=None, span=None, reference_heading=None,
        target_text=None,
    ):
        """Return local landmark logits, optional relation prior and target scores.

        No GT target/future observations are accepted. A direction prior alone
        does not certify true geographic north. Target appearance remains off
        until it receives independent training and validation.
        """
        landmark = self.pgvb(
            image_features=image_features,
            name_text=name_text, type_text=type_text,
            polygon_tokens=polygon_tokens, valid=valid,
        )
        result = dict(
            landmark_scores=landmark,
            direction_probability=None,
            target_visual_scores=None,
        )
        relation_inputs = (anchor_world, direction_id, span)
        if any(x is not None for x in relation_inputs):
            if not all(x is not None for x in relation_inputs):
                raise ValueError("Public Anchor, direction word and map span required")
            if not self.direction_ready:
                raise RuntimeError("Direction module is not calibrated in this checkpoint")
            result["direction_probability"] = self.direction(
                anchor_world, direction_id, span,
                reference_heading=reference_heading,
            )
        if target_text is not None:
            if not self.target_ready:
                raise RuntimeError("Target appearance head is not trained")
            if target_text.shape != name_text.shape:
                raise ValueError("Target text must share the SigLIP2 text dimension")
            pooled = self.pgvb.image(
                F.normalize(image_features.float(), dim=-1)
            ).mean(2)
            query = self.target_text(F.normalize(target_text.float(), dim=-1))
            raw = self.target_score(torch.cat(
                (pooled, query[:, None].expand_as(pooled)), dim=-1
            )).squeeze(-1)
            result["target_visual_scores"] = (
                self.target_gain.tanh() * raw
            ).masked_fill(~valid, 0)
        return result
