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
from .full_localization import GoalField
from .world_likelihood_evidence import world_evidence_logits
from .stage1 import preserved_proposal_order


class UnifiedPGVB(nn.Module):
    def __init__(self, dimension=1024, hidden=64, heads=4, geometry=None, direct_target=False, full_relation=False):
        super().__init__()
        self.geometry = geometry
        self._geometry_cache = None
        self.pgvb = PolygonGuidedVisualBelief(dimension, hidden, heads)
        self.direction = DirectionBelief()
        self.target_text = nn.Linear(dimension, hidden)
        self.target_score = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.target_gain = nn.Parameter(torch.zeros(()))
        # Joint geometric context + observed RGB evidence. The last layers
        # are zero-initialized so an untrained v2 exactly reproduces the
        # verified bounded PGVB rollout; enabling is explicit after training.
        geometry_hidden = getattr(geometry, "hidden", 128)
        self.joint_logits = nn.Sequential(
            nn.LayerNorm(geometry_hidden + 2),
            nn.Linear(geometry_hidden + 2, geometry_hidden),
            nn.GELU(),
            nn.Linear(geometry_hidden, 1),
        )
        self.joint_context = nn.Sequential(
            nn.LayerNorm(geometry_hidden + 2),
            nn.Linear(geometry_hidden + 2, geometry_hidden),
            nn.GELU(),
            nn.Linear(geometry_hidden, geometry_hidden),
        )
        for head in (self.joint_logits, self.joint_context):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.joint_ready = False
        self.direction_ready = False
        self.target_ready = False
        if direct_target:
            from .direct_target import DirectTargetHead
            self.direct_target = DirectTargetHead(hidden=geometry_hidden, visual_dim=dimension)
        self.direct_target_ready = False
        if full_relation:
            from .aerialvg_inspired_relation import CandidateAnchorRelation
            self.full_relation = CandidateAnchorRelation(
                context_dim=geometry_hidden, text_dim=768,
            )
        self.full_relation_ready = False

    def forward_full_relation(self, *, static_inputs, tokens, token_mask, node_emb,
                              observed=None, acquisition=None, anchor_labels=None,
                              relation_enabled=True):
        """GT-free path for ALL CityNav instructions; parsed one-hop links optional.

        Every instruction already has full BERT tokens and every public entity
        already has candidate-to-contour geometry. The null-anchor rows retain
        Geometry. Conditional PGVB still consumes only the current observed RGB.
        """
        if not hasattr(self, "full_relation"):
            raise RuntimeError("Full relation model was not installed")
        if not self.training and not self.full_relation_ready:
            raise RuntimeError("Untrained full relation model cannot run")
        evidence = self.forward_field(
            static_inputs=static_inputs, tokens=tokens, token_mask=token_mask,
            node_emb=node_emb, observed=observed, acquisition=acquisition,
            anchor_labels=anchor_labels,
        )
        base = evidence["geometry_field"]
        visual = evidence["field"]
        pair_geometry = static_inputs["pair_geometry"].float()
        # [B,Q,N,10] -> [B,N,Q,10]; N includes public named surface
        # candidates and optional mapped roads, never target-labelled anchors.
        references = pair_geometry.transpose(1, 2).contiguous()
        valid = static_inputs["entity_valid"].bool()
        # The whole original BERT instruction supplies relation semantics.
        # Predicate IDs are an optional cue, not an eligibility filter.
        predicates = torch.zeros_like(valid, dtype=torch.long)
        delta = (self.full_relation(
            base.dense_context, tokens, token_mask,
            references, predicates, valid,
            anchor_semantics=node_emb,
        ) if relation_enabled else torch.zeros_like(base.belief_logits, dtype=torch.float32))
        logits = visual.belief_logits.float() + delta.float()
        relation_logits = base.belief_logits.float() + delta.float()
        def make_field(scores, support=None):
            top = scores.topk(base.candidate_ids.shape[1], -1).indices
            context = visual.dense_context
            return GoalField(
                belief_logits=scores, candidate_ids=top,
                candidate_xy=static_inputs["grid_xy"].gather(
                    1, top[..., None].expand(-1, -1, 2)),
                candidate_context=context.gather(
                    1, top[..., None].expand(-1, -1, context.shape[-1])),
                dense_context=context, visual_coverage=support,
            )
        return dict(geometry_field=base,
                    relation_field=make_field(relation_logits),
                    field=make_field(logits, visual.visual_coverage),
                    relation_delta=delta, evidence=evidence.get("evidence"),
                    audit=evidence["audit"])

    def forward_direct_target(self, *, static_inputs, tokens, token_mask, node_emb,
                              relation_geometry, relation_ids, relation_valid,
                              observed=None, target_text=None):
        """Explicit trained target path. The named-landmark path is unchanged."""
        if not hasattr(self, 'direct_target'):
            raise RuntimeError('No direct-target head installed')
        if not self.training and not self.direct_target_ready:
            raise RuntimeError('Untrained direct-target head cannot run inference')
        base = self._geometry_field(static_inputs, tokens, token_mask, node_emb)
        head = self.direct_target
        context = head.relation_context(relation_geometry, relation_ids, relation_valid)
        delta = head.relation(base.dense_context, tokens, token_mask,
                              relation_geometry, relation_ids, relation_valid)
        related = base.belief_logits.float()+delta.float()
        full, coverage = head.visual(related, observed, self.pgvb, target_text,
                                     head.language(tokens, token_mask), context)
        def field(logits, support=None):
            # Unlike the historical preserved proposal list, this target path
            # must admit newly corrected target cells into the candidate set.
            ids = logits.topk(base.candidate_ids.shape[-1], dim=-1).indices
            return GoalField(belief_logits=logits, candidate_ids=ids,
                candidate_xy=static_inputs['grid_xy'].gather(1, ids[..., None].expand(-1,-1,2)),
                candidate_context=base.dense_context.gather(1, ids[...,None].expand(-1,-1,base.dense_context.shape[-1])),
                dense_context=base.dense_context, visual_coverage=support)
        return dict(geometry_field=base, relation_field=field(related),
                    field=field(full, coverage), coverage=coverage)

    def reset_episode(self):
        """Never reuse cached map/word geometry across different episodes."""
        self._geometry_cache = None

    def _geometry_field(self, static_inputs, tokens, token_mask, node_emb):
        if self.geometry is None:
            raise RuntimeError("UnifiedPGVB requires its internal geometry module")
        cacheable = (
            not self.training
            and not any(p.requires_grad for p in self.geometry.parameters())
        )
        if cacheable and self._geometry_cache is not None:
            return self._geometry_cache
        base = self.geometry(
            static_inputs, variant="geometry", tokens=tokens,
            token_mask=token_mask, node_emb=node_emb,
        )
        if cacheable:
            self._geometry_cache = base
        return base

    def forward_field(
        self, *, static_inputs, tokens, token_mask, node_emb,
        observed=None, bound=0.5, acquisition=None, anchor_labels=None,
    ):
        """ONE integrated geometry + PGVB field generator.

        Static A3 geometry is an internal submodule; the PGVB score and its
        rasterization are applied in this very module, not in a second policy.
        Returns the exact GoalField contract expected by frozen Trajectory/Stop.
        For frozen geometry, a per-episode cached field saves repeated BERT/GAT
        work; trainable geometry always runs without a detached cache.
        """
        if not 0 < bound <= 1:
            raise ValueError("Bounded visual evidence is required")
        base = self._geometry_field(
            static_inputs, tokens, token_mask, node_emb,
        )
        empty_audit = dict(
            active=False, covered_cells=0, max_abs_delta=0.,
            changed_top1=False,
        )
        if observed is None:
            if acquisition is not None:
                empty_audit["acquisition"] = acquisition.observe()
            return dict(
                field=base, geometry_field=base,
                landmark_scores=None, evidence=None, audit=empty_audit,
            )
        required = {
            "local_uv", "image_features", "name_text", "type_text",
            "polygon_tokens", "valid",
        }
        if set(observed) != required:
            raise ValueError("Observed visual input keys do not match UnifiedPGVB")
        uv, valid = observed["local_uv"], observed["valid"]
        scores = self.pgvb(
            image_features=observed["image_features"],
            name_text=observed["name_text"],
            type_text=observed["type_text"],
            polygon_tokens=observed["polygon_tokens"],
            valid=valid,
        ).float()
        if scores.shape != valid.shape or uv.shape != (*scores.shape, 2):
            raise ValueError("Observed world projection does not match patch scores")
        evidence = (uv, scores, valid)
        gain = 1.0  # legacy compatibility; no tracker means historical fusion
        if acquisition is not None:
            if self.training:
                raise RuntimeError("Stateful landmark tracking is inference-only")
            if anchor_labels is None:
                raise ValueError("Acquisition needs public-ROI weak patch labels")
            # Contrast with the SAME RGB frame's patches displaced in space.
            # Only another tiny PGVB head forward, NOT another SigLIP2 encode.
            with torch.no_grad():
                shifted = observed["image_features"].roll(
                    shifts=max(1, scores.shape[1] // 2), dims=1,
                )
                shuffled = self.pgvb(
                    image_features=shifted,
                    name_text=observed["name_text"],
                    type_text=observed["type_text"],
                    polygon_tokens=observed["polygon_tokens"],
                    valid=valid,
                ).float()
                tracked = acquisition.observe(
                    scores.detach(), shuffled.detach(),
                    anchor_labels.detach(), valid.detach(),
                )
            gain = float(tracked["visual_gain"])
            empty_audit["acquisition"] = tracked
        if (not valid.any()
                or scores[valid].float().std(unbiased=False) < 1e-6
                or gain <= 0):
            return dict(
                field=base, geometry_field=base,
                landmark_scores=scores, evidence=evidence, audit=empty_audit,
            )
        if base.dense_context is None:
            raise ValueError("Unified geometry must emit dense candidate context")
        zeros = torch.zeros_like(base.belief_logits, dtype=torch.float32)
        centered = scores.clone()
        for i in range(len(scores)):
            if valid[i].any():
                centered[i] -= centered[i, valid[i]].mean()
        neutral = zeros.log_softmax(-1)
        visual = world_evidence_logits(
            zeros, uv, centered, valid,
        ) - neutral
        coverage = world_evidence_logits(
            zeros, uv, torch.ones_like(centered), valid,
        ) - neutral
        covered = coverage > 0.5
        delta = (bound * gain) * torch.tanh(visual / 5.) * covered
        if not covered.any() or delta.abs().max() <= 1e-8:
            return dict(
                field=base, geometry_field=base,
                landmark_scores=scores, evidence=evidence, audit=empty_audit,
            )
        logits = base.belief_logits.float() + delta
        dense = base.dense_context
        if self.joint_ready:
            # This genuinely conditions the updated world Belief AND the
            # controller candidate context on geometry and observed vision.
            # Unseen cells remain unchanged, never assigned fake negatives.
            if dense.shape[-1] + 2 != self.joint_logits[0].normalized_shape[0]:
                raise ValueError("Joint geometry dimension mismatch")
            joint = torch.cat((
                dense.float(), visual.unsqueeze(-1),
                covered.float().unsqueeze(-1),
            ), dim=-1)
            logits = logits + (
                bound * torch.tanh(self.joint_logits(joint).squeeze(-1))
                * covered.float()
            )
            dense = dense.float() + (
                self.joint_context(joint) * covered[..., None]
            )
        ids = preserved_proposal_order(logits, base.candidate_ids)
        xy = static_inputs["grid_xy"].gather(
            1, ids[..., None].expand(-1, -1, 2),
        )
        context = dense.gather(
            1, ids[..., None].expand(-1, -1, dense.shape[-1]),
        )
        field = GoalField(
            belief_logits=logits,
            candidate_ids=ids,
            candidate_xy=xy,
            candidate_context=context,
            dense_context=dense,
            visual_coverage=covered.float(),
        )
        return dict(
            field=field, geometry_field=base,
            landmark_scores=scores, evidence=evidence,
            audit=dict(
                active=True, covered_cells=int(covered.sum()),
                max_abs_delta=float(delta.abs().max().detach()),
                changed_top1=bool(
                    (ids[:, 0] != base.candidate_ids[:, 0]).any()
                ),
                **({"acquisition": empty_audit["acquisition"]}
                   if "acquisition" in empty_audit else {}),
            ),
        )

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
        if "static_inputs" in encoded_inputs:
            return self.forward_field(**encoded_inputs)
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
