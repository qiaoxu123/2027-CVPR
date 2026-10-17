"""Full-token, candidate-relative geometry, typed GAT, observed-only RGB.

All 784 cells are scored before top-K. The RGB branch can only adjust ranked
candidates that have actual observed pixels; never uses future/oracle images.
Forward takes NO target_xy, teacher_future, or ground-truth candidate IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F

from .model import DenseGoalField
from .semantic_graph import TypedGeometryGAT


@dataclass
class GoalField:
    belief_logits: torch.Tensor        # [B,Q]
    candidate_ids: torch.Tensor        # [B,K]
    candidate_xy: torch.Tensor         # [B,K,2]
    candidate_context: torch.Tensor    # [B,K,H]
    visual_coverage: torch.Tensor | None = None
    dense_context: torch.Tensor | None = None


class FullTokenGeometryLocalizer(nn.Module):
    VARIANTS = ("building_prior", "prior", "rules", "tokens", "geometry", "gat", "rgb")

    def __init__(
        self, hidden=128, text_dim=768, heads=4, max_relations=8,
        vision_dim=768, top_k=20, dropout=0.1, rule_weight=1.0,
        relative_position=False,
    ):
        super().__init__()
        self.hidden, self.top_k, self.rule_weight = hidden, top_k, float(rule_weight)
        self.relative_position = bool(relative_position)
        self.gnn = TypedGeometryGAT(hidden=hidden, text_dim=text_dim, heads=heads, dropout=dropout)
        self.pos = nn.Sequential(
            nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.lang = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden))
        self.token_attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True,
        )
        self.rule_map = nn.Sequential(
            nn.Linear(max_relations, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.token_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 2), nn.Linear(hidden * 2 + 2, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )
        self.geometry_proj = nn.Sequential(
            nn.Linear(10, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.pair_query = nn.Linear(hidden, hidden)
        self.pair_key = nn.Linear(hidden, hidden)
        self.pair_logits = nn.Linear(hidden, 1)
        self.null_anchor = nn.Parameter(torch.zeros(hidden))
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(hidden * 3 + 2), nn.Linear(hidden * 3 + 2, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )
        self.token_gain = nn.Parameter(torch.tensor(0.10))
        self.geometry_gain = nn.Parameter(torch.tensor(0.10))
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.visual_score = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 1),
            nn.Linear(hidden * 2 + 1, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )
        # RGB delta is initially zero, preserving an independently trained GAT.
        nn.init.zeros_(self.visual_score[-1].weight)
        nn.init.zeros_(self.visual_score[-1].bias)

    def _candidate_positional_features(self, batch):
        """Optional center-relative query coords prevent absolute-city shortcuts.

        Anchor centroid derives ONLY from supplied referenced map polygons,
        never the target position or a GT-selected reference.
        """
        xy = batch["grid_xy"]
        if not self.relative_position:
            return DenseGoalField.positional_features(xy)
        valid = batch["entity_valid"].to(xy.dtype)
        centers = batch["nodes"][..., :2].to(xy.dtype)
        center = (centers * valid.unsqueeze(-1)).sum(1)
        center = center / valid.sum(1,keepdim=True).clamp_min(1.)
        centered_xy = xy - center[:, None] + .5
        return DenseGoalField.positional_features(centered_xy)

    def forward(
        self, batch: dict, *, variant="gat",
        tokens: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        node_emb: torch.Tensor | None = None,
    ) -> GoalField:
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        prior = batch["prior"]
        q = prior.shape[1]
        baseline = prior + self.rule_weight * batch["rules"]
        if variant == "building_prior":
            logits = batch["prior_buildings"]
        elif variant == "prior":
            logits = prior
        elif variant == "rules":
            logits = baseline
        else:
            if tokens is None or token_mask is None:
                raise ValueError("full token states and attention mask are required")
            if tokens.ndim != 3 or token_mask.shape != tokens.shape[:2]:
                raise ValueError("token dimensions mismatch")
            if not token_mask.any(dim=1).all():
                raise ValueError("every instruction needs at least one valid token")
            xy = batch["grid_xy"]
            b, q, _ = xy.shape
            query = self.pos(self._candidate_positional_features(batch))
            query = query + self.rule_map(batch["relation_fields"])
            word_context, _ = self.token_attn(
                query, self.lang(tokens), self.lang(tokens),
                key_padding_mask=~token_mask.bool(), need_weights=False
            )
            features = query + word_context
            token_scores = self.token_head(torch.cat((
                query, word_context, prior.unsqueeze(-1),
                batch["rules"].unsqueeze(-1),
            ), dim=-1)).squeeze(-1)
            logits = baseline + torch.tanh(self.token_gain) * token_scores
            if variant in ("geometry", "gat", "rgb"):
                if node_emb is None or node_emb.shape[:2] != batch["nodes"].shape[:2]:
                    raise ValueError("node name encodings required for geometric relation")
                # The geometry control uses the SAME node projections and scoring
                # layers as the GAT arm, but has NO message passing between nodes.
                if variant == "geometry":
                    node_features = batch["nodes"]
                    if self.relative_position:
                        # For P1, also eliminate absolute map XY from landmark
                        # embeddings; otherwise the query is relative but the
                        # anchor node still leaks its fixed training coordinates.
                        valid_entities = batch["entity_valid"].to(node_features.dtype)
                        reference_center = (
                            node_features[..., :2] * valid_entities.unsqueeze(-1)
                        ).sum(1) / valid_entities.sum(1, keepdim=True).clamp_min(1.)
                        node_features = node_features.clone()
                        node_features[..., :2] -= reference_center[:, None]
                    nodes = (
                        self.gnn.node_geometry(node_features)
                        + self.gnn.name_projection(node_emb)
                        + self.gnn.type_embedding(batch["node_types"])
                    )
                    nodes = nodes * batch["node_valid"].unsqueeze(-1)
                else:
                    nodes = self.gnn(
                        batch["nodes"], node_emb, batch["node_types"],
                        batch["adjacency"], batch["edges"], batch["node_valid"],
                    )
                geo = self.geometry_proj(batch["pair_geometry"].to(features.dtype))
                pair_context = nodes[:, None] + geo
                relevance = self.pair_logits(torch.tanh(
                    self.pair_query(features)[:, :, None] +
                    self.pair_key(nodes)[:, None] + geo
                )).squeeze(-1)
                # Add a learned null anchor: empty map still produces finite logits.
                valid = batch["entity_valid"].bool()
                relevance = relevance.masked_fill(~valid[:, None], -1e4)
                null_score = relevance.new_zeros((b, q, 1))
                relevance = torch.softmax(
                    torch.cat((relevance, null_score), -1).float(), -1
                ).to(features.dtype)
                null_value = self.null_anchor.view(1, 1, 1, -1).expand(b, q, 1, -1)
                attended = (
                    torch.cat((pair_context, null_value), 2)
                    * relevance.unsqueeze(-1)
                ).sum(2)
                # Candidate scores receive BOTH full-token and direct
                # candidate-to-each-landmark geometry, before Top-K.
                geometry_scores = self.geometry_head(torch.cat((
                    features, attended, query,
                    prior.unsqueeze(-1), batch["rules"].unsqueeze(-1),
                ), -1)).squeeze(-1)
                logits = logits + torch.tanh(self.geometry_gain) * geometry_scores
                features = features + attended
        if variant in ("prior", "rules", "building_prior"):
            # Proposal coordinates are deterministic from the global score.
            context = self.pos(self._candidate_positional_features(batch))
        else:
            context = features
        k = min(self.top_k, q)
        ids = logits.topk(k, dim=1).indices
        return GoalField(
            belief_logits=logits,
            candidate_ids=ids,
            candidate_xy=batch["grid_xy"].gather(
                1, ids.unsqueeze(-1).expand(-1, -1, 2)
            ),
            candidate_context=context.gather(
                1, ids.unsqueeze(-1).expand(-1, -1, self.hidden)
            ),
            dense_context=context,
        )

    def refine_rgb(self, field: GoalField, image_features, coverage, *, threshold=0.30):
        """Only trusted observed proposal patches modify global goal logits.

        RGB cannot alter any unproposed cell or a candidate without enough
        observed AND central pixels. Image features must come from an actual
        prefix of UAV observations, never a global orthomosaic crop.
        """
        b, k, h = field.candidate_context.shape
        if image_features.shape[:2] != (b, k) or coverage.shape != (b, k):
            raise ValueError("RGB features must align with model-selected proposals")
        visual = self.vision_proj(image_features)
        valid = (coverage >= threshold).to(field.belief_logits.dtype)
        inputs = torch.cat((
            field.candidate_context, visual, coverage.unsqueeze(-1)
        ), dim=-1)
        delta = self.visual_score(inputs).squeeze(-1) * valid
        logits = field.belief_logits.scatter_add(1, field.candidate_ids, delta)
        return replace(field, belief_logits=logits, visual_coverage=coverage)
