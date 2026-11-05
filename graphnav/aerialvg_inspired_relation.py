"""CityNav-native, AerialVG-inspired candidate/reference relation reasoning.

This is an independent lightweight reproduction of the IDEA of RAGM, not
copied AerialVG code or pretrained weights. Unlike AerialVG's all-object
pair matrix, a CityNav cell is paired only with *public* anchor hypotheses.

It consumes existing 28x28 Geometry context, BERT tokens, public polygon
geometry and parsed relation IDs. No target coordinates, GT identity,
observations from the future or extra image encoder enter forward.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CandidateAnchorRelation(nn.Module):
    """Learned text-conditioned target-cell <-> public-landmark pair scoring.

    Expected inputs:
        dense_context [B,Q,128] (Q=784 in current CityNav)
        tokens        [B,L,768] (existing frozen BERT embeddings)
        token_mask    [B,L] bool
        anchor_geom   [B,R,Q,10] (public Polygon geometry, *not* a hard mask)
        predicate_ids [B,R] int
        anchor_valid  [B,R] bool

    Returns bounded [B,Q] logit residual. Initialized to exactly zero so
    an untrained module is identical to the existing Geometry baseline.
    """

    def __init__(self, context_dim=128, text_dim=768, hidden=96,
                 heads=4, predicate_count=8, chunk_size=64,
                 max_logit_delta=2.0):
        super().__init__()
        if hidden % heads or chunk_size < 1 or max_logit_delta <= 0:
            raise ValueError("Invalid relation attention configuration")
        self.chunk_size = int(chunk_size)
        self.max_logit_delta = float(max_logit_delta)
        self.target = nn.Linear(context_dim, hidden)
        self.anchor = nn.Linear(context_dim, hidden)
        self.anchor_name = nn.Linear(text_dim, hidden)
        self.text = nn.Linear(text_dim, hidden)
        self.predicate = nn.Embedding(predicate_count, 24)
        self.pair = nn.Sequential(
            nn.Linear(hidden * 2 + 10 + 24, hidden),
            nn.GELU(), nn.LayerNorm(hidden),
        )
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.self_norm = nn.LayerNorm(hidden)
        self.anchor_weight = nn.Linear(hidden, 1)
        self.score = nn.Linear(hidden, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, dense_context, tokens, token_mask,
                anchor_geom, predicate_ids, anchor_valid,
                anchor_semantics=None):
        if (dense_context.ndim != 3 or tokens.ndim != 3
                or anchor_geom.ndim != 4
                or anchor_geom.shape[:2] != predicate_ids.shape
                or anchor_geom.shape[:2] != anchor_valid.shape
                or anchor_geom.shape[2] != dense_context.shape[1]
                or anchor_geom.shape[3] != 10
                or dense_context.shape[0] != tokens.shape[0]
                or token_mask.shape != tokens.shape[:2]):
            raise ValueError("Misaligned CityNav candidate/anchor/text tensors")
        if anchor_semantics is not None and (
                anchor_semantics.shape != (
                    dense_context.shape[0], anchor_valid.shape[1],
                    self.anchor_name.in_features)
                or not torch.isfinite(anchor_semantics).all()):
            raise ValueError("Public entity-name embeddings must align with map anchors")
        if not token_mask.any(-1).all():
            raise ValueError("at least one text token required")
        # Empty public references are legal CityNav examples. Their relation
        # residual stays exactly zero; they remain in training/eval denominator.
        no_anchor = ~anchor_valid.bool().any(-1)
        if anchor_valid.shape[1] == 0:
            return dense_context.new_zeros(dense_context.shape[:2], dtype=torch.float32)
        if no_anchor.any():
            anchor_valid = anchor_valid.clone()
            anchor_valid[no_anchor, 0] = True
        if (not torch.isfinite(dense_context).all()
                or not torch.isfinite(anchor_geom).all()
                or not torch.isfinite(tokens).all()):
            raise ValueError("Nonfinite candidate/reference inputs")

        # Small Belief network stays FP32 irrespective of the frozen
        # image/text encoder's mixed-precision caching.
        dense_context = dense_context.float()
        tokens = tokens.float()
        anchor_geom = anchor_geom.float()
        token_mask = token_mask.bool()
        anchor_valid = anchor_valid.bool()
        batch, cells, _ = dense_context.shape
        refs = anchor_valid.shape[1]
        length = tokens.shape[1]
        hidden = self.target.out_features

        # Approximate public anchor center by its nearest rasterized grid
        # location. This is *not* a true label and is never used as a target.
        centers = anchor_geom[..., :2].square().sum(-1).argmin(-1)
        anchor_context = dense_context.gather(
            1, centers[..., None].expand(-1, -1, dense_context.shape[-1])
        )
        anchor_features = self.anchor(anchor_context)
        if anchor_semantics is not None:
            # Explicit public entity identity; a building near two differently
            # named streets must be able to bind the correct language reference.
            anchor_features = anchor_features + self.anchor_name(anchor_semantics.float())
        target_features = self.target(dense_context)
        text_features = self.text(tokens)
        pred_features = self.predicate(predicate_ids)
        deltas = []
        for begin in range(0, cells, self.chunk_size):
            end = min(begin + self.chunk_size, cells)
            n = end - begin
            pair_features = torch.cat((
                target_features[:, begin:end, None].expand(-1, -1, refs, -1),
                anchor_features[:, None].expand(-1, n, -1, -1),
                anchor_geom[:, :, begin:end].transpose(1, 2),
                pred_features[:, None].expand(-1, n, -1, -1),
            ), dim=-1)
            pairs = self.pair(pair_features).reshape(batch * n, refs, hidden)
            valid = anchor_valid[:, None].expand(-1, n, -1).reshape(batch*n, refs)
            text = text_features[:, None].expand(-1, n, -1, -1).reshape(batch*n, length, hidden)
            text_valid = token_mask[:, None].expand(-1, n, -1).reshape(batch*n, length)

            cross, _ = self.cross(pairs, text, text,
                                  key_padding_mask=~text_valid,
                                  need_weights=False)
            pairs = self.cross_norm(pairs + cross)
            self_attention, _ = self.self_attn(
                pairs, pairs, pairs,
                key_padding_mask=~valid, need_weights=False,
            )
            pairs = self.self_norm(pairs + self_attention)
            weights = self.anchor_weight(pairs).squeeze(-1)
            weights = weights.masked_fill(~valid, -1e9).softmax(-1)
            scores = self.score(pairs).squeeze(-1)
            delta = (weights * scores).sum(-1).reshape(batch, n)
            deltas.append(delta)
        residual = self.max_logit_delta * torch.tanh(torch.cat(deltas, -1))
        return residual.masked_fill(no_anchor[:, None], 0.0)


def apply_relation_residual(geometry_logits, relation_delta):
    """Preserve the original Geometry ranking unless learned evidence changes it."""
    if geometry_logits.shape != relation_delta.shape:
        raise ValueError("Geometry and relation outputs must share the world grid")
    return geometry_logits.float() + relation_delta.float()
