"""Trainable direct target evidence, with explicit observed-world support.

No target labels, object IDs, future poses or destination-dependent crops are
accepted by forward. Public anchor geometry is a legitimate input.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .world_likelihood_evidence import world_evidence_logits


def project_patch_evidence(uv, scores, valid, cells):
    zeros = scores.new_zeros((scores.shape[0], cells), dtype=torch.float32)
    neutral = zeros.log_softmax(-1)
    delta = world_evidence_logits(zeros, uv, scores, valid) - neutral
    coverage = world_evidence_logits(zeros, uv, torch.ones_like(scores), valid) - neutral > .5
    return delta * coverage, coverage


class DirectTargetHead(nn.Module):
    def __init__(self, hidden=128, text_dim=768, visual_dim=1024):
        super().__init__()
        self.word = nn.Linear(text_dim, 64)
        self.predicate = nn.Embedding(8, 32)
        self.relative = nn.Sequential(nn.Linear(42, 64), nn.GELU(), nn.LayerNorm(64))
        self.relation_score = nn.Sequential(
            nn.Linear(hidden+128, 128), nn.GELU(), nn.Linear(128, 1))
        self.target_text = nn.Linear(visual_dim, 64)
        # PGVB's shared 64-d projection and anchor cross-attention are reused.
        self.visual_score = nn.Sequential(
            nn.Linear(64*4+1, 128), nn.GELU(), nn.Linear(128, 1))
        self.visual_scale = nn.Parameter(torch.tensor(1.))

    def language(self, tokens, mask):
        if not mask.any(-1).all():
            raise ValueError('Every original instruction needs valid BERT tokens')
        words = self.word(tokens)
        return (words * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

    def relation_context(self, geometry, ids, valid):
        if not valid.any(-1).all():
            raise ValueError('At least one text-bound public anchor required')
        emb = self.predicate(ids)[:, :, None].expand(-1, -1, geometry.shape[2], -1)
        context = self.relative(torch.cat((geometry.float(), emb), -1))
        return (context*valid[:, :, None, None]).sum(1)/valid.sum(1)[:, None, None]

    def relation(self, dense, tokens, mask, geometry, ids, valid):
        context = self.relation_context(geometry, ids, valid)
        text = self.language(tokens, mask)[:, None].expand(-1, dense.shape[1], -1)
        return self.relation_score(torch.cat((dense, text, context), -1)).squeeze(-1)

    def visual(self, logits, observed, pgvb, target_text, language, relation_context):
        if observed is None:
            return logits, torch.zeros_like(logits, dtype=torch.bool)
        valid = observed['valid']
        image = pgvb.image(F.normalize(observed['image_features'].float(), dim=-1)).mean(2)
        anchor_text = pgvb.text(torch.cat((observed['name_text'], observed['type_text']), -1))
        shape = pgvb.polygon(observed['polygon_tokens'])
        query = image + anchor_text[:, None]
        anchor_context, _ = pgvb.attention(query, shape, shape, need_weights=False)
        landmark_scores = pgvb.score(torch.cat((image, anchor_context), -1)).squeeze(-1)
        # Both original full-language context and target-specific semantic
        # description condition local evidence; no coordinate copy is a target.
        target = self.target_text(target_text) + language
        n = math.isqrt(logits.shape[-1])
        coordinates = observed['local_uv'].float()*2-1
        relative = F.grid_sample(relation_context.transpose(1, 2).reshape(-1, 64, n, n),
                                 coordinates[:, None], align_corners=False,
                                 mode='bilinear', padding_mode='zeros')[:, :, 0].transpose(1, 2)
        score = self.visual_score(torch.cat((image, anchor_context,
            target[:, None].expand_as(image), relative, landmark_scores[..., None]), -1)).squeeze(-1)
        mean = (score*valid).sum(-1, keepdim=True)/valid.sum(-1, keepdim=True).clamp_min(1)
        score = 3*torch.tanh((score-mean)/3)*F.softplus(self.visual_scale)
        delta, coverage = project_patch_evidence(observed['local_uv'], score, valid, logits.shape[-1])
        return logits.float()+delta, coverage
