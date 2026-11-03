"""Opt-in reference-to-world relation likelihood on a frozen visual baseline.

Only supplied current observations and public map features are accepted. Null
reference mass contributes exactly neutral likelihood, not target absence.
Unknown-frame directions are learned categories, never fixed world rules.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .relation_factor_roles import RELATIONS


class WorldRelationFactor(nn.Module):
    grid_output = True
    BASE_INPUTS = ('prior_logits', 'local_uv', 'image_features', 'target_text',
                   'context_text', 'public_geometry', 'quality', 'local_valid')

    def __init__(self, base, dimension=1024, hidden=32):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.reference_image = nn.Linear(dimension, hidden)
        self.reference_text = nn.Linear(dimension, hidden)
        self.reference_score = nn.Linear(hidden, 1)
        self.map_geometry = nn.Linear(31, hidden)
        self.map_relation = nn.Embedding(len(RELATIONS), hidden)
        self.map_score = nn.Linear(hidden, 1)
        self.pair_geometry = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 8))
        self.relation_embedding = nn.Embedding(len(RELATIONS), 8)
        self.target_condition = nn.Linear(dimension, 8)
        self.reference_null = nn.Parameter(torch.tensor(2.))
        self.gain_logit = nn.Parameter(torch.tensor(-2.))
        nn.init.normal_(self.pair_geometry[-1].weight, std=.01)
        nn.init.zeros_(self.pair_geometry[-1].bias)

    def forward(self, *, prior_logits, local_uv, image_features, target_text,
                context_text, public_geometry, quality, local_valid,
                world_hypotheses, reference_text, relation_ids, reference_valid,
                reference_geometry, reference_map_types, reference_support,
                relation_mode='typed', control='real', return_details=False):
        if relation_mode not in ('typed', 'hidden') or control not in ('real', 'shuffle-references', 'no-map-score'):
            raise ValueError('unsupported preregistered relation control')
        public = dict(prior_logits=prior_logits, local_uv=local_uv, image_features=image_features,
                      target_text=target_text, context_text=context_text,
                      public_geometry=public_geometry, quality=quality, local_valid=local_valid)
        # Frozen baseline uses the real observation in all factor-only controls.
        with torch.no_grad():
            base = self.base(**public)
        batch, count, _, dim = image_features.shape
        refs = reference_text.shape[1]
        cells = prior_logits.shape[-1]
        if world_hypotheses.shape != (batch, cells+count, 2):
            raise ValueError('public fixed world support and local positions required')
        if reference_geometry.shape != (batch, refs, count, 31) or reference_support.dtype != torch.bool:
            raise ValueError('reference-aligned public geometry/support required')
        if reference_valid.shape != (batch, refs) or relation_ids.shape != (batch, refs):
            raise ValueError('reference-aligned types/validity required')
        image = F.normalize(image_features[:, :, 2].float(), dim=-1)
        if control == 'shuffle-references':
            image = image.clone()
            for b in range(batch):
                valid = local_valid[b].nonzero().flatten()
                image[b, valid] = image[b, valid.flip(0)]
        text = F.normalize(reference_text.float(), dim=-1)
        projected_image = F.gelu(self.reference_image(image))
        projected_text = F.gelu(self.reference_text(text))
        match = self.reference_score(projected_image[:, None]*projected_text[:, :, None]).squeeze(-1)
        match = match+10*torch.einsum('bnd,brd->brn', image, text)
        map_types = reference_map_types.float()
        ids = relation_ids
        if relation_mode == 'hidden':
            ids = torch.zeros_like(ids)
            map_types = torch.zeros_like(map_types)
            # Same capacity, no observable predicate identity.
            map_types[..., 0] = reference_map_types.sum(-1)
        geometry = reference_geometry.float()
        if control == 'no-map-score':
            geometry = torch.zeros_like(geometry)
            map_types = torch.zeros_like(map_types)
        map_condition = map_types@self.map_relation.weight
        match = match+self.map_score(F.gelu(self.map_geometry(geometry))*map_condition[:, :, None]).squeeze(-1)
        allowed = reference_support & local_valid[:, None] & reference_valid[:, :, None]
        count_valid = allowed.sum(-1, keepdim=True).clamp_min(1)
        match = (match-count_valid.float().log()).masked_fill(~allowed, float('-inf'))
        null = self.reference_null.expand(batch, refs, 1)
        ref_logits = torch.cat((match, null), dim=-1)
        ref_logprob = ref_logits.log_softmax(-1)

        target_world = world_hypotheses[:, :cells].float()
        observed_world = world_hypotheses[:, cells:].float()
        delta = target_world[:, :, None]-observed_world[:, None]
        distance = delta.norm(dim=-1, keepdim=True)
        direction = delta/distance.clamp_min(1e-4)
        pair = torch.cat(((delta/50).clamp(-8, 8), (distance/50).clamp_max(8),
                          direction, torch.log1p(distance)/5), -1)
        basis = self.pair_geometry(pair)
        condition = self.relation_embedding(ids)+self.target_condition(F.normalize(target_text.float(), dim=-1))[:, None]
        potential = 2*torch.tanh(torch.einsum('bqnk,brk->brqn', basis, condition)/8**.5)
        observed = potential+ref_logprob[:, :, None, :-1]
        # Null reference -> exp(0) likelihood. No observation is NOT negative
        # evidence that the target does not exist at an unobserved location.
        factor = torch.logsumexp(torch.cat((observed, ref_logprob[:, :, None, -1:].expand(-1, -1, cells, -1)), -1), -1)
        usable = reference_valid & allowed.any(-1)
        factor = factor.masked_fill(~usable[:, :, None], 0)
        evidence = factor.sum(1)/usable.sum(-1, keepdim=True).clamp_min(1)
        updated = base[:, :cells]+self.gain_logit.sigmoid()*evidence
        packed = torch.cat((updated, base[:, cells:]), -1)
        if return_details:
            return packed, dict(reference_mass=ref_logprob[:, :, :-1].exp().sum(-1),
                                active_references=usable, evidence=evidence,
                                effective_gain=self.gain_logit.sigmoid())
        return packed
