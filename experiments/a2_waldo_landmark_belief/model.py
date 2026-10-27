"""Trainable A2 additions only; frozen image/text features enter as tensors."""
import torch
from torch import nn
from torch.nn import functional as F
from .core import InstructionLandmarkEncoder, BoxLandmarkRelationScorer


class LandmarkBeliefNetwork(nn.Module):
    def __init__(self, text_dim=768, vision_dim=1024, hidden=128, heads=4):
        super().__init__()
        self.config = dict(text_dim=text_dim, vision_dim=vision_dim, hidden=hidden, heads=heads)
        self.language = InstructionLandmarkEncoder(text_dim, hidden, heads)
        # Predict the destination's relation from instruction/reference context,
        # without giving this head the candidate geometry it must explain.
        self.relation = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.scorer = BoxLandmarkRelationScorer(vision_dim, hidden, pair_dim=16)
        self.offset = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.match = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, instruction_tokens, token_valid, mention_mask, landmark_name_tokens,
                name_valid, link_valid, box_rgb, box_valid, pair_geometry, pair_valid,
                offset_limit, control='real'):
        if box_rgb.shape[1] < 1 or mention_mask.shape[1] < 1:
            raise ValueError('Pad empty box/reference axes to one with false validity')
        context, refs = self.language(instruction_tokens, token_valid, mention_mask,
                                      landmark_name_tokens, name_valid, link_valid)
        if control == 'no_text':
            context = torch.zeros_like(context)
        if control == 'ambiguous':
            refs = torch.zeros_like(refs)
        relation = self.relation(context)
        angle = F.normalize(relation[...,:2], dim=-1, eps=1e-6)
        distance = F.softplus(relation[...,2])
        geometry = pair_geometry.float()
        if control == 'no_geometry':
            geometry = torch.zeros_like(geometry)
        if control == 'permuted_links':
            context = context.roll(1, dims=1)
            angle, distance = angle.roll(1, dims=1), distance.roll(1, dims=1)
            refs = refs.roll(1, dims=1)
        # Inputs 0:12 are world geometry; 12:14 are public projection quality.
        # Last channels compare observed box relation against language prediction.
        direction = torch.stack((geometry[...,4], geometry[...,3]), -1)
        alignment = (direction * angle[:,None]).sum(-1)
        mismatch = (geometry[...,5] - distance[:,None]).abs()
        pair = torch.cat((geometry, alignment[...,None], mismatch[...,None]), -1)
        scores, pair_scores, reliable = self.scorer(box_rgb, box_valid, context, refs, pair, pair_valid)
        box = self.scorer.box(box_rgb.float())
        ref_weights = refs.float() / refs.sum(-1,keepdim=True).clamp_min(1)
        sentence = (context * ref_weights[...,None]).sum(1)
        joint = torch.cat((box, sentence[:,None].expand(-1,box.shape[1],-1)), -1)
        offset = self.offset(joint).tanh() * offset_limit.float()
        candidate_match = self.match(joint).squeeze(-1)
        winner=scores.argmax(-1,keepdim=True)
        match=candidate_match.gather(1,winner).squeeze(-1)
        match=match.masked_fill(~reliable.any(-1),-1e4)
        return dict(scores=scores, pair_scores=pair_scores, reliable=reliable,
                    offset=offset, match_logit=match, candidate_match_logits=candidate_match, angle=angle,
                    distance=distance, reference_valid=refs)


def supervision_loss(output, labels, *, refine=False):
    """Unknown detections never enter the contrastive denominator or absence BCE.

    Target-derived tensors are private offline supervision, never network inputs.
    Direction and distance are supervised once per linked reference rather than
    duplicated across candidate boxes.
    """
    zero = output['scores'].sum()*0
    rank_terms = []
    for scores, reliable, truth, mask in zip(output['scores'], output['reliable'], labels['ranking'], labels['rank_label_mask']):
        known = mask.bool() & reliable
        positive = truth * known
        # Ranking requires both a verified positive and a verified nonmatch.
        if positive.sum() > 0 and (known & (truth <= 0)).any():
            target = positive[known] / positive[known].sum()
            rank_terms.append(-(target * scores[known].log_softmax(0)).sum())
    ranking = torch.stack(rank_terms).mean() if rank_terms else zero
    relation_valid = labels['angle_valid'].bool() & output['reference_valid']
    direction = (1-(output['angle']*labels['angle']).sum(-1))[relation_valid].mean() if relation_valid.any() else zero
    distance = F.smooth_l1_loss(output['distance'][relation_valid], labels['distance'][relation_valid]) if relation_valid.any() else zero
    # Trust is a property of a verified candidate's reachable local correction,
    # not a declaration that an entire scene with unknown boxes lacks a target.
    # This supplies both positive and negative labels without inventing absence.
    match_valid = labels['rank_label_mask'].bool() & output['reliable']
    match = F.binary_cross_entropy_with_logits(output['candidate_match_logits'][match_valid],labels['offset_valid'][match_valid].float()) if match_valid.any() else zero
    offset_valid = labels['offset_valid'].bool() & output['reliable']
    offset = F.smooth_l1_loss(output['offset'][offset_valid],labels['offset'][offset_valid]) if refine and offset_valid.any() else zero
    loss = ranking + .2*direction + .2*distance + .2*match + (.25*offset if refine else zero)
    return loss, dict(rank_views=len(rank_terms), ranking=float(ranking.detach()),
                      direction=float(direction.detach()),distance=float(distance.detach()),
                      match=float(match.detach()),offset=float(offset.detach()))
