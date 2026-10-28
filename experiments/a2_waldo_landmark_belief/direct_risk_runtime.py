"""Target-free head/proposal forward shared by runtime and diagnostics."""
import torch
from .revision_fusion import proposal_distribution
from .risk_probe import public_risk_features
from .direct_risk_head import apply_direct_risk


@torch.no_grad()
def risk_forward(spatial,rank_output,head,parameters):
    q,available=proposal_distribution(spatial,rank_output['scores'][0],rank_output['candidate_match_logits'][0],rank_output['reliable'][0])
    features=public_risk_features(spatial['base_logits'][None],q[None],spatial['grid_world'][None],
        rank_output['benefit_logit'].sigmoid(),torch.tensor([available],device=q.device))
    logits=head(features);corrected=head.corrected_logits(logits)
    mixed,info=apply_direct_risk(spatial['base_logits'],q,corrected[0],available,parameters)
    return mixed,info,dict(q=q,available=available,features=features,head_logits=logits,corrected_logits=corrected)
