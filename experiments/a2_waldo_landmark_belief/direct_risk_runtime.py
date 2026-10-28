"""Target-free head/proposal forward shared by runtime and diagnostics."""
import torch
from .revision_fusion import proposal_distribution,bounded_belief_mix,MAX_GATE
from .risk_probe import public_risk_features
from .direct_risk_head import apply_direct_risk


@torch.no_grad()
def risk_forward(spatial,rank_output,head,parameters):
    q,available=proposal_distribution(spatial,rank_output['scores'][0],rank_output['candidate_match_logits'][0],rank_output['reliable'][0])
    features=public_risk_features(spatial['base_logits'][None],q[None],spatial['grid_world'][None],
        rank_output['benefit_logit'].sigmoid(),torch.tensor([available],device=q.device))
    logits=head(features);corrected=head.corrected_logits(logits)
    mixed,info=apply_direct_risk(spatial['base_logits'],q,corrected[0],available,parameters)
    scores=(corrected[0].float()/float(parameters.get('temperature',1.))).softmax(-1)
    if float(scores[1])!=info['gain_score'] or float(scores[2])!=info['harm_score']:raise ValueError('Saved scores differ from actual gate')
    fixed=bounded_belief_mix(spatial['base_logits'],q,MAX_GATE) if available else spatial['base_logits']
    if info['updated'] and not torch.equal(mixed,fixed):raise ValueError('Actual action differs from fixed-action outcome contract')
    return mixed,info,dict(q=q,available=available,features=features,head_logits=logits,corrected_logits=corrected,
                          scores=scores,fixed_action_logits=fixed)
