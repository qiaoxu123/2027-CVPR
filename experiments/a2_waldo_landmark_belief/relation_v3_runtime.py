"""Public-only V3 proposal, score and exact fixed-action inference contract."""
import math
import torch
from .relation_v3_fusion import proposal_with_quality
from .revision_fusion import batch_belief_mix,MAX_GATE
from .risk_probe import public_risk_features

NETWORK_KEYS={'instruction_tokens','token_valid','mention_mask','landmark_name_tokens','name_valid','link_valid',
              'box_rgb','box_valid','pair_geometry','pair_valid','offset_limit'}


def public_network_inputs(pack,meta):
    source=pack['network']
    if set(source)-NETWORK_KEYS-{'a0_context','box_kind'} or not NETWORK_KEYS<=set(source):raise ValueError('Unexpected/missing public V3 input')
    result={k:source[k] for k in NETWORK_KEYS}
    kind=torch.zeros(len(source['box_rgb']),dtype=torch.long,device=source['box_rgb'].device)
    if len(meta['candidates'])>len(kind):raise ValueError('Public detector axes changed')
    for i,candidate in enumerate(meta['candidates']):
        if candidate['kind'] not in (0,1):raise ValueError('Unsupported public detector category')
        kind[i]=candidate['kind']
    result['box_kind']=kind
    return result


@torch.no_grad()
def evidence_forward(spatial,output,model):
    """Batched inference; no target, loss labels or identity associations."""
    q,available,quality=proposal_with_quality(spatial,output['scores'],output['candidate_match_logits'],output['reliable'])
    features=public_risk_features(spatial['base_logits'],q,spatial['grid_world'],quality,available)
    logits=model.predict_risk(features);corrected=model.corrected_risk_logits(logits);scores=corrected.softmax(-1)
    if not torch.isfinite(scores).all():raise ValueError('Nonfinite V3 inference scores')
    fixed=batch_belief_mix(spatial['base_logits'],q,torch.full_like(quality,MAX_GATE))
    return dict(q=q,available=available,quality=quality,features=features,head_logits=logits,
                corrected_logits=corrected,scores=scores,fixed_action_logits=fixed)


@torch.no_grad()
def runtime_forward(spatial,output,model,parameters):
    """Single observation: choose exact A0 or the saved fixed.25 action."""
    minimum=float(parameters.get('minimum_gain',1.));maximum=float(parameters.get('maximum_harm',0.))
    temperature=float(parameters.get('temperature',1.))
    if not all(math.isfinite(v) for v in (minimum,maximum,temperature)) or not 0<=minimum<=1 or not 0<=maximum<=1 or temperature!=1.:
        raise ValueError('V3 requires finite score thresholds and fixed T1')
    if output['scores'].shape[0]!=1:raise ValueError('One runtime observation required')
    packet=evidence_forward({k:v[None] for k,v in spatial.items()},output,model)
    gain=float(packet['scores'][0,1]);harm=float(packet['scores'][0,2])
    active=bool(parameters.get('enabled',False) and packet['available'][0] and gain>=minimum and harm<=maximum)
    base=spatial['base_logits'];mixed=packet['fixed_action_logits'][0] if active else base
    return mixed,dict(updated=not torch.equal(base,mixed),gate=MAX_GATE if active else 0.,
                      gain_score=gain,harm_score=harm),packet
