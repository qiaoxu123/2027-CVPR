"""Public-only runtime inputs and exact shared revision decision formula."""
import math
import torch
from .revision_fusion import a0_candidate_context,proposal_distribution,bounded_belief_mix,MAX_GATE


def public_network_inputs(pack,meta):
    network=dict(pack['network']);network['a0_context']=a0_candidate_context(pack['spatial'])
    kind=torch.zeros(len(network['box_rgb']),dtype=torch.long,device=network['box_rgb'].device)
    for i,candidate in enumerate(meta['candidates']):
        if candidate['kind'] not in (0,1):raise ValueError('Unsupported public detector category')
        kind[i]=candidate['kind']
    network['box_kind']=kind
    return network


def apply_revision(spatial,output,parameters):
    scale=float(parameters.get('scale',0.));threshold=float(parameters.get('threshold',1.));temperature=float(parameters.get('temperature',1.))
    if not all(math.isfinite(x) for x in (scale,threshold,temperature)) or not 0<=scale<=1 or not 0<=threshold<=1 or temperature<=0:
        raise ValueError('Invalid revision calibration')
    base=spatial['base_logits'];probability=float(torch.sigmoid(output['benefit_logit'][0]/temperature))
    info=dict(updated=False,gate=0.,benefit_probability=probability)
    if not parameters.get('enabled',False) or scale==0 or probability<threshold:return base,info
    q,available=proposal_distribution(spatial,output['scores'][0],output['candidate_match_logits'][0],output['reliable'][0])
    if not available:return base,info
    gate=MAX_GATE*probability*scale
    mixed=bounded_belief_mix(base,q,gate)
    info.update(updated=not torch.equal(base,mixed),gate=gate)
    return mixed,info
