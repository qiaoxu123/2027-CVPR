import pytest
import torch
from tests.test_a2_relation_v3_loss import batch
from experiments.a2_waldo_landmark_belief.relation_v3_model import RelationV3Network
from experiments.a2_waldo_landmark_belief.relation_v3_runtime import evidence_forward,runtime_forward,public_network_inputs


def test_runtime_uses_same_batched_stable_proposal_scores_and_fixed_action():
    torch.manual_seed(0);model=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2).eval();b=batch()
    out=model(**b['network']);packet=evidence_forward(b['spatial'],out,model)
    one={k:v[0] for k,v in b['spatial'].items()};output={k:v[:1] for k,v in out.items()}
    off,info,record=runtime_forward(one,output,model,dict(enabled=False))
    assert torch.equal(off,one['base_logits']) and not info['updated']
    on,info,record=runtime_forward(one,output,model,dict(enabled=True,minimum_gain=0.,maximum_harm=1.))
    assert torch.equal(on,packet['fixed_action_logits'][0]) and info['gate']==.25
    assert torch.allclose(record['scores'],packet['scores'][:1],atol=1e-7,rtol=1e-6)
    assert info['gain_score']==float(record['scores'][0,1]) and info['harm_score']==float(record['scores'][0,2])
    assert torch.equal(on,record['fixed_action_logits'][0])
    with pytest.raises(TypeError):runtime_forward(one,output,model,{},target_xy=torch.zeros(2))
    with pytest.raises(ValueError):runtime_forward(one,output,model,dict(temperature=2.))


def test_empty_observation_support_falls_back_even_if_forced_on():
    torch.manual_seed(0);model=RelationV3Network(text_dim=8,vision_dim=6,hidden=8,heads=2);b=batch()
    spatial={k:v[0] for k,v in b['spatial'].items()};spatial['candidate_support'].zero_()
    out={k:v[:1] for k,v in model(**b['network']).items()}
    mixed,info,packet=runtime_forward(spatial,out,model,dict(enabled=True,minimum_gain=0.,maximum_harm=1.))
    assert torch.equal(mixed,spatial['base_logits']) and info['gate']==0
    assert not packet['available'][0] and torch.isfinite(packet['scores']).all()


def test_runtime_inputs_rebuild_detector_kind_without_A0_context_or_GT():
    b=batch();network={k:v[0] for k,v in b['network'].items() if k!='box_kind'}
    result=public_network_inputs(dict(network=dict(network,a0_context=torch.ones(3,6))),dict(candidates=[dict(kind=1),dict(kind=0)]))
    assert result['box_kind'].tolist()==[1,0,0] and 'a0_context' not in result
    network['target_xy']=torch.zeros(2)
    with pytest.raises(ValueError):public_network_inputs(dict(network=network),dict(candidates=[]))
