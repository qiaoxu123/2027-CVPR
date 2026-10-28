import torch
from experiments.a2_waldo_landmark_belief.revision_runtime import public_network_inputs,apply_revision
from tests.test_a2_revision_fusion import spatial


def packet():
    s=spatial();return dict(network=dict(box_rgb=torch.zeros(2,16)),spatial=s)


def output():
    return dict(scores=torch.tensor([[10.,-10.]]),candidate_match_logits=torch.tensor([[4.,-4.]]),
      reliable=torch.tensor([[True,True]]),benefit_logit=torch.tensor([4.]),gate=torch.tensor([.2455]))


def test_runtime_off_is_exact_and_enabled_field_matches_training_formula():
    pack=packet();out=output();base=pack['spatial']['base_logits']
    mixed,info=apply_revision(pack['spatial'],out,dict(enabled=False,scale=1.,threshold=.5,temperature=1.))
    assert mixed is base and not info['updated']
    mixed,info=apply_revision(pack['spatial'],out,dict(enabled=True,scale=1.,threshold=.5,temperature=1.))
    assert mixed.argmax()==3 and info['updated'] and info['gate']<=.25
    out['benefit_logit'].fill_(-4.)
    mixed,info=apply_revision(pack['spatial'],out,dict(enabled=True,scale=1.,threshold=.5,temperature=1.))
    assert mixed is base


def test_public_runtime_inputs_use_actual_detection_kind_and_belief_only():
    pack=packet();meta=dict(candidates=[dict(kind=0),dict(kind=1)])
    x=public_network_inputs(pack,meta)
    assert x['box_kind'].tolist()==[0,1] and x['a0_context'].shape==(2,6)
    pack['private']=dict(target_xy=torch.ones(2)*9999);meta['target_kind']=9999
    y=public_network_inputs(pack,meta)
    assert torch.equal(x['a0_context'],y['a0_context']) and 'target_kind' not in y and 'target_xy' not in y
