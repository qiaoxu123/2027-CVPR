import torch
from graphnav.native_preserving_gate import (NativePreservingGate, incremental_targets,
                                              fit_probability_calibration, select_safe_threshold)


def test_incremental_targets_compare_native_not_a0_and_protect_success():
    native=torch.tensor([[2.,0.],[2.,0.],[2.,0.]])
    proposal=torch.tensor([[0.,2.],[0.,2.],[0.,2.]])
    world=torch.tensor([[[0.,0.],[30.,0.]]]*3)
    labels=incremental_targets(native,proposal,world,torch.tensor([[30.,0.],[0.,0.],[30.,0.]]),torch.tensor([True,True,False]))
    assert labels['target'].tolist()==[1.,0.,0.]
    assert labels['damaged'].tolist()==[False,True,False]


def test_calibration_can_select_threshold_below_half_without_native_damage():
    p=torch.tensor([.2,.3,.1]);eligible=torch.ones(3,dtype=torch.bool)
    result=select_safe_threshold(p,eligible,torch.tensor([40.,5.,50.]),torch.tensor([5.,40.,49.]),torch.tensor([40.,5.,50.]))
    # Any threshold accepting the beneficial .2 also harms the .3: reject all.
    assert result['selected']['accepted']==0
    result=select_safe_threshold(torch.tensor([.3,.2]),torch.ones(2,dtype=torch.bool),torch.tensor([40.,5.]),torch.tensor([5.,40.]),torch.tensor([40.,5.]))
    assert result['selected']['threshold']<.5
    assert result['selected']['net']==1 and result['selected']['damaged']==0


def test_platt_handles_no_positive_and_keeps_monotonicity():
    calibration=fit_probability_calibration(torch.tensor([-1.,0.,1.]),torch.zeros(3))
    assert calibration['slope']>0
    mixed=fit_probability_calibration(torch.tensor([-1.,0.,1.]),torch.tensor([0.,0.,1.]))
    assert mixed['slope']>0


def test_gate_rejected_disabled_or_no_relation_preserves_native_bitwise():
    native=torch.tensor([[1.,3.,0.]])
    kw=dict(native_logits=native,relation_query=torch.tensor([[1.,2.]]),null=torch.zeros(1),
            kernel=torch.tensor([[[1.,0.],[0.,0.],[0.,1.]]]),observed=torch.ones(1,3,dtype=torch.bool),
            valid=torch.ones(1,2,dtype=torch.bool),geometry=torch.zeros(1,2,8),
            grid_world=torch.tensor([[[0.,0.],[10.,0.],[20.,0.]]]),has_relation=torch.tensor([True]))
    gate=NativePreservingGate()
    for extra in [dict(enabled=False,threshold=.01),dict(enabled=True,threshold=1.),
                  dict(enabled=True,threshold=.01,has_relation=torch.tensor([False]))]:
        result=gate(**(kw|extra))
        assert torch.equal(result['logits'],native) and not result['accepted'].any()
    result=gate(**kw,enabled=True,threshold=.01)
    assert result['accepted'].all() and not torch.equal(result['logits'],native)


def test_probability_threshold_split_is_group_disjoint_and_public_stratified():
    from scripts.train_native_preserving_relation_gate import split_probability_threshold
    rows=[dict(group=str(i),reference_binding=torch.tensor([i<3])) for i in range(10)]
    a,b=split_probability_threshold(rows)
    assert not {r['group'] for r in a}&{r['group'] for r in b}
    assert all(any(r['reference_binding'].any() for r in side) for side in [a,b])
