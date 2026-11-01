import inspect
import numpy as np
import pytest
import torch
from graphnav.citynav_box_diagnostics import (authorize_box,OracleMode,public_selection_stages,bounded_fusion_limit,bbox_cxcywh,box_world_centers)


def annotation():
    return dict(sample_id='train:1:0',image_sha='image',image_shape=[100,100,3],auto_bbox_xyxy=[10,20,30,40],quality=dict(high_geometric_support=True))


def test_unreviewed_is_never_primary_oracle_and_no_box_in_normal_or_dev():
    a=annotation()
    assert authorize_box('verified',a) is None
    for partition,mode in [('train','normal'),('dev','verified'),('val_seen','projection-hypothesis')]:
        with pytest.raises(ValueError):authorize_box(mode,a,partition=partition)
    s=authorize_box('projection-hypothesis',a)
    assert s.provenance.startswith('UNVERIFIED') and not s.actual_visibility_verified


def test_invisible_rejected_or_low_support_never_inject_even_sensitivity():
    a=annotation()
    for r in [dict(status='not_visible'),dict(status='rejected'),dict(visibility='not_visible')]:
        assert authorize_box('projection-hypothesis',a,r) is None
    a['quality']['high_geometric_support']=False
    assert authorize_box('projection-hypothesis',a) is None


def test_real_human_attestation_and_identity_required():
    a=annotation();r=dict(sample_id=a['sample_id'],image_sha='image',status='corrected',bbox=[11,21,31,41],source='human',reviewer='reviewer',human_attested=True,visibility='visible')
    assert authorize_box('verified',a,r).bbox==(11.,21.,31.,41.)
    r['human_attested']=False;assert authorize_box('verified',a,r) is None
    r['human_attested']=True;r['image_sha']='other'
    with pytest.raises(ValueError):authorize_box('verified',a,r)


def test_public_selector_has_no_private_gt_parameter_and_budget_prefix_is_fixed():
    assert not any('target' in p or 'gt' in p for p in inspect.signature(public_selection_stages).parameters)
    boxes=torch.tensor([[.1,.1,.05,.05],[.8,.8,.05,.05],[.5,.5,.05,.05]])
    x=public_selection_stages(boxes,torch.tensor([.8,.7,.6]),torch.ones(3),torch.zeros(3))
    assert torch.equal(x['top15'],x['top60'])


def test_analytic_bound_detects_unmovable_prior_without_inventing_evidence():
    r=bounded_fusion_limit(torch.tensor([10.,0.]),torch.tensor([[0.,0.],[100.,0.]]),torch.ones(2,dtype=torch.bool),torch.tensor([100.,0.]),torch.tensor(.7741))
    assert r['bounded_impossible'] and not r['a0_correct']


def test_bbox_world_mapping_keeps_existing_normalized_box_convention():
    box=bbox_cxcywh([10,20,30,40],(100,100))
    world=box_world_centers(box[None],np.array([[0,100],[100,100],[100,0],[0,0]],np.float32),(100,100))
    assert torch.allclose(world[0],torch.tensor([20.,70.]))
