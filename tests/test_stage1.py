import inspect
import json
import numpy as np
import pytest
import torch
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_localization import FullTokenGeometryLocalizer
from graphnav.stage1 import GraphNavStage1,visual_supervision_regions,preserved_proposal_order
from graphnav.stage1_data import CandidateMemory
from graphnav.stage1_runtime import retrieval_ids


def small_system(tmp_path):
    row=dict(map_name='m',instruction='red car east of Church',world_bounds=[0,0,400,400],physical_map_sizes=[400,400],target_xy=[.7,.5],landmarks=[dict(name='Church',object_type='Building',contour=[[.1,.1],[.2,.1],[.2,.2],[.1,.2]])])
    path=tmp_path/'data.jsonl';path.write_text(json.dumps(row)+'\n')
    ds=FullCandidateDataset(path,grid_size=4,max_landmarks=2,max_roads=0,max_relations=2);static=collate_full([ds[0]])
    model=GraphNavStage1(FullTokenGeometryLocalizer(hidden=16,heads=4,max_relations=2,text_dim=8,top_k=8,dropout=0),vision_dim=8,matcher_hidden=4)
    model.configure_stage('A');model.eval()
    public=dict(static_inputs=static,tokens=torch.randn(1,5,8),token_mask=torch.ones(1,5,dtype=torch.bool),node_emb=torch.randn(1,static['nodes'].shape[1],8),native_text=torch.randn(1,8),appearance_text=torch.randn(1,8),roi_features=torch.randn(1,16,3,8),roi_valid=torch.ones(1,16,3,dtype=torch.bool),roi_quality=torch.ones(1,16,3,3),pose_world=torch.tensor([[200.,200.,50.,0.]]),bounds=torch.tensor([[0.,0.,400.,400.]]),ground_level=torch.zeros(1))
    return model,public


def test_stage1_exact_initial_control_and_bounded_unknown_fallback(tmp_path):
    model,x=small_system(tmp_path)
    initial=model(**x)
    assert initial.belief is initial.static
    assert not any(p.requires_grad for p in model.a3.parameters())
    assert not any(p.requires_grad for p in model.trajectory.parameters())
    with torch.no_grad():model.visual_gain.fill_(10)
    x['roi_valid'][:,8:]=False
    result=model(**x)
    assert result.delta.abs().max()<=.5
    assert torch.equal(result.delta[:,8:],torch.zeros(1,8))
    x['roi_valid'][:]=False
    absent=model(**x)
    assert absent.belief is absent.static
    disabled=model(**dict(x,visual_control='no-belief'))
    assert disabled.belief is disabled.static
    model.configure_stage('C')
    assert any(p.requires_grad for p in model.a3.parameters())
    assert not any(p.requires_grad for p in model.a3.gnn.layer1.parameters())
    assert all(p.requires_grad for p in model.trajectory.parameters())


def test_stage1_history_can_support_cells_missing_from_current(tmp_path):
    model,x=small_system(tmp_path);model.configure_stage('B');model.eval()
    with torch.no_grad():model.visual_gain.fill_(1)
    x['roi_valid'][:]=False
    hv=torch.zeros_like(x['roi_valid']);hv[:,2:5]=True
    out=model(**dict(x,use_history=True,history_features=x['roi_features'],history_valid=hv,history_quality=x['roi_quality']))
    assert torch.equal(out.observed,hv.any(-1))
    assert not out.current_observed.any()
    assert out.delta[:,5:].eq(0).all()
    assert not {'target_world','teacher_world','target_id','future_frames'}&set(inspect.signature(model.forward).parameters)


def test_quality_memory_rejects_reordering_and_deduplicates_views():
    memory=CandidateMemory(4,8)
    def frame(step,x,value,quality):
        return dict(step=step,pose=torch.tensor([x,0.,50.,0.]),roi_ids=torch.tensor([1]),roi_features=torch.full((1,3,8),value),roi_valid=torch.ones(1,3,dtype=torch.bool),roi_quality=torch.tensor([[[quality,1.,.5]]*3]))
    memory.update(frame(0,0,1.,1.));memory.update(frame(1,0,100.,1.))
    assert memory.duplicates==1
    memory.update(frame(2,10,3.,.5))
    features,valid,_=memory.tensors()
    torch.testing.assert_close(features[1].float(),torch.full((3,8),5/3),atol=.001,rtol=.001)
    assert valid[1].all() and not valid[0].any()
    with pytest.raises(ValueError):memory.update(frame(1,20,5.,1.))


def test_retrieval_preserves_legacy_first_eight_tie_order():
    scores=torch.zeros(3,28*28)
    ids=retrieval_ids(scores)
    assert torch.equal(ids[:,:8],scores.topk(8,-1).indices)
    for row in ids:assert len(set(row.tolist()))==16


def test_hard_visual_negatives_exclude_overlapping_target_crops():
    world=torch.tensor([[[10.,0.],[25.,0.],[80.,0.],[100.,0.]]])
    target=torch.zeros(1,3);valid=torch.ones(1,4,3,dtype=torch.bool);valid[:,3]=False
    positive,negative=visual_supervision_regions(world,target,valid)
    assert positive.tolist()==[[True,False,False,False]]
    # 25m is outside navigation success but its 64/128m ROI includes GT.
    assert negative.tolist()==[[False,False,True,False]]


def test_visual_zero_residual_preserves_original_proposal_ties():
    scores=torch.zeros(2,784,dtype=torch.bfloat16)
    original=scores.topk(20,-1).indices
    assert torch.equal(preserved_proposal_order(scores.float(),original),original)
    changed=scores.float();changed[:,10]=1
    updated=preserved_proposal_order(changed,original)
    assert updated[:,0].eq(10).all()
