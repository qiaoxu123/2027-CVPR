import json
import torch
import pytest
from graphnav.candidate_geometry import FullCandidateDataset, collate_full
from graphnav.full_localization import FullTokenGeometryLocalizer
from graphnav.dynamic_belief import DynamicGoalBelief, LandmarkHistory


def inputs(tmp_path):
    path=tmp_path/'public.jsonl'
    path.write_text(json.dumps(dict(map_name='map',instruction='east of hall',
        world_bounds=[0,0,400,400],physical_map_sizes=[400,400],target_xy=[0,0],
        landmarks=[dict(name='hall',object_type='Building',contour=[[.2,.2],[.3,.2],[.3,.3],[.2,.3]])]))+'\n')
    batch=collate_full([FullCandidateDataset(path,grid_size=4,max_landmarks=2,max_roads=0,max_relations=2)[0]])
    model=FullTokenGeometryLocalizer(hidden=16,text_dim=12,heads=4,max_relations=2,top_k=3,dropout=0).eval()
    kwargs=dict(tokens=torch.randn(1,5,12),token_mask=torch.ones(1,5,dtype=torch.bool),node_emb=torch.randn(1,4,12))
    return batch,model,kwargs


def test_zero_residual_is_exactly_a3_and_gate_has_gradients(tmp_path):
    batch,a3,kwargs=inputs(tmp_path)
    net=DynamicGoalBelief(a3,vision_dim=6).eval()
    reference=a3(batch,variant='geometry',**kwargs)
    vectors=torch.randn(1,4,6);confidence=batch['entity_valid'].float()
    out=net(batch,**kwargs,visual_vectors=vectors,visual_confidence=confidence,use_graph=True,use_visual=True)
    assert torch.equal(reference.belief_logits,out.belief_logits)
    assert torch.equal(reference.candidate_ids,out.candidate_ids)
    out.belief_logits.sum().backward()
    assert net.graph_gate.grad is not None and net.visual_gate.grad is not None
    assert all(p.grad is None for p in a3.parameters())


def test_no_visual_observation_is_unknown_and_all_cells_can_change(tmp_path):
    batch,a3,kwargs=inputs(tmp_path);net=DynamicGoalBelief(a3,vision_dim=6).eval()
    with torch.no_grad():net.visual_gate.fill_(1)
    base=net(batch,**kwargs)
    no_observation=net(batch,**kwargs,use_visual=True,visual_vectors=torch.randn(1,4,6),visual_confidence=torch.zeros(1,4))
    assert torch.equal(base.belief_logits,no_observation.belief_logits)
    observed=net(batch,**kwargs,use_visual=True,visual_vectors=torch.randn(1,4,6),visual_confidence=batch['entity_valid'].float())
    outside=torch.ones(16,dtype=torch.bool);outside[base.candidate_ids[0]]=False
    assert (observed.belief_logits-base.belief_logits)[0,outside].abs().sum()>0


def test_history_preserves_unknown_and_rejects_noncausal_steps():
    history=LandmarkHistory.empty(1,2,6,'cpu');v=torch.ones(1,2,6)
    history.update(v,torch.tensor([[.8,0.]]),step=0)
    old=history.vectors.clone()
    history.update(v*3,torch.tensor([[0.,.5]]),step=1)
    assert torch.equal(history.vectors[:,0],old[:,0])
    assert torch.equal(history.vectors[:,1],v[:,1]*3)
    with pytest.raises(ValueError):history.update(v,torch.ones(1,2),step=1)
    with pytest.raises(ValueError):history.update(v,torch.full((1,2),2.),step=2)


def test_precomputed_frozen_field_matches_uncached_residual(tmp_path):
    batch,a3,kwargs=inputs(tmp_path);net=DynamicGoalBelief(a3,vision_dim=6).eval()
    reference=a3(batch,variant='geometry',**kwargs)
    cached=net(batch,**kwargs,base_field=reference,use_graph=True)
    uncached=net(batch,**kwargs,use_graph=True)
    assert torch.equal(cached.belief_logits,uncached.belief_logits)
    a3.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen'):net(batch,**kwargs,base_field=reference)


def test_visual_geometry_binding_survives_pooling_and_supports_unreferenced_observations(tmp_path):
    batch,a3,kwargs=inputs(tmp_path);net=DynamicGoalBelief(a3,vision_dim=6).eval()
    with torch.no_grad():net.visual_gate.fill_(1)
    vectors=torch.randn(1,2,6)*3;geometry=torch.randn(1,16,2,10)*2;confidence=torch.ones(1,2)
    first=net(batch,**kwargs,use_visual=True,observed_vectors=vectors,observed_geometry=geometry,observed_confidence=confidence)
    swapped=net(batch,**kwargs,use_visual=True,observed_vectors=vectors.flip(1),observed_geometry=geometry,observed_confidence=confidence)
    assert not torch.allclose(first.belief_logits,swapped.belief_logits,atol=1e-6)
    base=net(batch,**kwargs)
    missing=net(batch,**kwargs,use_visual=True,observed_vectors=vectors,observed_geometry=geometry,observed_confidence=confidence*0)
    assert torch.equal(missing.belief_logits,base.belief_logits)


def test_history_weighted_mean_does_not_lose_mass_when_confidence_is_bounded():
    history=LandmarkHistory.empty(1,1,2,'cpu')
    for step,value in enumerate((1.,3.,5.)):
        history.update(torch.full((1,1,2),value),torch.tensor([[.8]]),step=step)
    assert torch.allclose(history.vectors,torch.full((1,1,2),3.))
    assert torch.allclose(history.confidence,torch.tensor([[.8]]))
    assert torch.allclose(history.total_weight,torch.tensor([[2.4]]))


def test_low_registration_confidence_attenuates_visual_evidence(tmp_path):
    batch,a3,kwargs=inputs(tmp_path);net=DynamicGoalBelief(a3,vision_dim=6).eval()
    with torch.no_grad():net.visual_gate.fill_(1)
    vectors=torch.randn(1,4,6);confidence=batch['entity_valid'].float()
    base=net(batch,**kwargs).belief_logits
    strong=net(batch,**kwargs,use_visual=True,visual_vectors=vectors,visual_confidence=confidence).belief_logits-base
    weak=net(batch,**kwargs,use_visual=True,visual_vectors=vectors,visual_confidence=confidence*.1).belief_logits-base
    assert torch.allclose(weak,strong*.1,atol=2e-6)


def test_gat_node_state_actually_updates_with_observed_visual_evidence(tmp_path):
    batch,a3,kwargs=inputs(tmp_path);net=DynamicGoalBelief(a3,vision_dim=6).eval()
    with torch.no_grad():net.graph_gate.fill_(1);net.visual_gate.zero_()
    baseline=net(batch,**kwargs,use_graph=True)
    vectors=torch.randn(1,4,6)*3;confidence=batch['entity_valid'].float()
    observed=net(batch,**kwargs,use_graph=True,use_visual=True,visual_vectors=vectors,visual_confidence=confidence)
    assert not torch.allclose(observed.belief_logits,baseline.belief_logits,atol=1e-6)
    unknown=net(batch,**kwargs,use_graph=True,use_visual=True,visual_vectors=vectors,visual_confidence=confidence*0)
    assert torch.equal(unknown.belief_logits,baseline.belief_logits)
