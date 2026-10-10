import inspect
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_localization import FullTokenGeometryLocalizer
from graphnav.joint_active import JointActiveGraphNav,joint_losses
from graphnav.observed_patch_scales import masked_native_pool,observed_region_masks


def test_joint_a3_visual_trajectory_and_stop_receive_gradients(tmp_path):
    torch.manual_seed(7)
    record=dict(map_name='m',instruction='east of Church',world_bounds=[0,0,400,400],
        physical_map_sizes=[400,400],target_xy=[.7,.5],landmarks=[dict(name='Church',
        object_type='Building',contour=[[.1,.1],[.2,.1],[.2,.2],[.1,.2]])])
    path=tmp_path/'data.jsonl';path.write_text(json.dumps(record)+'\n')
    ds=FullCandidateDataset(path,grid_size=4,max_landmarks=2,max_roads=0,max_relations=2)
    static=collate_full([ds[0]])
    model=JointActiveGraphNav(FullTokenGeometryLocalizer(hidden=16,heads=4,max_relations=2,
                              text_dim=8,top_k=8,dropout=0),vision_dim=8,matcher_hidden=4)
    inputs=dict(static_inputs=static,tokens=torch.randn(1,5,8),token_mask=torch.ones(1,5,dtype=torch.bool),
        node_emb=torch.randn(1,static['nodes'].shape[1],8),native_text=torch.randn(1,8),
        roi_features=torch.randn(1,16,3,8),roi_valid=torch.ones(1,16,3,dtype=torch.bool),
        roi_quality=torch.ones(1,16,3,3),pose_world=torch.tensor([[200.,200.,50.,0.]]),
        bounds=torch.tensor([[0.,0.,400.,400.]]),ground_level=torch.zeros(1))
    model.eval();before=model(**inputs)
    torch.testing.assert_close(before.static.belief_logits,before.belief.belief_logits)
    # No private ground truth/future arguments can enter forward.
    assert not {'target_world','teacher_world','target_id'} & set(inspect.signature(model.forward).parameters)
    labels=dict(grid_world=static['grid_xy']*400,target_world=torch.tensor([[280.,200.,50.]]),
        pose_world=inputs['pose_world'],teacher_world=torch.tensor([[[220.,210.,50.]]*8]),
        valid_teacher_z=torch.ones(1,8))
    losses=joint_losses(before,**labels);losses['total'].backward()
    for group in (model.a3,model.matcher,model.trajectory):
        assert sum(float(p.grad.abs().sum()) for p in group.parameters() if p.grad is not None)>0
    assert model.visual_gain.grad is not None and abs(float(model.visual_gain.grad))>0
    # Changing loss labels cannot change deterministic inference.
    labels['target_world']+=30
    joint_losses(before,**labels)
    after=model(**inputs)
    torch.testing.assert_close(before.belief.belief_logits,after.belief.belief_logits)
    assert before.trajectory.paths_world.shape==(1,8,3,8,3)


def test_masked_pool_all_support_matches_native_attention():
    torch.manual_seed(2)
    head=SimpleNamespace(probe=torch.randn(1,1,8),attention=nn.MultiheadAttention(8,2,batch_first=True),
                         layernorm=nn.LayerNorm(8),mlp=nn.Sequential(nn.Linear(8,16),nn.GELU(),nn.Linear(16,8)))
    patches=torch.randn(12,8)
    pooled=head.attention(head.probe,patches[None],patches[None])[0][:,0]
    expected=pooled+head.mlp(head.layernorm(pooled))
    actual=masked_native_pool(head,patches,torch.ones(1,12,dtype=torch.bool))
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_patch_scales_reject_outside_and_depth_holes():
    observation=dict(bgr=np.full((384,384,3),127,np.uint8),depth_valid=np.ones((384,384),bool),
        footprint_world=np.array([[0,100],[100,100],[100,0],[0,0]],np.float32))
    masks,_=observed_region_masks(observation,[[50,50],[0,0],[200,200]])
    assert masks[0,:2].any() and not masks[0,2].any()
    assert not masks[1:].any()
    observation['depth_valid'][:]=False
    masks,_=observed_region_masks(observation,[[50,50]])
    assert not masks.any()


def test_shard_shuffle_visits_all_states_once_reproducibly():
    from graphnav.joint_active_data import ShardShuffleSampler
    dataset=SimpleNamespace(manifest={'chunks':[{'rows':8},{'rows':11},{'rows':9}]})
    sampler=ShardShuffleSampler(dataset,seed=0)
    first=list(sampler)
    assert sorted(first)==list(range(28)) and list(sampler)==first
    sampler.epoch=1
    assert list(sampler)!=first and sorted(sampler)==list(range(28))


def test_legacy_a3_missing_text_length_uses_verified_96(tmp_path,monkeypatch):
    import graphnav.joint_active_runtime as runtime
    class FakeModel:
        def to(self,*args):return self
        def eval(self):return self
        def load_state_dict(self,*args,**kwargs):pass
    captured={}
    def bert(**kwargs):captured.update(kwargs);return None
    monkeypatch.setattr(runtime,'new_model',lambda *args:None)
    monkeypatch.setattr(runtime,'JointActiveGraphNav',lambda *args:FakeModel())
    monkeypatch.setattr(runtime,'FrozenBERTTokens',bert)
    monkeypatch.setattr(runtime,'ObservedSigLIPMatchingEncoder',lambda *args,**kwargs:
        SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash='rev'))))
    checkpoint=tmp_path/'joint.pt'
    torch.save(dict(a3_args=dict(text_backbone='bert-base-uncased'),model={},protocol=dict(
        name='joint-active-v1',trainable_a3=True,backbone='siglip',backbone_revision='rev',
        active_max_actions=2)),checkpoint)
    runtime.JointActivePolicy(checkpoint,tmp_path/'inputs',device='cpu')
    assert captured['max_length']==96
