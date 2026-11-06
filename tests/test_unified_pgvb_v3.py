"""CPU-only V3 invariants: co-registration, no-GT forward, private Loss, yaw."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from graphnav.unified_pgvb_v3 import (
    COARSE, PATCHES, SIZE, UnifiedPGVBV3, offline_topk,
    project_global_to_frame, raster_target_polygon, v3_losses,
)
from scripts.unified_pgvb_v3_pilot import fixed_yaw_public_poses


class TinyGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(.1))

    def forward(self, static, *, variant, tokens, token_mask, node_emb):
        assert variant == "geometry"
        assert not static["rules"].abs().any()
        assert not static["relation_fields"].abs().any()
        p = static["grid_xy"]
        return SimpleNamespace(belief_logits=(
            p[..., 0] * (1 + self.bias) + p[..., 1]*self.bias),
        )


def _inputs(batch=1, frames=1, all_invalid=False):
    g = (torch.arange(COARSE) + .5) / COARSE
    y,x = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack((x.flatten(),y.flatten()),-1)[None].expand(batch,-1,-1)
    fp = torch.tensor([[20.,20.],[80.,20.],[80.,80.],[20.,80.]])
    return dict(
        static_inputs=dict(grid_xy=grid, rules=torch.ones(batch,COARSE**2),
                           relation_fields=torch.ones(batch,COARSE**2,8)),
        bert_tokens=torch.randn(batch,5,768),
        bert_mask=torch.ones(batch,5,dtype=torch.bool),
        landmark_embeddings=torch.randn(batch,4,768),
        siglip_text=torch.randn(batch,1024),
        rgb_patches=torch.randn(batch,frames,PATCHES*PATCHES,1024),
        depth_patches=torch.ones(batch,frames,PATCHES*PATCHES,2),
        patch_valid=torch.full((batch,frames,PATCHES*PATCHES),
                               not all_invalid,dtype=torch.bool),
        footprints_world=fp[None,None].expand(batch,frames,-1,-1).clone(),
        frame_poses=torch.tensor([50.,50.,85.,0.])[None,None].expand(
            batch,frames,-1).clone(),
        world_bounds=torch.tensor([0.,0.,100.,100.])[None].expand(batch,-1),
    )


def test_global_registration_coordinates_and_support():
    i = _inputs()
    grid, inside = project_global_to_frame(i["world_bounds"],i["footprints_world"])
    assert grid.shape == (1,1,SIZE,SIZE,2)
    # All maps use ROW/Y increasing with WORLD +Y.
    assert grid[0,0,0,0,1] < grid[0,0,-1,-1,1]
    assert bool(inside[0,0,SIZE//2,SIZE//2])
    assert not bool(inside[0,0,0,0])


def test_same_global_footprint_with_90_degree_yaw_rotation():
    i = _inputs()
    first = project_global_to_frame(i["world_bounds"],i["footprints_world"])[1]
    rotated = torch.tensor([[80.,20.],[80.,80.],[20.,80.],[20.,20.]])
    latter = project_global_to_frame(i["world_bounds"],rotated[None,None])[1]
    torch.testing.assert_close(first, latter)


def test_no_observation_does_not_inject_fake_visual_evidence():
    torch.manual_seed(3)
    m=UnifiedPGVBV3(TinyGeometry()).eval()
    x=_inputs(all_invalid=True)
    out=m(**x)
    assert out.geometry_logits.shape==(1,COARSE**2)
    assert out.fused_logits.shape==(1,SIZE,SIZE)
    assert not out.observed_mask.any()
    torch.testing.assert_close(out.fused_logits,out.geometry_global,atol=0,rtol=0)
    assert out.visual_global.abs().max()==0


def test_causal_visual_has_dense_world_support_and_grads():
    torch.manual_seed(42)
    m=UnifiedPGVBV3(TinyGeometry()).train()
    args=_inputs()
    out=m(**args)
    assert out.observed_mask.any()
    assert out.observed_mask.sum()<SIZE*SIZE
    loss=out.fused_logits.square().mean()
    loss.backward()
    assert m.vision_proj[0].weight.grad is not None
    assert m.vision_proj[0].weight.grad.abs().sum()>0
    assert m.geometry.bias.grad is not None


def test_world_gt_polygon_is_training_only_and_correctly_aligned():
    ring=[[48,48],[52,48],[52,52],[48,52]]
    mask,valid=raster_target_polygon(ring,[0,0,100,100],SIZE)
    assert valid
    assert mask[SIZE//2,SIZE//2]>.5
    assert mask[10,10]==0
    mask,valid=raster_target_polygon(None,[0,0,100,100],SIZE)
    assert not valid and mask.sum()==0


def test_loss_uses_private_teacher_only_after_forward():
    torch.manual_seed(5)
    m=UnifiedPGVBV3(TinyGeometry()).train()
    out=m(**_inputs())
    labels=[[35,35],[65,35],[65,65],[35,65]]
    train = v3_losses(
        out,torch.tensor([[.5,.5]]),[labels],
        torch.tensor([[0.,0.,100.,100.]]),
        torch.randn(1,1024),m.teacher_head)
    assert torch.isfinite(train["total"])
    train["total"].backward()
    assert m.teacher_head.weight.grad is not None
    assert m.teacher_head.weight.grad.abs().sum()>0
    assert int(train["target_polygon_valid"])==1
    results=offline_topk(out,torch.tensor([[.5,.5]]),
                         torch.tensor([[0.,0.,100.,100.]]))
    assert results["n"]==1 and "R@5/20m_count" in results


def test_fixed_yaw_sampling_public_only_and_same_for_different_gt():
    r=dict(
        map_name="cambridge_block_0",
        world_bounds=[0.,0.,400.,400.],
        landmarks=[dict(name="One Stop", contour=[
            [.4,.4],[.6,.4],[.6,.6],[.4,.6]])],
        target_xy=[.9,.9],
    )
    a=fixed_yaw_public_poses(r,ground=15.,altitude_m=80.,
                             yaw=1.2,offset_m=25.,views=5)
    r["target_xy"]=[.05,.05]
    b=fixed_yaw_public_poses(r,ground=15.,altitude_m=80.,
                             yaw=1.2,offset_m=25.,views=5)
    np.testing.assert_array_equal(np.stack(a),np.stack(b))
    assert len(a)==5
    assert all(abs(p[3]-1.2)<1e-9 for p in a)
    r["landmarks"]=[]
    assert fixed_yaw_public_poses(r,ground=15.,altitude_m=80.,
                                  yaw=0.,offset_m=25.,views=5)==[]


def test_rejects_malformed_world_registration():
    args=_inputs()
    args["world_bounds"][0,2]=0
    with pytest.raises(ValueError):
        UnifiedPGVBV3(TinyGeometry())(**args)


def test_blind_public_polygon_binding_cannot_use_annotation_target_id():
    from scripts.unified_pgvb_v3_pilot import blind_public_landmarks
    records = [{
        "map_name":"cambridge_block_0",
        "world_bounds":[0.,0.,100.,100.],
        "instruction":"The red car next to One Stop",
        "landmarks":[dict(id="999",name="GT-picked wrong landmark")],
        "target_xy":[.1,.9],
        "episode_id":"cambridge_block_0:target42:0:0",
    }]
    catalog = {"cambridge_block_0":{
        "1":dict(name="One Stop",object_type="Building",
                 contour=[[10,10],[20,10],[20,20],[10,20]]),
        "2":dict(name="One Stop",object_type="Building",
                 contour=[[40,40],[50,40],[50,50],[40,50]]),
        "target42":dict(name="Unnamed Car",object_type="Car",
                        contour=[[80,80],[82,80],[82,82],[80,82]]),
    }}
    first=records[0]
    counts=blind_public_landmarks(records,catalog,16)
    assert counts["bound"]==1
    assert {x["id"] for x in first["landmarks"]}=={"1","2"}
    expected=[x["contour"] for x in first["landmarks"]]
    first["target_xy"]=[.9,.1]
    first["episode_id"]="cambridge_block_0:target999:0:0"
    blind_public_landmarks(records,catalog,16)
    assert [x["contour"] for x in first["landmarks"]]==expected


def test_private_polygon_scoring_is_not_20m_radius_scoring():
    torch.manual_seed(6)
    model=UnifiedPGVBV3(TinyGeometry()).eval()
    out=model(**_inputs(all_invalid=True))
    poly=[[48,48],[52,48],[52,52],[48,52]]
    r=offline_topk(out,torch.tensor([[.5,.5]]),
                   torch.tensor([[0.,0.,100.,100.]]),
                   branch="geometry",target_polygons=[poly])
    assert r["Polygon_valid"]==[True]
    assert "1" in r["Polygon_hit_flags"]
    assert "5" in r["hit_flags"]


def test_dense_controller_candidate_coordinates_and_context_are_aligned():
    from graphnav.unified_pgvb_v3_runtime import V3CausalBelief
    coarse=(torch.arange(28)+.5)/28
    yy,xx=torch.meshgrid(coarse,coarse,indexing='ij')
    context=torch.stack((xx.flatten(),yy.flatten()),-1)[None]
    logits=torch.zeros(1,384,384)
    logits[0,100,200]=4.;logits[0,0,383]=3.
    output=SimpleNamespace(geometry_global=logits,fused_logits=logits,
                           geometry_context=context,observed_mask=torch.zeros_like(logits,dtype=torch.bool))
    field=V3CausalBelief.controller_field(output,top_k=2)
    assert field.candidate_ids.tolist()==[[100*384+200,383]]
    torch.testing.assert_close(field.candidate_xy[0,0],torch.tensor([200.5/384,100.5/384]))
    expected=field.candidate_xy.clamp(.5/28,1-.5/28)
    torch.testing.assert_close(field.candidate_context,expected,atol=1e-6,rtol=1e-6)
    assert field.belief_logits.gather(1,field.candidate_ids).tolist()==[[4.,3.]]


def test_posthoc_correction_capacity_distinguishes_bound_and_evidence_failures():
    from graphnav.v3_diagnostics import visual_correction_capacity
    bounds=torch.tensor([[0.,0.,100.,100.]])
    target=torch.tensor([[.25,.25]])
    geom=torch.tensor([[[0.,3.],[0.,0.]]])
    out=SimpleNamespace(geometry_global=geom,fused_logits=geom.clone(),observed_mask=torch.ones_like(geom,dtype=torch.bool))
    row=visual_correction_capacity(out,target,bounds,.5,radius_m=1)[0]
    assert not row['geometry_correct'] and row['correct_region_observed']
    assert not row['can_flip_under_independent_bounded_updates']
    geom[0,0,1]=.8
    row=visual_correction_capacity(out,target,bounds,.5,radius_m=1)[0]
    assert row['can_flip_under_independent_bounded_updates'] and not row['fused_correct']
    out.observed_mask.zero_()
    row=visual_correction_capacity(out,target,bounds,.5,radius_m=1)[0]
    assert not row['can_flip_under_independent_bounded_updates'] and not row['correct_region_observed']


def test_physical_partition_does_not_load_rgb_tensors(tmp_path):
    from scripts.unified_pgvb_v3_pilot import fit_dev_partition
    (tmp_path/'private').mkdir()
    keys=[]
    for object_id in range(64):
        for query_id in range(2):
            key=f'{len(keys):06d}';keys.append(key)
            torch.save(dict(episode_id=f'map:{object_id}:{query_id}:0'),tmp_path/'private'/f'{key}.pt')
    class MetadataDataset:
        root=tmp_path
        def __len__(self):return len(self.keys)
        def __getitem__(self,index):raise AssertionError('RGB must not be loaded for partitioning')
    dataset=MetadataDataset();dataset.keys=keys
    fit,dev=fit_dev_partition(dataset,42)
    assert fit and dev and len(fit)+len(dev)==len(keys)
    assert not {i//2 for i in fit}&{i//2 for i in dev}


def test_capacity_actual_goal_uses_formal_top20_tie_order():
    from graphnav.v3_diagnostics import visual_correction_capacity
    geom=torch.zeros(1,384,384)
    idx=int(geom.flatten().topk(20).indices[0])
    target=torch.tensor([[(idx%384+.5)/384,(idx//384+.5)/384]])
    output=SimpleNamespace(geometry_global=geom,fused_logits=geom.clone(),observed_mask=torch.zeros_like(geom,dtype=torch.bool))
    row=visual_correction_capacity(output,target,torch.tensor([[0.,0.,100.,100.]]),.5,radius_m=.01)[0]
    assert row['geometry_correct'] and row['fused_correct']
    assert row['capacity_tie_undetermined'] and not row['bound_insufficient']
    assert not row['can_flip_under_independent_bounded_updates']


def test_evaluation_failure_never_reuses_previous_prediction():
    from scripts.evaluate_unified_pgvb_v3_initial import failed_row
    record=SimpleNamespace(episode_id=('map',3,7),map_name='map')
    row=failed_row(8,record,RuntimeError('forward failed'))
    assert row['index']==8 and row['episode_id']==['map',3,7]
    assert row['correction_capacity'] is None and 'forward failed' in row['error']
    for branch in ('geometry','fused'):
        assert row[branch]['NE'] is None
        assert not any(row[branch]['hits'].values())
