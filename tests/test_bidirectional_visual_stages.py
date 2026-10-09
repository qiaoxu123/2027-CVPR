"""P0-P2 regression: geometry correspondence, inverse-only labels, visual pairs."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from graphnav.landmark_projection import (
    project_landmark, world_to_frame, frame_to_world, make_roi_image
)
from scripts.prepare_landmark_views import objects_by_map, process
from graphnav.bidirectional import generate_pairs, cardinal_relation, INVERSE
from graphnav.bidirectional_train import (
    PairedGeometryDataset, paired_collate, batch_loss
)
from graphnav.landmark_visual_pretrain import (
    PairedROIs, VisualLandmarkAligner, roi_collate, match_loss
)


def bounds():
    return {"test_block_0":[0.,0.,400.,400.]}


def objects():
    def square(x,y,r=7.):
        return [[x-r,y-r],[x+r,y-r],[x+r,y+r],[x-r,y+r]]
    return {"test_block_0":{
        "a":{"name":"Church","object_type":"Building","contour":square(100,200)},
        "b":{"name":"Library","object_type":"Building","contour":square(200,200)},
        "c":{"name":"Main Road","object_type":"TrafficRoad","contour":square(310,200)},
    }}


def footprint():
    return [[0.,400.],[400.,400.],[400.,0.],[0.,0.]]


def test_projection_roundtrip_world_to_view_and_crop():
    points=np.asarray([[80,80],[320,80],[320,320],[80,320]],np.float32)
    xy=world_to_frame(points,footprint(),(400,400,3))
    reconstructed=frame_to_world(xy,footprint(),(400,400,3))
    np.testing.assert_allclose(points,reconstructed,atol=.01)
    obj={"id":"a","name":"Test","object_type":"Building",
         "contour":[[100,100],[150,100],[150,150],[100,150]]}
    roi=project_landmark(obj,footprint(),(400,400,3),min_pixels=20)
    assert roi is not None and roi.visible_fraction>.98
    image=np.full((400,400,3),127,np.uint8)
    out=make_roi_image(image,roi,output_size=128)
    assert out.shape==(128,128,3)
    assert out.max()==127


def test_projection_clipped_object_rejected_without_gt():
    obj={"id":"b","name":"Far","object_type":"Building",
         "contour":[[380,380],[430,380],[430,430],[380,430]]}
    assert project_landmark(obj,footprint(),(400,400,3),
                            min_visible=.8,min_pixels=20) is None


def test_p0_exports_real_frame_roi_and_refuses_future(tmp_path):
    image=np.zeros((256,256,3),np.uint8)
    image[:,:,1]=200
    path=tmp_path/"frame.png"
    assert cv2.imwrite(str(path),image)
    record={"map_name":"test_block_0","observation_cutoff_step":0,
            "rgb_observations":[{"step":0,"path":str(path),
                                 "footprint_world":footprint()}],
            "target_xy":[0.75,0.75],
            "teacher_future":[[.8,.8]]}
    info=process([record],objects_by_map(objects()),tmp_path/"p0",
                 save_rois=True,overlay_frames=1,min_pixels=20)
    assert info["frames"]==1
    assert info["rois"]>=2
    rows=[json.loads(x) for x in (tmp_path/"p0"/"landmark_rois.jsonl").read_text().splitlines()]
    assert all("target_xy" not in x and "teacher_future" not in x for x in rows)
    assert all(Path(x["roi_path"]).exists() for x in rows)
    record["rgb_observations"][0]["step"]=1
    with pytest.raises(ValueError,match="future"):
        process([record],objects_by_map(objects()),tmp_path/"bad")


def test_p1_pairs_reverse_identity_and_no_target_name_in_instruction(tmp_path):
    data=list(generate_pairs(objects(),bounds(),{"test_block_0"},
                             max_objects_per_map=3,max_pairs_per_map=1,seed=0,
                             augment_shift_count=1))
    assert len(data) in (2,4)
    f,r=data[:2]
    assert f["pair_role"]=="forward" and r["pair_role"]=="reverse"
    assert r["relation_direction"]==INVERSE[f["relation_direction"]]
    assert f["source_anchor_id"]==r["hidden_target_id"]
    assert r["source_anchor_id"]==f["hidden_target_id"]
    assert f["landmarks"][0]["id"]==f["source_anchor_id"]
    target=objects()["test_block_0"][f["hidden_target_id"]]["name"]
    assert target.lower() not in f["instruction"].lower()
    assert all("rgb_observations" not in x and "teacher_future" not in x for x in data)
    if len(data)==4:
        assert f["landmarks"][0]["contour"]!=data[2]["landmarks"][0]["contour"]
    file=tmp_path/"pairs.jsonl"
    file.write_text("".join(json.dumps(x)+"\n" for x in data))
    dataset=PairedGeometryDataset(file,grid_size=4,max_relations=8)
    batch=paired_collate([dataset[0]])
    assert batch["target_xy"].shape==(2,2)


def test_p1_reverse_loss_has_gradients_and_finite_values(tmp_path):
    data=list(generate_pairs(objects(),bounds(),{"test_block_0"},
                             max_objects_per_map=3,max_pairs_per_map=1,seed=0))
    file=tmp_path/"pairs.jsonl"
    file.write_text("".join(json.dumps(x)+"\n" for x in data))
    batch=paired_collate([PairedGeometryDataset(file,grid_size=4)[0]])
    class FakeEncoder:
        def encode_tokens(self,text):
            return torch.ones(len(text),7,768),torch.ones(len(text),7,dtype=torch.bool)
        def encode(self,text):
            return torch.zeros(len(text),768)
    from graphnav.full_localization import FullTokenGeometryLocalizer
    torch.manual_seed(1)
    torch.set_num_threads(2)
    model=FullTokenGeometryLocalizer(
        hidden=32,heads=4,max_relations=8,relative_position=True,dropout=0
    )
    loss,components,_=batch_loss(model,batch,FakeEncoder(),torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.geometry_proj[0].weight.grad is not None
    assert components["inverse"]>=0


def test_p2_needs_distinct_observations_and_uses_only_patches(tmp_path):
    image=np.full((64,64,3),140,np.uint8)
    paths=[]
    for i in range(3):
        path=tmp_path/f"{i}.png"
        cv2.imwrite(str(path),image)
        paths.append(path)
    rois=tmp_path/"rois.jsonl"
    entries=[]
    for obj_id in ("1","2"):
        for i,path in enumerate(paths):
            entries.append(dict(map_name="test_block_0",object_id=obj_id,
                object_type="Building",name=f"Building {obj_id}",
                observed_step=i,observed_frame=f"{obj_id}_{i}.png",
                roi_path=str(path),visible_fraction=.9,
                annotation_kind="geometry_projected_roi_not_verified_segmentation"))
    rois.write_text("".join(json.dumps(x)+"\n" for x in entries))
    ds=PairedROIs(rois)
    assert len(ds)==2
    batch=roi_collate([ds[0],ds[1]])
    assert len(batch["first"])==2
    assert batch["source_ids"][0]!=batch["source_ids"][1]
    model=VisualLandmarkAligner(vision_dim=24,text_dim=16,hidden=32,heads=4)
    p=torch.randn(2,16,24)
    valid=torch.ones(2,16,dtype=torch.bool)
    a=model(p,valid)
    b=model(p+torch.randn_like(p)*.01,valid)
    loss,_=match_loss(model,a,b,batch["type"],torch.randn(1,16))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.patch[0].weight.grad.abs().sum()>0



def test_sparse_causal_frame_prefix_cannot_contain_future():
    from scripts.prepare_citynav import convert
    import tempfile
    class FakeRenderer:
        def __init__(self):
            self.calls=[]
        def render(self,map_name,episode_key,step,pose,ground):
            self.calls.append(step)
            return {"step":int(step),"path":f"dummy_{step}.png",
                    "footprint_world":footprint()}
    renderer=FakeRenderer()
    objs={"test_block_0":{
        "t":{"name":"Target","descriptions":["east of Church"],
             "contour":[[200,200],[220,200],[210,210]]},
        "a":{"name":"Church","object_type":"Building",
             "contour":[[90,190],[110,190],[100,210]]}}}
    processed={"test_block_0":{"t":[{"landmarks":["Church"]}]}}
    track=[[10+i*10,10+i*10,80,0.,0.] for i in range(7)]
    episodes=[dict(area="test",block=0,object_ids=["t"],ann_ids=[0],
                   target_positions=[[210,210,80]],trajectory=track)]
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)/"samples.jsonl"
        count,skipped=convert(
            objs,processed,episodes,bounds(),out,
            stride=2,max_steps=7,ground_levels={"test_block_0":0},
            renderer=renderer,render_observed_only=True
        )
        assert (count,skipped)==(4,0)
        assert renderer.calls==[0,2,4,6]
        rows=[json.loads(s) for s in out.read_text().splitlines()]
        assert [[f["step"] for f in row["rgb_observations"]] for row in rows]==[
            [0],[0,2],[0,2,4],[0,2,4,6]
        ]


def test_frozen_inference_patch_inputs_support_trainable_roi_head():
    from graphnav.landmark_visual_train import encode_twoview
    class FakeFrozen:
        def encode(self,paths):
            with torch.inference_mode():
                return (torch.ones(len(paths),9,24),
                        torch.ones(len(paths),9,dtype=torch.bool))
    model=VisualLandmarkAligner(vision_dim=24,hidden=32,heads=4)
    a,b=encode_twoview(
        {"first":["a","b"],"second":["c","d"]},
        FakeFrozen(),model,torch.device("cpu")
    )
    (a.square().sum()+b.square().sum()).backward()
    assert model.patch[0].weight.grad is not None
