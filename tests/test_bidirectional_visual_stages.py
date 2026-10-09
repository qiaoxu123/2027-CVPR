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
    PairedGeometryDataset, paired_collate, batch_loss, family_split
)
from graphnav.landmark_visual_pretrain import (
    PairedROIs, VisualLandmarkAligner, roi_collate, match_loss, SameMapTypeBatchSampler
)


def test_p1_family_split_never_leaks_shifted_pair_to_dev():
    from types import SimpleNamespace
    records = [{"pair_id": f"map:a{i}:b{i}{suffix}"}
               for i in range(20) for suffix in ("", ":aug0", ":aug1")]
    dataset = SimpleNamespace(samples=SimpleNamespace(records=records),
                              pairs=[(i, i) for i in range(len(records))])
    train, dev = family_split(dataset)
    families = lambda ids: {records[i]["pair_id"].split(":aug")[0] for i in ids}
    assert not families(train) & families(dev)
    assert sorted(train+dev) == list(range(len(records)))


def test_p1_inverse_metric_uses_pair_denominator_and_reports_meters(monkeypatch):
    from types import SimpleNamespace
    from graphnav.bidirectional_train import evaluate
    batches=[]
    for n in (2,4):
        batches.append({'text':['query']*n,'grid_xy':torch.full((n,1,2),.5),
            'target_xy':torch.zeros(n,2),'map_sizes':torch.full((n,2),400.),
            'nodes':torch.zeros(n,1,2),'direction_label':torch.zeros(n,dtype=torch.long)})
    def fake_loss(model,batch,encoder,device):
        n=len(batch['text'])
        return None,{'inverse':1. if n==2 else 3.},SimpleNamespace(belief_logits=torch.zeros(n,1))
    monkeypatch.setattr('graphnav.bidirectional_train.batch_loss',fake_loss)
    metric=evaluate(SimpleNamespace(eval=lambda:None),batches,None,torch.device('cpu'))
    assert metric['samples']==6 and metric['inverse_error']==pytest.approx(7/3)
    assert metric['inverse_displacement_error_m']==pytest.approx(400*2**.5)


def test_p2_hard_batches_do_not_mix_city_or_category():
    from types import SimpleNamespace
    groups = [((f"city{city}_block_{i}", str(i)), [{"object_type": typ}])
              for city in (0, 1) for typ in ("Building", "Road") for i in range(5)]
    dataset = SimpleNamespace(groups=groups)
    batches = list(SameMapTypeBatchSampler(dataset, range(len(groups)), batch_size=4))
    for batch in batches:
        labels = {(groups[i][0][0].split("_block_")[0], groups[i][1][0]["object_type"])
                  for i in batch}
        assert len(labels) == 1


def test_p2_rejects_future_roi_metadata(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"observed_step": 8, "observation_cutoff_step": 0}) + "\n")
    with pytest.raises(ValueError, match="future"):
        PairedROIs(path)


def test_p0_quality_filter_measures_rgb_holes_not_semantic_iou():
    from scripts.curate_p0_rois import pixel_support
    image = np.full((32,32,3),120,np.uint8)
    poly = [[0,0],[31,0],[31,31],[0,31]]
    assert pixel_support(image,poly) == 1
    image[:16] = 0
    assert pixel_support(image,poly) == .5


def test_invalid_surface_excluded_from_p0_and_p1():
    from graphnav.semantic_geometry import valid_surface_polygon
    from graphnav.bidirectional import source_landmark
    contour=[[0,0],[40,20],[0,20],[40,0],[50,50]]
    assert not valid_surface_polygon(contour)
    bad={"id":"bad","name":"bad","object_type":"Building","contour":contour}
    assert project_landmark(bad,footprint(),(384,384,3)) is None
    assert source_landmark('bad',bad,np.asarray([0,0,400,400])) is None


def test_p2_identical_pose_with_different_paths_is_not_cross_view(tmp_path):
    row = dict(map_name="map",object_id="1",object_type="Building",roi_path="roi.png",
               visible_fraction=1,footprint_world=footprint(),
               annotation_kind="geometry_projected_roi_not_verified_segmentation")
    path=tmp_path/"rows.jsonl"
    path.write_text("".join(json.dumps({**row,"observed_frame":f"episode{i}.png"})+"\n"
                            for i in range(2)))
    with pytest.raises(ValueError,match="DIFFERENT"):
        PairedROIs(path)


def test_patch_disk_cache_and_small_lru_keep_all_requested_inputs(tmp_path):
    from collections import OrderedDict
    from types import SimpleNamespace
    from graphnav.landmark_visual_pretrain import FrozenSigLIP2Patches
    class Vision:
        def forward(self,pixel_values):
            return SimpleNamespace(last_hidden_state=torch.ones(len(pixel_values),9,4))
        __call__=forward
    def fake():
        obj=FrozenSigLIP2Patches.__new__(FrozenSigLIP2Patches)
        obj.model=SimpleNamespace(vision_model=Vision())
        obj.processor=lambda images,return_tensors: {'pixel_values':torch.ones(len(images),3,32,32)}
        obj.device=torch.device('cpu')
        obj.cache=OrderedDict();obj.cache_limit=1;obj.microbatch=2;obj.hidden_size=4
        obj.cache_directory=tmp_path;obj.protocol=('test','revision','fp32')
        obj.encoded_images=0;obj.disk_hits=0
        return obj
    paths=[]
    for i in range(2):
        path=tmp_path/f'image{i}.png';cv2.imwrite(str(path),np.full((32,32,3),127,np.uint8))
        paths.append(str(path))
    first=fake();tokens,valid=first.encode(paths)
    assert tokens.shape==(2,9,4) and valid.all() and len(first.cache)==1
    second=fake();again,_=second.encode(paths)
    assert torch.equal(tokens,again) and second.disk_hits==2 and second.encoded_images==0


def test_patch_validity_excludes_letterbox_without_polygon_mask():
    from graphnav.landmark_visual_pretrain import observed_patch_mask
    image=np.full((32,32,3),127,np.uint8)
    image[:,16:] = 0
    assert observed_patch_mask([image],(32,32),16).tolist()==[[True,False,True,False]]


def test_retrieval_handles_pin_memory_list_ids_and_reports_categories():
    from types import SimpleNamespace
    from graphnav.landmark_visual_pretrain import retrieval_metrics
    class Model(torch.nn.Module):
        def forward(self,patches,valid):return patches[:,0]
    class Encoder:
        def encode(self,paths):return torch.eye(2)[:,None],torch.ones(2,1,dtype=torch.bool)
    class Loader(list):pass
    loader=Loader([{'first':['a','b'],'second':['aa','bb'],
                    'source_ids':[['city_block_0','1'],['city_block_1','2']]}])
    loader.dataset=SimpleNamespace(groups=[(('city_block_0','1'),[{'object_type':'Building'}]),
                                          (('city_block_1','2'),[{'object_type':'Building'}])])
    result=retrieval_metrics(Model(),loader,Encoder(),torch.device('cpu'))
    assert result['R@1']==1 and result['by_type']['Building']['queries']==2
    assert result['same_city_type_hard']['queries']==2


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
