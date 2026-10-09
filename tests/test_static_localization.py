"""Static localization must be trainable and independent from UAV/RGB."""
import json
import numpy as np
import torch

from graphnav.static_localization import (
    StaticLocalizationDataset, StaticGoalLocalizer, collate_static,
    cross_entropy_goal, static_recall,
)


def _make_record():
    return {
        "episode_id":"cambridge_block_3:1:0:0",
        "map_name":"cambridge_block_3",
        "world_bounds":[0, 0, 400, 400],
        "physical_map_sizes":[400, 400],
        "instruction":"east of St Mary's Church near Main Road",
        "landmarks":[{
            "name":"St Mary's Church", "id":"church",
            "contour":[[.25,.25],[.35,.25],[.35,.35],[.25,.35]]
        }],
        "target_xy":[.75,.32],
        # Must not influence any model-facing ablation tensor:
        "uav_pose":[.05,.05,0.,1.],
        "teacher_future":[[.1,.1],[.2,.2]],
        "rgb_observations":[{"path":"do_not_open_future.png"}],
    }


def test_static_dataset_ignores_future_rgb_and_pose_and_uses_roads(tmp_path):
    original = _make_record()
    inputfile = tmp_path/"static.jsonl"
    inputfile.write_text(json.dumps(original)+"\n",encoding="utf-8")
    roads={"cambridge_block_3":[{"id":"road1","name":"Main Road",
         "polyline":[[100,125],[320,125]],
         "coordinate_system":"world"}]}
    roadsfile=tmp_path/"roads.json"
    roadsfile.write_text(json.dumps(roads),encoding="utf-8")
    d=StaticLocalizationDataset(inputfile,grid_size=4,max_landmarks=2,max_roads=2,
        max_relations=3,roads_json=roadsfile)
    x=d[0]
    assert x["relation_count"]==2
    assert x["nodes"].shape==(7,10)
    assert x["prior"].shape==(16,)
    assert "uav_pose" not in x and "rgb_observations" not in x
    before=x["prior"].clone(),x["rules"].clone(),x["nodes"].clone()
    # Modify forbidden data and reinstantiate. Predictions must stay identical.
    original["uav_pose"]=[.97,.97,1.,0.]
    original["teacher_future"]=[[.9,.9]]
    original["rgb_observations"]=[{"path":"different.png"}]
    inputfile.write_text(json.dumps(original)+"\n",encoding="utf-8")
    y=StaticLocalizationDataset(inputfile,grid_size=4,max_landmarks=2,max_roads=2,
        max_relations=3,roads_json=roadsfile)[0]
    for a,b in zip(before,(y["prior"],y["rules"],y["nodes"])):
        assert torch.allclose(a,b)


def test_nested_ablation_is_trainable_and_uses_same_784_candidates(tmp_path):
    torch.set_num_threads(2)
    rec=_make_record()
    file=tmp_path/"one.jsonl"
    file.write_text(json.dumps(rec)+"\n",encoding="utf-8")
    d=StaticLocalizationDataset(file,grid_size=4,max_landmarks=2,max_roads=2,max_relations=3)
    batch=collate_static([d[0],d[0]])
    model=StaticGoalLocalizer(hidden=32,text_dim=16,heads=4,relations=3,dropout=0)
    prior=model(batch,variant="prior")
    rules=model(batch,variant="rules")
    assert torch.equal(rules, prior+batch["rules"])
    assert prior.shape==(2,16)
    with torch.no_grad():
        inst=torch.randn(2,16)
        names=torch.randn(2,batch["nodes"].shape[1],16)
    output=model(batch,variant="gat",instruction_emb=inst,node_emb=names)
    assert output.shape==(2,16) and torch.isfinite(output).all()
    loss=cross_entropy_goal(output,batch["target_xy"],grid_size=4)
    loss.backward()
    assert model.gnn.layer1.edge.weight.grad is not None
    assert model.graph_head[-1].weight.grad is not None
    metrics=static_recall(prior,batch["target_xy"],batch["map_sizes"],batch["grid_xy"])
    assert metrics["count"]==2
    assert metrics["R@1/20m_count"]<=metrics["R@4/20m_count"]<=metrics["R@8/20m_count"]


def test_north_south_rule_changes_even_when_reference_prior_identical(tmp_path):
    a=_make_record()
    b=dict(a,instruction="north of St Mary's Church")
    c=dict(a,instruction="south of St Mary's Church")
    file=tmp_path/"northsouth.jsonl"
    file.write_text("\n".join(json.dumps(x) for x in (b,c))+"\n",encoding="utf-8")
    d=StaticLocalizationDataset(file,grid_size=8,max_landmarks=2,max_roads=2,max_relations=2)
    north,south=d[0],d[1]
    assert torch.allclose(north["prior"],south["prior"])
    assert not torch.allclose(north["rules"],south["rules"])
    northmost=8*7+4
    southmost=4
    assert north["rules"][northmost]>north["rules"][southmost]
    assert south["rules"][northmost]<south["rules"][southmost]


def test_static_query_deduplicates_human_trajectory_prefixes(tmp_path):
    record=_make_record()
    prefix=dict(record,episode_id="city_block_0:7:2:0")
    later=dict(record,episode_id="city_block_0:7:2:4",
               uav_pose=[.7,.8,1,0])
    filename=tmp_path/"dupe.jsonl"
    filename.write_text(
        json.dumps(prefix)+"\n"+json.dumps(later)+"\n",encoding="utf-8"
    )
    dataset=StaticLocalizationDataset(filename,grid_size=4,
                                      max_landmarks=2,max_roads=2,max_relations=2)
    assert len(dataset)==1


def test_forward_batch_strips_ground_truth_from_model_inputs(tmp_path):
    from graphnav.static_experiment import forward_batch

    record=_make_record()
    filename=tmp_path/"one.jsonl"
    filename.write_text(json.dumps(record)+"\n",encoding="utf-8")
    ds=StaticLocalizationDataset(filename,grid_size=4,max_landmarks=2,
                                 max_roads=2,max_relations=2)
    batch=collate_static([ds[0]])
    class Guard(torch.nn.Module):
        def forward(self, kwargs, **other):
            assert "target_xy" not in kwargs
            assert "map_sizes" not in kwargs
            return kwargs["prior"] + kwargs["rules"]
    pred, labels=forward_batch(
        Guard(),None,batch,torch.device("cpu"),"rules",False
    )
    assert pred.shape==(1,16)
    assert "target_xy" in labels
