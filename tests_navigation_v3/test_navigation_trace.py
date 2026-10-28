import copy,base64,zlib
import numpy as np
import pytest
import torch
from experiments.a2_waldo_landmark_belief.relation_v3_navigation_trace import save_belief,validate_observed_artifacts,latency_summary
from experiments.a2_waldo_landmark_belief.relation_v3_confirmation import save_native_rgb
from experiments.a2_waldo_landmark_belief.revision_eval_protocol import sensor_payload
from experiments.a2_waldo_landmark_belief.paths import sha
from graphnav.geometry import grid_centers
from graphnav.memory import ObservedRGBMemory


def validate(row,directory):
    return validate_observed_artifacts(row,directory,bounds=(0.,0.,1000.,1000.),ground=0.)


def record(tmp_path):
    world=(grid_centers(28)*1000)[None]
    logits=torch.arange(784).float()[None];ids=torch.arange(783,767,-1)[None]
    captured=dict(static_logits=logits,belief_logits=logits,world=world,static_ids=ids,belief_ids=ids)
    belief=save_belief(captured,tmp_path,index=3,step=0,signature='s')
    rgb=save_native_rgb(np.ones((384,384,3),dtype=np.uint8),tmp_path)
    points=world[0,ids[0]].tolist()
    sensor=sensor_payload(dict(bgr=np.ones((384,384,3),dtype=np.uint8),footprint_world=ObservedRGBMemory.view_corners([0.,0.],0.,50.)),[])
    row=dict(index=3,signature_sha256='s',inference_seconds=[.1],decision_seconds=[.2],poses=[[0.,0.,50.,0.]],
        step_diagnostics=[dict(observation_step=0,static_top16_world_xy=points,belief_top16_world_xy=points)],
        observations=[dict(sensor,native_rgb=rgb,native_belief=belief)])
    return row


def test_observed_rgb_and_native_beliefs_are_bound_to_exact_episode_step(tmp_path):
    row=record(tmp_path);validate(row,tmp_path)
    bad=copy.deepcopy(row);bad['index']=4
    with pytest.raises(ValueError):validate(bad,tmp_path)
    bad=copy.deepcopy(row);bad['step_diagnostics'][0]['belief_top16_world_xy'][0][0]+=1
    with pytest.raises(ValueError):validate(bad,tmp_path)


def test_rgb_or_belief_mutation_and_path_escape_are_rejected(tmp_path):
    row=record(tmp_path);bad=copy.deepcopy(row)
    bad['observations'][0]['native_belief']['filename']='../elsewhere.pt'
    with pytest.raises(ValueError):validate(bad,tmp_path)
    path=tmp_path/row['observations'][0]['native_belief']['filename'];path.write_bytes(b'changed')
    with pytest.raises(ValueError):validate(row,tmp_path)


def test_self_consistent_forged_world_grid_still_fails_original_public_bounds(tmp_path):
    row=record(tmp_path);path=tmp_path/row['observations'][0]['native_belief']['filename']
    data=torch.load(path,weights_only=True);data['world'].zero_();torch.save(data,path)
    row['observations'][0]['native_belief']['sha256']=sha(path)
    for key in ('static_top16_world_xy','belief_top16_world_xy'):row['step_diagnostics'][0][key]=[[0.,0.]]*16
    with pytest.raises(ValueError,match='grid'):validate(row,tmp_path)


def test_rgb_bitmap_and_pose_footprint_cannot_be_changed_independently(tmp_path):
    row=record(tmp_path);bad=copy.deepcopy(row)
    bad['observations'][0]['valid_bitmap']=base64.b64encode(zlib.compress(bytes(384*384//8))).decode()
    with pytest.raises(ValueError,match='bitmap'):validate(bad,tmp_path)
    bad=copy.deepcopy(row);bad['observations'][0]['footprint_world']=[[0.,0.]]*4
    with pytest.raises(ValueError,match='footprint'):validate(bad,tmp_path)


def test_actual_tied_gpu_selection_is_not_replaced_by_cpu_topk_order(tmp_path):
    row=record(tmp_path);path=tmp_path/row['observations'][0]['native_belief']['filename']
    data=torch.load(path,weights_only=True)
    for name in ('static','belief'):
        data[name+'_logits'].zero_();data[name+'_ids']=torch.arange(16)[None]
        row['step_diagnostics'][0][name+'_top16_world_xy']=data['world'][0,:16].tolist()
    torch.save(data,path);row['observations'][0]['native_belief']['sha256']=sha(path)
    validate(row,tmp_path)


def test_latency_uses_every_synchronized_predict_and_excludes_trace_io():
    rows=[dict(inference_seconds=[.01,.03]),dict(inference_seconds=[.02])]
    result=latency_summary(rows)
    assert result['decisions']==3 and result['mean_ms']==pytest.approx(20.)
    assert result['median_ms']==pytest.approx(20.)
    with pytest.raises(ValueError):latency_summary([dict(inference_seconds=[float('nan')])])
