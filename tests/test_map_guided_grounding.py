import inspect
import numpy as np
import torch
from graphnav.map_guided_grounding import (project_polygon,permute_visual_correspondence,
    geometry_scores,score_candidates,registered_instance_evaluation)

FOOTPRINT=np.array([[0,100],[100,100],[100,0],[0,0]],np.float32)


def test_full_partial_outside_and_invalid_depth_are_distinct():
    depth=np.ones((101,101),bool)
    full=project_polygon([[20,20],[40,20],[40,40],[20,40]],FOOTPRINT,depth.shape,depth)
    partial=project_polygon([[-20,20],[40,20],[40,40],[-20,40]],FOOTPRINT,depth.shape,depth)
    outside=project_polygon([[-40,20],[-20,20],[-20,40],[-40,40]],FOOTPRINT,depth.shape,depth)
    assert full['status']=='full' and full['valid_roi']
    assert partial['status']=='partial' and abs(partial['coverage']-2/3)<1e-6
    assert outside['status']=='outside' and not outside['valid_roi']
    invalid=project_polygon([[20,20],[40,20],[40,40],[20,40]],FOOTPRINT,depth.shape,np.zeros_like(depth))
    assert invalid['status']=='full' and not invalid['valid_roi']
    assert full['roundtrip_error_m']<1e-4


def test_inference_has_no_private_target_or_instance_label_argument():
    for f in [geometry_scores,score_candidates,permute_visual_correspondence]:
        assert not any('target_world' in k or 'target_contour' in k or 'object_id' in k for k in inspect.signature(f).parameters)


def test_shuffle_only_changes_correspondence_and_uses_no_other_frame():
    target=torch.arange(12).reshape(3,4).float();refs=torch.arange(4).reshape(1,4).float()+100
    valid=torch.tensor([True,True,False]);before=target.clone()
    t,r,p=permute_visual_correspondence(target,refs,valid)
    assert torch.equal(target,before) and torch.equal(t[2],target[2])
    assert torch.equal(t[0],refs[0]) and torch.equal(r[0],target[0])
    assert sorted(p.tolist())==list(range(3))


def test_instance_match_uses_iou_not_nearby_world_center():
    depth=np.ones((101,101),bool)
    contour=[[20,20],[40,20],[40,40],[20,40]]
    # Wrong tiny box can have exactly the correct center but near-zero IoU.
    boxes=torch.tensor([[.30,.70,.01,.01],[.30,.70,.21,.21]])
    result=registered_instance_evaluation(contour,FOOTPRINT,depth.shape,depth,boxes,torch.tensor([2.,1.]))
    assert result['eligible'] and not result['top1'] and result['top5']


def test_citynav_yaw_zero_maps_east_up_and_north_left():
    from graphnav.memory import ObservedRGBMemory
    from graphnav.landmark_projection import world_to_frame
    footprint=ObservedRGBMemory.view_corners([0,0],0,50)
    pixels=world_to_frame(np.array([[25,0],[0,25]],np.float32),footprint,(101,101))
    assert np.allclose(pixels,[[50,25],[25,50]])


def test_shuffled_control_preserves_map_only_scores_and_fixed_geometry():
    prior=torch.tensor([0.,-1.,-2.]);fields=torch.empty(3,0);features=torch.eye(3)
    valid=torch.ones(3,dtype=torch.bool);lang=torch.tensor([1.,2.,3.])
    empty=torch.empty(0,3)
    map0,rgb0,_=score_candidates(prior,fields,[],features,lang,empty,empty,[],valid)
    changed,ref,_=permute_visual_correspondence(features,empty,valid)
    map1,rgb1,_=score_candidates(prior,fields,[],changed,lang,ref,empty,[],valid)
    assert torch.equal(map0,map1) and not torch.equal(rgb0,rgb1)
