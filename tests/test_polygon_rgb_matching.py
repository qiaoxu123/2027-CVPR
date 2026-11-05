import cv2
import numpy as np
from scripts.polygon_rgb_matching import candidate_inputs, image_evidence, explicit_score, rank_credit


def test_full_polygon_change_updates_masks_tokens_and_geometry():
    obs=dict(bgr=np.ones((384,384,3),np.uint8)*120,depth_valid=np.ones((384,384),bool),
             footprint_world=[[0,0],[100,0],[100,100],[0,100]])
    a=candidate_inputs([[[10,10],[40,10],[40,40],[10,40]]],obs)
    b=candidate_inputs([[[55,10],[85,10],[85,40],[55,40]]],obs)
    assert a['eligible'] and b['eligible']
    assert not np.array_equal(a['anchor']['geometry_mask'],b['anchor']['geometry_mask'])
    assert not np.array_equal(a['tokens'],b['tokens'])
    assert not np.array_equal(a['anchor']['labels'],b['anchor']['labels'])


def test_boundary_alignment_synthetic_and_camera_clip_not_fake_edge():
    im=np.zeros((384,384,3),np.uint8)+40;im[80:220,80:220]=200
    ev=image_evidence(im,np.ones((384,384),bool))
    aligned=np.zeros((384,384),bool);aligned[80:220,80:220]=True
    shifted=np.zeros_like(aligned);shifted[105:245,105:245]=True
    assert explicit_score(aligned,ev)['score']>explicit_score(shifted,ev)['score']
    assert explicit_score(np.ones_like(aligned),ev) is None


def test_ties_do_not_guarantee_positive_top1():
    assert rank_credit([1,1,1])['top1']==1/3
    assert rank_credit([.5,.7,.3])['top1']==0
    assert rank_credit([.8,.7,.3])['margin']>0


def test_pure_geometry_score_is_independent_of_rgb_values_and_outside_negative_rejected():
    obs=dict(bgr=np.ones((384,384,3),np.uint8)*100,depth_valid=np.ones((384,384),bool),
             footprint_world=[[0,0],[100,0],[100,100],[0,100]])
    contours=[[[10,10],[40,10],[40,40],[10,40]]]
    a=candidate_inputs(contours,obs)
    b=candidate_inputs(contours,{**obs,'bgr':np.zeros((384,384,3),np.uint8)})
    assert a['geometry_score']==b['geometry_score']
    from scripts.polygon_rgb_matching import comparable
    outside=candidate_inputs([[[200,200],[230,200],[230,230],[200,230]]],obs)
    assert not comparable(a,outside)


def test_union_of_fragments_keeps_interior_hole_in_roi():
    from shapely.geometry import box
    obs=dict(bgr=np.ones((384,384,3),np.uint8)*100,depth_valid=np.ones((384,384),bool),
             footprint_world=[[0,0],[100,0],[100,100],[0,100]])
    pieces=[box(10,10,70,20),box(10,60,70,70),box(10,10,20,70),box(60,10,70,70)]
    ci=candidate_inputs([list(p.exterior.coords) for p in pieces],obs)
    assert not ci['anchor']['geometry_mask'][154,154]
    assert ci['anchor']['geometry_mask'][58,154]
