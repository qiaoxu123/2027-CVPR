import numpy as np
from scripts.yolo_world_citynav import tile_windows, iou_matrix, match_boxes, family_nms
from graphnav.landmark_projection import world_to_frame, frame_to_world

def test_tile_coverage():
    for w,h in [(800,800),(2048,1600),(320,400)]:
        seen=np.zeros((h,w),bool)
        for x,y,x2,y2 in tile_windows(w,h): seen[y:y2,x:x2]=1
        assert seen.all()

def test_matching_and_empty():
    p=[[0,0,10,10],[0,0,10,10]];g=[[0,0,10,10]]
    assert len(match_boxes(p,g,.5))==1
    assert match_boxes([],g,.5)==[]
    assert iou_matrix(p,g).shape==(2,1)

def test_family_nms():
    rows=[dict(box=[0,0,10,10],score=.9,label='car'),dict(box=[0,0,10,10],score=.8,label='white car'),dict(box=[0,0,10,10],score=.7,label='building')]
    assert len(family_nms(rows))==2

def test_world_roundtrip():
    f=np.array([[10,10],[10,30],[30,30],[30,10]],float);p=np.array([[15,18],[22,26]],float)
    assert np.max(np.abs(frame_to_world(world_to_frame(p,f,(800,800)),f,(800,800))-p))<1e-3

def test_center_proxy_not_iou_truth():
    from scripts.yolo_world_proxy_diagnostics import center_association
    d=[dict(label='car',box=[4,4,6,6]),dict(label='car',box=[4,4,6,6])]
    g=[dict(eligible=True,category='vehicle',contours=[[[0,0],[10,0],[10,10],[0,10]]],box=[0,0,10,10])]
    z=center_association(d,g,'vehicle')
    assert z['associated']==1
    assert z['proxy_to_detection_bbox_area_ratios']==[25]
    assert match_boxes([x['box'] for x in d],[g[0]['box']],.5)==[]

def test_maximum_cardinality_before_iou(monkeypatch):
    import scripts.yolo_world_citynav as m
    # Greedy or plain max-sum IoU would prefer .99 and lose the second pair.
    monkeypatch.setattr(m,'iou_matrix',lambda a,b:np.array([[.99,.50],[.50,.49]]))
    assert len(m.match_boxes([1,2],[1,2],.5))==2

def test_no_cross_family_suppression():
    rows=[dict(box=[0,0,10,10],score=.9,label='car'),dict(box=[0,0,10,10],score=.8,label='parking lot')]
    assert len(family_nms(rows))==2

def test_resume_rejects_changed_weights(tmp_path):
    import pytest
    from scripts.yolo_world_citynav import ensure_config
    p=tmp_path/'config.json'
    ensure_config(p,{'weights_sha256':'a'})
    ensure_config(p,{'weights_sha256':'a'})
    with pytest.raises(ValueError,match='weights_sha256'):
        ensure_config(p,{'weights_sha256':'b'})
