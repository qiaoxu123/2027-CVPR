import copy
import json
import numpy as np
from shapely.geometry import box, mapping
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tools.osm_map.landmark_groups import grouped_fit
from tools.osm_map.landmark_review import install


def row(i, x, token=None):
    return dict(match_id=str(i), high_confidence=True, physical_tokens=[token or str(i)], source_polygon=mapping(box(x, 0, x+1, 1)), osm_polygon_en=mapping(box(x+100, 200, x+101, 201)), source_anchor_xy=[x+.5,.5], osm_anchor_en=[x+100.5,200.5], kind='building', target_entity=str(i), confidence=.9)


def test_touching_building_parts_and_shared_road_do_not_cross_groups():
    rows=[row(0,0),row(1,1),row(2,20,'road:one'),row(3,40,'road:one')]
    grouped_fit(rows, [[1,0,100],[0,1,200]])
    assert rows[0]['independence_group']==rows[1]['independence_group']
    assert rows[2]['independence_group']==rows[3]['independence_group']
    assert rows[0]['split']==rows[1]['split']


def test_validation_values_never_change_fit():
    rows=[row(i,10*i) for i in range(30)]
    a,_,_,val=grouped_fit(rows,[[1,0,100],[0,1,200]])
    assert val
    modified=copy.deepcopy(rows)
    # Changing an evaluated anchor does not change physical grouping or fit data.
    for r in modified:
        if r['split']=='validation':r['osm_anchor_en']=[999999.,999999.]
    b,_,_,_=grouped_fit(modified,[[1,0,100],[0,1,200]])
    np.testing.assert_allclose(a,b)


def test_review_persistence_keeps_split_and_never_approves_transform(tmp_path):
    r=dict(row(0,0),city='cambridge',map_name='cambridge_block_3',split='validation',independence_group='building0')
    root=tmp_path/'landmark_matches';root.mkdir();(root/'matches.json').write_text(json.dumps([r]));(root/'summary.json').write_text(json.dumps({'cities':{'cambridge':{'matrix':[[1,0,544000],[0,1,257000]]}}}))
    app=FastAPI();install(app,tmp_path);c=TestClient(app)
    assert c.get('/api/landmark-matches').json()['counts']['pending']==1
    z=c.post('/api/landmark-matches/0/review',json={'decision':'confirmed','note':'unit test'})
    assert z.status_code==200 and not z.json()['geographic_validation_approved']
    app2=FastAPI();install(app2,tmp_path);c2=TestClient(app2)
    d=c2.get('/api/landmark-matches/0').json()
    assert d['review_status']=='confirmed' and d['split']=='validation'
    assert json.loads((root/'matches.json').read_text())[0]==json.loads(json.dumps(r))
    assert c2.post('/api/landmark-matches/0/review',json={'decision':'approve_all'}).status_code==400


def test_spatial_nms_uses_prediction_order_and_world_distance_only():
    from scripts.evaluate_v2_public_candidate_recall import spatial_nms
    xy=np.array([[0.,0.],[1.,1.],[20.,0.],[40.,0.],[40.,1.]])
    assert spatial_nms([1,0,2,3,4],xy,radius=20,k=5)==[1,3]
    assert spatial_nms([0,1,2,3,4],xy,radius=20,k=5)==[0,2,3]
    assert spatial_nms([],xy)==[]
