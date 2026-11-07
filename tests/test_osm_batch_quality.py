import numpy as np,pytest
from scripts.summarize_citynav_osm_batch import validated_export
from tools.osm_map.auto_registration import run_job

def test_structure_score_cannot_authorize_export(tmp_path):
 with pytest.raises(ValueError,match='Independent'):
  validated_export({},dict(grade='pending_validation',structure_cost_initial=0,validation=None),tmp_path)

def test_other_scene_validation_cannot_verify_current_scene(tmp_path,monkeypatch):
 import tools.osm_map.auto_registration as ar
 from rasterio.warp import transform
 xy=np.array([[i%5*40.,i//5*50.] for i in range(15)]);en=xy+[545000,258000];lon,lat=transform('EPSG:27700','EPSG:4326',en[:,0],en[:,1]);points=[dict(id=str(i),x=float(p[0]),y=float(p[1]),lon=lon[i],lat=lat[i],map_name='cambridge_block_3',split='fit' if i<10 else 'validation') for i,p in enumerate(xy)]
 M=[[1,0,545000],[0,1,258000]]
 monkeypatch.setattr(ar,'estimate',lambda *a:({'matrix':M,'initial_matrix':M},np.zeros((2,2,3),np.uint8),np.zeros((2,2),np.uint8),np.zeros((2,2),np.uint8)))
 monkeypatch.setattr(ar,'render_preview',lambda *a:None)
 r=run_job(tmp_path,'cambridge','cambridge_block_4',dict(points=points,revision=0),tmp_path/'result')
 assert r['validation'] is None and not r['verified'] and r['per_map_validation']=={}


def test_cityrefer_traffic_road_is_not_compared_to_buildings():
 from scripts.summarize_citynav_osm_batch import public_polygon_kind
 assert public_polygon_kind('TrafficRoad')=='road'
 assert public_polygon_kind('Road')=='road'
 assert public_polygon_kind('Building')=='building'
 assert public_polygon_kind('Car') is None

def test_structural_selection_and_independent_grading_are_separate():
 from tools.osm_map.batch_policy import select_public_transform,grade_independent
 initial=[[1,0,0],[0,1,0]];local=[[1,0,40],[0,1,0]]
 r=select_public_transform(initial,local,[0,0,400,400],10,1,2500,20)
 assert r['selected']=='city_uniform' and 'unstable_local_correction' in r['risk_flags']
 assert grade_independent(None,[],10)=='pending_validation'
 assert grade_independent({'n':5,'max_m':4},[],10)=='verified'
 assert grade_independent({'n':5,'max_m':6},[],10)=='unreliable'
 assert grade_independent({'n':1,'max_m':0},[],10)=='pending_validation'


def test_web_policy_matches_frozen_real_batch():
 from pathlib import Path
 import json,rasterio
 from tools.osm_map.batch_policy import select_public_transform,grade_independent
 root=Path('/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/osm_batch34_20261107');data=Path('/home/20261024_1/Workspace/DATA/sbfnav-data')
 if not (root/'summary.json').exists():pytest.skip('Real batch not available')
 rows=json.load(open(root/'summary.json'))['rows'];assert len(rows)==34
 for row in rows:
  r=json.load(open(root/row['map_name']/'result.json'));points=json.load(open(root/(row['city']+'_controls_snapshot.json')))['points']
  with rasterio.open(data/'rgbd'/(row['map_name']+'.tif')) as f:bounds=list(f.bounds)
  pred=select_public_transform(r['initial_matrix'],r['matrix'],bounds,r['structure_cost_initial'],r['structure_cost_final'],r['edge_samples'],r['building_proxy_polygons'])
  assert pred['selected']==row['selected'];np.testing.assert_allclose(pred['matrix'],row['matrix'])
  assert grade_independent(row['validation'],pred['risk_flags'],sum(p['split']=='fit' for p in points))==row['grade']
