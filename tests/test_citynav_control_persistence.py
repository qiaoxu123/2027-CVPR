import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from tools.osm_map.control_app import build
DATA=Path('/home/20261024_1/Workspace/DATA/sbfnav-data')
@pytest.mark.skipif(not DATA.exists(),reason='Requires real CityNav map catalog')
def test_authenticated_persistence_and_conflict(tmp_path):
 app=build(DATA,tmp_path/'overlays',tmp_path/'controls','tester','long-test-password-42');c=TestClient(app)
 assert c.get('/api/controls/cambridge').status_code==401
 auth=('tester','long-test-password-42');h={'X-Control-Write':'1'}
 payload={'revision':0,'points':[],'mode':'similarity'}
 assert c.post('/api/controls/cambridge',auth=auth,json=payload).status_code==403
 r=c.post('/api/controls/cambridge',auth=auth,headers=h,json=payload);assert r.status_code==200 and r.json()['revision']==1
 assert c.get('/api/controls/cambridge',auth=auth).json()['revision']==1
 assert c.post('/api/controls/cambridge',auth=auth,headers=h,json=payload).status_code==409
 c2=TestClient(build(DATA,tmp_path/'overlays',tmp_path/'controls','tester','long-test-password-42'))
 assert c2.get('/api/controls/cambridge/export',auth=auth).json()['revision']==1

@pytest.mark.skipif(not DATA.exists(),reason='Requires real CityNav map catalog')
def test_fit_report_and_corners_are_persisted(tmp_path):
 import numpy as np
 from rasterio.warp import transform
 xy=np.array([[i%5*40.,i//5*50.+1300] for i in range(15)]);en=xy+np.array([545000.,258000.]);lon,lat=transform('EPSG:27700','EPSG:4326',en[:,0],en[:,1]);points=[dict(id=str(i),x=float(p[0]),y=float(p[1]),lon=lon[i],lat=lat[i],map_name='cambridge_block_3',split='fit' if i<10 else 'validation',source='public_building_corner') for i,p in enumerate(xy)]
 c=TestClient(build(DATA,tmp_path/'overlays',tmp_path/'controls','tester','long-test-password-42'));auth=('tester','long-test-password-42');r=c.post('/api/controls/cambridge',auth=auth,headers={'X-Control-Write':'1'},json=dict(revision=0,mode='similarity',points=points));assert r.status_code==200 and r.json()['registration']['verified'];assert c.get('/api/project-corners/cambridge/cambridge_block_3',auth=auth).status_code==200
 assert c.get('/api/controls/cambridge/export',auth=auth).json()['registration']['validation_n']==5

@pytest.mark.skipif(not DATA.exists(),reason='Requires real CityNav map catalog')
def test_auto_accept_reject_preserves_controls_and_rejects_stale(tmp_path):
 import json
 store=tmp_path/'controls';c=TestClient(build(DATA,tmp_path/'overlays',store,'tester','long-test-password-42'));auth=('tester','long-test-password-42');h={'X-Control-Write':'1'}
 job=store/'auto_proposals'/'cambridge_block_3';job.mkdir(parents=True)
 proposal=dict(matrix=[[1,0,545000],[0,1,258000]],base_revision=0,map_name='cambridge_block_3',status='proposal',verified=False,per_map_validation={})
 (job/'result.json').write_text(json.dumps(proposal))
 r=c.post('/api/auto/cambridge/cambridge_block_3/accept',auth=auth,headers=h);assert r.status_code==200 and not r.json()['verified']
 current=c.get('/api/controls/cambridge',auth=auth).json();assert current['points']==[] and current['revision']==1
 assert c.get('/api/project-corners/cambridge/cambridge_block_4',auth=auth).status_code==400  # Scene-specific correction must not leak to another scene.
 assert c.post('/api/auto/cambridge/cambridge_block_3/accept',auth=auth,headers=h).status_code==409
 r=c.post('/api/auto/cambridge/cambridge_block_3/reject',auth=auth,headers=h);assert r.status_code==200
 current=c.get('/api/controls/cambridge',auth=auth).json();assert current['points']==[] and 'active_registration' not in current

@pytest.mark.skipif(not DATA.exists(),reason='Requires real CityNav map catalog')
def test_multiple_scene_proposals_accept_with_unchanged_controls(tmp_path):
 import json,hashlib
 store=tmp_path/'controls';c=TestClient(build(DATA,tmp_path/'overlays',store,'tester','long-test-password-42'));auth=('tester','long-test-password-42');h={'X-Control-Write':'1'}
 for n in ['cambridge_block_3','cambridge_block_4']:
  job=store/'auto_proposals'/n;job.mkdir(parents=True)
  proposal=dict(matrix=[[1,0,545000],[0,1,258000]],base_revision=0,map_name=n,status='proposal',verified=False,per_map_validation={},controls_fingerprint=hashlib.sha256(b'[]').hexdigest())
  (job/'result.json').write_text(json.dumps(proposal))
  preview=c.get('/api/project-corners/cambridge/'+n+'?preview=auto',auth=auth);assert preview.status_code==200 and not preview.json()['verified']
  assert c.post('/api/auto/cambridge/'+n+'/accept',auth=auth,headers=h).status_code==200
 current=c.get('/api/controls/cambridge',auth=auth).json();assert len(current['scene_registrations'])==2 and current['points']==[]
 # A changed public control invalidates all old proposals even if a caller knows the current revision.
 current['points']=[dict(id='new',x=1,y=2,lon=.1,lat=52.2,map_name='cambridge_block_3',split='fit',source='public_building_corner')]
 (store/'cambridge.json').write_text(json.dumps(current))
 assert c.post('/api/auto/cambridge/cambridge_block_3/accept',auth=auth,headers=h).status_code==409
 exports=store/'validated_osm_exports';exports.mkdir();(exports/'cambridge_block_3.json').write_text('{}')
 assert c.get('/api/osm-export/cambridge_block_3',auth=auth).status_code==403
