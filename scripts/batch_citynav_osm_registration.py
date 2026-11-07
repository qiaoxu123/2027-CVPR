"""Resumable 34-scene public structural registration. Independent grading is map-specific."""
import argparse,json,time,hashlib,traceback
from pathlib import Path
import numpy as np
from tools.osm_map.auto_registration import run_job,metrics
from graphnav.checkpointing import atomic_json
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--controls',type=Path,required=True);p.add_argument('--seeds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--proxy');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
# Frozen public-only selection; never choose a transform by heldout error.
policy=dict(local_min_relative_structure_improvement=.10,local_max_corner_shift_m=15,grade_max_m=5,precise_max_m=2,seed=42)
manifest=dict(policy=policy,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),optimizer_sha256=hashlib.sha256(Path('tools/osm_map/auto_registration.py').read_bytes()).hexdigest(),n=34,split='public scene registration; no task labels',started_at=time.time())
if not (a.output/'manifest.json').exists():atomic_json(manifest,a.output/'manifest.json')
controls={};seeds={}
for city in ['cambridge','birmingham']:
 snap=a.output/(city+'_controls_snapshot.json')
 if not snap.exists():atomic_json(json.loads((a.controls/(city+'.json')).read_text()),snap)
 controls[city]=json.loads(snap.read_text());seed=a.output/(city+'_city_seed.json')
 if not seed.exists():atomic_json(json.loads((a.seeds/city/'result.json').read_text()),seed)
 seeds[city]=json.loads(seed.read_text())
paths=sorted((a.data/'rgbd').glob('*.tif'));assert len(paths)==34
for i,path in enumerate(paths):
 name=path.stem;city=name.split('_')[0];dest=a.output/name;dest.mkdir(exist_ok=True);status=dest/'batch_status.json'
 if status.exists():print('resume',name,flush=True);continue
 t=time.perf_counter()
 try:
  import rasterio
  r=run_job(a.data,city,name,controls[city],dest,a.proxy,seeds[city]['matrix'])
  M=np.array(r['matrix']);M0=np.array(seeds[city]['matrix'])
  with rasterio.open(path) as f:b=f.bounds
  corners=np.array([[b.left,b.bottom],[b.right,b.bottom],[b.right,b.top],[b.left,b.top]]);shift=float(np.linalg.norm(corners@M[:,:2].T+M[:,2]-(corners@M0[:,:2].T+M0[:,2]),axis=1).max());improvement=(r['structure_cost_initial']-r['structure_cost_final'])/max(r['structure_cost_initial'],1e-6)
  local=improvement>=.10 and shift<=15;chosen=M if local else M0;val=[x for x in controls[city]['points'] if x['split']=='validation' and x['map_name']==name];v=metrics(chosen,val);city_v=metrics(M0,val)
  unreliable=(r['edge_samples']<100 or r['building_proxy_polygons']<3 or r['structure_cost_final']>10 or shift>30)
  grade='verified' if v and v['max_m']<=5 else 'unreliable' if v or unreliable else 'pending_validation'
  entry=dict(map_name=name,city=city,status='complete',grade=grade,precise_2m=bool(v and v['max_m']<=2),selected='scene_similarity' if local else 'city_uniform',matrix=chosen.tolist(),city_uniform_validation=city_v,validation=v,structure_cost_initial=r['structure_cost_initial'],structure_cost_local=r['structure_cost_final'],relative_improvement=improvement,local_max_corner_shift_m=shift,building_proxy_n=r['building_proxy_polygons'],osm=r['osm'],elapsed_s=time.perf_counter()-t,base_revision=controls[city]['revision'],explanation='Independent metrics are current-scene only; structural score is not geographic accuracy')
  atomic_json(entry,status)
 except Exception as e:atomic_json(dict(map_name=name,city=city,status='failed',grade='unreliable',precise_2m=False,error=str(e),elapsed_s=time.perf_counter()-t),status)
 print(json.dumps(dict(completed=i+1,scene=name,grade=json.loads(status.read_text())['grade'],seconds=round(time.perf_counter()-t,2))),flush=True)
rows=[json.loads((a.output/p.stem/'batch_status.json').read_text()) for p in paths]
atomic_json(dict(n=34,completed=sum(r['status']=='complete' for r in rows),verified=sum(r['grade']=='verified' for r in rows),pending=sum(r['grade']=='pending_validation' for r in rows),unreliable=sum(r['grade']=='unreliable' for r in rows),precise_2m=sum(r.get('precise_2m',False) for r in rows),rows=rows,policy=policy),a.output/'summary.json');print('DONE',flush=True)
