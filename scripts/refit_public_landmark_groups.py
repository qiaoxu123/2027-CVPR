"""Mechanical independence correction; reuses frozen matches/scores, not tuned by errors."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from tools.osm_map.landmark_groups import grouped_fit
from scripts.match_public_citynav_landmarks import error
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=json.load(open(a.source/'matches.json'));summary=json.load(open(a.source/'summary.json'));cities={}
for city,old in summary['cities'].items():
 rr=[r for r in rows if r['city']==city];M0=np.array(old['initial_matrix']);M,status,fit,val=grouped_fit(rr,M0);(a.output/(city+'_prediction_frozen.json')).write_text(json.dumps(dict(matrix=M.tolist(),status=status,fit_ids=[r['match_id'] for r in fit],validation_ids=[r['match_id'] for r in val]),indent=2))
 for r in rr:r['anchor_residual_m']=float(np.linalg.norm(np.array(r['source_anchor_xy'])@M[:,:2].T+M[:,2]-r['osm_anchor_en']))
 permap={m:dict(auto_heldout_proxy=error(M,[r for r in val if r['map_name']==m]),manual_independent=None,status='pending_identity_review') for m in old['per_scene']};cities[city]=dict(old,matrix=M.tolist(),status=status,fit_objects=len(fit),validation_objects=len(val),fit_groups=len({r['independence_group'] for r in fit}),validation_groups=len({r['independence_group'] for r in val}),initial_validation_proxy=error(M0,val),optimized_validation_proxy=error(M,val),per_scene=permap)
summary.update(cities=cities,group_correction='Merge touching/nested OSM building parts <=0.5m before seed42 group split. Scores and confidence thresholds unchanged.',initial_source_matches_sha256=hashlib.sha256((a.source/'matches.json').read_bytes()).hexdigest());(a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'matches.json').write_text(json.dumps(rows,indent=2));print('Group-isolated refit complete')
