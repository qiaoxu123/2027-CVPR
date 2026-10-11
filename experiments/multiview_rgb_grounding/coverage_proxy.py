"""GT-sealed geometric FOV diagnostics for all existing Train/Dev states.

No rendering, no matching predictions, and no flight decisions are made here.
Sphere/cone intersection is a loose upper proxy, NOT observed visual evidence.
"""
import itertools,json,time
from pathlib import Path
import numpy as np,torch
from .render import OBJECTS,basis,project
from .geometry import world_grid

def run():
 root=Path('runs/multiview_rgb_grounding');objects=json.loads(OBJECTS.read_text());start=time.time();rows=[];stats={}
 for split in ['train','dev']:
  plan=torch.load(Path('runs/instance_grounding')/('plan_'+split)/'plan.pt',weights_only=True)
  for s in plan['states']:
   m=s['metadata'];lab=plan['labels'][m['sample_id']];o=objects[m['map_name']][lab['target_object_id']];center=np.asarray(o['position']);dim=np.asarray(o['dimension']);radius=np.linalg.norm(dim)/2;pose=np.asarray(s['public']['pose_world']);grid=world_grid(s['public']);static=np.asarray(s['public']['static_logits']);error=np.linalg.norm(grid[np.argsort(static,kind='stable')[-1]]-lab['target_world'])>20
   corners=center+np.array(list(itertools.product([-1,1],repeat=3)))*dim/2
   for angle in [0,20,40,60]:
    cam=(center-pose[:3])@basis(pose[3],angle).T
    loose=bool(cam[2]+radius>.5 and np.linalg.norm(cam)-radius<=400 and abs(cam[0])-cam[2]<=radius*np.sqrt(2) and abs(cam[1])-cam[2]<=radius*np.sqrt(2))
    uv,z,good=project(np.r_[center[None],corners],pose,angle);front=bool((z>.5).all());side=None
    if front:
     low=np.maximum(uv[1:].min(0),0);high=np.minimum(uv[1:].max(0),512);side=float(max(0,np.min(high-low)))
    rows.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],split=split,tilt=angle,geometry_error=bool(error),object_type=o['object_type'],center_in_fov=bool(good[0]),box_sphere_fov_possible=loose,projected_box_min_side=side,pixel16_proxy=bool(good[0] and side is not None and side>=16),source_step=m['cutoff_step'],status='GT offline geometry proxy only; occlusion, measured points, ROI quality and semantics unverified'))
  stats[split]={}
  for angle in [0,20,40,60]:
   chosen=[r for r in rows if r['split']==split and r['tilt']==angle];error_families={r['family_id'] for r in chosen if r['geometry_error']};stats[split][str(angle)]={'states':len(chosen),'families':len({r['family_id'] for r in chosen}),'static_error_families':len(error_families),'center_fov_error_families_any_prefix':len({r['family_id'] for r in chosen if r['geometry_error'] and r['center_in_fov']}),'sphere_fov_error_families_any_prefix':len({r['family_id'] for r in chosen if r['geometry_error'] and r['box_sphere_fov_possible']}),'pixel16_center_fov_error_families_any_prefix':len({r['family_id'] for r in chosen if r['geometry_error'] and r['pixel16_proxy']})}
 (root/'coverage_geometric_proxy.json').write_text(json.dumps(dict(statistics=stats,seconds=time.time()-start,strictly_offline_GT=True,observed_credible_coverage=False,no_seen_or_unseen=True),indent=2)+'\n');(root/'coverage_geometric_proxy_predictions.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(stats),flush=True)

if __name__=='__main__':run()
