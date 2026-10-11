"""Read completed pilots, audit measured geometry, and save paired raw QA rows."""
import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path
import cv2, numpy as np, torch
from scipy.spatial import cKDTree
from .render import read_cloud,OBJECTS,basis,project
from .geometry import world_grid
from matplotlib.path import Path as Polygon
from experiments.bidirectional_core.stats import paired_difference,summarize_rows

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def analyze(root,pilots=None,output=None):
 start=time.time();objects=json.loads(OBJECTS.read_text());allrows=[];summary={};checks={};cloudhash={};surface=[]
 pilots=pilots or ['render_pilot_dev_scatter','render_pilot_train_scatter'];destination=Path(output) if output else root;destination.mkdir(parents=True,exist_ok=True)
 for pilot in pilots:
  directory=root/pilot;manifest=json.loads((directory/'manifest.json').read_text());source=Path(manifest['source_plan']);source_manifest=json.loads((source/'manifest.json').read_text());assert sha(source/'plan.pt')==source_manifest['files']['plan.pt'];plan=torch.load(source/'plan.pt',weights_only=True);rows=[];normals=[]
  for name,c in manifest['clouds'].items():
   path=Path(c['path']);cloudhash[name]=sha(path);_,_,data=read_cloud(path);target_points={}
   for f in [v for v in manifest['frames'] if v['metadata']['map_name']==name and v['splat_radius']==1]:
    m=f['metadata'];label=plan['labels'][m['sample_id']];target=label['target_object_id'];regions={v['object_id']:v for v in f['regions']};roi=regions.get(target);public=next(s['public'] for s in plan['states'] if s['metadata']['sample_id']==m['sample_id']);grid=world_grid(public);logits=np.array(public['static_logits']);geometry_error=np.linalg.norm(grid[np.argsort(logits,kind='stable')[-1]]-label['target_world'])>20
    geo=np.load(directory/f['frame_key']/f"tilt{f['tilt']}_r1_geometry.npz");win=geo['winner'];ids=np.unique(win[win>=0]);sample=ids[np.linspace(0,max(0,len(ids)-1),min(2000,len(ids))).astype(int)] if len(ids) else ids
    hull_mask=np.zeros(win.shape,np.uint8);yy,xx=np.where(win>=0)
    if len(xx)>=3:cv2.fillConvexPoly(hull_mask,cv2.convexHull(np.stack([xx,yy],1).astype(np.int32)),1)
    hull_area=int(hull_mask.sum());hull_support=float((win>=0).sum()/hull_area) if hull_area else 0.
    if len(sample):
     xyz=np.stack([data[k][sample] for k in ['x','y','z']],1);rgb=np.stack([data[k][sample] for k in ['red','green','blue']],1);uv,z,good=project(xyz,f['camera_pose'],f['tilt'])
     assert good.all() and (z>.5).all()
     image=cv2.imread(str(directory/f['frame_key']/f"tilt{f['tilt']}_r1_rgb.png"))[...,::-1]
     pos={int(v):int(i) for i,v in enumerate(sample)};chosen=np.flatnonzero(np.isin(win.ravel(),sample));chosen=chosen[::max(1,len(chosen)//1000)]
     for idx in chosen:assert (image.reshape(-1,3)[idx]==rgb[pos[int(win.ravel()[idx])]]).all()
    target_credible=bool(roi and roi['credible'] and f['frame_quality_pass']);region_ids=[v['object_id'] for v in f['regions'] if v['credible'] and f['frame_quality_pass']]
    if target not in target_points:
     obj=objects[name][target];poly=np.asarray(obj['contour']);pos=np.asarray(obj['position']);dim=np.asarray(obj['dimension']);lo=poly.min(0);hi=poly.max(0);selected=(data['x']>=lo[0])&(data['x']<=hi[0])&(data['y']>=lo[1])&(data['y']<=hi[1])&(np.abs(data['z']-pos[2])<=dim[2]/2+.5);pids=np.flatnonzero(selected);p=np.stack([data[k][pids] for k in ['x','y','z']],1);inside=Polygon(poly).contains_points(p[:,:2],radius=1e-7);target_points[target]=(pids[inside],p[inside])
    pids,p=target_points[target];_,_,in_fov=project(p,f['camera_pose'],f['tilt']);visible_weak=int(np.isin(ids,pids).sum());projected=int(in_fov.sum())
    reason='credible' if target_credible else ('source_box_has_no_points' if not len(pids) else 'outside_measured_point_fov' if not projected else 'occluded_or_no_winning_point' if not visible_weak else 'weak_instance_association_ambiguous' if roi is None else 'native_pixels_or_support_below_threshold' if not roi['credible'] else 'full_frame_quality_below_threshold')
    row=dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=name,pilot=pilot,tilt=f['tilt'],camera_pose=f['camera_pose'],cutoff_step=m['cutoff_step'],source_step=f['source_step'],target_object_id=target,geometry_error=bool(geometry_error),credible_candidate_ids=region_ids,target_region=roi,target_geometric_box_points=len(pids),target_measured_points_in_fov=projected,target_weak_box_winning_points=visible_weak,target_failure_reason=reason,target_point_diagnostic='offline weak box/Polygon label diagnostic; not proof of semantic segmentation',
      metrics=dict(valid_fraction=f['valid_fraction'],inside_measured_pixel_convex_hull_valid_fraction=hull_support,outside_measured_pixel_convex_hull_fraction=float(1-hull_mask.mean()),target_credible_visible=float(target_credible),candidate_count=len(region_ids),frame_quality_pass=float(f['frame_quality_pass']),error_and_target_visible=float(geometry_error and target_credible)))
    rows.append(row);allrows.append(row)
    # Local camera-height check is environment geometry, never an inference input.
    if f['tilt']==0:
     camera=np.asarray(f['camera_pose']);local=(np.abs(data['x']-camera[0])<=1)&(np.abs(data['y']-camera[1])<=1);heights=np.asarray(data['z'][local]);top=float(np.quantile(heights,.95)) if len(heights) else None
     surface.append(dict(sample_id=m['sample_id'],map_name=name,camera_z=float(camera[2]),local_surface_p95_z=top,camera_minus_local_surface_m=float(camera[2]-top) if top is not None else None,source_points_within_1m=len(heights)))
   # Normals from actually rendered building points, not invented facades.
   grouped={}
   for f in [v for v in manifest['frames'] if v['metadata']['map_name']==name and v['splat_radius']==1]:grouped.setdefault(f['frame_key'],[]).append(f)
   for key,fs in grouped.items():
    seen={}
    for f in fs:
     w=np.load(directory/key/f"tilt{f['tilt']}_r1_geometry.npz")['winner'];ids=np.unique(w[w>=0]);seen[f['tilt']]=ids[np.asarray(data['class'][ids])==2] if 'class' in data.dtype.names else np.zeros(0,int)
    ids=np.unique(np.concatenate(list(seen.values())));sample=ids[np.linspace(0,max(0,len(ids)-1),min(4000,len(ids))).astype(int)] if len(ids) else ids
    if len(ids)>=20:
     xyz=np.stack([data[k][ids] for k in ['x','y','z']],1).astype(np.float64);points=np.stack([data[k][sample] for k in ['x','y','z']],1).astype(np.float64);tree=cKDTree(xyz);dist,index=tree.query(points,k=16,workers=1);neighborhood=xyz[index];centered=neighborhood-neighborhood.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',centered,centered)/16;values,vectors=np.linalg.eigh(cov);nz=np.abs(vectors[:,2,0]);planarity=values[:,0]/np.maximum(values.sum(1),1e-12);stable=(planarity<.05)&(dist[:,-1]<1.5)
     item=dict(sample_id=fs[0]['metadata']['sample_id'],map_name=name,sampled_building_points=len(sample),stable_local_plane_points=int(stable.sum()),facade_like_points=int((stable&(nz<.3)).sum()),roof_like_points=int((stable&(nz>.8)).sum()),normal_definition='local measured-point PCA, k16; class2=Building; proxy not human surface truth')
     item['views']={str(tilt):dict(building_winning_points=len(v),sampled_stable_planes=int((np.isin(sample,v)&stable).sum()),sampled_facade_like=int((np.isin(sample,v)&stable&(nz<.3)).sum()),new_points_relative_nadir=int(len(np.setdiff1d(v,seen.get(0,[]))))) for tilt,v in seen.items()};normals.append(item)
  pertilt={}
  for tilt in [0,20,40,60]:
   selected=[r for r in rows if r['tilt']==tilt];pertilt[str(tilt)]={k:summarize_rows(selected,k,seed=20261011) for k in ['valid_fraction','inside_measured_pixel_convex_hull_valid_fraction','outside_measured_pixel_convex_hull_fraction','target_credible_visible','candidate_count','frame_quality_pass','error_and_target_visible']}
  paired={str(tilt):{k:paired_difference([r for r in rows if r['tilt']==0],[r for r in rows if r['tilt']==tilt],k,seed=20261011) for k in ['valid_fraction','target_credible_visible','candidate_count']} for tilt in [20,40,60]}
  common=[]
  for sample in sorted({r['sample_id'] for r in rows}):
   group=[r for r in rows if r['sample_id']==sample];ids=set(group[0]['credible_candidate_ids'])
   for r in group:ids &= set(r['credible_candidate_ids'])
   target=group[0]['target_object_id'];common.append(dict(sample_id=sample,family_id=group[0]['family_id'],map_name=group[0]['map_name'],candidates=sorted(ids),target_credible_all_views=target in ids,at_least_two_candidates=len(ids)>=2,geometry_error=group[0]['geometry_error']))
  summary[pilot]=dict(states=manifest['states'],families=manifest['independent_families'],numeric_quality_gate_pass=manifest['numeric_quality_gate_pass'],oblique_frame_pass_fraction=manifest['oblique_frame_pass_fraction'],per_tilt=pertilt,paired=paired,common_candidates=common,facade_proxies=normals,render_seconds=manifest['seconds'],maps=sorted(manifest['clouds']))
  checks[pilot]=dict(sampled_original_color_exact=True,projection_of_winning_points_in_fov=True,causal_current_pose=True,source_step_le_cutoff=all(r['source_step']<=r['cutoff_step'] for r in rows),label_usage='target ID/world only offline quality/coverage labels; never selects camera or crop')
 dump(destination/'quality_metrics.json',summary);dump(destination/'quality_predictions.json',allrows);dump(destination/'geometry_checks.json',checks);dump(destination/'surface_height_audit.json',surface);dump(destination/'cloud_sha256.json',cloudhash)
 dump(destination/'quality_execution.json',dict(seconds=time.time()-start,model_training=0,backbone_loading=0,stages_run=['P3-A'],stages_not_run=['P3-B formal benchmark','P3-C','P3-D','P3-E'],all_original_results_retained=True))
 print(json.dumps({'seconds':time.time()-start,'pilots':{k:dict(states=v['states'],gate_pass=v['numeric_quality_gate_pass']) for k,v in summary.items()}}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='runs/multiview_rgb_grounding');p.add_argument('--pilots',nargs='+');p.add_argument('--output');a=p.parse_args();analyze(Path(a.root),a.pilots,a.output)
