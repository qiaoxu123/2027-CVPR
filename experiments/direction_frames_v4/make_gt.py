"""Phase 2 starts here. Require frozen predictions BEFORE opening target GT."""
from shared import *
from geometry_frames import poly
from shapely.ops import unary_union

def run():
 manifest=readj(OUT/'freeze_manifest.json');assert manifest['frame_predictions_sha256']==sha(OUT/'frozen_frames.jsonl.gz')
 R=readlines(OUT/'relations.jsonl.gz');train=readj(DATA/'original_citynav/citynav_train_seen.json');O=readj(DATA/'cityrefer_original/objects.json');E=[]
 for r in R:
  if r['status']!='bound':continue
  ep=train[r['episode']];obs=O[r['map']];target=obs[r['subject_ids'][0]];tp=poly(target);refs=[obs[i] for i in r['reference_ids']];ps=[poly(o) for o in refs];rp=unary_union(ps) if all(p is not None for p in ps) else None;gt=ep['target_positions'][0][:2];rxy=refs[0]['position'][:2] if len(refs)==1 else list(rp.centroid.coords[0]) if rp is not None else np.mean([z['position'][:2] for z in refs],axis=0).tolist();delta=np.array(gt)-np.array(rxy);status='invalid_polygon'
  if tp is not None and rp is not None:status='disjoint' if not tp.intersects(rp) and tp.distance(rp)>1 and np.linalg.norm(delta)>1 else 'ambiguous_overlap_or_boundary'
  pose=np.array(ep['trajectory'][0]);yaw=angle(pose[3:5]) if len(pose)==6 else float(np.degrees(pose[3])%360)
  E.append({'occurrence_id':r['occurrence_id'],'cluster':r['cluster'],'map':r['map'],'word':r['semantic_word'],'parse_tier':r['confidence_tier'],'origin':r['origin'],'subject_xy':gt,'reference_xy':rxy,'target_polygon':target['contour'],'reference_polygon':[list(z.exterior.coords) for z in (list(rp.geoms) if rp and rp.geom_type=='MultiPolygon' else [rp]) if z is not None],'reference_category':refs[0]['object_type'],'subject_category':target['object_type'],'geometry_status':status,'angle_world':angle(delta),'initial_yaw_deg':yaw,'start_xy':pose[:2].tolist()})
 writelines(OUT/'evaluation_gt.jsonl.gz',E);writej(OUT/'GT_phase_manifest.json',{'phase':2,'frozen_predictions_sha256_before_GT':manifest['frame_predictions_sha256'],'GT_sha256':sha(OUT/'evaluation_gt.jsonl.gz'),'future_trajectory_used':False,'initial_pose_read_only_for_baseline':True,'new_disjoint':sum(e['geometry_status']=='disjoint' and e['origin'].startswith('v4') for e in E)})
 print('GT materialized after frame freeze:',len(E))
if __name__=='__main__':run()
