"""Select one frozen formal episode; extract archived data without model inference."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union,nearest_points
ROOT=Path(__file__).resolve().parents[4]
ASSETS=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from graphnav.observed_patch_scales import _small_grid_masks

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
def main():
 cid='formal_val_unseen_success'
 c=next(json.loads(s) for s in (ASSETS/'cases.jsonl').read_text().splitlines() if json.loads(s)['case_id']==cid)
 meta=json.loads((ASSETS/f'fields/{cid}.json').read_text()); path=ASSETS/f'fields/{cid}.npz'
 assert sha(path)==meta['arrays_sha256']
 a=np.load(path);xy=a['world_xy'];s0=a['real_static'][0].astype(float);st=a['real_belief'][-1].astype(float)
 assert np.array_equal(a['poses'],np.array(c['trace']['poses']))
 assert np.max(np.abs(a['real_belief']-a['real_static']-a['real_residual']))<1e-5
 obsindex=json.loads((ASSETS/'observation_index.json').read_text()); rows=[r for r in obsindex['frames'] if r['case_id']==cid]
 current=[];crop_files=[]
 for row in rows:
  op=ASSETS/f"local_only/observations/{cid}_step{row['step']}.npz"
  assert sha(op)==row['crop_sha256']
  obs=dict(np.load(op));m,q=_small_grid_masks(obs,xy);current.append(m.any(-1))
  crop_files.append({'step':row['step'],'local_path':str(op),'sha256':sha(op),'footprint_world':obs['footprint_world'].tolist(),'pose':row['pose']})
 current=np.array(current)
 # Runtime's accepted-history criterion: strictly advancing step, >2m XYZ from last accepted pose.
 hist=np.zeros_like(current[0]); last=None
 for i,pose in enumerate(a['poses']):
  combined=(current[i]|hist).any(-1)
  assert np.array_equal(combined,a['real_observed'][i].astype(bool))
  if current[i].any() and (last is None or np.linalg.norm(pose[:3]-last[:3])>=2):hist|=current[i];last=pose
 lo=np.array(c['public_map']['world_bounds'][:2]);span=np.array(c['public_map']['world_bounds'][2:])-lo
 polygons=[lo+np.array(r['contour'])*span for r in c['public_map']['landmarks']]
 road=unary_union([Polygon(p).buffer(0) for p in polygons]);top=xy[s0.argmax()];bp=np.array(nearest_points(Point(top),road.boundary)[1].coords[0])
 view=[560,830,800,1070]
 ids=[]
 for i in np.argsort(-s0,kind='stable'):
  q=xy[i]
  if view[0]<=q[0]<=view[2] and view[1]<=q[1]<=view[3] and all(np.linalg.norm(q-xy[j])>=60 for j in ids):ids.append(int(i))
  if len(ids)==3:break
 def prob(x):v=np.exp(x-x.max());return v/v.sum()
 b0,bt=prob(s0),prob(st)
 np.savez_compressed(OUT/'source/overview_fields.npz',world_xy=xy,B0=b0,Bt=bt,s0=s0,st=st,residual=a['real_residual'][-1],poses=a['poses'],target_xyz=a['target_xyz'],current_roi_valid=current[-1],formal_observed=a['real_observed'][-1],footprint_world=np.array(rows[-1]['footprint_world']))
 context=json.loads((ASSETS/'public_map_context.json').read_text())
 write(OUT/'source/map.json',{'context':context['maps'][c['map_name']],'references':[{'name':r['name'],'contour':p.tolist()} for r,p in zip(c['public_map']['landmarks'],polygons)],'attribution':context['attribution']})
 write(OUT/'source/case.json',c)
 sources=[]
 for f in ['cases.jsonl',f'fields/{cid}.npz',f'fields/{cid}.json','replay_manifest.json','observation_index.json','public_map_context.json','final_evidence/MODEL_MECHANISM_AUDIT.md','final_evidence/RGB_FRAME_INDEX.json']:
  sources.append({'file':f,'sha256':sha(ASSETS/f)})
 tif=ROOT.parent.parent/'DATA/rgbd/birmingham_block_5.tif';png=tif.with_suffix('.png')
 import rasterio
 with rasterio.open(tif) as ds:affine=list(ds.transform); raster_extent=[ds.bounds.left,ds.bounds.right,ds.bounds.bottom,ds.bounds.top]
 source_raster=next(r for r in obsindex['source_assets'] if Path(r['path']).name==png.name)
 assert sha(png)==source_raster['sha256']
 evidence={'episode_id':c['episode_id'],'case_id':cid,'split':c['split'],'dataset_version':'Revised CityNav (processed_citynav archive)','dataset_sha256':c['dataset_sha256'],'map_name':c['map_name'],'instruction_verbatim':c['instruction'],'checkpoint_sha256':c['checkpoint_sha256'],'residual_checkpoint':None,'model_identity':'Formal GeomBelief / GraphNav-V1 frozen C epoch1; no independent CVR','source_predictions':c['source_predictions'],'source_predictions_sha256':c['source_predictions_sha256'],'sources':sources,'step':7,'display_world_bounds':view,'model_world_bounds':c['public_map']['world_bounds'],'candidate_grid':[28,28],'candidate_ids_for_display':ids,'candidate_selection':'Descending static score, deterministic 60m spatial separation inside display crop; examples are actual grid cells, not object detections. No GT used.','reference_landmark':'Aldridge Road','dirt_track_note':'Instruction text only: no verified dirt-track polygon exists in the supplied reference map; no invented annotation.','geometry':{'top1_xy':top.tolist(),'nearest_road_boundary_xy':bp.tolist(),'boundary_distance_m':float(np.linalg.norm(top-bp)),'direction_from_boundary_plus_x_deg':float(np.degrees(np.arctan2(*(top-bp)[::-1]))),'direction_note':'Measured world-XY angle, not a compass bearing or an asserted language direction.'},'GT_xyz':c['target_xyz'],'selected_goal_xy':c['trace']['step_diagnostics'][-1]['selected_goal_xy'],'final_pose':c['trace']['poses'][-1],'metrics':{'NE_m':c['trace']['metrics']['NE'],'SR':c['trace']['metrics']['SR'],'L1_B0_Bt':float(np.abs(b0-bt).sum()),'KL_B0_Bt_nats':float(np.sum(b0*np.log(b0/bt))),'top1_shift_m':float(np.linalg.norm(xy[b0.argmax()]-xy[bt.argmax()]))},'ROI':{'current_supported_candidates':int(current[-1].any(-1).sum()),'current_or_history_supported_candidates':int(a['real_observed'][-1].sum()),'method':'Exact archived _small_grid_masks on SHA-matched observation; current/history union independently matches every saved formal observed mask. Not semantic visibility.','source_code_sha256':sha(ROOT/'graphnav/observed_patch_scales.py')},'observation_crops':crop_files,'raster':{'local_path':str(png),'sha256':sha(png),'surface_tif':str(tif),'surface_sha256':sha(tif),'pixel_corner_to_world_affine':affine,'extent':raster_extent,'origin':'upper','observation_registration':'Four-corner world-to-frame homography to (0,0),(383,0),(383,383),(0,383); cv2.INTER_LINEAR; constant black border.','public_redistribution_permission':'Unconfirmed: RGB full figure remains local_only.'},'mechanism':'Bt=softmax(s0+Delta_t); fixed s0, current/history feature evidence; no recursive probability update.','replay_scope':'Read saved arrays; recompute only deterministic ROI support and figure statistics; no neural inference/training.','public_figure':'No orthophoto or input RGB pixels. Real map geometry, archived beliefs and executed path.','GT_usage':'Plot and evaluation only.','plot_normalization':'B0 and Bt normalized over all 784 candidates; shared logarithmic color normalization and fixed opacity; no per-panel rescaling or interpolated cells.','source_array_sha256':sha(OUT/'source/overview_fields.npz')}
 write(OUT/'OVERVIEW_EVIDENCE.json',evidence)
 print(json.dumps({'episode':c['episode_id'],'candidate_ids':ids,'metrics':evidence['metrics'],'ROI':evidence['ROI']},indent=2))
if __name__=='__main__':main()
