"""Real RGB extraction, frozen Geometry timing, full-field comparisons and resume proof."""
import argparse,json,sys,time,gc
from pathlib import Path
import cv2,numpy as np,torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.unified_pgvb_runtime import UnifiedPGVBNamedLandmarkPolicy
from graphnav.unified_pgvb_training import load_unified,configure_visual_training,CachedLandmarkFrames,collate_cached_frames,move_batch,weak_landmark_loss
from graphnav.pgvb_observation import encode_landmark_observation
from graphnav.closedloop import PublicEpisode


def main():
 p=argparse.ArgumentParser()
 for k in ['data','before','after','stage1','sbf-source','data-root','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);sys.path.insert(0,str(a.sbf_source));from sbfnav.coordinates import MapTransform
 from sbfnav.dataset import CityReferCatalog
 a.output.mkdir(parents=True,exist_ok=False);data=torch.load(a.data/'cache.pt',map_location='cpu',weights_only=True);before=torch.load(a.before,map_location='cpu',weights_only=True);after=torch.load(a.after,map_location='cpu',weights_only=True);torch.cuda.synchronize();t=time.perf_counter();runtime=UnifiedPGVBNamedLandmarkPolicy(a.stage1,a.before,a.output/'runtime');torch.cuda.synchronize();init_s=time.perf_counter()-t;native=runtime.native;catalog=CityReferCatalog(a.data_root/'cityrefer');count=[0];native.model.vision_model.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1));fresh=[]
 chosen=[]
 for typ in ['Building','Road']:
  chosen.append(next(r for r in data['records'] if r['subset']=='dev' and r['type']==typ))
 for r in chosen:
  f=data['frames'][r['frame_id']];q=data['queries'][r['query_key']];path=Path(f['image_path']);depth=path.parent/Path(f['frame']['depth']).name
  with np.load(depth) as z:valid=z['depth_valid'].copy()
  obs=dict(bgr=cv2.imread(str(path)),depth_valid=valid,footprint_world=np.asarray(f['frame']['footprint_world']));landmark=dict(name=q['name'],typ=q['type'],contours=q['contours']);native.texts([q['name']]);native.texts([q['type']]);torch.cuda.synchronize();t=time.perf_counter();oldcount=count[0];encoded,reason=encode_landmark_observation(obs,landmark,dict(world_bounds=f['bounds']),native,torch.device('cuda'));torch.cuda.synchronize();seconds=time.perf_counter()-t;assert encoded is not None,reason;difference=(encoded['image_features'][0].float().cpu()-f['image_features'].float()).abs();fresh.append(dict(query_key=r['query_key'],frame_id=r['frame_id'],seconds=seconds,siglip2_rgb_forwards=count[0]-oldcount,feature_mae=float(difference.mean()),feature_max_error=float(difference.max()),input_rgb_shape=list(obs['bgr'].shape),cached_shape=list(f['image_features'].shape)));assert count[0]-oldcount==1
 # Deterministic public-only 24-query city-Dev comparison; no task outcomes select it.
 seen=set();cases=[]
 for typ in ['Building','Road']:
  choices=sorted([r for r in data['records'] if r['subset']=='dev' and r['type']==typ],key=lambda r:(r['query_key'],r['frame_id']))
  choices=sorted(choices,key=lambda r:len(data['queries'][r['query_key']]['contours']),reverse=True)
  num=0
  for r in choices:
   if r['query_key'] in seen:continue
   seen.add(r['query_key']);cases.append(r);num+=1
   if num==12:break
 field_rows=[];geotimes=[];berttimes=[];snapshots=[]
 for j,r in enumerate(cases):
  f=data['frames'][r['frame_id']];q=data['queries'][r['query_key']];mp=f['map_name'];pose=f['frame'].get('camera_pose',f['frame'].get('agent_pose'));episode=PublicEpisode(mp,'Navigate to '+q['name'],np.asarray(pose,float),(),(q['name'],));runtime.set_episode(episode,MapTransform.from_raster(a.data_root/'rgbd'/f'{mp}.tif'),catalog);torch.cuda.synchronize();t=time.perf_counter();runtime._initialize_geometry_inputs();torch.cuda.synchronize();berttimes.append(time.perf_counter()-t);runtime.unified.load_state_dict(before['model']);runtime.unified.reset_episode();torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):base=runtime.unified.forward_field(static_inputs=runtime.inputs,tokens=runtime.tokens,token_mask=runtime.mask,node_emb=runtime.names,observed=None)['field']
  torch.cuda.synchronize();geotimes.append(time.perf_counter()-t);observed=dict(local_uv=f['local_uv'][None].cuda(),image_features=f['image_features'][None].cuda(),valid=f['valid'][None].cuda(),name_text=q['name_text'][None].cuda(),type_text=q['type_text'][None].cuda(),polygon_tokens=r['polygon_tokens'][None].cuda());fields={'geometry_only':base};scores={}
  for name,payload in [('before',before),('after',after)]:
   runtime.unified.load_state_dict(payload['model'])
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=runtime.unified.forward_field(static_inputs=runtime.inputs,tokens=runtime.tokens,token_mask=runtime.mask,node_emb=runtime.names,observed=observed)
   fields[name]=out['field'];scores[name]=out['landmark_scores'][0].float().cpu().numpy()
  row=dict(query_key=r['query_key'],frame_id=r['frame_id'],type=r['type'],polygons=len(q['contours']),metrics={});snapshot=dict(frame_id=r['frame_id'],query_key=r['query_key'],map_name=mp,bounds=f['bounds'],image_path=f['image_path'],footprint_world=f['frame']['footprint_world'],polygons=q['contours'],fields={},scores={k:v.tolist() for k,v in scores.items()},labels=r['labels'].tolist(),local_uv=f['local_uv'].tolist())
  for method,field in fields.items():
   prob=field.belief_logits[0].float().softmax(-1).cpu();top=int(prob.argmax());row['metrics'][method]=dict(polygon_mass=float((prob*r['area_fraction']).sum()),R20=bool(r['distance'][top]<=20),NE=float(r['distance'][top]));snapshot['fields'][method]=prob.tolist()
  field_rows.append(row);snapshots.append(snapshot)
 # Genuine save/load continuation: one in-memory trained update -> save -> load
 # optimizer/model -> compare the next update to the uninterrupted live model.
 ds=CachedLandmarkFrames(data,'fit');batch=collate_cached_frames([ds[i] for i in ds.frame_ids[:4]]);inputs,labels,fids,types=move_batch(batch,'cuda');m1=load_unified(after);configure_visual_training(m1);o1=torch.optim.AdamW(m1.pgvb.parameters(),lr=1e-4,weight_decay=.001);o1.load_state_dict(after['optimizer'])
 def update(model,opt):
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):s=model(**inputs)['landmark_scores']
  loss=weak_landmark_loss(s,labels,fids,types)['total'];loss.backward();torch.nn.utils.clip_grad_norm_(model.pgvb.parameters(),5.);opt.step();return float(loss)
 update(m1,o1);state=dict(after,model=m1.state_dict(),optimizer=o1.state_dict());atomic_save(state,a.output/'resume_probe.pt');restored=torch.load(a.output/'resume_probe.pt',weights_only=True,map_location='cpu');m2=load_unified(restored);configure_visual_training(m2);o2=torch.optim.AdamW(m2.pgvb.parameters(),lr=1e-4,weight_decay=.001);o2.load_state_dict(restored['optimizer']);l1=update(m1,o1);l2=update(m2,o2);delta=max(float((x-y).abs().max()) for x,y in zip(m1.pgvb.parameters(),m2.pgvb.parameters()));assert delta<1e-6
 summary=dict(runtime_cold_load_seconds=init_s,fresh_extraction=fresh,Geometry_GPU_forward_seconds=geotimes,static_BERT_and_data_setup_seconds=berttimes,geometry_note='Frozen Geometry one forward per named query; cached during episode; never recomputed in PGVB-only training steps',resume_one_step_equivalence=dict(max_parameter_error=delta,loss_uninterrupted=l1,loss_restored=l2,passes=delta<1e-6,probe_not_used_for_training_or_selection=True),field_cases=field_rows,field_summary={method:{key:float(np.mean([r['metrics'][method][key] for r in field_rows])) for key in ['polygon_mass','R20','NE']} for method in ['geometry_only','before','after']},scope='24 deterministic public named queries, independent TRAIN city; weak Polygon supervision only; no SR claim',before_sha256=file_sha256(a.before),after_sha256=file_sha256(a.after));atomic_json(summary,a.output/'metrics.json');atomic_json(snapshots,a.output/'field_snapshots.json');print(json.dumps({k:v for k,v in summary.items() if k not in ['field_cases','Geometry_GPU_forward_seconds','static_BERT_and_data_setup_seconds']},indent=2))
if __name__=='__main__':main()
