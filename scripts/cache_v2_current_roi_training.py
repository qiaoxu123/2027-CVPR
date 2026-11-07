"""TRAIN-only current-pose ROI preflight; frozen SigLIP2, no learning or Val.
World20m matches are diagnostic location proxies, never instance matching GT.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch,cv2
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.evidence_grounding import appearance_query
from graphnav.observed_candidate_rois import CurrentCandidateROIEnvironment
from scripts.unified_pgvb_v3_pilot import _load_base
from scripts.evaluate_v2_public_candidate_recall import spatial_nms


def main():
 p=argparse.ArgumentParser()
 for k in ['prepared','data-root','sbf-source','source-v2','trained-v2','stage1','bert-cache','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--split',choices=['fit','development','confirmation','all'],default='all');p.add_argument('--limit',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'public_features').mkdir(exist_ok=True);(a.output/'views').mkdir(exist_ok=True);(a.output/'private_rows').mkdir(exist_ok=True);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False
 labels=json.load(open(a.prepared/'labels_private.json'));ids=[r['index'] for r in labels if a.split=='all' or r['split']==a.split];selected=sorted(np.random.default_rng(42).choice(ids,min(a.limit,len(ids)),replace=False).tolist()) if a.limit else sorted(ids);source=[json.loads(s) for s in (a.prepared/'public_train.jsonl').read_text().splitlines()];assert len(source)==21878 and all(q['target_xy']==[0.,0.] for q in source);path=a.output/'selected_public.jsonl';path.write_text(''.join(json.dumps(source[i])+'\n' for i in selected))
 sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
 from sbfnav.mapdata import GROUND_LEVEL
 ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav');assert len(ds)==21878
 geometry,cfg=_load_base(a.source_v2,'cuda');state=torch.load(a.trained_v2,weights_only=True,map_location='cpu');geometry.load_state_dict({k[len('geometry.'):]:v for k,v in state['model'].items() if k.startswith('geometry.')},strict=True);geometry.eval().requires_grad_(False);bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));bert.load_cache(a.bert_cache);protocol=torch.load(a.stage1,weights_only=True,map_location='cpu')['protocol'];native=ObservedSigLIPMatchingEncoder(protocol['backbone'],device='cuda',microbatch=16)
 if getattr(native.model.config,'_commit_hash',None)!=protocol['backbone_revision']:raise ValueError('Native frozen backbone mismatch')
 env=CurrentCandidateROIEnvironment(a.data_root/'rgbd',a.output/'observations',min_support=.9);dataset=FullCandidateDataset(path,grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False)
 config=dict(seed=42,n=len(selected),selected_indices=selected,selection='TRAIN '+a.split+'; seed42 subset only if limited; no GT visibility/distance filtering',source_v2_sha256=file_sha256(a.source_v2),trained_v2_sha256=file_sha256(a.trained_v2),parser_sha256=file_sha256(Path('graphnav/public_landmark_mentions.py')),backbone=protocol['backbone'],backbone_revision=protocol['backbone_revision'],processor_size=getattr(native.processor.image_processor,'size',{}),matching_precision='frozen SigLIP2 BF16, normalized features storedFP16, similarityFP32; GeometryFP32',extents_m=[20,40],candidate_budget=20,spatial_nms_m=20,observed_support_min=.9,camera='Official actual TRAIN start XYZYaw only; no teacher/planned/future images',resolutions='Same physical ROIs, current384 sampling vs source-native GSD, max2048; common validity mask for fair comparison',normal_model_inputs='Geometry public fields, original instruction, current-pose ROI RGB only',GT='post-feature location-proxy scoring only; no GT crop or candidate selection',source_sha256=file_sha256(__file__),environment_sha256=file_sha256(Path('graphnav/observed_candidate_rois.py')),renderer_sha256=file_sha256(Path('graphnav/renderer.py')),public_queries_sha256=file_sha256(path))
 mp=a.output/'manifest.json'
 if mp.exists() and json.load(open(mp))!=config:raise ValueError('Refuse overwrite changed preflight')
 atomic_json(config,mp);rows=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 for j,batch in enumerate(DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_full)):
  i=selected[j];ep=ds[i];q=source[i]
  completed=a.output/'private_rows'/f'{i:05d}.json'
  if completed.exists() and (a.output/'public_features'/f'{i:05d}.npz').exists():
   rows.append(json.load(open(completed)));continue
  with torch.no_grad():
   obs,tokens,mask,nodes=_batch_inputs(batch,bert,'cuda','geometry');field=geometry(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes);z=field.belief_logits.float()[0].cpu().numpy();bounds=np.array(q['world_bounds']);world=bounds[:2]+batch['grid_xy'][0].numpy()*(bounds[2:]-bounds[:2]);prefix=field.candidate_ids[0].cpu().tolist();order=prefix+[int(x) for x in np.argsort(-z,kind='stable') if x not in prefix];cand=spatial_nms(order,world,20,20);centers=world[cand]
  t=time.perf_counter();current,rois=env.current(ep.map_name,ep.start_pose.copy(),GROUND_LEVEL[ep.map_name],centers);render_s=time.perf_counter()-t;valid=np.zeros((len(cand),2),bool);features=np.zeros((2,len(cand),2,native.dimension),np.float16);views=[];positions=[];meta=[]
  for roi in rois:
   meta.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in roi.items() if k not in ['native','low']})
   if not roi['valid']:continue
   meta[-1]['native_image_sha256']=native.image_key(roi['native']);meta[-1]['low_image_sha256']=native.image_key(roi['low'])
   c,s=roi['candidate'],roi['scale'];valid[c,s]=True
   for res,key in [(0,'low'),(1,'native')]:views.append(roi[key]);positions.append((res,c,s))
  t=time.perf_counter()
  if views:
   values=native.images(views).detach().float().cpu().numpy()
   for position,value in zip(positions,values):features[position]=value
  text=native.texts([ep.instruction,appearance_query(ep.instruction)]).detach().float().cpu().numpy();torch.cuda.synchronize();encoding_s=time.perf_counter()-t;known=valid.any(-1);scores={}
  for res,key in enumerate(['current384','native_roi']):
   pooled=(features[res].astype(np.float32)*valid[:,:,None]).sum(1)/np.maximum(valid.sum(1)[:,None],1);pooled/=np.maximum(np.linalg.norm(pooled,axis=-1,keepdims=True),1e-9);scores[key]=pooled@text.T
  # Public feature cache contains no private labels/target identity/coordinates.
  np.savez_compressed(a.output/'public_features'/f'{i:05d}.npz',features=features,valid=valid,text_features=text,candidate_world=centers.astype(np.float32),geometry_logits=z[cand],base_logits_full=z,base_world_grid=world.astype(np.float32),candidate_grid_ids=np.array(cand),candidate_context=field.dense_context[0,cand].detach().float().cpu().numpy().astype(np.float16),start_pose=ep.start_pose.copy(),footprint=current['footprint_world'])
  atomic_json(dict(index=i,map_name=ep.map_name,instruction=ep.instruction,appearance_query=appearance_query(ep.instruction),roi_metadata=meta),a.output/'public_features'/f'{i:05d}.json')
  # Private endpoint is accessed only now, after crop/feature extraction.
  target=np.array(labels[i]['target_world']);distance=np.linalg.norm(centers-target,axis=-1);visible_correct=bool(((distance<=20)&known).any());count=int(known.sum());metrics={};rng=np.random.default_rng(42+i);known_ids=np.flatnonzero(known);perm=np.roll(known_ids,int(rng.integers(1,count))) if count>1 else known_ids
  for key,score in scores.items():
   for channel,name in enumerate(['full','appearance']):
    if count:
     winner=int(np.where(known,score[:,channel],-np.inf).argmax());shuffle=score[:,channel].copy();shuffle[known_ids]=score[perm,channel];shuf_winner=int(np.where(known,shuffle,-np.inf).argmax());metrics[key+'_'+name]=dict(hit20=bool(distance[winner]<=20),winner=winner,NE=float(distance[winner]),shuffled_hit20=bool(distance[shuf_winner]<=20))
    else:metrics[key+'_'+name]=None
  row=dict(index=i,map_name=ep.map_name,original_geometry_hit20=bool(distance[0]<=20),observed_candidate_count=count,correct_candidate_with_support=visible_correct,geometry_first_observed_hit20=bool(distance[known_ids[0]]<=20) if count else None,metrics=metrics,render_s=render_s,encode_s=encoding_s,ROI_count=len(rois),supported_scales=int(valid.sum()))
  rows.append(row);atomic_json(row,completed)
  if views:
   roi=next(r for r in rois if r['valid']);base=a.output/'views'/f'{i:05d}';cv2.imwrite(str(base)+'_current.png',current['bgr']);cv2.imwrite(str(base)+'_low.png',roi['low']);cv2.imwrite(str(base)+'_native.png',roi['native'])
  if (j+1)%10==0 or j<2:print(json.dumps(dict(completed=j+1,expected=len(selected),encoded=native.encoded_images,elapsed_s=time.perf_counter()-start)),flush=True)
 atomic_json(rows,a.output/'diagnostic_rows_private.json');summary=dict(n=len(rows),no_supported_candidate=sum(r['observed_candidate_count']==0 for r in rows),any_supported_candidate=sum(r['observed_candidate_count']>0 for r in rows),at_least_two_supported=sum(r['observed_candidate_count']>=2 for r in rows),correct_candidate_supported=sum(r['correct_candidate_with_support'] for r in rows),conditional={},seconds=time.perf_counter()-start,new_encoded_images_this_invocation=native.encoded_images,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,scope='TRAIN frozen-feature preflight; no global Goal improvement, instance accuracy, Val or navigation claim')
 for subset_name,predicate in [('all_observed',lambda r:r['observed_candidate_count']>0),('correct_exists_at_least2',lambda r:r['correct_candidate_with_support'] and r['observed_candidate_count']>=2)]:
  rr=[r for r in rows if predicate(r)];entry=dict(n=len(rr),geometry_first_observed=float(np.mean([r['geometry_first_observed_hit20'] for r in rr])) if rr else None)
  for key in ['current384','native_roi']:
   for name in ['full','appearance']:
    k=key+'_'+name;entry[k]=float(np.mean([r['metrics'][k]['hit20'] for r in rr])) if rr else None;entry[k+'_shuffled']=float(np.mean([r['metrics'][k]['shuffled_hit20'] for r in rr])) if rr else None
  summary['conditional'][subset_name]=entry
 atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
