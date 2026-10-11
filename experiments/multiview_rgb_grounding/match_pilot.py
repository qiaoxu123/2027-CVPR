"""Frozen, Train-only exploratory identity rankings on supported true 3D ROIs."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from .render import OBJECTS
from experiments.rgb_grounding.city_features import CropEncoder,visual_phrase
from experiments.visual_grounding.adapters.remoteclip import RemoteCLIPEncoder
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from experiments.rgb_feasibility.core import percentile
from graphnav.checkpointing import file_sha256
from .geometry import world_grid

def identity(scores,ids,target):
 if target not in ids:return None,None
 x=np.asarray(scores);j=ids.index(target);higher=int((x>x[j]+1e-7).sum());tie=int(np.isclose(x,x[j],atol=1e-7,rtol=0).sum())
 return float(1/tie if higher==0 else 0),float(higher+(tie+1)/2)

def run(a):
 out=Path(a.output);assert not out.exists();out.mkdir(parents=True);start=time.time();cv2.setNumThreads(1);torch.set_num_threads(1)
 protocol_bytes=Path(__file__).with_name('pilot_matching_protocol.json').read_bytes();protocol=json.loads(protocol_bytes);root=Path(a.render);manifest=json.loads((root/'manifest.json').read_text())
 if a.allow_internal_dev:assert 'plan_dev_error_probe' in manifest['source_plan']
 else:assert 'plan_train' in manifest['source_plan']
 plan=torch.load(Path(manifest['source_plan'])/'plan.pt',weights_only=True);state_lookup={s['metadata']['sample_id']:s for s in plan['states']};objects=json.loads(OBJECTS.read_text());cases=[];coverage=[]
 grouped={}
 for frame in manifest['frames']:
  if frame['splat_radius']==1:grouped.setdefault(frame['metadata']['sample_id'],{})[frame['tilt']]=frame
 for sample,frames in sorted(grouped.items()):
  assert set(frames)=={0,20,40,60};meta=frames[0]['metadata'];target=plan['labels'][sample]['target_object_id'];state=state_lookup[sample];grid=world_grid(state['public']);static=np.asarray(state['public']['static_logits'])
  for scope,tilts in protocol['views'].items():
   common=set(v['object_id'] for v in frames[0]['regions'] if v['credible'])
   for tilt in tilts:common &= set(v['object_id'] for v in frames[tilt]['regions'] if v['credible'])
   good=all(frames[t]['frame_quality_pass'] for t in tilts);allids=sorted(common,key=lambda x:int(x)) if good else []
   centers=np.array([np.mean(objects[meta['map_name']][oid]['contour'],axis=0) for oid in allids]);geo=static[np.linalg.norm(centers[:,None]-grid[None],axis=-1).argmin(1)] if allids else np.zeros(0)
   chosen=sorted(range(len(allids)),key=lambda j:(-float(geo[j]),int(allids[j])))[:20];ids=[allids[j] for j in chosen];geo=geo[chosen]
   coverage.append(dict(sample_id=sample,family_id=meta['family_id'],map_name=meta['map_name'],scope=scope,frame_quality_pass=good,common_candidate_ids=allids,candidate_ids=ids,target_common=target in allids,target_top20=target in ids))
   if len(ids)<2:continue
   crops={};regions={}
   for tilt in tilts:
    frame=frames[tilt];image=cv2.imread(str(root/frame['frame_key']/f'tilt{tilt}_r1_rgb.png'));lookup={r['object_id']:r for r in frame['regions']};regions[tilt]=[lookup[oid] for oid in ids];tight=[];context=[]
    for oid in ids:
     x0,y0,x1,y1=lookup[oid]['bbox'];w=x1-x0;h=y1-y0;tight.append(image[y0:y1,x0:x1].copy());context.append(image[max(0,int(y0-.25*h)):min(512,int(y1+.25*h)),max(0,int(x0-.25*w)):min(512,int(x1+.25*w))].copy())
    crops[tilt]=(tight,context)
   phrase,supported=visual_phrase(meta['instruction']);groups={}
   for j,oid in enumerate(ids):groups.setdefault(objects[meta['map_name']][oid]['object_type'],[]).append(j)
   perm=list(range(len(ids)))
   for group in groups.values():
    if len(group)>1:
     for left,right in zip(group,group[1:]+group[:1]):perm[left]=right
   views=[('nadir',[0]),('oblique'+str(tilts[1]),[tilts[1]]),('dual_nadir'+str(tilts[1]),[0,tilts[1]])]
   if scope=='four_view':views=[('nadir',[0]),('oblique40',[40]),('dual_nadir40',[0,40]),('multi_0_20_40_60',tilts)]
   cases.append(dict(meta=meta,ids=ids,target=target,target_common=target in allids,crops=crops,regions=regions,phrase=phrase,supported=supported,perm=perm,geometry=geo,scope=scope,tilts=tilts,views=views))
 (out/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
 rows=[];models={};feature_cache={};torch.cuda.reset_peak_memory_stats()
 for name in protocol['models']:
  loaded=time.time()
  if name.startswith('SigLIP'):
   enc=CropEncoder();assert enc.native.model.config._commit_hash=='1b426889ea62b5a72bf9839009a1b184bfc9c178';encode=lambda ims:enc.encode(ims)[0].float();texts=lambda ss:enc.native.texts(ss).float().cpu();dim=1024
   from huggingface_hub import hf_hub_download
   weights=hf_hub_download('google/siglip2-large-patch16-384','model.safetensors',revision=enc.native.model.config._commit_hash,local_files_only=True)
   provenance=dict(revision=enc.native.model.config._commit_hash,checkpoint_sha256=file_sha256(weights),all_weights_frozen=all(not p.requires_grad for p in enc.native.model.parameters()))
  else:
   enc=RemoteCLIPEncoder('ViT-L-14','/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/weights/RemoteCLIP-ViT-L-14.pt');encode=lambda ims:torch.tensor(enc.images(ims));texts=lambda ss:torch.tensor(enc.texts(ss));dim=768
   provenance=dict(checkpoint_sha256=enc.checkpoint_sha256,all_weights_frozen=all(not p.requires_grad for p in enc.model.parameters()))
  raw_encode=encode;timing={'images':0,'seconds':0.}
  def measured_encode(images):
   torch.cuda.synchronize();t=time.time();value=raw_encode(images);torch.cuda.synchronize();timing['images']+=len(images);timing['seconds']+=time.time()-t;return value
  encode=measured_encode
  black=encode([np.zeros((384,384,3),np.uint8)])[0];assert black.shape==(dim,) and torch.isfinite(black).all()
  for case in cases:
   ids=case['ids'];n=len(ids);vectors={};query=texts([case['meta']['instruction'],case['phrase']]);smoke=True
   for tilt in case['tilts']:
    for kind,k in [('tight',0),('context',1)]:
     vec=F.normalize(encode(case['crops'][tilt][k]).float(),dim=-1);assert torch.isfinite(vec).all() and vec.shape==(n,dim);vectors[(tilt,kind)]=vec
   feature_cache[(name,case['meta']['sample_id'],case['scope'])]=dict(vectors=vectors,query=query,black=black)
   for language in protocol['languages']:
    kind='context' if language=='full_with_context' else 'tight';q=query[1 if language=='conservative_attribute' else 0]
    for control in protocol['controls']:
     z={tilt:(torch.stack([black]*n) if control=='encoded_black' else vectors[(tilt,kind)][case['perm']] if control=='same_class_cyclic_shuffle' else vectors[(tilt,kind)]) for tilt in case['tilts']}
     for view,angles in case['views']:
      v=F.normalize(sum(z[t] for t in angles),dim=-1)
      scores=(v@q).tolist()
      for modality in protocol['modalities']:
       s=scores if modality=='rgb_only' else ((percentile(scores)+percentile(case['geometry']))/2).tolist();r1,rank=identity(s,ids,case['target']);hard=[j for j in range(n) if objects[case['meta']['map_name']][ids[j]]['object_type']==objects[case['meta']['map_name']][case['target']]['object_type']];hr1,_=identity([s[j] for j in hard],[ids[j] for j in hard],case['target']) if len(hard)>1 else (None,None)
       if case['target_common'] and r1 is None:
        r1=0.
        if hard:hr1=0.
       rows.append(dict(sample_id=case['meta']['sample_id'],family_id=case['meta']['family_id'],map_name=case['meta']['map_name'],scope=case['scope'],model=name,language=language,control=control,view=view,modality=modality,candidate_ids=ids,candidate_scores=s,target_object_id=case['target'],target_rank=rank,phrase_supported=case['supported'],metrics=dict(instance_r1=r1,hard_r1=hr1)))
  models[name]=dict(**provenance,seconds=time.time()-loaded,image_encoding=timing,peak_gpu_bytes=torch.cuda.max_memory_allocated(),dimension=dim,finite_embeddings=True,embedding_norm_checked=True,backbone_training=0)
  del enc;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 summary={};contrasts={}
 for name in protocol['models']:
  for language in protocol['languages']:
   for scope in protocol['views']:
    viewnames=['nadir','oblique20','dual_nadir20'] if scope=='pair20' else ['nadir','oblique40','dual_nadir40','multi_0_20_40_60'];dual='dual_nadir20' if scope=='pair20' else 'dual_nadir40'
    for modality in protocol['modalities']:
     for control in protocol['controls']:
      for view in viewnames:
       key='/'.join([name,language,scope,modality,control,view]);selected=[r for r in rows if (r['model'],r['language'],r['scope'],r['modality'],r['control'],r['view'])==(name,language,scope,modality,control,view)];summary[key]={k:summarize_rows(selected,k,seed=20261011) for k in ['instance_r1','hard_r1']}
     base=lambda view,control:[r for r in rows if (r['model'],r['language'],r['scope'],r['modality'],r['control'],r['view'])==(name,language,scope,modality,control,view)]
     key='/'.join([name,language,scope,modality]);contrasts[key]={'dual_minus_nadir':paired_difference(base('nadir','real'),base(dual,'real'),'instance_r1',seed=20261011),'real_minus_black':paired_difference(base(dual,'encoded_black'),base(dual,'real'),'instance_r1',seed=20261011),'real_minus_shuffle':paired_difference(base(dual,'same_class_cyclic_shuffle'),base(dual,'real'),'instance_r1',seed=20261011)}
 torch.save(feature_cache,out/'features.pt');(out/'predictions.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n');(out/'metrics.json').write_text(json.dumps(dict(models=models,summary=summary,contrasts=contrasts,seconds=time.time()-start,protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),dev_probe_protocol_sha256=hashlib.sha256(Path(__file__).with_name('dev_probe_protocol.json').read_bytes()).hexdigest() if a.allow_internal_dev else None,formal_gate_evidence=False,train_only=not a.allow_internal_dev,internal_dev=a.allow_internal_dev),indent=2,allow_nan=False)+'\n');print(json.dumps({'cases':len(cases),'seconds':time.time()-start,'models':models}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--render',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-internal-dev',action='store_true');run(p.parse_args())
