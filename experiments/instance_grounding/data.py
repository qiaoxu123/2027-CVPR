"""Causal history selection public-only; sealed referring labels for supervision."""
import argparse, hashlib, json, time
from pathlib import Path
import cv2, numpy as np, torch
from experiments.bidirectional_core.cache import CachedCoreDataset, DEFAULT_DATA, jsonl, public_landmarks, conservative_object_attributes
from experiments.visual_goal_association.coverage import frame_geometry
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.checkpointing import file_sha256
from experiments.rgb_grounding.city_features import CropEncoder, visual_phrase
from experiments.rgb_grounding.evaluate_features import load

SEED=20261011

def write(path,data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def rank_view(region):
    return (float(region['depth_support']>=.9),float(region['native_min_side'])*float(region['quality'][1])*float(region['quality'][2]),-region['source_step'])
def update_memory(memory, regions, cutoff):
    for r in regions:
        if r['source_step']>cutoff: raise ValueError('future observation')
        if r['object_id'] not in memory or rank_view(r)>rank_view(memory[r['object_id']]):memory[r['object_id']]=r

def choose(memory,current_ids,max_instances=128):
    current=set(current_ids)
    return sorted(memory.values(),key=lambda r:(r['object_id'] not in current,tuple(-x for x in rank_view(r)),r['object_id']))[:max_instances]

def plan(args):
    root=Path(args.output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);cv2.setNumThreads(1);start=time.time();ds=CachedCoreDataset(args.cache)
    objects=json.loads((Path(DEFAULT_DATA)/'cityrefer/objects.json').read_text());lm=public_landmarks(objects)
    ep_path=next(p for p in ds.manifest['frozen_resources'] if Path(p).name=='pilot_episodes.jsonl');eps={e['episode_id']:e for e in jsonl(ep_path)}
    rows=[ds[i] for i in range(len(ds)) if ds[i]['metadata']['direction']=='forward']
    exclude=set()
    if args.exclude_maps: exclude=set(json.loads(Path(args.exclude_maps).read_text())['maps'])
    families=sorted({r['metadata']['family_id'] for r in rows if r['metadata']['map_name'] not in exclude},key=lambda f:hashlib.sha256(('p2-family:'+str(f)).encode()).hexdigest())
    if args.limit:families=families[:args.limit]
    fs=set(families);rows=[r for r in rows if r['metadata']['family_id'] in fs];rows.sort(key=lambda r:(r['metadata']['map_name'],r['metadata']['episode_id'],r['metadata']['cutoff_step']))
    renderer=CityNavObservedViewRenderer(Path(DEFAULT_DATA)/'rgbd',root/'environment_private',image_size=384)
    memory={};last={};plans=[];views={};sealed={};frames=0
    for i,r in enumerate(rows):
        m=r['metadata'];p=r['public'];ep=eps[m['episode_id']];bank=memory.setdefault(m['episode_id'],{});current=[]
        for step in range(last.get(m['episode_id'],-1)+1,m['cutoff_step']+1):
            pose=np.asarray(ep['trajectory_xyzyaw'][step]);ground=float(p['ground_level'])
            if pose[2]<=ground:continue
            obs=renderer.current_rgbd(m['map_name'],pose,ground);regs,qual=frame_geometry(obs,lm[m['map_name']]);frames+=1
            for rg in regs:
                rg.pop('mask',None);rg.update(source_step=step,episode_id=m['episode_id'],map_name=m['map_name'],actual_camera_pose=pose.tolist(),ground_level=ground,depth_support=qual[rg['object_id']]['depth_support'])
                rg['view_key']=hashlib.sha256((m['episode_id']+':'+str(step)+':'+rg['object_id']).encode()).hexdigest()[:24]
            update_memory(bank,regs,step)
            if step==m['cutoff_step']:current=[v['object_id'] for v in regs]
        last[m['episode_id']]=m['cutoff_step'];selected=choose(bank,current)
        for v in selected:views[v['view_key']]=v
        pub={k:p[k].tolist() for k in ['static_logits','grid_xy','bounds','pose_world']}
        plans.append(dict(metadata={k:m[k] for k in ['sample_id','family_id','episode_id','map_name','cutoff_step','instruction','direction']},public=pub,regions=selected,current_ids=current))
        sealed[m['sample_id']]=dict(target_object_id=str(ep['private_labels']['target_object_id']),target_world=r['labels']['target_world'][:2].tolist(),annotation_status='weak original referring annotation, not human visual verification')
        if i%100==0:print(json.dumps(dict(states=i+1,total=len(rows),frames=frames,views=len(views),seconds=time.time()-start)),flush=True)
    torch.save(dict(states=plans,views=views,labels=sealed),root/'plan.pt')
    write(root/'manifest.json',dict(complete=True,states=len(plans),families=len(fs),maps=sorted({r['metadata']['map_name'] for r in rows}),selected_views=len(views),replayed_frames=frames,seconds=time.time()-start,files={'plan.pt':file_sha256(root/'plan.pt')},source_manifest_sha256=file_sha256(Path(args.cache)/'manifest.json'),source_episodes_sha256=file_sha256(ep_path),source_cache=str(Path(args.cache).resolve()),target_conditioned_selection=False))

def encode(args):
    root=Path(args.output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);start=time.time();torch.set_num_threads(2);cv2.setNumThreads(1)
    planroot=Path(args.plan);manifest=json.loads((planroot/'manifest.json').read_text());assert file_sha256(planroot/'plan.pt')==manifest['files']['plan.pt']
    plan=torch.load(planroot/'plan.pt',weights_only=True);reuse={};texts={}
    for path in args.reuse:
        for r in load(path):
            m=r['metadata'];texts[m['instruction']]=r
            for j,roi in enumerate(m['roi_objects']):reuse[(m.get('episode_id'),m.get('cutoff_step'),roi['object_id'])]=(r,j)
    renderer=CityNavObservedViewRenderer(Path(DEFAULT_DATA)/'rgbd',root/'environment_private',image_size=384);encoder=CropEncoder();bank={};hits=0;new=0;lastkey=None;obs=None
    zero_vectors,zero_tokens=encoder.encode([np.zeros((384,384,3),np.uint8)],tokens=True)
    pixel_zero=dict(image=zero_vectors[0],patch_tokens=zero_tokens[0]);smoke=None
    if encoder.native.model.config._commit_hash!='1b426889ea62b5a72bf9839009a1b184bfc9c178':raise ValueError('frozen SigLIP2 revision mismatch')
    for i,(key,v) in enumerate(sorted(plan['views'].items(),key=lambda kv:(kv[1]['map_name'],kv[1]['episode_id'],kv[1]['source_step'],kv[1]['object_id']))):
        old=reuse.get((v['episode_id'],v['source_step'],v['object_id']))
        if old:
            r,j=old;bank[key]=dict(tight=r['features']['tight'][j],context=r['features']['context'][j],patch_tokens=r['patch_tokens'][j]);hits+=1
        else:
            view=(v['episode_id'],v['source_step'])
            if lastkey!=view:obs=renderer.current_rgbd(v['map_name'],np.array(v['actual_camera_pose']),v['ground_level']);lastkey=view
            a,b,c,d=v['bbox'];tight=obs['bgr'][b:d,a:c].copy();w,h=c-a,d-b
            context=obs['bgr'][max(0,int(b-.25*h)):min(384,int(d+.25*h)),max(0,int(a-.25*w)):min(384,int(c+.25*w))].copy()
            vec,tok=encoder.encode([tight,context],tokens=True)
            if smoke is None:
                batch=encoder.native.processor(images=[cv2.cvtColor(tight,cv2.COLOR_BGR2RGB)],return_tensors='pt').to(encoder.device)
                with torch.inference_mode():
                    hidden=encoder.native.model.vision_model(**batch).last_hidden_state
                    exact=torch.nn.functional.normalize(encoder.native.model.vision_model.head(hidden).float(),dim=-1).cpu()[0]
                cosine=float(torch.nn.functional.cosine_similarity(vec[0].float(),exact,dim=0))
                smoke=dict(dimension=len(exact),finite=bool(torch.isfinite(vec).all()),fp16_vs_fp32_cosine=cosine,zero_rgb_is_actual_encoded_black=True,backbone_all_frozen=all(not p.requires_grad for p in encoder.native.model.parameters()))
                if cosine<.995 or len(exact)!=1024:raise ValueError('CUDA precision smoke failed')
                write(root/'cuda_smoke.json',smoke)
            if not torch.isfinite(vec).all() or not torch.isfinite(tok).all():raise ValueError('CUDA smoke invalid')
            bank[key]=dict(tight=vec[0],context=vec[1],patch_tokens=tok[0]);new+=1
        if i%200==0:print(json.dumps(dict(views=i+1,total=len(plan['views']),reused=hits,new=new,seconds=time.time()-start)),flush=True)
    queries={}
    for ins in sorted({r['metadata']['instruction'] for r in plan['states']}):
        if ins in texts:
            old=texts[ins];queries[ins]=dict(texts=old['texts'],language_tokens=old['language_tokens'],language_valid=old['language_valid'],phrase=old['phrase']);continue
        phrase,_=visual_phrase(ins);vec=encoder.native.texts([ins,phrase]).cpu().half()
        batch=encoder.native.processor(text=[ins],padding='max_length',truncation=True,max_length=64,return_tensors='pt').to(encoder.device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): words=encoder.native.model.text_model(**batch).last_hidden_state[0].cpu().half()
        queries[ins]=dict(texts=dict(full=vec[0],phrase=vec[1]),language_tokens=words,language_valid=batch['input_ids'][0].cpu()!=encoder.native.processor.tokenizer.pad_token_id,phrase=phrase)
    torch.save(dict(plan=plan,bank=bank,queries=queries,pixel_zero=pixel_zero),root/'features.pt')
    write(root/'manifest.json',dict(complete=True,states=len(plan['states']),families=manifest['families'],maps=manifest['maps'],unique_views=len(bank),reused_views=hits,new_encoded_views=new,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),siglip_revision=encoder.native.model.config._commit_hash,source_plan_sha256=file_sha256(planroot/'plan.pt'),feature_sha256=file_sha256(root/'features.pt'),precision='FP16 vision transformer FP32 attention pooling',human_semantic_verified=0))

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);q=s.add_parser('plan');q.add_argument('--cache',required=True);q.add_argument('--output',required=True);q.add_argument('--exclude-maps');q.add_argument('--limit',type=int)
    q=s.add_parser('encode');q.add_argument('--plan',required=True);q.add_argument('--output',required=True);q.add_argument('--reuse',action='append',default=[]);a=p.parse_args();(plan if a.mode=='plan' else encode)(a)
if __name__=='__main__':main()
