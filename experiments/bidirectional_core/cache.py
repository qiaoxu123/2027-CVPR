"""Causal current-camera feature cache; private supervision never enters encoders."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from graphnav.checkpointing import atomic_save, atomic_json, file_sha256
from graphnav.a3_runtime import public_static_record
from graphnav.closedloop import PublicEpisode
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs,new_model
from graphnav.token_language import FrozenBERTTokens
from graphnav.observed_landmarks import LandmarkSpatialIndex
from graphnav.renderer import CityNavObservedViewRenderer
from experiments.visual_grounding.data.audit_attributes import object_attributes,bind_phrase,COLORS
from .vision import FrozenROIEncoder
from .model import CoreModel,collate_core,core_losses

DEFAULT_DATA='/home/rental/20260922_1/Workspace/DATA/sbfnav-data'
DEFAULT_SBF='/home/rental/20260922_1/Workspace/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'
DEFAULT_A3='/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/runs/static_final_analysis_20261010/E1_seed1/best.pt'


def jsonl(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def public_landmarks(raw):
    """Retain valid unnamed polygons; identity is bookkeeping, not a feature."""
    result={}
    for name,entries in raw.items():
        valid=[]
        for oid,item in entries.items():
            p=np.asarray(item.get('contour',[]),np.float32)
            if p.ndim!=2 or p.shape[1:]!=(2,) or len(p)<3 or not np.isfinite(p).all():continue
            valid.append(dict(id=str(oid),name=str(item.get('name','')),object_type=item.get('object_type','Unknown'),
                contour=p,low=p.min(0),high=p.max(0)))
        result[name]=LandmarkSpatialIndex(valid)
    return result


class CachedCoreDataset(Dataset):
    def __init__(self,root,condition='E1',verify=True):
        if condition not in ('E0','E1','E2'):raise ValueError('invalid matched condition')
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['complete']:raise ValueError('incomplete core cache')
        if not {'queries.pt','queries.jsonl',*self.manifest['chunks']} <= set(self.manifest['files']):
            raise ValueError('unbound public query/chunk cache')
        if verify:
            for path,sha in self.manifest['files'].items():
                if file_sha256(self.root/path)!=sha:raise ValueError('cache changed: '+path)
        self.queries=torch.load(self.root/'queries.pt',map_location='cpu',weights_only=True)
        self.rows=[]
        for path in self.manifest['chunks']:
            self.rows.extend(torch.load(self.root/path,map_location='cpu',weights_only=True))
        lookup={r['metadata']['sample_id']:i for i,r in enumerate(self.rows)}
        if len(lookup)!=len(self.rows):raise ValueError('duplicate cache state')
        self.indices=[lookup[r['E0_forward_sample_id'] if condition=='E0' else r['E1_sample_id']]
                      for r in self.manifest['matched_budget']]
        self.condition=condition
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        row=self.rows[self.indices[index]];public=dict(self.queries[row['query_index']]);public.update(row['public'])
        return dict(public=public,labels=row['labels'],metadata=row['metadata'])


def conservative_attributes(phrase,typ):
    attrs=bind_phrase(phrase,typ)
    colors=set(re.findall(r'\b('+COLORS+r')\b',str(phrase).lower().replace('grey','gray')))
    if typ=='Building' and len(colors)>1:
        attrs={k:v for k,v in attrs.items() if k not in ('roof_color','building_color')}
    return attrs


def conservative_object_attributes(descriptions,typ):
    attrs=object_attributes(descriptions,typ)
    if typ=='Building':
        for kind in ('roof_color','building_color'):
            if kind in attrs and any(len(set(re.findall(r'\b('+COLORS+r')\b',p.lower().replace('grey','gray'))))>1 for p in attrs[kind]['support_phrases']):
                del attrs[kind]
    return attrs


def supervision(ep,source_record,regions,attrs,processed,objects):
    positive=np.zeros(len(regions),bool);judged=np.zeros(len(regions),bool)
    unknown=(torch.zeros(len(regions),dtype=torch.bool),torch.zeros(len(regions),dtype=torch.bool))
    target=ep['private_labels']['target_object_id']
    if target is None or not any(r['object_id']==str(target) for r in regions):return unknown
    if ep['direction']=='reverse':
        desc=ep.get('reverse_geometry_description') or {}
        pos=set(map(str,desc.get('label_positive_object_ids',[])))
        neg=set(map(str,desc.get('label_negative_object_ids',[])))
        for i,r in enumerate(regions):
            positive[i]=r['object_id'] in pos;judged[i]=positive[i] or r['object_id'] in neg
    else:
        target=str(ep['private_labels']['target_object_id']);name=ep['map_name'];typ=objects[name].get(target,{}).get('object_type','Unknown')
        # Description target field is supplied language; labels remain weak and separate.
        descriptions=processed.get(name,{}).get(target,[])
        did=int(source_record.episode_id[2])
        phrase=descriptions[did].get('target','') if did<len(descriptions) else ''
        query=conservative_attributes(phrase,typ)
        for i,r in enumerate(regions):
            if r['object_type']!=typ:continue
            candidate=attrs.get((name,r['object_id']),{})
            shared=[k for k in query if k in candidate]
            if not shared:continue
            # Conjunction: complete required attributes for positives; any supported
            # contrary value establishes a weak negative, missing attributes unknown.
            same=len(shared)==len(query) and all(candidate[k]['label']==query[k] for k in shared)
            contrary=any(candidate[k]['label']!=query[k] for k in shared)
            positive[i]=same;judged[i]=same or contrary
    if ep['direction']=='forward' and not any(positive[i] and r['object_id']==str(target) for i,r in enumerate(regions)):
        return unknown
    return torch.from_numpy(positive),torch.from_numpy(judged)


def finalize_cache_integrity(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('cannot finalize incomplete cache')
    if manifest.get('target_positive_guard_finalized') and 'queries.jsonl' in manifest['files']:return manifest
    for p,sha in manifest['files'].items():
        if file_sha256(root/p)!=sha:raise ValueError('pre-finalization cache hash mismatch: '+p)
    episodepaths=[p for p in manifest['frozen_resources'] if Path(p).name=='pilot_episodes.jsonl']
    if len(episodepaths)!=1:raise ValueError('cannot locate frozen episode identities')
    if file_sha256(episodepaths[0])!=manifest['frozen_resources'][episodepaths[0]]:
        raise ValueError('frozen episode identities changed before finalization')
    episodes={r['episode_id']:r for r in jsonl(episodepaths[0])}
    changed=eligible=0
    for path in manifest['chunks']:
        rows=torch.load(root/path,map_location='cpu',weights_only=True);dirty=False
        for row in rows:
            target=episodes[row['metadata']['episode_id']]['private_labels']['target_object_id']
            positives=row['labels']['positive_roi_mask'];judged=row['labels']['semantic_valid_mask']
            if row['metadata']['direction']=='forward' and not any(bool(positives[i]) and r['object_id']==str(target) for i,r in enumerate(row['metadata']['roi_objects'])):
                if judged.any():
                    row['labels']['positive_roi_mask']=torch.zeros_like(positives);row['labels']['semantic_valid_mask']=torch.zeros_like(judged);changed+=1;dirty=True
            pos=row['labels']['positive_roi_mask'];valid=row['labels']['semantic_valid_mask'];eligible+=int(bool(pos.any() and (valid & ~pos).any()))
        if dirty:atomic_save(rows,root/path);manifest['files'][path]=file_sha256(root/path)
    manifest['files']['queries.jsonl']=file_sha256(root/'queries.jsonl')
    manifest['target_positive_guard_finalized']=True
    manifest['integrity_finalize']=dict(label_only_rows_changed=changed,vl_eligible=eligible,public_features_reencoded=False,queries_jsonl_late_bound=True)
    manifest['counts']['vl_eligible']=eligible
    atomic_json(manifest,root/'manifest.json');return manifest


def build(args):
    started=time.perf_counter();root=Path(args.output)
    if root.exists():raise ValueError('refusing cache overwrite')
    root.mkdir(parents=True)
    data=Path(args.metadata)/args.split
    episodes={r['episode_id']:r for r in jsonl(data/'pilot_episodes.jsonl')}
    prefixes=jsonl(data/'pilot_prefixes.jsonl')
    if args.limit:prefixes=prefixes[:args.limit]
    prefix_ids={r['sample_id'] for r in prefixes}
    budgets=[r for r in jsonl(data/'pilot_matched_e0_e1.jsonl') if r['E0_forward_sample_id'] in prefix_ids and r['E1_sample_id'] in prefix_ids]
    # Smoke selection may omit matched pairs; it still runs real causal states.
    if args.limit:budgets=[dict(E0_forward_sample_id=r['sample_id'],E1_sample_id=r['sample_id']) for r in prefixes]
    resources={str(Path(args.a3_checkpoint).resolve()):file_sha256(args.a3_checkpoint)}
    for p in [data/'pilot_episodes.jsonl',data/'pilot_prefixes.jsonl',data/'pilot_matched_e0_e1.jsonl',
              Path(args.data_root)/'cityrefer/objects.json',Path(args.data_root)/'cityrefer/processed_descriptions.json']:
        resources[str(p.resolve())]=file_sha256(p)
    manifest=dict(complete=False,hidden=None,top_k=None,files={},chunks=[],matched_budget=budgets,frozen_resources=resources,
        protocol=dict(split=args.split,cache_builder_sha256=file_sha256(__file__),vision_encoder_sha256=file_sha256(Path(__file__).with_name('vision.py')),cache_a3=True,freeze_a3=True,freeze_siglip=True,current_camera_only=True,
            reverse_camera='rerender actual visited xyz with recomputed reverse travel yaw; not recorded return flight',
            invalid_agl='<=0 gives unknown RGB, altitude never corrected',target_coordinates='private labels only',
            feature='one real current384 frame, native SigLIP contextual patch masked head, max32ROI public geometry',
            annotations='geometry weak projection; language weak attributes; no human verified attribute labels',building_multicolor='conservative drop any roof/building color labels with multiple distinct target/support phrase colors',vl_target_visibility='actual target object must have a current valid ROI; reverse needs reliable launch target identity'))
    atomic_json(manifest,root/'manifest.json')
    sys.path.insert(0,args.sbf_source)
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.coordinates import MapTransform
    source=CityNavDataset(args.data_root,args.split,version='processed_citynav')
    catalog=CityReferCatalog(Path(args.data_root)/'cityrefer')
    manifest['frozen_resources'][str(Path(source.annotation_path).resolve())]=file_sha256(source.annotation_path)
    raw=json.loads((Path(args.data_root)/'cityrefer/objects.json').read_text())
    processed=json.loads((Path(args.data_root)/'cityrefer/processed_descriptions.json').read_text())
    landmarks=public_landmarks(raw)
    attrs={(m,str(oid)):conservative_object_attributes(processed.get(m,{}).get(str(oid),[]),o.get('object_type','Unknown'))
           for m,entries in raw.items() for oid,o in entries.items()}
    state=torch.load(args.a3_checkpoint,map_location='cpu',weights_only=True)
    config=SimpleNamespace(**state['args']);device=torch.device(args.device)
    a3=new_model(config,device).eval();a3.load_state_dict(state['model'],strict=True);a3.requires_grad_(False)
    bert=FrozenBERTTokens(name=config.text_backbone,max_length=getattr(config,'max_text_length',96),device=args.device)
    native=FrozenROIEncoder(device=args.device,max_rois=32)
    manifest.update(hidden=a3.hidden,top_k=a3.top_k,backbone_revision=native.revision)
    # Pin exact local SigLIP weight files, never download another revision here.
    hfhome=Path.home()/'.cache/huggingface/hub/models--google--siglip2-large-patch16-384/snapshots'/native.revision
    for p in sorted(hfhome.glob('*.safetensors')):
        manifest['frozen_resources'][str(p.resolve())]=file_sha256(p)
    if not list(hfhome.glob('*.safetensors')):
        raise ValueError('cannot pin local SigLIP checkpoint files')
    transforms={};queries=[];querylookup={};rowqueries=[]
    for row in prefixes:
        ep=episodes[row['episode_id']];name=ep['map_name'];record=source[ep['source_episode_index']]
        if ep['split']!=args.split or row['split']!=args.split or record.map_name!=name or row['map_name']!=name:
            raise ValueError('source split/map mismatch')
        if ep['direction']=='forward' and ep['instruction']!=record.instruction:
            raise ValueError('forward source instruction mismatch')
        traj=np.asarray(ep['trajectory_xyzyaw']);cutoff=row['cutoff_step']
        if not 0<=cutoff<len(traj) or not np.allclose(row['pose_world'],traj[cutoff],rtol=0,atol=1e-6):
            raise ValueError('direction current pose/cutoff mismatch')
        if row['source_step']!=(cutoff if ep['direction']=='forward' else len(traj)-1-cutoff):
            raise ValueError('source-direction time mismatch')
        if name not in transforms:
            transforms[name]=MapTransform.from_raster(Path(args.data_root)/'rgbd'/f'{name}.tif',canvas_size=224,field_size=28)
        if ep['direction']=='forward':
            refs=tuple(catalog.referenced_landmarks(record));names=tuple(catalog.referenced_landmark_names(record))
        else:
            anchor=(ep.get('reverse_geometry_description') or {}).get('public_anchor',{})
            refs=();names=(str(anchor['name']),) if anchor.get('name') else ()
        public=PublicEpisode(name,ep['instruction'],np.asarray(row['pose_world']),refs,names)
        query=public_static_record(public,transforms[name],catalog,max_landmarks=config.max_landmarks)
        key=json.dumps(query,sort_keys=True)
        if key not in querylookup:querylookup[key]=len(queries);queries.append(query)
        rowqueries.append(querylookup[key])
    (root/'queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
    manifest['files']['queries.jsonl']=file_sha256(root/'queries.jsonl')
    ds=FullCandidateDataset(root/'queries.jsonl',grid_size=config.grid_size,max_landmarks=config.max_landmarks,
        max_roads=config.max_roads,max_relations=config.max_relations,prior_sigma_m=config.prior_sigma_m,
        rule_sigma_m=config.rule_sigma_m,deduplicate_queries=False)
    cachedqueries=[]
    for begin in range(0,len(ds),args.batch_size):
        batch=collate_full([ds[i] for i in range(begin,min(len(ds),begin+args.batch_size))])
        inputs,tokens,mask,names=_batch_inputs(batch,bert,device,'geometry')
        with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            field=a3(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
        texts=native.texts([q['instruction'] for q in queries[begin:begin+args.batch_size]])
        for i in range(len(texts)):
            cachedqueries.append(dict(static_logits=field.belief_logits[i].float().cpu(),
                static_context=field.dense_context[i].half().cpu(),grid_xy=inputs['grid_xy'][i].float().cpu(),
                bounds=torch.tensor(queries[begin+i]['world_bounds'],dtype=torch.float32),native_text=texts[i]))
        if begin%256==0:print(json.dumps(dict(phase='static',completed=len(cachedqueries),total=len(queries),seconds=time.perf_counter()-started)),flush=True)
    atomic_save(cachedqueries,root/'queries.pt');manifest['files']['queries.pt']=file_sha256(root/'queries.pt')
    del a3,bert,field,tokens,names;torch.cuda.empty_cache()
    renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',root/'environment_private',image_size=384)
    counts=Counter();rowsbuffer=[]
    for begin in range(0,len(prefixes),args.batch_size):
        chunk=prefixes[begin:begin+args.batch_size];observations=[];indices=[]
        for i,row in enumerate(chunk):
            pose=np.asarray(row['pose_world']);counts['states']+=1
            if pose[2]-row['ground_level']<=0:counts['agl_nonpositive_unknown']+=1;continue
            observations.append(renderer.current_rgbd(row['map_name'],pose,row['ground_level']));indices.append(i)
        encoded=native.encode(observations,[landmarks[chunk[i]['map_name']] for i in indices]) if indices else []
        byindex=dict(zip(indices,encoded))
        for i,row in enumerate(chunk):
            ep=episodes[row['episode_id']];regions,vectors=byindex.get(i,([],torch.zeros(0,1024,dtype=torch.float16)))
            qid=rowqueries[begin+i];query=cachedqueries[qid];bounds=query['bounds'].numpy();gridworld=bounds[:2]+query['grid_xy'].numpy()*(bounds[2:]-bounds[:2])
            rid=torch.tensor([int(np.square(gridworld-np.asarray(r['center_world'])).sum(-1).argmin()) for r in regions],dtype=torch.long)
            positives,judged=supervision(ep,source[ep['source_episode_index']],regions,attrs,processed,raw)
            counts['rois']+=len(regions);counts['vl_eligible']+=int(bool(positives.any() and (judged & ~positives).any()))
            labels={k:torch.tensor(row['labels'][k],dtype=torch.float32) for k in ('target_world','teacher_world','valid_teacher_z')}
            labels.update(positive_roi_mask=positives,semantic_valid_mask=judged)
            public=dict(pose_world=torch.tensor(row['pose_world'],dtype=torch.float32),ground_level=torch.tensor(row['ground_level'],dtype=torch.float32),
                roi_features=vectors,roi_grid_ids=rid,roi_valid=torch.ones(len(regions),dtype=torch.bool),
                roi_quality=torch.tensor([r['quality'] for r in regions],dtype=torch.float32).reshape(-1,3))
            for region in regions:
                region.update(roi_pixel_size=region['pixel_size'],observed_step=row['cutoff_step'],observation_cutoff_step=row['cutoff_step'],roi_visibility=region['quality'][2],attributes=attrs.get((row['map_name'],region['object_id']),{}))
            metadata={k:row[k] for k in ('sample_id','episode_id','family_id','map_name','source_episode_index','direction','cutoff_step','source_step','slot','static_language_eligible')}
            metadata.update(roi_objects=regions,current_observation_step=row['cutoff_step'],
                actual_camera_pose=row['pose_world'],instruction=row['instruction'],annotation='language_weak_not_visual_verified')
            rowsbuffer.append(dict(query_index=qid,public=public,labels=labels,metadata=metadata))
        if len(rowsbuffer)>=256 or begin+len(chunk)==len(prefixes):
            filename=f'rows_{begin+len(chunk):06}.pt';atomic_save(rowsbuffer,root/filename)
            manifest['chunks'].append(filename);manifest['files'][filename]=file_sha256(root/filename);rowsbuffer=[]
            manifest.update(counts=dict(counts),seconds=time.perf_counter()-started);atomic_json(manifest,root/'manifest.json')
            print(json.dumps(dict(phase='visual',completed=counts['states'],total=len(prefixes),seconds=manifest['seconds'],counts=dict(counts))),flush=True)
    manifest.update(complete=True,seconds=time.perf_counter()-started,counts=dict(counts),target_positive_guard_finalized=True)
    atomic_json(manifest,root/'manifest.json')
    if args.limit:
        dataset=CachedCoreDataset(root);public,labels,_=collate_core([dataset[i] for i in range(len(dataset))])
        public={k:v.to(device) for k,v in public.items()};labels={k:v.to(device) for k,v in labels.items()}
        torch.manual_seed(0);model=CoreModel(hidden=manifest['hidden'],top_k=manifest['top_k']).to(device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):output=model(public)
        losses=core_losses(output,public,labels,vl_weight=.1);losses['total'].backward()
        smoke=dict(real_states=len(dataset),finite=bool(torch.isfinite(losses['total'])),loss=float(losses['total'].detach()),
            matcher_grad=sum(float(p.grad.norm()) for p in model.matcher.parameters() if p.grad is not None),
            trajectory_grad=sum(float(p.grad.norm()) for p in model.trajectory.parameters() if p.grad is not None),
            frozen_resources_unchanged=all(file_sha256(p)==sha for p,sha in manifest['frozen_resources'].items()),counts=dict(counts))
        if not smoke['finite'] or smoke['trajectory_grad']<=0 or not smoke['frozen_resources_unchanged']:raise ValueError('real CUDA smoke failed')
        atomic_json(smoke,root/'cuda_smoke.json');print(json.dumps(smoke),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',default='runs/bidirectional_core_test/data');p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['train_seen','val_seen'],required=True);p.add_argument('--data-root',default=DEFAULT_DATA)
    p.add_argument('--sbf-source',default=DEFAULT_SBF);p.add_argument('--a3-checkpoint',default=DEFAULT_A3)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--limit',type=int);p.add_argument('--device',default='cuda');build(p.parse_args())
if __name__=='__main__':main()
