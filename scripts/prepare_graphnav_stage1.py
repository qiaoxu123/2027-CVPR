"""Prepare actual current and causal prefix observations; never reuse averaged views as current RGB."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256,provenance
from graphnav.observed_patch_scales import observed_region_masks,masked_native_pool
from graphnav.evidence_grounding import appearance_query

_WORKER=None

def initialize_worker(source,data_root):
    global _WORKER
    sys.path.insert(0,source)
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.renderer import CityNavObservedViewRenderer
    _WORKER=(CityNavDataset(data_root,'train_seen',version='processed_citynav'),
             CityNavObservedViewRenderer(Path(data_root)/'rgbd',Path(data_root)/'unused_stage1',image_size=384),GROUND_LEVEL)
    torch.set_num_threads(1)


def prepare_episode(task):
    index,steps,world=task;source,renderer,ground=_WORKER
    record=source[index];frames=[]
    for step in steps:
        pose=record.trajectory[step].astype(np.float32)
        if pose[2]-ground[record.map_name]<=0:
            frames.append(dict(step=step,pose=pose,empty=True));continue
        observation=renderer.current_rgbd(record.map_name,pose,ground[record.map_name])
        masks,quality=observed_region_masks(observation,world)
        valid=masks.any(-1);ids=np.flatnonzero(valid.any(-1))
        # NumPy IPC avoids creating hundreds of shared-memory file descriptors
        # per returned episode in PyTorch's multiprocessing tensor reducer.
        frames.append(dict(step=step,pose=pose,empty=False,bgr=observation['bgr'],
            roi_ids=ids,roi_valid=valid[ids],roi_quality=quality[ids],pool_masks=masks[valid]))
    return index,frames


def main():
    p=argparse.ArgumentParser()
    for key in ('source-cache','a3-checkpoint','sbf-source','data-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--microbatch',type=int,default=24)
    p.add_argument('--smoke-chunks',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--resume-compatible-transport-fix',action='store_true')
    a=p.parse_args();root=Path(a.output);src=Path(a.source_cache)
    if root.exists() and not a.resume:raise ValueError('refusing cache overwrite')
    root.mkdir(parents=True,exist_ok=True)
    old=json.loads((src/'manifest.json').read_text());baseline=torch.load(a.a3_checkpoint,map_location='cpu',weights_only=True)
    if not old['complete'] or old['protocol']['split']!='train_seen':raise ValueError('complete real Train Seen source required')
    for key in ('grid_size','max_landmarks','max_roads','max_relations','prior_sigma_m','rule_sigma_m'):
        if old['protocol']['a3_args'][key]!=baseline['args'][key]:raise ValueError('cached public geometry mismatch: '+key)
    manifest=dict(complete=False,chunks=[],protocol=dict(old['protocol'],a3_args=baseline['args'],
        a3_sha256=file_sha256(a.a3_checkpoint),history='Separate actual frames; past-only quality-weighted memory with pose dedup',
        cache_kind='stage1-current-and-prefix-v1'),source_manifest_sha256=file_sha256(src/'manifest.json'),
        source_queries_sha256=file_sha256(src/'queries.jsonl'),backbone_revision=old['backbone_revision'],
        preparation_sha256=file_sha256(__file__),provenance=provenance(),smoke_only=bool(a.smoke_chunks))
    if a.resume:
        saved=json.loads((root/'manifest.json').read_text())
        for key in ('protocol','source_manifest_sha256','source_queries_sha256','backbone_revision','smoke_only'):
            if saved[key]!=manifest[key]:raise ValueError('resume source/implementation mismatch: '+key)
        if saved['preparation_sha256']!=manifest['preparation_sha256']:
            if not a.resume_compatible_transport_fix:raise ValueError('preparation code changed; explicit compatible IPC migration required')
            saved.setdefault('compatible_transport_revisions',[]).append(dict(previous=saved['preparation_sha256'],current=manifest['preparation_sha256'],reason='Tensor IPC replaced by identical NumPy values to avoid ancillary file-descriptor exhaustion'))
            saved['preparation_sha256']=manifest['preparation_sha256']
        manifest=saved
        for chunk in manifest['chunks']:
            if file_sha256(root/chunk['file'])!=chunk['sha256']:raise ValueError('corrupt frame cache')
        if manifest['complete']:return
    (root/'queries.jsonl').write_bytes((src/'queries.jsonl').read_bytes());manifest['queries_sha256']=file_sha256(root/'queries.jsonl')
    native=ObservedSigLIPMatchingEncoder(microbatch=a.microbatch)
    if getattr(native.model.config,'_commit_hash',None)!=manifest['backbone_revision']:raise ValueError('SigLIP revision mismatch')
    queries=[json.loads(line)for line in (src/'queries.jsonl').read_text().splitlines()]
    grid=np.array(old['canonical_grid_xy'],np.float32);started=time.perf_counter()
    limit=min(len(old['chunks']),a.smoke_chunks or len(old['chunks']))
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker,initargs=(a.sbf_source,a.data_root)) as pool:
        for ci in range(len(manifest['chunks']),limit):
            chunk=old['chunks'][ci]
            if file_sha256(src/chunk['file'])!=chunk['sha256']:raise ValueError('original frozen cache changed')
            rows=torch.load(src/chunk['file'],map_location='cpu',weights_only=True)['rows'];by_episode={}
            for row in rows:by_episode.setdefault(row['episode_index'],[]).append(row)
            tasks=[]
            for index,items in by_episode.items():
                bounds=np.array(queries[items[0]['query_index']]['world_bounds'],np.float32)
                world=bounds[:2]+grid*(bounds[2:]-bounds[:2]);steps=sorted(set(s for r in items for s in r['history_steps']))
                tasks.append((index,steps,world))
            episodes=dict(pool.map(prepare_episode,tasks))
            for frames in episodes.values():
                for f in frames:
                    for key in ('pose','roi_ids','roi_valid','roi_quality','pool_masks'):
                        if key in f:f[key]=torch.from_numpy(f[key])
            flat=[f for frames in episodes.values() for f in frames if not f['empty']]
            for start in range(0,len(flat),a.microbatch):
                batch=flat[start:start+a.microbatch]
                import cv2
                images=native.processor(images=[cv2.cvtColor(f['bgr'],cv2.COLOR_BGR2RGB)for f in batch],return_tensors='pt').to(native.device)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    patches=native.model.vision_model(**images).last_hidden_state
                    for f,x in zip(batch,patches):
                        valid=f['roi_valid'];vectors=torch.zeros(*valid.shape,native.dimension,device=native.device)
                        if valid.any():vectors[valid.to(native.device)]=F.normalize(masked_native_pool(native.model.vision_model.head,x,f['pool_masks'].to(native.device)).float(),dim=-1)
                        f['roi_features']=vectors.cpu().half().clone()
                        del f['bgr'],f['pool_masks']
            for frames in episodes.values():
                for f in frames:
                    if f['empty']:f.update(roi_ids=torch.empty(0,dtype=torch.long),roi_valid=torch.zeros(0,3,dtype=torch.bool),roi_quality=torch.zeros(0,3,3),roi_features=torch.zeros(0,3,native.dimension,dtype=torch.float16))
            clean=[{k:v for k,v in r.items()if k not in ('roi_ids','roi_features','roi_valid','roi_quality')}for r in rows]
            phrases=[appearance_query(queries[r['query_index']]['instruction']) for r in clean]
            appearance=native.texts(phrases).cpu().half()
            for r,text in zip(clean,appearance):r['appearance_text']=text.clone()
            filename=f'frames_{ci:05d}.pt';atomic_save(dict(rows=clean,episodes=episodes),root/filename)
            manifest['chunks'].append(dict(file=filename,sha256=file_sha256(root/filename),rows=len(rows),episodes=len(episodes)))
            manifest.update(seconds=time.perf_counter()-started,total_rows=sum(c['rows']for c in manifest['chunks']),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            atomic_json(manifest,root/'manifest.json')
            if ci%8==0:print(json.dumps(dict(chunks=ci+1,total_chunks=limit,states=manifest['total_rows'],seconds=manifest['seconds'])),flush=True)
    manifest['complete']=True;atomic_json(manifest,root/'manifest.json')


if __name__=='__main__':main()
