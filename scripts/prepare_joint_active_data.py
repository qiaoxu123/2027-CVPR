"""Actual current/past RGB features; future and destination labels kept separate.

Four fixed interior trajectory quantiles follow the local SBF training sampler.
Only frozen backbone representations are persisted, never A3 model outputs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from graphnav.a3_runtime import public_static_record
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256,provenance
from graphnav.closedloop import PublicEpisode
from graphnav.observed_patch_scales import ObservedPatchScaleEncoder
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.trajectory_policy import remaining_waypoints


def prefix_indices(length, samples=4):
    return [max(1,min(length,int(round((slot+1)/(samples+1)*length))))-1 for slot in range(samples)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('sbf-source','data-root','a3-checkpoint','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--split',choices=['train_seen','val_seen'],default='train_seen')
    parser.add_argument('--max-episodes',type=int)
    parser.add_argument('--microbatch',type=int,default=8)
    parser.add_argument('--data-audit',default='reports/sbf_protocol/full_input_audit.json')
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();root=Path(a.output);manifest_path=root/'manifest.json'
    if root.exists() and not a.resume:raise ValueError('refusing prepared dataset overwrite')
    root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,a.sbf_source)
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.coordinates import MapTransform
    from sbfnav.mapdata import GROUND_LEVEL
    from sbfnav.validation import evenly_spaced_indices
    source=CityNavDataset(a.data_root,a.split,version='processed_citynav')
    catalog=CityReferCatalog(Path(a.data_root)/'cityrefer')
    indices=evenly_spaced_indices(len(source),a.max_episodes)
    original=torch.load(a.a3_checkpoint,map_location='cpu',weights_only=True)
    protocol=dict(split=a.split,source_sha256=file_sha256(source.annotation_path),
        objects_sha256=file_sha256(Path(a.data_root)/'cityrefer/objects.json'),
        descriptions_sha256=file_sha256(Path(a.data_root)/'cityrefer/processed_descriptions.json'),
        a3_sha256=file_sha256(a.a3_checkpoint),a3_args=original['args'],
        episode_indices=[int(i) for i in indices],samples_per_episode=4,
        prefix_sampler='clamp(round((slot+1)/5 * trajectory_length),1,length)-1',
        backbone='google/siglip2-large-patch16-384',extents_m=[32,64,128],
        feature_kind='native frozen contextual patches + exact masked native pooling; no ROI resize',
        history='initial pose + preceding fixed quantile observations only; no future observations',
        no_cached_a3_outputs=True,causal_observations=True,
        preparation_script_sha256=file_sha256(__file__),
        patch_pooling_script_sha256=file_sha256('graphnav/observed_patch_scales.py'),
        data_audit_sha256=file_sha256(a.data_audit))
    manifest=dict(protocol=protocol,complete=False,chunks=[],queries=[],provenance=provenance())
    if a.resume:
        manifest=json.loads(manifest_path.read_text())
        if manifest['protocol']!=protocol:raise ValueError('resume source/config mismatch')
        for chunk in manifest['chunks']:
            if file_sha256(root/chunk['file'])!=chunk['sha256']:raise ValueError('corrupt prepared feature chunk')
        if manifest['complete']:return
    native=ObservedSigLIPMatchingEncoder(microbatch=a.microbatch)
    revision=getattr(native.model.config,'_commit_hash',None)
    if 'backbone_revision' in manifest and manifest['backbone_revision']!=revision:
        raise ValueError('cache resume visual model revision changed')
    manifest['backbone_revision']=revision
    audit=json.loads(Path(a.data_audit).read_text())
    checked={}
    for name in sorted({source[int(i)].map_name for i in indices}):
        hashes=dict(height_sha256=file_sha256(Path(a.data_root)/'rgbd'/f'{name}.tif'),
                    rgb_sha256=file_sha256(Path(a.data_root)/'rgbd'/f'{name}.png'))
        if name not in audit['rasters'] or any(hashes[k]!=audit['rasters'][name][k] for k in hashes):
            raise ValueError('actual RGB/height SHA differs from previously verified SBF data: '+name)
        checked[name]=hashes
    manifest['verified_rasters']=checked
    atomic_json(manifest,manifest_path)
    encoder=ObservedPatchScaleEncoder(native)
    renderer=CityNavObservedViewRenderer(Path(a.data_root)/'rgbd',root/'unused',image_size=384)
    args=original['args'];grid_size=args['grid_size']
    # Use the canonical A3 grid and ordering from its existing dataset class.
    from graphnav.static_localization import StaticLocalizationDataset
    transforms={};query_lookup={json.dumps(q,sort_keys=True):i for i,q in enumerate(manifest['queries'])}
    completed=sum(c['episodes'] for c in manifest['chunks']);started=time.perf_counter()
    for chunk_start in range(completed,len(indices),16):
        rows=[]
        for raw_index in indices[chunk_start:chunk_start+16]:
            record=source[int(raw_index)];name=record.map_name
            if name not in transforms:
                transforms[name]=MapTransform.from_raster(Path(a.data_root)/'rgbd'/f'{name}.tif',canvas_size=224,field_size=28)
            public=PublicEpisode(name,record.instruction,record.start_pose,
                tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
            query=public_static_record(public,transforms[name],catalog,max_landmarks=args['max_landmarks'])
            query.update(match_mode='fuzzy',episode_id='public-query')
            key=json.dumps(query,sort_keys=True)
            if key not in query_lookup:
                query_lookup[key]=len(manifest['queries']);manifest['queries'].append(query)
            qid=query_lookup[key];bounds=np.array(query['world_bounds'],np.float32)
            # Dataset defines canonical ordering, rather than approximating it here.
            if 'canonical_grid_xy' not in manifest:
                query_path=root/'grid_probe.jsonl';query_path.write_text(json.dumps(query)+'\n')
                probe=StaticLocalizationDataset(query_path,grid_size=grid_size,max_landmarks=args['max_landmarks'],
                    max_roads=args['max_roads'],max_relations=args['max_relations'])
                manifest['canonical_grid_xy']=probe.grid_xy.tolist();query_path.unlink()
            grid=np.asarray(manifest['canonical_grid_xy'],np.float32)
            world=bounds[:2]+grid*(bounds[2:]-bounds[:2])
            prefixes=prefix_indices(len(record.trajectory))
            steps=sorted(set([0]+prefixes))
            observations=[]
            for step in steps:
                pose=record.trajectory[step]
                if pose[2]-GROUND_LEVEL[name]<=0:
                    observations.append(None)
                else:observations.append(renderer.current_rgbd(name,pose,GROUND_LEVEL[name]))
            encoded={}
            for offset in range(0,len(steps),a.microbatch):
                selected=[i for i in range(offset,min(len(steps),offset+a.microbatch)) if observations[i] is not None]
                if selected:
                    values=encoder.encode([observations[i] for i in selected],[world for _ in selected])
                    encoded.update({steps[i]:v for i,v in zip(selected,values)})
            accumulated=torch.zeros(len(grid),3,native.dimension);counts=torch.zeros(len(grid),3)
            quality_sum=torch.zeros(len(grid),3,3);history={}
            for step in steps:
                if step in encoded:
                    features,valid,quality=encoded[step]
                    accumulated+=features.float()*valid[...,None];counts+=valid
                    quality_sum+=quality*valid[...,None]
                valid=counts>0
                ids=valid.any(-1).nonzero().flatten()
                features=accumulated/counts[...,None].clamp_min(1)
                quality=quality_sum/counts[...,None].clamp_min(1)
                history[step]=dict(roi_ids=ids,roi_features=features[ids].half().clone(),
                    roi_valid=valid[ids].clone(),roi_quality=quality[ids].clone())
            text=native.texts([record.instruction])[0].cpu().half().clone()
            for slot,step in enumerate(prefixes):
                pose=record.trajectory[step].astype(np.float32)
                teacher=remaining_waypoints(record.trajectory,step)
                ground=GROUND_LEVEL[name]
                labels=dict(target_world=torch.tensor(record.target_position,dtype=torch.float32),
                    teacher_world=torch.from_numpy(teacher),valid_teacher_z=torch.from_numpy(
                    ((teacher[:,2]-ground>=10)&(teacher[:,2]-ground<=100)).astype(np.float32)))
                rows.append(dict(query_index=qid,episode_index=int(raw_index),slot=slot,cutoff_step=step,
                    history_steps=[s for s in steps if s<=step],pose_world=torch.from_numpy(pose),
                    ground_level=torch.tensor(ground,dtype=torch.float32),native_text=text,
                    **history[step],labels=labels))
        filename=f'features_{chunk_start:06d}.pt';atomic_save(dict(rows=rows),root/filename)
        manifest['chunks'].append(dict(file=filename,sha256=file_sha256(root/filename),rows=len(rows),
            episodes=min(16,len(indices)-chunk_start)))
        manifest['seconds']=time.perf_counter()-started
        atomic_json(manifest,manifest_path)
        print(json.dumps(dict(completed_episodes=chunk_start+min(16,len(indices)-chunk_start),
            total_episodes=len(indices),seconds=manifest['seconds'],cached_rows=sum(c['rows'] for c in manifest['chunks']))),flush=True)
    (root/'queries.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in manifest['queries']))
    manifest['queries_sha256']=file_sha256(root/'queries.jsonl')
    manifest['backbone_revision']=getattr(native.model.config,'_commit_hash',None)
    manifest['complete']=True;atomic_json(manifest,manifest_path)


if __name__=='__main__':main()
