"""Frozen TRAIN reference-frame partial-ROI experiment, not navigation."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.observed_candidate_rois import CurrentCandidateROIEnvironment
from graphnav.partial_observed_roi import observed_intersection_crop
from scripts.audit_reference_roi_supervision import reference_indices, reference_observation


def score_candidates(features, valid, text):
    pooled=(features*valid[...,None]).sum(1)/np.maximum(valid.sum(-1,keepdims=True),1)
    pooled/=np.maximum(np.linalg.norm(pooled,axis=-1,keepdims=True),1e-9)
    return pooled@text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','stages','data','cache','data-root','sbf-source','stage1','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',choices=['fit','development'],default='fit')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for directory in ('rows','public_features','views'):(a.output/directory).mkdir(exist_ok=True)
    torch.manual_seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    labels=json.loads((a.data/'labels_private.json').read_text())
    audit=json.loads((a.audit/'manifest.json').read_text())
    ids=json.loads((a.stages/'summary.json').read_text())['selected'] if a.split=='fit' else [i for i in audit['selected'] if labels[i]['split']=='development']
    ids=sorted(ids)[:a.limit] if a.limit else sorted(ids)
    assert all(labels[i]['split']==a.split for i in ids)
    cache_manifest=json.loads((a.cache/'manifest.json').read_text())
    assert cache_manifest['public_queries_sha256']==file_sha256(a.data/'public_train.jsonl')
    protocol=torch.load(a.stage1,map_location='cpu',weights_only=True)['protocol']
    encoder=ObservedSigLIPMatchingEncoder(protocol['backbone'],device='cuda',microbatch=16)
    if getattr(encoder.model.config,'_commit_hash',None)!=protocol['backbone_revision']:
        raise ValueError('Frozen backbone revision mismatch')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav')
    config=dict(selected=ids,split=a.split,seed=42,script_sha256=file_sha256(__file__),
                clip_sha256=file_sha256('graphnav/partial_observed_roi.py'),
                environment_sha256=file_sha256('graphnav/observed_candidate_rois.py'),
                cache_manifest_sha256=file_sha256(a.cache/'manifest.json'),
                annotation_sha256=file_sha256(ds.annotation_path),
                stage1_sha256=file_sha256(a.stage1),backbone=protocol['backbone'],
                backbone_revision=protocol['backbone_revision'],quality=.9,
                scope='TRAIN reference frames only; each frame is scored independently; no closed-loop or future-frame inputs')
    path=a.output/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Changed experiment protocol')
    atomic_json(config,path)
    env=CurrentCandidateROIEnvironment(a.data_root/'rgbd',a.output/'observations')
    started=time.perf_counter();rows=[];torch.cuda.reset_peak_memory_stats()
    for ordinal,i in enumerate(ids):
        episode=ds[i]
        source_metadata=json.loads((a.cache/'public_features'/f'{i:05d}.json').read_text())
        if episode.instruction!=source_metadata['instruction'] or episode.map_name!=source_metadata['map_name']:
            raise ValueError('Reference episode and cached public candidates are misaligned')
        with np.load(a.cache/'public_features'/f'{i:05d}.npz') as z:
            centers=z['candidate_world'].copy()
        text=encoder.texts([episode.instruction])[0].cpu().numpy()
        for phase,step in reference_indices(episode.trajectory):
            key=f'{i:05d}_{phase}';row_path=a.output/'rows'/f'{key}.json'
            if row_path.exists():rows.append(json.loads(row_path.read_text()));continue
            pose=episode.trajectory[step].copy()
            obs,rois,reason=reference_observation(env,episode.map_name,pose,GROUND_LEVEL[episode.map_name],centers)
            valid=np.zeros((2,len(centers),2),bool)  # original vs intersection
            features=np.zeros((2,2,len(centers),2,encoder.dimension),np.float32)  # full/crop, low/native
            views=[];positions=[];metadata=[]
            if reason is None:
                for roi in rois:
                    c,s=roi['candidate'],roi['scale'];clipped=[];quality=[];fractions=[]
                    for res,name in enumerate(('low','native')):
                        crop,bounds,fraction=observed_intersection_crop(roi[name],roi['quad'],obs['footprint_world'])
                        clipped.append(crop);fractions.append(fraction)
                        quality.append(roi[name+'_support']/max(fraction,1e-9))
                    old=bool(roi['valid']);new=bool(min(quality)>=.9 and all(x is not None for x in clipped))
                    valid[0,c,s]=old;valid[1,c,s]=new
                    metadata.append(dict(candidate=c,scale=s,original_valid=old,intersection_valid=new,
                                         observed_fraction=fractions,intersection_quality=quality,
                                         source_sizes=[roi['low_size'],roi['native_size']]))
                    if old or new:
                        for res,name in enumerate(('low','native')):
                            views.append(roi[name]);positions.append((0,res,c,s))
                            if new:views.append(clipped[res]);positions.append((1,res,c,s))
                    if ordinal<4 and new and min(fractions)<.95:
                        cv2.imwrite(str(a.output/'views'/f'{key}_c{c}_s{s}_full.png'),roi['native'])
                        cv2.imwrite(str(a.output/'views'/f'{key}_c{c}_s{s}_clip.png'),clipped[1])
                if views:
                    vectors=encoder.images(views).cpu().numpy()
                    for position,vector in zip(positions,vectors):features[position]=vector
                if ordinal<4:cv2.imwrite(str(a.output/'views'/f'{key}_current.png'),obs['bgr'])
            # All public choices are made before target-label evaluation.
            choices={'geometry':0};observed={};availability={}
            for resolution,res in [('low',0),('native',1)]:
                for arm,form,mask_id in [('A',0,0),('B',0,1),('C',1,1)]:
                    mask=valid[mask_id];known=mask.any(-1);scores=score_candidates(features[form,res],mask,text)
                    name=f'{arm}_{resolution}';availability[name]=int(known.sum())
                    eligible=bool(known[0] and known.sum()>=2)
                    winner=int(np.where(known,scores,-np.inf).argmax()) if known.any() else None
                    observed[name]=winner;choices[name]=winner if eligible else 0
                    if arm=='C':
                        permutation=np.arange(len(known));rng=np.random.default_rng(42+i*3+step)
                        for pattern in np.unique(mask,axis=0):
                            selected=np.flatnonzero((mask==pattern).all(-1)&known)
                            permutation[selected]=rng.permutation(selected)
                        shuffled=scores[permutation]
                        winner=int(np.where(known,shuffled,-np.inf).argmax()) if known.any() else None
                        observed[name+'_shuffle']=winner;choices[name+'_shuffle']=winner if eligible else 0
            np.savez_compressed(a.output/'public_features'/f'{key}.npz',features=features.astype(np.float16),
                                valid=valid,text=text,candidate_world=centers,pose=pose)
            # Private labels for scoring only; no source cropping/choice uses them.
            distance=np.linalg.norm(centers-labels[i]['target_world'],axis=1)
            row=dict(index=i,phase=phase,step=step,unknown=reason,choice=choices,
                     NE={k:float(distance[v]) for k,v in choices.items()},availability=availability,
                     observed_NE={k:float(distance[v]) if v is not None else None for k,v in observed.items()},
                     correct_supported={str(j):bool(((distance<=20)&valid[j].any(-1)).any()) for j in (0,1)},
                     roi_metadata=metadata)
            atomic_json(row,row_path);rows.append(row)
        if ordinal%16==0:print(json.dumps(dict(completed=ordinal+1,expected=len(ids),encoded_images=encoder.encoded_images,seconds=time.perf_counter()-started)),flush=True)
    result={}
    for phase in ('all','start','middle','end'):
        rr=[r for r in rows if phase=='all' or r['phase']==phase];base=np.array([r['NE']['geometry'] for r in rr])
        metrics={}
        for arm in rows[0]['choice']:
            e=np.array([r['NE'][arm] for r in rr]);metrics[arm]=dict(n=len(e),R20=float((e<=20).mean()),NE=float(e.mean()),
                  corrections=int(((e<=20)&(base>20)).sum()),damage=int(((e>20)&(base<=20)).sum()))
        result[phase]=metrics
    conditional={}
    for arm in rows[0]['observed_NE']:
        mask_id='0' if arm.startswith('A_') else '1'
        rr=[r for r in rows if r['correct_supported'][mask_id] and r['observed_NE'][arm] is not None]
        conditional[arm]=dict(n=len(rr),R20=float(np.mean([r['observed_NE'][arm]<=20 for r in rr])) if rr else None)
    atomic_json(dict(metrics=result,conditional=conditional,groups=len(ids),frames=len(rows),
                     unknown_frames=sum(r['unknown'] is not None for r in rows),
                     seconds=time.perf_counter()-started,encoded_images_this_invocation=encoder.encoded_images,
                     peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                     scope='Position proxies on TRAIN reference frames; repeated frames are not independent tasks; no SR'),a.output/'summary.json')
    print(json.dumps(dict(all=result['all'],conditional=conditional)),flush=True)


if __name__=='__main__':main()
