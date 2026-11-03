"""Frozen current-view SigLIP public-name retrieval proxy, not instance gold."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.checkpointing import atomic_json, atomic_save, file_sha256
from graphnav.observed_landmarks import current_landmark_rois
from scripts.analyze_instruction_landmark_association import PublicNames


def top1_credit(scores, positive):
    """Fractional credit for ties, avoiding ID-order leakage into a baseline."""
    scores=np.asarray(scores);positive=np.asarray(positive,dtype=bool)
    if scores.ndim!=1 or scores.shape!=positive.shape or not positive.any():
        raise ValueError('nonempty aligned candidate scores and positive set required')
    best=np.isclose(scores,scores.max(),rtol=0,atol=1e-7)
    return float(positive[best].mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','roi-audit','observations','catalog','backbone','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads((a.audit/'records.json').read_text())
    roirows=json.loads((a.roi_audit/'records.json').read_text())
    selected={r['frame_key'] for r in roirows if r['accepted_current_roi']}
    framepath=a.observations/'public_frames.jsonl'
    frames={x['frame_key']:x for x in map(json.loads,framepath.read_text().splitlines())}
    catalog=json.loads(a.catalog.read_text());names=PublicNames(catalog)
    public={mapname:sorted({v['id'] for items in table.values() for v in items}) for mapname,table in names.maps.items()}
    grouped={}
    for r in source:
        if r['frame_key'] in selected:grouped.setdefault(r['frame_key'],[]).append(r)
    torch.set_num_threads(4);torch.manual_seed(0);rng=np.random.default_rng(0)
    a.output.mkdir(parents=True)
    manifest=dict(private_target_labels_read=False,normal_inference_modified=False,
        candidate_source='all public named polygons in the same CURRENT observed frame; same-type hard negatives',
        labels='geometry-associated public name set, unverified pixel identity',seed=0,
        backbone_sha256={p.name:file_sha256(p) for p in a.backbone.glob('*.safetensors')},
        inputs_sha256={k:file_sha256(v) for k,v in dict(audit=a.audit/'records.json',roi=a.roi_audit/'records.json',frames=framepath,catalog=a.catalog,producer=Path(__file__)).items()})
    atomic_json(manifest,a.output/'manifest.json')
    started=time.perf_counter();native=ObservedSigLIPMatchingEncoder(str(a.backbone),microbatch=8)
    records=[];features={};timings=[]
    for no,(key,queries) in enumerate(sorted(grouped.items())):
        frame=frames[key];mp=frame['map_name'];image=a.observations/frame['image'];depth=a.observations/frame['depth']
        assert file_sha256(image)==frame['image_sha256'] and file_sha256(depth)==frame['depth_sha256']
        with np.load(depth) as d:valid=d['depth_valid'].copy()
        observation=dict(bgr=cv2.imread(str(image)),depth_valid=valid,footprint_world=frame['footprint_world'])
        landmarks=[dict(id=i,**{k:catalog[mp][i][k] for k in ('name','object_type','contour')}) for i in public[mp]]
        rois=current_landmark_rois(observation,landmarks,max_rois=max(1,len(landmarks)))
        if not rois:continue
        torch.cuda.synchronize();t=time.perf_counter()
        visual=native.images([r.roi_image for r in rois]).float()
        torch.cuda.synchronize();timings.append(time.perf_counter()-t)
        features[key]=dict(ids=[r.object_id for r in rois],types=[r.object_type for r in rois],images=visual.cpu().half())
        for query in queries:
            members=set(query['group']['public_ids'])
            observed=[r for r in rois if r.object_id in members]
            if not observed:continue
            types={r.object_type for r in observed}
            eligible=[i for i,r in enumerate(rois) if r.object_type in types]
            pos=np.array([rois[i].object_id in members for i in eligible])
            # Multiple polygons of a name remain a set, not fabricated identities.
            if pos.all():continue
            name=' or '.join(query['group']['names'])
            type_text='An overhead view of a '+ ' or '.join(sorted(types)) + '.'
            texts=[name,'An overhead view of '+name+'.',type_text]
            language=native.texts(texts).float()
            score=(visual[eligible]@language.T).cpu().numpy()
            perm=rng.permutation(len(eligible));shuffled=score[perm,1]
            records.append(dict(frame_key=key,subset=query['subset'],map_name=mp,name=name,
                candidate_ids=[rois[i].object_id for i in eligible],positive_set=pos.tolist(),
                candidates=len(pos),positives=int(pos.sum()),group_kind=query['group']['kind'],
                raw_name_scores=score[:,0].tolist(),named_scores=score[:,1].tolist(),type_scores=score[:,2].tolist(),
                shuffled_permutation=perm.tolist(),named_credit=top1_credit(score[:,1],pos),
                raw_name_credit=top1_credit(score[:,0],pos),type_credit=top1_credit(score[:,2],pos),
                shuffled_credit=top1_credit(shuffled,pos),chance=float(pos.mean())))
        if (no+1)%25==0:print(json.dumps(dict(frames=no+1,total=len(grouped),queries=len(records))),flush=True)
    torch.cuda.synchronize()
    summary=dict(complete=True,seconds=time.perf_counter()-started,encoded_images=native.encoded_images,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,median_frame_encoding_seconds=float(np.median(timings)),
        by_subset={},warning='Public polygon/name association proxy only; not verified instance accuracy, no Goal or navigation result.')
    for subset in ('fit','dev'):
        rr=[r for r in records if r['subset']==subset]
        summary['by_subset'][subset]=dict(queries=len(rr),frames=len({r['frame_key'] for r in rr}),
            maps=len({r['map_name'] for r in rr}),group_kinds=dict(Counter(r['group_kind'] for r in rr)),
            **{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ('named_credit','raw_name_credit','type_credit','shuffled_credit','chance')})
    atomic_save(features,a.output/'features_public.pt');atomic_json(records,a.output/'records.json')
    summary['records_sha256']=file_sha256(a.output/'records.json');summary['features_sha256']=file_sha256(a.output/'features_public.pt')
    atomic_json(summary,a.output/'summary.json');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
