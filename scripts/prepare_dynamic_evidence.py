"""Assemble actual current/past landmark evidence; keep goal labels separate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from graphnav.candidate_geometry import relative_entity_features
from graphnav.checkpointing import atomic_save,file_sha256,provenance


def roi_confidence(row):
    mask=np.zeros(row['frame_resolution'],np.uint8)
    cv2.fillPoly(mask,[np.rint(row['contour_px']).astype(np.int32)],1)
    stable=cv2.erode(mask,np.ones((5,5),np.uint8)).sum()/max(mask.sum(),1)
    return float(row['visible_fraction']*row['rgb_pixel_support']*stable)


def causal_history(rows,indices,*,episode_index,cutoff):
    """Source IDs only group observed evidence; no destination identity is read."""
    selected=[i for i in indices if rows[i]['source_episode_index']==episode_index and rows[i]['observed_step']<=cutoff]
    if any(rows[i]['observed_step']>rows[i]['observation_cutoff_step'] for i in selected):
        raise ValueError('future/untracked ROI in causal history')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('vectors','queries','labels','objects','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=['train_seen','val_seen'],required=True);args=p.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('refusing dynamic-evidence overwrite')
    out.mkdir(parents=True);root=Path(args.vectors);manifest=json.loads((root/'manifest.json').read_text())
    if not manifest.get('complete') or file_sha256(root/'vectors.pt')!=manifest['vectors_sha256']:
        raise ValueError('incomplete/changed visual vectors')
    rows=[json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines() if line]
    queries=[json.loads(line) for line in Path(args.queries).read_text().splitlines() if line]
    if any(query['target_xy']!=[0.,0.] for query in queries):raise ValueError('public queries must not contain true targets')
    labels={}
    with Path(args.labels).open() as stream:
        for line in stream:
            row=json.loads(line);episode=row['source_episode_index']
            if episode in labels:continue
            target=row['labels']['target_world'] if args.split=='train_seen' else row['target_world']
            labels[episode]=dict(query_index=row['query_index'],target_world=target)
    objects=json.loads(Path(args.objects).read_text());episodes=defaultdict(list);frames=defaultdict(list)
    for i,row in enumerate(rows):
        if row['observed_step']>row['observation_cutoff_step']:raise ValueError('future visual ROI rejected')
        episodes[(row['map_name'],row['source_episode_index'])].append(i)
        frames[(row['map_name'],row['source_episode_index'],row['observed_step'])].append(i)
    axis=(np.arange(28,dtype=np.float32)+.5)/28;xx,yy=np.meshgrid(axis,axis);grid=np.stack((xx.ravel(),yy.ravel()),-1)
    geometry={};descriptors={};map_bounds={};qualities=torch.tensor([roi_confidence(row) for row in rows],dtype=torch.float32)
    records=[];skipped=0
    for (map_name,episode,step),current in sorted(frames.items()):
        if episode not in labels:skipped+=1;continue
        qid=labels[episode]['query_index'];query=queries[qid]
        if query['map_name']!=map_name:raise ValueError('source episode/public query map mismatch')
        history=causal_history(rows,episodes[(map_name,episode)],episode_index=episode,cutoff=step)
        bounds=np.asarray(query['world_bounds'],np.float32);world=bounds[:2]+grid*(bounds[2:]-bounds[:2])
        if map_name in map_bounds and not np.array_equal(map_bounds[map_name],bounds):
            raise ValueError('inconsistent public bounds within one map')
        map_bounds[map_name]=bounds
        for i in history:
            key=map_name+':'+rows[i]['object_id']
            if key not in geometry:
                contour=np.asarray(objects[map_name][rows[i]['object_id']]['contour'],np.float32)
                # Every CityRefer entry is a region surface, never a fabricated road centerline.
                feature=relative_entity_features(world,[dict(geometry_m=contour,kind='surface')],1)[:,0]
                geometry[key]=torch.from_numpy(feature.astype(np.float16))
                source=objects[map_name][rows[i]['object_id']]
                descriptors[key]=dict(name=source['name'],object_type=source['object_type'],contour_world=torch.from_numpy(contour))
        records.append(dict(map_name=map_name,source_episode_index=episode,query_index=qid,step=step,
            current_roi_indices=current,history_roi_indices=history,
            labels=dict(target_world=labels[episode]['target_world'])))
    atomic_save(dict(geometry=geometry,confidence=qualities,entity_descriptors=descriptors),out/'geometry_quality.pt')
    (out/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    result=dict(split=args.split,state_queries=len(records),episodes=len(set(row['source_episode_index'] for row in records)),
        public_queries=len(set(row['query_index'] for row in records)),named_regions=len(geometry),skipped_nonapproved_frames=skipped,
        provenance=provenance(),sha256={key:file_sha256(getattr(args,key)) for key in ('queries','labels','objects')},
        vector_manifest_sha256=file_sha256(root/'manifest.json'),samples_sha256=file_sha256(out/'samples.jsonl'),
        geometry_sha256=file_sha256(out/'geometry_quality.pt'),
        note='Human current/past observation diagnostic subset, not full benchmark or initial-only RGB. Destination stays in labels; history only within same source episode and ≤current step. RGB/boundary confidence; no semantic segmentation GT or unobserved negative evidence.')
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
