"""Whitelist TRAIN public inputs for an annotation-selected frozen grid pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from scripts.analyze_instruction_landmark_association import PublicNames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','annotations','catalog','static-queries','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    # TRAIN annotation-side selection only; boxes are NOT exported to the model.
    annotations=json.loads(a.annotations.read_text());keys={x['frame_key'] for x in annotations if x['assessment']=='candidate' and x['boxes_px']}
    if len(keys)!=27:raise ValueError('frozen27 weak appearance selection expected')
    root=a.root;frames={};features={};rolevec={};sources={}
    for cache,obs,roles in [('chain_train400_features','fixed_yaw_train400_s1','reference_vectors_v2.pt'),('scoped_between_train294_features_v1','scoped_between_train294_observations_v1','scoped_between_train294_reference_vectors_v1.pt')]:
        meta=json.loads((root/cache/'manifest.json').read_text());assert meta['complete'] and meta['split']=='train_seen'
        sources[cache]=file_sha256(root/cache/'manifest.json')
        rolevec.update(torch.load(root/roles,weights_only=True))
        for f in map(json.loads,(root/obs/'public_frames.jsonl').read_text().splitlines()):
            if f['frame_key'] in keys:
                assert f['camera_pose'][3]==0;frames[f['frame_key']]=(f,root/obs)
        for c in meta['chunks']:
            path=root/cache/c['file'];assert file_sha256(path)==c['sha256']
            for f in torch.load(path,weights_only=True)['records']:
                if f['frame_key'] in keys:
                    assert f['subset']=='fit' and not {'target_world','target_id','physical_group'}&f.keys();features[f['frame_key']]=f
    wanted={(f['map_name'],f['instruction']) for f,_ in frames.values()};bounds={}
    for line in a.static_queries.open():
        q=json.loads(line);key=(q['map_name'],q['instruction'])
        if key in wanted:bounds[key]=torch.tensor(q['world_bounds'],dtype=torch.float32)
    catalog=json.loads(a.catalog.read_text());names=PublicNames(catalog);public={}
    for key in sorted(keys):
        frame,obs=frames[key];f=features[key];mp=frame['map_name'];text=frame['instruction']
        ids=list(dict.fromkeys(i for m in names.match(mp,text) for i in m['public_ids']))[:16]
        entities=[dict(kind='region',geometry_m=torch.tensor(catalog[mp][i]['contour'],dtype=torch.float32)) for i in ids]
        public[key]=dict(frame=frame,observations=str(obs),prior_logits=f['prior_logits'],target_text=f['target_text'],
            context_text=rolevec[f['task_key']]['reference_vector'],world_bounds=bounds[(mp,text)],global_world=f['candidate_world'][:784],
            public_entities=entities,cached_world=f['candidate_world'][784:],cached_uv=f['local_uv'],cached_quality=f['quality'],cached_valid=f['local_valid'])
    a.output.mkdir(parents=True);atomic_save(public,a.output/'inputs_public.pt')
    atomic_json(dict(cases=len(public),all_train_fit=True,selection_uses_train_annotation_hints=True,
        inference_reads_boxes=False,inference_reads_private_goal_labels=False,annotation_selection_sha256=file_sha256(a.annotations),
        public_inputs_sha256=file_sha256(a.output/'inputs_public.pt'),source_manifests=sources,catalog_sha256=file_sha256(a.catalog),
        producer_sha256=file_sha256(__file__)),a.output/'manifest.json')

if __name__=='__main__':main()
