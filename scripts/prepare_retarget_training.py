"""Separate public query inputs from private TRAIN weak-instance teacher labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.retarget_training_data import select_training_query,weak_instance_mask
from scripts.cache_local_chain_training import public_query


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('proposals','feature-cache','static-queries','data-root','sbf-source','review','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier teacher dataset')
    manifest=json.loads((args.feature_cache/'manifest.json').read_text())
    proposal_summary=json.loads((args.proposals/'summary.json').read_text())
    if proposal_summary['selected_profile']!='n144_r1_width4' or manifest['split']!='train_seen':raise ValueError('declared strict TRAIN profile only')
    if proposal_summary['links_sha256']!=file_sha256(args.proposals/'links_private.json'):raise ValueError('proposals changed')
    links=json.loads((args.proposals/'links_private.json').read_text())
    review=json.loads(args.review.read_text())['rows']
    decision={(r['frame_key'],r['private_training_instance_id']):r['candidate_same_instance_support'] for r in review}
    sys.path.insert(0,str(args.sbf_source))
    from sbfnav.dataset import CityNavDataset
    dataset=CityNavDataset(args.data_root,'train_seen',version='processed_citynav')
    catalog=json.loads((args.data_root/'cityrefer/objects.json').read_text())
    public_tasks={};truths={};by_frame=defaultdict(dict)
    for link in links:
        name=link['map_name'];oid=link['private_training_instance_id']
        if manifest['map_assignment'].get(name)!='fit':raise ValueError('non-fitting teacher target')
        index=select_training_query(link['available_train_query_indices'],name,oid)
        record=dataset[index]
        if record.map_name!=name or str(record.object_id)!=oid:raise ValueError('TRAIN query/instance mismatch')
        if not np.allclose(record.target_position[:2],catalog[name][oid]['position'][:2],atol=1e-3):raise ValueError('annotation/registry point mismatch')
        key=f'retarget_{index:05d}'
        public_tasks[key]=dict(task_key=key,map_name=name,instruction=record.instruction)
        truths[key]=dict(target_world=record.target_position[:2].tolist(),physical_group=[name,oid])
        if decision.get((link['frame_key'],oid)) not in ('uncertain','not_supported'):
            by_frame[link['frame_key']][oid]=link['candidate_indices']
    selected={(row['map_name'],row['instruction']) for row in public_tasks.values()}
    static={}
    with args.static_queries.open() as stream:
        for line in stream:
            record=json.loads(line);key=(record['map_name'],record['instruction'])
            if key in selected and key not in static:static[key]=public_query(record)
    if set(static)!=selected:raise ValueError('new human instructions missing static protocol; never reuse old query prior')
    private={};public_links=[];eligible=0;uncertain=0
    for link in links:
        name=link['map_name'];oid=link['private_training_instance_id']
        index=select_training_query(link['available_train_query_indices'],name,oid);key=f'retarget_{index:05d}'
        sample=link['frame_key']+'__'+key
        mask=weak_instance_mask(by_frame[link['frame_key']],oid,144)
        status=decision.get((link['frame_key'],oid),'unreviewed_geometry_proxy')
        # A soft preference, not calibrated visibility probability or hard GT.
        preference=.8 if status=='supported' else .5 if status in ('uncertain','not_supported') else .65
        if not (mask==1).any() or not (mask==0).any():preference=.5
        eligible+=preference>.5;uncertain+=preference==.5
        private[sample]=dict(**truths[key],category='Car',weak_instance_labels=mask,
            soft_preference=preference,annotation_status=status,pixel_gt=False)
        public_links.append(dict(sample_key=sample,source_frame_key=link['frame_key'],task_key=key,subset='fit'))
    args.output.mkdir(parents=True)
    tasks=list(public_tasks.values())
    (args.output/'public_tasks.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in tasks))
    (args.output/'public_queries.jsonl').write_text(''.join(json.dumps(dict(**static[(row['map_name'],row['instruction'])],task_key=row['task_key']),ensure_ascii=False)+'\n' for row in tasks))
    atomic_json(public_links,args.output/'public_links.json');atomic_save(private,args.output/'loss_labels_private.pt')
    summary=dict(tasks=len(tasks),samples=len(public_links),eligible_soft_instance_samples=eligible,
        skipped_uncertain_or_missing_core_samples=uncertain,soft_preference_unreviewed=.65,
        soft_preference_ai_supported=.8,soft_preference_uncertain=.5,
        warning='Weak TRAIN teachers only. Soft preferences are design weights, not calibrated correctness probabilities. No new RGB/BERT/A0 features computed yet.',
        inference_gt_fields=False)
    for key,path in [('proposals',args.proposals/'links_private.json'),('review',args.review),('annotations',dataset.annotation_path),
        ('feature_manifest',args.feature_cache/'manifest.json'),('static_queries',args.static_queries),
        ('public_tasks',args.output/'public_tasks.jsonl'),('public_queries',args.output/'public_queries.jsonl'),
        ('public_links',args.output/'public_links.json'),('private_labels',args.output/'loss_labels_private.pt')]:summary[key+'_sha256']=file_sha256(path)
    atomic_json(summary,args.output/'summary.json');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
