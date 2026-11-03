"""Fresh language-conditioned A0/text/map inputs for TRAIN re-pairing.

This stage has no private labels argument and never reads target IDs/positions.
The image-only SigLIP2 descriptors are reused later; no RGB is encoded here.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.token_language import FrozenBERTTokens
from scripts.analyze_instruction_landmark_association import PublicNames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('public-tasks','public-queries','roles','catalog','checkpoint','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--backbone',required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior public input cache')
    torch.set_num_threads(4)
    tasks=[json.loads(line) for line in args.public_tasks.read_text().splitlines()]
    queries=[json.loads(line) for line in args.public_queries.read_text().splitlines()]
    roles=json.loads(args.roles.read_text())
    if len(tasks)!=len(queries) or set(roles)!={row['task_key'] for row in tasks}:raise ValueError('query/role alignment differs')
    for task,query in zip(tasks,queries):
        if task['task_key']!=query['task_key'] or (task['map_name'],task['instruction'])!=(query['map_name'],query['instruction']):raise ValueError('new-language task mismatch')
        if query['target_xy']!=[0.,0.] or {'target_world','object_id','target_id','physical_group'}&query.keys():raise ValueError('private target field in public query')
    started=time.perf_counter()
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model_args=SimpleNamespace(**source['a3_args'])
    model=new_model(model_args,torch.device('cuda')).eval().requires_grad_(False)
    model.load_state_dict({key.removeprefix('a3.'):value for key,value in source['model'].items() if key.startswith('a3.')},strict=True)
    bert=FrozenBERTTokens(model_args.text_backbone,device='cuda',max_length=getattr(model_args,'max_text_length',96))
    dataset=FullCandidateDataset(args.public_queries,grid_size=model_args.grid_size,max_landmarks=model_args.max_landmarks,
        max_roads=model_args.max_roads,max_relations=model_args.max_relations,prior_sigma_m=model_args.prior_sigma_m,
        rule_sigma_m=model_args.rule_sigma_m,deduplicate_queries=False)
    inputs={}
    with torch.inference_mode():
        for start in range(0,len(tasks),8):
            batch=collate_full([dataset[i] for i in range(start,min(start+8,len(tasks)))])
            public,tokens,mask,names=_batch_inputs(batch,bert,torch.device('cuda'),'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                field=model(public,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            for i,task in enumerate(tasks[start:start+8]):
                inputs[task['task_key']]=dict(prior_logits=field.belief_logits[i].float().cpu().clone(),
                    map_name=task['map_name'],world_bounds=queries[start+i]['world_bounds'],
                    instruction=task['instruction'])
    del model,bert,dataset
    torch.cuda.empty_cache()
    native=ObservedSigLIPMatchingEncoder(args.backbone)
    phrases=list(dict.fromkeys(text for task in tasks for text in
        (roles[task['task_key']]['target_text'],roles[task['task_key']]['reference_text'],task['instruction'])))
    vectors={}
    for start in range(0,len(phrases),32):
        chunk=phrases[start:start+32];encoded=native.texts(chunk).cpu().half()
        for text,vector in zip(chunk,encoded):vectors[text]=vector.clone()
    catalog=json.loads(args.catalog.read_text());public_names=PublicNames(catalog)
    for task in tasks:
        key=task['task_key'];role=roles[key]
        ids=list(dict.fromkeys(oid for mention in public_names.match(task['map_name'],task['instruction']) for oid in mention['public_ids']))[:16]
        entities=[dict(kind='region',geometry_m=catalog[task['map_name']][oid]['contour']) for oid in ids]
        inputs[key].update(target_text=vectors[role['target_text']],context_text=vectors[role['reference_text']],
            full_text=vectors[task['instruction']],target_phrase=role['target_text'],public_entities=entities)
    args.output.mkdir(parents=True)
    atomic_save(inputs,args.output/'public_inputs.pt')
    summary=dict(complete=True,tasks=len(inputs),new_BERT_A0_queries=len(inputs),new_SigLIP_texts=len(phrases),
        rgb_images_encoded=0,private_labels_read=False,seconds=time.perf_counter()-started,
        backbone=args.backbone,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    for key,path in [('public_tasks',args.public_tasks),('public_queries',args.public_queries),('roles',args.roles),
        ('catalog',args.catalog),('checkpoint',args.checkpoint),('public_inputs',args.output/'public_inputs.pt'),('producer',Path(__file__))]:summary[key+'_sha256']=file_sha256(path)
    atomic_json(summary,args.output/'summary.json');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
