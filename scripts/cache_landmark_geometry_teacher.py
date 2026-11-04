"""Fresh frozen original A3 beliefs for named TRAIN queries and polygon swaps."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_region_supervision import named_union,public_type
from graphnav.a3_runtime import public_static_record
from graphnav.closedloop import PublicEpisode
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','data-root','sbf-source','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityReferCatalog
    from sbfnav.coordinates import MapTransform
    from graphnav.stage1_runtime import GraphNavStage1Policy
    queries=json.loads((a.data/'queries_public.json').read_text());source=json.loads((a.data/'manifest.json').read_text())
    if file_sha256(a.data/'queries_public.json')!=source['queries_sha256']:raise ValueError('Data query source changed')
    catalog=CityReferCatalog(a.data_root/'cityrefer');runtime=GraphNavStage1Policy(a.checkpoint,a.output/'unused')
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    all_groups={};transforms={}
    for mp,objects in catalog.objects.items():
        grouped=defaultdict(list)
        for oid,obj in objects.items():
            if obj.get('name'):grouped[canonical_name(obj['name'])].append((oid,obj))
        all_groups[mp]=grouped
    entries=[];mapping=[];metadata={};skipped=[]
    for key,q in sorted(queries.items()):
        if len(q['public_ids'])>runtime.args.max_landmarks:
            skipped.append(dict(query_key=key,reason='teacher_landmark_capacity'));continue
        mp=q['map_name']
        if mp not in transforms:transforms[mp]=MapTransform.from_raster(a.data_root/'rgbd'/f'{mp}.tif')
        transform=transforms[mp]
        ep=PublicEpisode(mp,q['instruction'],None,(),(q['name'],))
        record=public_static_record(ep,transform,catalog,max_landmarks=runtime.args.max_landmarks)
        if len(record['landmarks'])!=len(q['public_ids']):raise ValueError('Teacher named binding differs')
        entries.append(record);mapping.append((key,'geometry'))
        options=[]
        for name,group in all_groups[mp].items():
            if name==canonical_name(q['name']) or public_type([o['object_type'] for _,o in group])!=q['type']:continue
            try:named_union([o['contour'] for _,o in group])
            except ValueError:continue
            options.append((name,group))
        options.sort(key=lambda x:hashlib.sha256(('0:'+key+':'+x[0]).encode()).hexdigest())
        metadata[key]=dict(world_bounds=record['world_bounds'],type=q['type'],instruction=q['instruction'],
            wrong_polygon_ids=None)
        if options:
            _,group=options[0];wrong=copy.deepcopy(record);bounds=torch.tensor(record['world_bounds'])
            for i,landmark in enumerate(wrong['landmarks']):
                contour=torch.tensor(group[i%len(group)][1]['contour'])
                landmark['contour']=((contour-bounds[:2])/(bounds[2:]-bounds[:2])).tolist()
            # Keep query/name/type/node count identical; change only contours.
            entries.append(wrong);mapping.append((key,'wrong_polygon_geometry'))
            metadata[key]['wrong_polygon_ids']=[str(i) for i,_ in group]
    path=a.output/'teacher_queries_public.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in entries))
    args=runtime.args
    ds=FullCandidateDataset(path,grid_size=args.grid_size,max_landmarks=args.max_landmarks,max_roads=args.max_roads,
        max_relations=args.max_relations,prior_sigma_m=args.prior_sigma_m,rule_sigma_m=args.rule_sigma_m,deduplicate_queries=False)
    if len(ds)!=len(mapping):raise ValueError('Teacher counterfactual query was deduplicated')
    output={k:{} for k in metadata};started=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(ds),8):
            batch=collate_full([ds[i] for i in range(start,min(start+8,len(ds)))])
            inputs,tokens,mask,names=_batch_inputs(batch,runtime.bert,runtime.device,'geometry')
            with torch.autocast('cuda',dtype=torch.bfloat16):field=runtime.model.a3(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
            for j,(key,mode) in enumerate(mapping[start:start+8]):output[key][mode]=field.belief_logits[j].float().cpu().clone()
            if start%80==0:print(json.dumps(dict(teacher_queries=min(start+8,len(ds)),total=len(ds),seconds=time.perf_counter()-started)),flush=True)
        keys=list(metadata)
        type_phrases={'Building':'an aerial photograph of a building','Road':'an aerial photograph of a road',
            'Ground':'an aerial photograph of a grassy ground area','Parking':'an aerial photograph of a parking lot'}
        for start in range(0,len(keys),32):
            selected=keys[start:start+32]
            name_vectors=runtime.native.texts([queries[k]['instruction'] for k in selected]).cpu().half().clone()
            type_vectors=runtime.native.texts([type_phrases[queries[k]['type']] for k in selected]).cpu().half().clone()
            for j,key in enumerate(selected):output[key].update(name_text=name_vectors[j],type_text=type_vectors[j],metadata=metadata[key])
    assert all(not p.requires_grad for p in runtime.model.parameters())
    atomic_save(output,a.output/'teacher_public.pt')
    atomic_json(dict(complete=True,queries=len(output),teacher_queries=len(ds),skipped=skipped,seconds=time.perf_counter()-started,
        teacher_sha256=file_sha256(a.output/'teacher_public.pt'),data_manifest_sha256=file_sha256(a.data/'manifest.json'),
        checkpoint_sha256=file_sha256(a.checkpoint),producer_sha256=file_sha256(__file__),
        teacher_frozen=True,old_target_conditioned_cache_used=False,original_target_labels_read=False),a.output/'manifest.json')
    print(json.dumps(dict(complete=True,queries=len(output),teacher_queries=len(ds),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
