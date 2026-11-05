"""TRAIN-only binding/identity proxies, causal cached evidence and synchronized text probes.

Private destination coordinates/IDs are accessed for scoring only. Counterfactual
directions have no invented target labels. Pixel instance truth is unavailable.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import numpy as np
import shapely
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.direct_target_data import load_model,batch_inputs
from graphnav.direct_target_runtime import bind_public_anchors,admissible_target_rgb
from graphnav.landmark_acquisition import LandmarkAcquisition
from graphnav.landmark_region_supervision import named_union
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens


def main():
    p=argparse.ArgumentParser()
    for key in ('source','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--belief-precision',choices=['fp32','bf16'],default='fp32')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    tasks=json.loads((a.source/'inventory_validated/public_tasks.json').read_text())
    labels=json.loads((a.source/'inventory_validated/private_labels.json').read_text())
    catalog=json.loads((a.data_root/'cityrefer/objects.json').read_text())
    checkpoint=a.source/'training_aligned/best.pt'
    model=load_model(torch.load(checkpoint,map_location='cpu',weights_only=True)).eval().requires_grad_(False)
    counts=Counter();rows=[];bindings=[];cardinal=[];conflicts=[]
    with torch.inference_mode():
        for task in tasks:
            anchors,audit=bind_public_anchors(task['anchors'],catalog[task['map_name']])
            bindings.append(dict(index=task['index'],bindings=audit))
            counts['queries']+=1;counts['single_public_ID_queries']+=int(len(anchors)==1 and len(anchors[0]['landmark']['public_ids'])==1)
            counts['changed_public_bindings']+=sum(x['public_IDs_changed'] or x['contours_changed'] for x in audit)
            label=labels[task['index']];region=named_union(anchors[0]['contours']);gt=np.array(label['target_world'])
            same=bool(set(label['target_ids'])&set(anchors[0]['landmark']['public_ids']))
            inside=bool(region.covers(shapely.Point(gt)))
            counts['target_ID_overlaps_anchor_ID']+=int(same);counts['GT_point_inside_anchor_union']+=int(inside)
            if same:conflicts.append(dict(index=task['index'],instruction=task['instruction'],map_name=task['map_name'],
                anchor=anchors[0]['landmark'],target_ids=label['target_ids'],
                status='annotation_or_name_scope_ambiguity_not_resolved_by_GT_selection'))
            predicate=anchors[0]['predicate'];unverified=predicate in ('north','south','east','west')
            if unverified:
                axis=np.array(dict(north=(0,1),south=(0,-1),east=(1,0),west=(-1,0))[predicate])
                center=np.array(region.centroid.coords[0]);corners=np.concatenate([np.array(c) for c in anchors[0]['contours']])
                cardinal.append(dict(index=task['index'],predicate=predicate,delta_world=(gt-center).tolist(),
                    center_side_consistent=bool((gt-center)@axis>0),
                    beyond_full_polygon=bool(gt@axis>max(corners@axis)),
                    reference_frame='engineering axes only; geographic north NOT established'))
            item=torch.load(a.source/'cache_aligned'/f"{task['index']:05d}.pt",map_location='cpu',weights_only=True)
            tracker=LandmarkAcquisition();model.reset_episode()
            unique=len(anchors)==1 and len(anchors[0]['landmark']['public_ids'])==1
            supported=anchors[0]['landmark']['type'] in ('Building','Road')
            for obs in item['observations']:
                sample=dict(query=item['query'],**obs);inputs=batch_inputs([sample])
                tracking=None;observed=inputs['observed'];eligible=unique and supported and not unverified
                weak=obs['anchor_labels'];roi=(weak is not None and int((weak==1).sum())>=2)
                if eligible and roi and observed is not None:
                    kw={k:v for k,v in observed.items() if k!='local_uv'}
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        score=model.pgvb(**kw)
                        shifted=dict(kw,image_features=kw['image_features'].roll(score.shape[1]//2,1))
                        counter=model.pgvb(**shifted)
                    tracking=tracker.observe(score.float(),counter.float(),weak[None].cuda(),kw['valid'])
                else:tracking=tracker.observe()
                admitted,reason=admissible_target_rgb(tracking,{'registration_only':True} if roi else None,
                                                     observed if eligible and roi else None)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.belief_precision=='bf16'):raw=model.forward_direct_target(**inputs)
                fields={'Geometry':raw['geometry_field'], 'RawRelation':raw['relation_field'], 'RawRGB':raw['field']}
                fields['SafeRelation']=fields['Geometry'] if unverified else fields['RawRelation']
                fields['SafeRGB']=fields['Geometry'] if unverified else fields['RawRGB'] if admitted else fields['RawRelation']
                bounds=np.array(task['world_bounds']);world=bounds[:2]+inputs['static_inputs']['grid_xy'][0].cpu().numpy()*(bounds[2:]-bounds[:2])
                d=np.linalg.norm(world-gt,axis=-1)
                row=dict(index=task['index'],subset=task['subset'],step=obs['step'],predicate=predicate,
                    category=task['structured']['target']['category'],admitted=admitted,reason=reason,
                    tracking=tracking['state'],unique_public_ID=unique,GT_inside_anchor=inside,
                    uncalibrated_direction_fallback=unverified)
                for name,field in fields.items():
                    logits=field.belief_logits[0].float();top=int(logits.argmax());prob=logits.softmax(-1).cpu().numpy()
                    row[name]=dict(R20=bool(d[top]<=20),NE=float(d[top]),mass20=float(prob[d<=20].sum()))
                rows.append(row)
            if task['index']%100==0:print(json.dumps(dict(query=task['index'],states=len(rows))),flush=True)
    # Synchronized language intervention, unlike previous structure-ID-only probe.
    # Original target labels NEVER score rewritten counterfactual instructions.
    bert=FrozenBERTTokens('bert-base-uncased',device='cuda',max_length=96)
    records=[json.loads(l) for l in (a.source/'cache_aligned/public_static.jsonl').read_text().splitlines()]
    probes=[];args=torch.load(checkpoint,map_location='cpu',weights_only=True)['config']['geometry_args']
    with torch.inference_mode():
        for task in [t for t in tasks if t['anchors'][0]['predicate'] in ('north','south')]:
            phrase=task['anchors'][0]['predicate']
            pattern=r'\b'+phrase+r'\s+of\b'
            if len(re.findall(pattern,task['instruction'],re.I))!=1:
                counts['direction_probe_ambiguous_phrase_skipped']+=1;continue
            item=torch.load(a.source/'cache_aligned'/f"{task['index']:05d}.pt",map_location='cpu',weights_only=True)
            means={};locations={}
            for word,word_id in [('north',0),('south',1)]:
                text=re.sub(pattern,word+' of',task['instruction'],flags=re.I)
                record=dict(records[task['index']],instruction=text)
                path=a.output/'counterfactual_public_query.jsonl';path.write_text(json.dumps(record)+'\n')
                ds=FullCandidateDataset(path,grid_size=28,max_landmarks=args['max_landmarks'],max_roads=args['max_roads'],
                    max_relations=args['max_relations'],prior_sigma_m=args['prior_sigma_m'],rule_sigma_m=args['rule_sigma_m'])
                static,tokens,mask,nodes=_batch_inputs(collate_full([ds[0]]),bert,'cuda','geometry')
                sample=dict(query=item['query'],**item['observations'][0]);inputs=batch_inputs([sample],visual=False)
                inputs.update(static_inputs=static,tokens=tokens,token_mask=mask,node_emb=nodes,
                    relation_ids=torch.full_like(inputs['relation_ids'],word_id))
                model.reset_episode()
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.belief_precision=='bf16'):out=model.forward_direct_target(**inputs)
                bounds=torch.tensor(task['world_bounds'],device='cuda');world=bounds[:2]+static['grid_xy'][0]*(bounds[2:]-bounds[:2])
                means[word]={}
                for method,key in [('Geometry','geometry_field'),('Relation','relation_field')]:
                    logits=out[key].belief_logits[0].float();prob=logits.softmax(-1)
                    means[word][method]=(prob[:,None]*world).sum(0).cpu().tolist()
                locations[word]=text
            probes.append(dict(index=task['index'],texts=locations,world_expectations=means,
                north_minus_south_Y={m:means['north'][m][1]-means['south'][m][1] for m in ('Geometry','Relation')},
                language_and_structure_both_changed=True,counterfactual_target_GT_used=False))
    metrics={m:dict(R20=float(np.mean([q[m]['R20'] for q in rows])),NE=float(np.mean([q[m]['NE'] for q in rows])))
             for m in ('Geometry','RawRelation','RawRGB','SafeRelation','SafeRGB')}
    summary=dict(counts=dict(counts),cached_states=len(rows),raw_and_safe_state_metrics=metrics,
        belief_precision=a.belief_precision,
        state_metrics_by_subset={subset:dict(queries=len({q['index'] for q in rows if q['subset']==subset}),
            states=sum(q['subset']==subset for q in rows),
            **{m:dict(R20=float(np.mean([q[m]['R20'] for q in rows if q['subset']==subset])),
                       NE=float(np.mean([q[m]['NE'] for q in rows if q['subset']==subset])))
               for m in ('Geometry','RawRelation','RawRGB','SafeRelation','SafeRGB')}) for subset in ('fit','dev')},
        gate_accepts=sum(q['admitted'] for q in rows),probe_queries=len(probes),
        synchronized_north_above_south={m:float(np.mean([q['north_minus_south_Y'][m]>0 for q in probes]))
            for m in ('Geometry','Relation')},
        cardinal_label_side_statistics={p:dict(n=len([c for c in cardinal if c['predicate']==p]),
            center_side_consistent=sum(c['center_side_consistent'] for c in cardinal if c['predicate']==p),
            beyond_full_polygon=sum(c['beyond_full_polygon'] for c in cardinal if c['predicate']==p))
            for p in ('north','south')},
        checkpoint_sha256=file_sha256(checkpoint),pixel_instance_accuracy=None,
        visibility='unknown; public Polygon registration is weak support only',
        replay='TRAIN start/middle/last cached causal sparse replay; NOT contiguous closed-loop tracking',
        geometry_geography='TIF CRS absent; keep existing axes and do not flip from GT scores')
    for name,value in [('summary.json',summary),('states.json',rows),('public_bindings.json',bindings),
                       ('TRAIN_ID_conflicts.json',conflicts),('direction_GT_proxy.json',cardinal),('synchronized_direction_probes.json',probes)]:
        atomic_json(value,a.output/name)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
