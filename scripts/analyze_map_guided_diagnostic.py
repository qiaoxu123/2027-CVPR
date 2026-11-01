"""Post-hoc read-only analysis of frozen predictions; never runs a model."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch

from graphnav.checkpointing import atomic_json
from graphnav.map_guided_grounding import public_landmarks,reference_candidates
from scripts.train_object_query_belief import load


def group_interval(records,left,right,instance=False,seed=0):
    """Paired group bootstrap; fixed predictions, no model/threshold selection."""
    grouped=defaultdict(list)
    for row in records:
        if instance and not row['instance']['native']['eligible']:continue
        if instance:
            delta=float(row['instance'][left]['top1'])-float(row['instance'][right]['top1'])
        else:delta=float(row['methods'][left]['error']<=20)-float(row['methods'][right]['error']<=20)
        grouped[row['group']].append(delta)
    if not grouped:return dict(states=0,groups=0,delta=None,CI95=None)
    values=list(grouped.values());sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(values),size=(10000,len(values)))
    draws=sums[indices].sum(-1)/counts[indices].sum(-1)
    return dict(states=int(counts.sum()),groups=len(values),delta=float(sums.sum()/counts.sum()),
                CI95=np.quantile(draws,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['records','cache','catalog','train-annotations','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Do not overwrite analysis')
    records=[json.loads(line) for line in Path(a.records).read_text().splitlines()]
    meta,rows=load(a.cache)
    if meta['partition'] not in ['train','dev']:raise ValueError('Only TRAIN/internal Dev')
    lookup={(r['episode'],r['step']):r for r in rows}
    catalog=json.loads(Path(a.catalog).read_text())
    if Path(a.train_annotations).name!='citynav_train_seen.json':raise ValueError('Only TRAIN labels')
    annotations=json.loads(Path(a.train_annotations).read_text())
    named={m:public_landmarks(catalog,m) for m in {r['map'] for r in records}}
    types={};mismatch=[];ref_counts=Counter();cohorts=Counter();eligibility=Counter();spatial_vs_instance=Counter()
    for record in records:
        row=lookup[(record['episode'],record['step'])];ann=annotations[record['episode']]
        obj=catalog[record['map']][record['official_instance_id']]
        difference=float(np.linalg.norm(np.array(obj['position'])[:2]-row['labels']['target_world'][:2].numpy()))
        if difference>1e-3:mismatch.append(dict(episode=record['episode'],step=record['step'],difference_m=difference))
        cohorts[record['visibility']]+=1
        landmarks=named[record['map']]
        _,relations,_,_=reference_candidates(landmarks,[dict(valid_roi=False) for _ in landmarks],record['instruction'])
        selected={i for i,e in enumerate(landmarks) if e['key'] in record['reference_ids']}
        active=any(selected.intersection(a for group in relation.anchors for a in group) for relation in relations)
        ref_counts['states_with_supported_world_relation']+=bool(relations)
        ref_counts['states_where_reference_RGB_can_modulate_relation']+=active
        ref_counts['states_with_reference_crops']+=bool(record['reference_ids'])
        ref_counts['total_reference_crops']+=len(record['reference_ids'])
        ref_counts['states_with_appearance_tokens']+=record['appearance_tokens']>0
        if record['methods']['A0']['error']<=20 and record['visibility']=='neither_projected':ref_counts['A0_correct_neither_projected']+=1
        if record['instance']['native']['eligible']:
            kind=obj['object_type'];items=types.setdefault(kind,dict(states=0,methods={}))
            items['states']+=1
            for name in ['native','map_only','map_rgb','shuffled_rgb']:
                m=items['methods'].setdefault(name,dict(top1=0,top5=0,top15=0))
                for k in m:m[k]+=bool(record['instance'][name][k])
            top=record['methods']['native']['rank']
            spatial=bool((row['query_world'][top]-row['labels']['target_world'][:2]).norm()<=20)
            spatial_vs_instance['eligible']+=1;spatial_vs_instance['native_center20_correct']+=spatial
            spatial_vs_instance['center20_but_not_instance_IoU50']+=spatial and not record['instance']['native']['top1']
            eligibility['eligible']+=1
        else:
            q=record['instance']['native']['projection']
            eligibility['excluded_'+q['reason']]+=1
    if mismatch:raise ValueError('Official instance labels disagree with localization coordinates: '+str(mismatch[:5]))
    comparisons={}
    for right in ['map_only','shuffled_rgb','native']:
        comparisons['map_rgb_minus_'+right]=dict(R1=group_interval(records,'map_rgb',right),
                                                instance_top1=group_interval(records,'map_rgb',right,instance=True))
    for value in types.values():
        for m in value['methods'].values():
            for k in m:m[k]/=value['states']
    atomic_json(dict(states=len(records),visibility=dict(cohorts),reference_evidence=dict(ref_counts),
                     instance_by_type=types,instance_eligibility=dict(eligibility),spatial_vs_instance=dict(spatial_vs_instance),
                     official_instance_label_mismatches=0,comparisons=comparisons,
                     max_footprint_center_error_m=max(r['footprint_center_error_m'] for r in records),
                     physical_semantic_visibility='unverified for every state; geometric support only',
                     human_projection_error=None),out)
    review=out.with_suffix('.human_review.csv')
    with review.open('w') as handle:
        writer=csv.writer(handle);writer.writerow(['episode','step','official_instance_id','geometric_visibility','human_target_visible','human_reference_visible','landmark_registration_error_px','human_verified_instance_box','notes'])
        for r in records:writer.writerow([r['episode'],r['step'],r['official_instance_id'],r['visibility'],'UNREVIEWED','UNREVIEWED','','',''])


if __name__=='__main__':main()
