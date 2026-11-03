"""TRAIN-only identity manifest for public two-anchor between proposals.

Target IDs are used ONLY to deduplicate physical training examples. No target
position/contour, model outcome, Val/Test record or GT-nearest anchor is read.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from scripts.analyze_instruction_landmark_association import iter_train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('proposals','annotations','map-manifest','catalog','exclude-tasks','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    proposals=json.loads(a.proposals.read_text())
    eligible={(x['map_name'],x['instruction']):x for x in proposals if any(g['kind']=='two-public-anchors' for g in x['groups'])}
    maps=json.loads(a.map_manifest.read_text())['map_assignment']
    objects=json.loads(a.catalog.read_text())
    excluded={tuple(x['physical_group']) for x in json.loads(a.exclude_tasks.read_text())}
    candidates=[];matched=Counter()
    for i,r in iter_train(a.annotations):
        mp=f"{r['area']}_block_{r['block']}";key=(mp,r['descriptions'][0])
        if key not in eligible:continue
        matched[maps.get(mp,'unassigned')]+=1
        if maps.get(mp)!='fit':continue
        oid=int(r['object_ids'][0]);group=(mp,oid)
        if group in excluded:continue
        candidates.append(dict(index=i,physical_group=list(group),category=objects[mp][str(oid)]['object_type']))
    ordered=sorted(candidates,key=lambda x:hashlib.sha256(('0:'+str(x['index'])).encode()).hexdigest())
    selected=[];seen=set()
    for x in ordered:
        group=tuple(x['physical_group'])
        if group not in seen:seen.add(group);selected.append(x)
    a.output.mkdir(parents=True)
    (a.output/'selected_tasks_private.json').write_text(json.dumps(selected,indent=2))
    summary=dict(eligible_public_texts=len(eligible),matched_processed_tasks=dict(matched),
        new_fit_physical_targets=len(selected),categories=dict(Counter(x['category'] for x in selected)),
        maps=len({x['physical_group'][0] for x in selected}),no_gt_target_positions_or_contours_read=True,
        selection='two-public-anchor text proposal, exact map+instruction join; fit maps only; exclude all prior400 physical targets; deterministic seed0 identity dedup',
        no_observations_rendered=True,no_training_performed=True,
        sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in dict(proposals=a.proposals,annotations=a.annotations,map_manifest=a.map_manifest,exclude_tasks=a.exclude_tasks,selected=a.output/'selected_tasks_private.json',producer=Path(__file__)).items()})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
