"""Check actual before/after public inputs without loading any target labels into selection."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import sys

import torch

from graphnav.a3_runtime import public_static_record
from graphnav.closedloop import PublicCatalog, PublicEpisode
from graphnav.landmark_ablation import instruction_landmarks_only


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.main_run);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    report=json.loads((root/'full_seen/metrics.json').read_text());settings=report['args']
    sys.path.insert(0,settings['sbf_source'])
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    state=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
    catalog=CityReferCatalog(Path(settings['data_root'])/'cityrefer')
    builder=PlanningStateBuilder(settings['data_root'],PublicCatalog(catalog))
    result={}
    for split in ['val_seen','val_unseen']:
        dataset=CityNavDataset(settings['data_root'],split)
        rows=[];counts=Counter();original=Counter();retained=Counter()
        for ep in dataset:
            public=PublicEpisode(ep.map_name,ep.instruction,ep.start_pose.copy(),
                tuple(catalog.referenced_landmarks(ep)),tuple(catalog.referenced_landmark_names(ep)))
            record=public_static_record(public,builder.transform(ep.map_name),catalog,
                max_landmarks=state['a3_args']['max_landmarks'])
            filtered,audit=instruction_landmarks_only(record,public.reference_names)
            digest=lambda data:hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
            before_sha,after_sha=digest(record),digest(filtered)
            changed=before_sha!=after_sha
            counts['changed' if changed else 'unchanged']+=1
            original[audit['original_polygons']]+=1;retained[audit['retained_polygons']]+=1
            rows.append(dict(index=ep.index,map_name=ep.map_name,instruction=ep.instruction,changed=changed,
                original_public_input_sha256=before_sha,filtered_public_input_sha256=after_sha,**audit))
        result[split]=dict(episodes=len(rows),**counts,original_polygons=dict(sorted(original.items())),
            retained_polygons=dict(sorted(retained.items())))
        (output/(split+'_input_audit.json')).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (output/'input_audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
