"""Fit-only necessary support bound; no new model or outcome-conditioned method."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from graphnav.evidence_reachability import local_evidence_coverage,position_support_bound
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.train_local_chain_position import records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior diagnostic')
    torch.set_num_threads(4);r=a.root;allowed=set(json.loads((r/'public_deployment_filter.json').read_text())['frame_keys'])
    data=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_fit.jsonl','fit',None,'full',allowed)
    baseline=json.loads((r/'region_chain_feasibility_v2/results.json').read_text())['rows']['FrozenV3'];base={x['frame_key']:x for x in baseline}
    counts=Counter();rows=[]
    for row in data:
        covered=local_evidence_coverage(row['local_uv'][None],row['local_valid'][None],784)[0]
        success=(row['candidate_world'][:784]-row['target_world']).norm(dim=-1)<=20
        bound=position_support_bound(row['prior_logits'],covered,success,diagnostic=True)
        current=base[row['frame_key']];a0_correct=bool(success[row['prior_logits'].argmax()])
        # An observed actual argmax cannot contradict the necessary support bound.
        if current['success'] and not bound['possible']:raise RuntimeError('support bound inconsistent with measured head')
        counts['frames']+=1;counts['optimistically_possible']+=int(bound['possible']);counts[bound['kind']]+=1
        if not current['success']:
            counts['current_failed']+=1;counts['failed_'+bound['kind']]+=1
            if a0_correct:counts['failed_but_A0_correct']+=1
        rows.append(dict(frame_key=row['frame_key'],task_key=row['task_key'],category=row['category'],current_success=current['success'],a0_success=a0_correct,
            bound=bound,covered_world_cells=int(covered.sum()),covered_success_cells=int((covered&success).sum())))
    a.output.mkdir(parents=True)
    atomic_json(dict(summary=dict(counts),rows=rows,source_sha256=file_sha256(__file__),
        warning='Single-frame Goal argmax necessary bound only. Covered success cells may be coupled by splatting; not an achievable Oracle SR or complete-trajectory bound.'),a.output/'results.json')
    print(json.dumps(dict(counts)))


if __name__=='__main__':main()
