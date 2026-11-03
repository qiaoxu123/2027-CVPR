"""Fit-only region-chain counterfactuals; no training or deployable rule change."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import torch
import numpy as np
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.region_chain_oracle import oracle_region_rejection
from graphnav.region_semantic_probe import RegionSemanticProbe,shuffle_observed_features
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records,collate,batch_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--requirements',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve original counterfactual')
    r=a.root;torch.set_num_threads(4);started=time.perf_counter()
    allowed=set(json.loads((r/'public_deployment_filter.json').read_text())['frame_keys'])
    role_vectors=torch.load(r/'reference_vectors_v2.pt',map_location='cpu',weights_only=True)
    rows=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_fit.jsonl','fit',role_vectors,'reference',allowed)
    assert len(rows)==900
    requirements=json.loads(a.requirements.read_text())
    private=torch.load(r/'region_semantic_probe_labels_v1/loss_labels_private.pt',map_location='cpu',weights_only=True)
    head_path=r/'world_v3_envelope_s0/best.pt';probe_path=r/'region_semantic_probe_suite_v1_s0/pilot/final.pt'
    head=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False);head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=True)['model'])
    probe=RegionSemanticProbe().cuda().eval().requires_grad_(False);probe.load_state_dict(torch.load(probe_path,map_location='cpu',weights_only=True)['model'])
    methods=('FrozenV3','OracleRegionOnly','RGBRegionOnly','ShuffleRegionOnly');scores={m:[] for m in methods};coverage=Counter();known_confusion=np.zeros((2,2),int)
    for start in range(0,len(rows),8):
        batch_rows=rows[start:start+8];batch=collate(batch_rows);inputs=batch_inputs(batch,'cuda')
        with torch.inference_mode():
            logits=head(**inputs)[:,:784].cpu()
            predicted=probe(image_features=inputs['image_features']).argmax(-1).cpu()
            shuffled=torch.stack([shuffle_observed_features(row['image_features'],row['local_valid']) for row in batch_rows]).cuda()
            predicted_shuffle=probe(image_features=shuffled).argmax(-1).cpu()
        for i,row in enumerate(batch_rows):
            proposal=requirements[row['task_key']]['proposal'];coverage['frames']+=1
            masks={m:torch.zeros(784,dtype=torch.bool) for m in methods}
            if proposal:
                kind=('Building','Parking','Ground').index(proposal['region']);coverage['applicable_frames']+=1;coverage[proposal['kind']]+=1
                region=private[row['frame_key']];valid=row['local_valid'];known=(region>=0)&valid
                np.add.at(known_confusion,((region[known]==kind).long().numpy(),(predicted[i,known]==kind).long().numpy()),1)
                args=(row['candidate_world'][:784],row['candidate_world'][784:])
                width=float(row['quality'][0,3])*100
                for name,label in [('OracleRegionOnly',region),('RGBRegionOnly',predicted[i]),('ShuffleRegionOnly',predicted_shuffle[i])]:
                    masks[name]=oracle_region_rejection(*args,label,valid,width,kind,diagnostic=True)
                    if masks[name].all():masks[name].zero_();coverage[name+'_all_rejected_fallback']+=1
                choice=int(logits[i].argmax())
                for name in methods[1:]:coverage[name+'_baseline_peak_rejected']+=int(masks[name][choice])
            else:coverage['abstain_frames']+=1
            for name in methods:
                choice=int(logits[i].masked_fill(masks[name],float('-inf')).argmax())
                distance=float((row['candidate_world'][choice]-row['target_world']).norm())
                scores[name].append(dict(frame_key=row['frame_key'],task_key=row['task_key'],category=row['category'],
                    requirement=proposal,ne_m=distance,success=distance<=20.,rejected_cells=int(masks[name].sum()),chosen_index=choice))
    summaries={};pairs={}
    for name,scored in scores.items():
        summaries[name]=dict(frames=len(scored),correct=sum(x['success'] for x in scored),goal_r1=float(np.mean([x['success'] for x in scored])),goal_ne_m=float(np.mean([x['ne_m'] for x in scored])))
        pairs[name]=dict(corrected=[x['frame_key'] for x,b in zip(scored,scores['FrozenV3']) if x['success'] and not b['success']],harmed=[x['frame_key'] for x,b in zip(scored,scores['FrozenV3']) if not x['success'] and b['success']])
    a.output.mkdir(parents=True)
    atomic_json(dict(summary=summaries,paired=pairs,coverage=dict(coverage),binary_region_confusion=known_confusion.tolist(),rows=scores,
        probe_sha256=file_sha256(probe_path),head_sha256=file_sha256(head_path),requirements_sha256=file_sha256(a.requirements),
        source_sha256=file_sha256(__file__),seconds=time.perf_counter()-started,
        warning='Fit-only hard-filter mechanism diagnosis, not an optimal upper bound, verified instance accuracy or closed-loop SR'),a.output/'results.json')
    print(json.dumps(dict(summary=summaries,paired=pairs,coverage=dict(coverage),binary_region_confusion=known_confusion.tolist())))


if __name__=='__main__':main()
