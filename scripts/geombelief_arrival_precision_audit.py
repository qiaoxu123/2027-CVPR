"""Offline audit of numerical arrival-gate precision; never alter trajectories."""
import json
import torch
from scripts.geombelief_map_experiments import OUT
from graphnav.checkpointing import atomic_json

def main():
    result=[]
    for scope in ('controller','controller_fixed_goal'):
        for split in ('val_seen','val_unseen'):
            for folder in sorted((OUT/scope).glob(f'*/{split}')):
                positions=[];goals=[];metadata=[]
                for line in (folder/'predictions.jsonl').open():
                    row=json.loads(line)
                    for step,d in enumerate(row['step_diagnostics']):
                        positions.append(row['poses'][step][:2]);goals.append(d['selected_goal_xy']);metadata.append((row['episode_id'],step,d))
                delta=torch.tensor(goals,device='cuda',dtype=torch.float32)-torch.tensor(positions,device='cuda',dtype=torch.float32)
                arrival=(delta.norm(dim=-1)<=.001).cpu().tolist();changed=[]
                for a,(eid,step,d) in zip(arrival,metadata):
                    if a!=d['predicted_arrival']:
                        changed.append(dict(episode_id=eid,step=step,float32_arrival=a,float64_arrival=d['predicted_arrival'],would_change_stop_request=bool(d['stop_altitude_guard'] and (folder.parent.name=='rule_arrival' or d['learned_stop_requested']))))
                result.append(dict(scope=scope,split=split,arm=folder.parent.name,decisions=len(metadata),arrival_disagreements=len(changed),potential_stop_request_disagreements=sum(x['would_change_stop_request'] for x in changed),details=changed))
    atomic_json(dict(threshold_m=.001,current_control='float64 world-distance recomputed from selected goal and pose; identical precision for every controlled arm',archived_runtime='float32 tensor-distance guard',note='Offline float32 GPU norm comparison at recorded states. This is not a counterfactual replay. No metric, model, threshold or trajectory is changed.',results=result),OUT/'arrival_precision_audit.json')
if __name__=='__main__':main()
