"""Post-rollout diagnosis of arrival evidence and cross-view candidate stability."""
import argparse,json
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    directory=Path(a.collection);manifest=json.loads((directory/'collection.json').read_text())
    if not manifest.get('complete') or manifest['partition']!='train':raise ValueError('Complete Train collection required')
    total=near=covered=miss=0;changes=[];stable_changes=[];tp=fp=fn=0;episodes=[]
    for episode in manifest['episodes']:
        path=directory/episode['file']
        if file_sha256(path)!=episode['sha256']:raise ValueError('Evidence hash mismatch')
        rows=torch.load(path,map_location='cpu',weights_only=True);target=torch.tensor(episode['target_world'][:2]);previous=None;local_near=0;local_miss=0
        for row in rows:
            frame=row['frame'];valid=frame['valid'][0];world=frame['query_world'][0];pose=torch.as_tensor(row['pose'])[:2]
            q=row['query_logits'][0].float().masked_fill(~valid,-1e4)
            probability=torch.cat((q,row['null_logits'].reshape(1).float())).softmax(-1)
            distance=(world-pose).norm(dim=-1);at_goal=bool((pose-target).norm()<=20)
            confirmed=bool((probability[:-1]*(distance<=10)*valid).sum()>=.7)
            tp+=int(confirmed and at_goal);fp+=int(confirmed and not at_goal);fn+=int(not confirmed and at_goal)
            has_target=bool((((world-target).norm(dim=-1)<=20)&valid).any())
            total+=1;near+=int(at_goal);local_near+=int(at_goal);covered+=int(has_target)
            if valid.any():
                chosen=world[q.argmax()];wrong=bool((chosen-target).norm()>20)
                miss+=int(at_goal and wrong);local_miss+=int(at_goal and wrong)
                if previous is not None:
                    jump=float((chosen-previous[0]).norm());changes.append(jump)
                    if float((pose-previous[1]).norm())<=5:stable_changes.append(jump)
                previous=(chosen.clone(),pose.clone())
        episodes.append(dict(index=episode['index'],states=len(rows),near_goal_states=local_near,near_goal_wrong_topquery=local_miss))
    def stats(x):
        return dict(count=len(x),median_m=float(torch.tensor(x).median()),jump_gt20m=sum(v>20 for v in x)) if x else dict(count=0)
    report=dict(partition='train',episodes=len(episodes),states=total,near_goal_states=near,query_coverage20m=covered,near_goal_wrong_topquery=miss,consecutive_query_motion=stats(changes),query_motion_when_agent_moves_le5m=stats(stable_changes),frozen_arrival_rule=dict(radius_m=10,threshold=.7,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None),per_episode=episodes,collection_sha=file_sha256(directory/'collection.json'),note='Training diagnostics only; dependent frames are not independent samples. Top-query motion alone does not prove global goal instability.')
    atomic_json(report,Path(a.output));print(json.dumps({k:v for k,v in report.items() if k!='per_episode'},indent=2))


if __name__=='__main__':main()
