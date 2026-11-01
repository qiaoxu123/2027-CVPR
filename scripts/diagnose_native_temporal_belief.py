"""Train-only causal window evidence diagnosis; no navigation gain is inferred."""
import argparse,json
from pathlib import Path
import torch
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.temporal_native_evidence import CausalResidualWindow
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.collection);manifest=json.loads((root/'collection.json').read_text())
    if not manifest.get('complete') or manifest['partition']!='train' or manifest['native_sha']!=file_sha256(a.checkpoint):raise ValueError('Completed Train collection/checkpoint required')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);config=state['config']
    if config['relation']:raise ValueError('This CPU diagnostic currently requires the plain checkpoint')
    ranker=ObjectQueryRanker();ranker.load_state_dict(state['model']);ranker.eval()
    windows=[1,2,4,8];totals={w:dict(states=0,correct=0,error_m=0.,near_states=0,near_correct=0,goal_jumps_gt20m=0,transitions=0) for w in windows}
    per_episode=[]
    with torch.no_grad():
        for episode in manifest['episodes']:
            path=root/episode['file']
            if file_sha256(path)!=episode['sha256']:raise ValueError('Evidence checksum mismatch')
            target=torch.tensor(episode['target_world'][:2]);memories={w:CausalResidualWindow(w) for w in windows};previous={};local={w:0 for w in windows}
            for row in torch.load(path,map_location='cpu',weights_only=True):
                f=row['frame'];base=row['base_logits'].float();world=row['grid_world'].float()
                _,delta=ranker.update_belief(base,row['query_logits'],row['null_logits'],f['kernel'],f['observed'],f['valid'],bound=config['bound'])
                for w in windows:
                    residual=memories[w].update(delta,f['observed'])[0]
                    goal=world[(base[0]+residual).argmax()];error=float((goal-target).norm());correct=int(error<=20);near=bool((torch.as_tensor(row['pose'])[:2]-target).norm()<=20)
                    total=totals[w];total['states']+=1;total['correct']+=correct;total['error_m']+=error;total['near_states']+=int(near);total['near_correct']+=int(near)*correct;local[w]+=correct
                    if w in previous:total['transitions']+=1;total['goal_jumps_gt20m']+=int((goal-previous[w]).norm()>20)
                    previous[w]=goal
            per_episode.append(dict(index=episode['index'],correct_by_window=local))
    for value in totals.values():
        value['R1']=value['correct']/value['states'];value['NE_m']=value['error_m']/value['states'];value['near_R1']=value['near_correct']/max(value['near_states'],1)
    atomic_json(dict(partition='train',windows=totals,episodes=per_episode,protocol='Causal arithmetic mean of bounded residuals over observed support within last W actual views; GT scores afterward. Offline state ranking, not a navigation counterfactual.'),Path(a.output))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
