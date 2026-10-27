"""Policy-hypothesis endpoint Stop supervision, with label-free pose generation."""
import math
import torch
from torch.utils.data import Dataset


class CandidateStopRecoveryDataset(Dataset):
    def __init__(self,base,*,candidates=8):
        self.features=base.features;self.candidates=candidates
        if not 1<=candidates<=self.features['candidate_xy'].shape[1]:raise ValueError('invalid candidate count')
        episodes={}
        for row in base.rows:episodes.setdefault(row['source_episode_index'],row)
        self.rows=list(episodes.values())

    def __len__(self):return len(self.rows)*self.candidates*2

    def __getitem__(self,index):
        row=self.rows[index//(self.candidates*2)];candidate=(index//2)%self.candidates;jitter=index%2
        qid=row['query_index'];inputs={k:v[qid] for k,v in self.features.items()}
        bounds=inputs['bounds'];xy=bounds[:2]+inputs['candidate_xy'][candidate]*(bounds[2:]-bounds[:2])
        # Offset and bearing use public hypothesis/start and deterministic index, not labels.
        angle=(row['source_episode_index']*131+candidate*17)%360*math.pi/180
        offset=torch.tensor([math.cos(angle),math.sin(angle)])*(10.*jitter)
        xy=(xy+offset).maximum(bounds[:2]).minimum(bounds[2:])
        delta=xy-torch.tensor(row['initial_pose_world'][:2]);yaw=torch.atan2(delta[1],delta[0])
        moves=max(1,math.ceil(float(delta.norm())/50.));initial_z=row['initial_pose_world'][2]
        legal_z=min(max(initial_z,row['ground_level']+10),row['ground_level']+100)
        z=initial_z+min(max(legal_z-initial_z,-10*moves),10*moves)
        if jitter:z=row['ground_level']+50.
        inputs.update(pose_world=torch.tensor([xy[0],xy[1],z,yaw]),
            ground_level=torch.tensor(row['ground_level']),observed_fraction=torch.tensor(0.))
        target=torch.tensor(row['labels']['target_world'][:2])
        labels=dict(candidate_index=torch.tensor(candidate),stop=((xy-target).norm()<=20.).float())
        return inputs,labels
