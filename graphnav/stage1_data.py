"""Actual current RGB features and strictly earlier, candidate-aligned quality memory."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .candidate_geometry import FullCandidateDataset,collate_full
from .checkpointing import file_sha256


class CandidateMemory:
    def __init__(self,cells,dimension,device='cpu'):
        self.sum=torch.zeros(cells,3,dimension,device=device)
        self.weight=torch.zeros(cells,3,device=device)
        self.quality_sum=torch.zeros(cells,3,3,device=device)
        self.last_step=-1;self.last_pose=None;self.accepted=0;self.duplicates=0;self.unobserved=0

    def update(self,frame):
        step=int(frame['step'])
        if step<=self.last_step:raise ValueError('noncausal or repeated memory timestep')
        self.last_step=step;pose=frame['pose'].detach().cpu().numpy()
        if not frame['roi_valid'].any():self.unobserved+=1;return
        if self.last_pose is not None and np.linalg.norm(pose[:3]-self.last_pose[:3])<2.:
            self.duplicates+=1;return
        self.last_pose=pose.copy();self.accepted+=1
        ids=frame['roi_ids'].to(self.sum.device);valid=frame['roi_valid'].to(self.sum.device)
        quality=frame['roi_quality'].to(self.sum.device).float();images=frame['roi_features'].to(self.sum.device).float()
        w=(quality[...,0].clamp(0,1)*quality[...,1].clamp(0,1)).clamp_min(.01)*valid
        self.sum[ids]+=images*w[...,None];self.weight[ids]+=w;self.quality_sum[ids]+=quality*w[...,None]

    def tensors(self):
        return (self.sum/self.weight.clamp_min(1e-9)[...,None]).half(),self.weight>0,self.quality_sum/self.weight.clamp_min(1e-9)[...,None]


class Stage1Dataset(Dataset):
    def __init__(self,root,verify=True):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['complete'] or self.manifest['protocol']['cache_kind']!='stage1-current-and-prefix-v1':raise ValueError('actual complete frame cache required')
        if verify:
            assert file_sha256(self.root/'queries.jsonl')==self.manifest['queries_sha256']
            for c in self.manifest['chunks']:
                if file_sha256(self.root/c['file'])!=c['sha256']:raise ValueError('frame cache corrupt')
        args=self.manifest['protocol']['a3_args']
        self.static=FullCandidateDataset(self.root/'queries.jsonl',grid_size=args['grid_size'],max_landmarks=args['max_landmarks'],max_roads=args['max_roads'],max_relations=args['max_relations'],prior_sigma_m=args['prior_sigma_m'],rule_sigma_m=args['rule_sigma_m'],deduplicate_queries=False)
        self.lookup=[(ci,ri)for ci,c in enumerate(self.manifest['chunks'])for ri in range(c['rows'])];self.cache=OrderedDict()

    def __len__(self):return len(self.lookup)

    def __getitem__(self,index):
        ci,ri=self.lookup[index]
        if ci not in self.cache:self.cache[ci]=torch.load(self.root/self.manifest['chunks'][ci]['file'],map_location='cpu',weights_only=True)
        self.cache.move_to_end(ci)
        while len(self.cache)>2:self.cache.popitem(last=False)
        chunk=self.cache[ci];row=chunk['rows'][ri];frames=chunk['episodes'][row['episode_index']]
        if row['episode_index']!=self.manifest['protocol']['episode_indices'][index//4]:raise ValueError('expert prefix indexing differs from deterministic split')
        current=[f for f in frames if f['step']==row['cutoff_step']]
        if len(current)!=1:raise ValueError('current observation must match exact cutoff')
        past=[f for f in frames if f['step']<row['cutoff_step']]
        if any(s>row['cutoff_step']for s in row['history_steps']):raise ValueError('future history')
        return self.static[row['query_index']],row,current[0],past


def collate_stage1(items):
    static=collate_full([x[0]for x in items]);rows=[x[1]for x in items];q=static['grid_xy'].shape[1];d=rows[0]['native_text'].shape[-1]
    images=torch.zeros(len(items),q,3,d,dtype=torch.float16);valid=torch.zeros(len(items),q,3,dtype=torch.bool);quality=torch.zeros(len(items),q,3,3)
    past_images=torch.zeros_like(images);past_valid=torch.zeros_like(valid);past_quality=torch.zeros_like(quality)
    for i,(_,row,frame,past)in enumerate(items):
        ids=frame['roi_ids'];images[i,ids]=frame['roi_features'];valid[i,ids]=frame['roi_valid'];quality[i,ids]=frame['roi_quality']
        memory=CandidateMemory(q,d)
        for f in past:memory.update(f)
        past_images[i],past_valid[i],past_quality[i]=memory.tensors()
    public=dict(native_text=torch.stack([r['native_text']for r in rows]).float(),appearance_text=torch.stack([r['appearance_text']for r in rows]).float(),roi_features=images,roi_valid=valid,roi_quality=quality,history_features=past_images,history_valid=past_valid,history_quality=past_quality,pose_world=torch.stack([r['pose_world']for r in rows]),bounds=torch.tensor(static['world_bounds'],dtype=torch.float32),ground_level=torch.stack([r['ground_level']for r in rows]))
    labels={k:torch.stack([r['labels'][k]for r in rows])for k in ('target_world','teacher_world','valid_teacher_z')}
    return static,public,labels
