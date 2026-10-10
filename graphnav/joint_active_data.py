"""Frozen-backbone feature shards, with public inputs and private labels apart."""
from collections import OrderedDict
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset,Sampler

from .candidate_geometry import FullCandidateDataset,collate_full
from .checkpointing import file_sha256


class JointActiveDataset(Dataset):
    def __init__(self,root,*,verify=True):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if not self.manifest['complete']:raise ValueError('incomplete observed backbone cache')
        protocol=self.manifest['protocol']
        if not protocol['causal_observations'] or not protocol['no_cached_a3_outputs']:
            raise ValueError('joint training requires causal backbone-only features')
        if verify:
            if file_sha256(self.root/'queries.jsonl')!=self.manifest['queries_sha256']:
                raise ValueError('public query source changed')
            for chunk in self.manifest['chunks']:
                if file_sha256(self.root/chunk['file'])!=chunk['sha256']:raise ValueError('observed cache SHA mismatch')
        args=protocol['a3_args']
        self.static=FullCandidateDataset(self.root/'queries.jsonl',grid_size=args['grid_size'],
            max_landmarks=args['max_landmarks'],max_roads=args['max_roads'],
            max_relations=args['max_relations'],prior_sigma_m=args['prior_sigma_m'],
            rule_sigma_m=args['rule_sigma_m'],deduplicate_queries=False)
        self.lookup=[(ci,ri) for ci,c in enumerate(self.manifest['chunks']) for ri in range(c['rows'])]
        self.cache=OrderedDict()

    def __len__(self):return len(self.lookup)

    def __getitem__(self,index):
        ci,ri=self.lookup[index]
        if ci not in self.cache:
            self.cache[ci]=torch.load(self.root/self.manifest['chunks'][ci]['file'],map_location='cpu',weights_only=True)['rows']
        self.cache.move_to_end(ci)
        while len(self.cache)>4:self.cache.popitem(last=False)
        row=self.cache[ci][ri]
        static=self.static[row['query_index']]
        if any(step>row['cutoff_step'] for step in row['history_steps']):raise ValueError('future image in feature history')
        return static,row


def collate_joint(items):
    static=collate_full([i[0] for i in items]);rows=[i[1] for i in items]
    q=static['grid_xy'].shape[1];dimension=rows[0]['native_text'].shape[-1]
    images=torch.zeros(len(rows),q,3,dimension,dtype=torch.float16)
    valid=torch.zeros(len(rows),q,3,dtype=torch.bool)
    quality=torch.zeros(len(rows),q,3,3)
    for i,row in enumerate(rows):
        ids=row['roi_ids'];images[i,ids]=row['roi_features'];valid[i,ids]=row['roi_valid'];quality[i,ids]=row['roi_quality']
    public=dict(native_text=torch.stack([r['native_text'] for r in rows]).float(),roi_features=images.float(),
        roi_valid=valid,roi_quality=quality,pose_world=torch.stack([r['pose_world'] for r in rows]),
        bounds=torch.tensor(static['world_bounds'],dtype=torch.float32),
        ground_level=torch.stack([r['ground_level'] for r in rows]))
    labels={key:torch.stack([r['labels'][key] for r in rows])
            for key in ('target_world','teacher_world','valid_teacher_z')}
    return static,public,labels


class ShardShuffleSampler(Sampler):
    """Seeded shard and within-shard shuffle avoids rereading GB per mini-batch.

    Every source state is used once per epoch, no dropping or target-dependent
    sampling. This order differs from SBF's row permutation and is disclosed.
    """
    def __init__(self,dataset,seed=0):
        self.seed=seed;self.epoch=0;self.sizes=[c['rows'] for c in dataset.manifest['chunks']]
        self.offsets=[];offset=0
        for size in self.sizes:self.offsets.append(offset);offset+=size

    def __len__(self):return sum(self.sizes)

    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        for ci in torch.randperm(len(self.sizes),generator=generator).tolist():
            for ri in torch.randperm(self.sizes[ci],generator=generator).tolist():yield self.offsets[ci]+ri
