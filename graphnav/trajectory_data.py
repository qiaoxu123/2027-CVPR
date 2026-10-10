"""Immutable frozen-query features and label-isolated temporal supervision."""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .checkpointing import file_sha256

FEATURE_KEYS=('candidate_xy','candidate_context','candidate_logits','belief_entropy','bounds')


def query_is_dev(query, seed=0):
    # All annotation variants and temporal/recovery derivatives of a query stay together.
    family=json.dumps([query['map_name'],query['instruction']],ensure_ascii=False)
    digest=hashlib.sha256(f'{seed}:{family}'.encode()).hexdigest()
    return int(digest[:8],16)%10==0


class FrozenTrajectoryDataset(Dataset):
    def __init__(self,samples,queries,cache,*,split='train',seed=0):
        self.samples_path=Path(samples);self.queries_path=Path(queries);self.cache_path=Path(cache)
        manifest=json.loads((self.cache_path/'manifest.json').read_text())
        if not manifest.get('complete'):raise ValueError('frozen feature cache is incomplete')
        if manifest['protocol']['query_sha256']!=file_sha256(queries):
            raise ValueError('frozen feature/public query SHA256 mismatch')
        query_rows=[json.loads(line) for line in self.queries_path.read_text().splitlines() if line]
        features={key:[] for key in FEATURE_KEYS}
        first=0
        for chunk in manifest['chunks']:
            path=self.cache_path/chunk['file']
            if chunk['first_query']!=first or file_sha256(path)!=chunk['sha256']:
                raise ValueError('cache chunk order or SHA256 mismatch')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            for key in FEATURE_KEYS:features[key].append(payload[key])
            first+=chunk['queries']
            del payload # Dense candidate contexts are deliberately not retained by this head.
        if first!=len(query_rows):raise ValueError('feature/query count mismatch')
        self.features={key:torch.cat(parts).float() for key,parts in features.items()}
        self.rows=[]
        with self.samples_path.open() as stream:
            for line in stream:
                row=json.loads(line);index=row['query_index']
                if not 0<=index<len(query_rows):raise ValueError('invalid public query index')
                dev=query_is_dev(query_rows[index],seed)
                if split=='all' or dev==(split=='dev'):self.rows.append(row)
        if split not in ('train','dev','all'):raise ValueError('unknown split')

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];qid=row['query_index']
        inputs={key:value[qid] for key,value in self.features.items()}
        inputs.update(pose_world=torch.tensor(row['pose_world'],dtype=torch.float32),
            ground_level=torch.tensor(row['ground_level'],dtype=torch.float32),
            # This geometry/pose baseline has no rendered history or visual supervision.
            observed_fraction=torch.tensor(0.,dtype=torch.float32))
        labels={key:torch.tensor(row['labels'][key],dtype=torch.float32)
                for key in ('target_world','teacher_world','valid_teacher_z')}
        return inputs,labels


def move_batch(batch,device):
    return tuple({key:value.to(device,non_blocking=True) for key,value in part.items()} for part in batch)
