"""Offline label/candidate coverage audit; no GT enters the inference network."""
from common import *
import numpy as np
import torch
from train import load_groups,audit_groups


def main():
    groups=load_groups();summary=audit_groups(groups);examples=[]
    for split,rows in groups.items():
        stats=dict(candidates=0,trusted_candidates=0,unknown_candidates=0,empty_views=0,capped_views=0,
            weak_nearby_target_views=0,observed_height_pair_views=0,supervised_height_views=0,match_positive_views=0,match_negative_views=0,
            target_visible_views=0,duplicate_reference_views=0,coarse_candidate_collision_views=0)
        for r in rows:
            y=r['labels'];x=r['public']['network'];s=r['public']['spatial'];meta=r['meta'];n=len(meta['candidates'])
            stats['candidates']+=n;trusted=int(y['rank_label_mask'].sum());stats['trusted_candidates']+=trusted;stats['unknown_candidates']+=n-trusted
            stats['empty_views']+=n==0;stats['capped_views']+=n==32;stats['supervised_height_views']+=bool(y['height_valid'].any())
            stats['observed_height_pair_views']+=bool((x['pair'][...,12]>.5).any());stats['target_visible_views']+=r['private']['target_visible']
            if y['match_valid']:stats['match_positive_views' if y['match']>0 else 'match_negative_views']+=1
            if n:
                dist=(s['candidate_xy']-y['target_xy']).norm(dim=-1)
                kind=torch.tensor([c['kind'] for c in meta['candidates']])
                stats['weak_nearby_target_views']+=bool(((dist<=8)&(kind==r['private']['target_kind'])).any())
                cell=torch.cdist(s['candidate_xy'],s['grid_world']).argmin(-1)
                stats['coarse_candidate_collision_views']+=len(cell.unique())<n
            names=[v['name'] for v in meta['references']];stats['duplicate_reference_views']+=len(set(names))<len(names)
            for j,a in enumerate(r['private']['association']):
                examples.append(dict(split=split,episode=r['episode'],step=r['step'],candidate=j,**a))
        summary[split].update(stats)
    rng=np.random.default_rng(17);ids=rng.permutation(len(examples))[:100]
    sample=[examples[int(i)] for i in ids]
    result=dict(scope='frozen train/calibration/internal-dev only',coverage=summary,association_audit_sample=sample,
        manually_verified=False,caveats=['IoU/center association is a conservative rule, not human-verified instance GT.',
            'Weak nearby counts are diagnostic only and never promoted to ranking labels.',
            'Duplicate canonical reference names are retained to match the original public packing; investigate deduplication separately.',
            'Most boxes share coarse cells; finer instance ranking need not change28x28 navigation.'])
    save(result,ROOT/'reports/data_audit.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
