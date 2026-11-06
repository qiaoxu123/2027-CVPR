import json,torch,random,numpy as np
from pathlib import Path
from types import SimpleNamespace
from scripts.unified_pgvb_v3_pilot import V3CacheDataset,fit_dev_partition,_load_base,evaluate,_collate
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3,v3_losses
from graphnav.checkpointing import atomic_json,file_sha256
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--source-v2',type=Path,required=True)
args=parser.parse_args();R=args.run;S=args.source_v2
torch.manual_seed(42);ds=V3CacheDataset(R/'cache128');fit,dev=fit_dev_partition(ds,42)
g,c=_load_base(S,'cuda');m=UnifiedPGVBV3(g).cuda();a,rows=evaluate(m,ds,dev,'cuda',1)
atomic_json(dict(metrics=a,paired=rows),R/'train128/before_replayed.json')
allstats=[]
for i in range(len(ds)):
 kw,lb=_collate([ds[i]],'cuda')
 with torch.no_grad():
  o=m(**kw);ls=v3_losses(o,lb['target_xy'],lb['target_polygons'],kw['world_bounds'],lb['teacher_features'],m.teacher_head)
 gtworld=kw['world_bounds'][:,:2]+lb['target_xy']*(kw['world_bounds'][:,2:]-kw['world_bounds'][:,:2])
 uv=(lb['target_xy'][0]*384).long().clamp(0,383)
 allstats.append(dict(index=i,fit=i in fit,episode_id=ds[i][1]['episode_id'],observed=bool(o.observed_mask.any()),gt_center_observed=bool(o.observed_mask[0,uv[1],uv[0]]),visual_eligible=int(ls['visual_eligible'])))
kw,lb=_collate([ds[fit[0]]],'cuda');m.train();o=m(**kw);ls=v3_losses(o,lb['target_xy'],lb['target_polygons'],kw['world_bounds'],lb['teacher_features'],m.teacher_head);ls['total'].backward()
grads={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.grad is not None}
atomic_json(dict(before=a,source_sha256=file_sha256(S),stats=allstats,gradients=grads,fit_physical_ids=sorted(set(ds[i][1]['episode_id'].split(':')[1] for i in fit)),dev_physical_ids=sorted(set(ds[i][1]['episode_id'].split(':')[1] for i in dev))),R/'audit.json')
print(json.dumps(a))
