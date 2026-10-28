from pathlib import Path
from collections import Counter,defaultdict
import json
import torch,numpy as np
from experiments.a2_waldo_landmark_belief.collected_snapshot import load_cache_row
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold
from experiments.a2_waldo_landmark_belief.paths import save

torch.set_num_threads(2);b=Path('runs/a2_waldo_landmark_20261027');cache=b/'collected_snapshot';cal=b/'calibration_collected'
m=json.loads((cache/'manifest.json').read_text());counts=Counter();by_kind=defaultdict(Counter);rr=[];top=[];full=[];offset_groups=defaultdict(list)
for e in m['entries']:
 if e['split']!='calibration':continue
 p=torch.load(cal/'predictions'/e['filename'],map_location='cpu',weights_only=False)
 if calibration_fold(*p['group'])!='verify':continue
 row=load_cache_row(cache,e,'calibration');out=p['output'];y=row['labels'];positive=(y['ranking']>0) & out['reliable'];known=p['known'];scores=out['scores'];winner=int(scores.argmax());target_kind=row['private']['target_kind']
 if row['meta']['candidates']:
  winner_kind=row['meta']['candidates'][winner]['kind']
  counts['winner_cross_category_all_views']+=int(winner_kind!=target_kind)
 if positive.any():
  winner_kind=row['meta']['candidates'][winner]['kind'];by_kind[str(target_kind)]['positive_views']+=1
  by_kind[str(target_kind)]['winner_positive']+=int(positive[winner]);by_kind[str(target_kind)]['winner_cross_category']+=int(winner_kind!=target_kind)
  counts['positive_views']+=1;counts['positive_views_cross_category_winner']+=int(winner_kind!=target_kind)
  same=torch.tensor([c['kind']==target_kind for c in row['meta']['candidates']],dtype=torch.bool)&out['reliable']
  # GT category mask is ONLY an exploratory oracle, NEVER an inference change.
  masked=scores.masked_fill(~same,-torch.inf)
  counts['oracle_target_category_only_winner_positive']+=int(positive[int(masked.argmax())])
 if (positive & known).any() and (known & ~positive).any():
  order=scores[known].argsort(descending=True,stable=True);pos=positive[known][order].nonzero().flatten();rr.append(1/(int(pos[0])+1));top.append(int(pos[0])==0);full.append(bool(positive[winner]))
 valid=y['offset_valid'].bool() & out['reliable']
 if valid.any():
  xy=p['spatial']['candidate_xy'][valid];target=p['target_xy'];new=xy+out['offset'][valid]
  offset_groups[tuple(p['group'])].extend(((new-target).norm(dim=-1)-(xy-target).norm(dim=-1)).tolist())
 group=row['public']['network'];pair=group['pair_valid'].bool()
 counts['valid_pairs']+=int(pair.sum());counts['height_valid_pairs']+=int(((group['pair_geometry'][...,11]>0)&pair).sum())
 if len(rr) and len(rr)%100==0:pass
values=np.asarray([np.mean(v) for v in offset_groups.values()]);rng=np.random.default_rng(0)
ci=np.quantile(rng.choice(values,size=(2000,len(values)),replace=True).mean(1),[.025,.975]).tolist() if len(values) else []
r=dict(scope='Frozen verify teacher cache; diagnostic only, no parameters changed',counts=dict(counts),by_target_kind={k:dict(v) for k,v in by_kind.items()},
 common_contrast_population=dict(views=len(rr),known_candidate_mrr=float(np.mean(rr)),known_candidate_top1=float(np.mean(top)),full_pool_top1_verified_target=float(np.mean(full))),
 oracle_category_mask_not_deployable='Uses private target_kind for attribution only. No runtime GT category mask applied.',
 offset_target_group_statistics=dict(groups=len(values),mean_euclidean_error_change_m=float(values.mean()),bootstrap95ci_m=ci))
save(r,b/'ranking_diagnosis.json');save(r,Path('reports/a2_execution_20261027/RANKING_DIAGNOSIS.json'));print(json.dumps(r),flush=True)
