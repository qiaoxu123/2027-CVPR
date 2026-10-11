import json,numpy as np,torch
from pathlib import Path
from experiments.multiview_rgb_grounding.geometry import world_grid
from experiments.multiview_rgb_grounding.render import OBJECTS
from experiments.multiview_rgb_grounding.match_pilot import identity
r=Path('runs/multiview_rgb_grounding');a=Path('docs/assets/multiview_rgb_grounding');objs=json.loads(OBJECTS.read_text());out=[]
for split,folder,pilot in [('train','matching_train_pilot_world','render_pilot_train_scatter'),('dev','matching_dev_error_probe','render_dev_error_probe')]:
 man=json.loads((r/pilot/'manifest.json').read_text());plan=torch.load(Path(man['source_plan'])/'plan.pt',weights_only=True);state={s['metadata']['sample_id']:s for s in plan['states']}
 rows=json.loads((r/folder/'predictions.json').read_text())
 for x in rows:
  if (x['model'],x['language'],x['control'],x['view'],x['modality'])!=('SigLIP2-Large384','full','real','nadir','rgb_only'):continue
  p=state[x['sample_id']]['public'];grid=world_grid(p);centers=np.array([np.mean(objs[x['map_name']][o]['contour'],axis=0) for o in x['candidate_ids']]);scores=np.array(p['static_logits'])[np.linalg.norm(centers[:,None]-grid[None],axis=-1).argmin(1)];rr,rank=identity(scores.tolist(),x['candidate_ids'],x['target_object_id']);out.append(dict(split=split,sample_id=x['sample_id'],scope=x['scope'],candidate_ids=x['candidate_ids'],target_object_id=x['target_object_id'],r1=rr,rank=rank))
(a/'geometry_instance_baseline.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
