"""Fixed offline error-cohort probe; GT selects evaluation errors, never camera."""
import hashlib,json,re,time
from pathlib import Path
import numpy as np,torch
from .geometry import world_grid

def run():
 root=Path('runs/multiview_rgb_grounding/plan_dev_error_probe');assert not root.exists();root.mkdir();source=Path('runs/instance_grounding/plan_dev');plan=torch.load(source/'plan.pt',weights_only=True);families={}
 for s in plan['states']:families.setdefault(s['metadata']['family_id'],[]).append(s)
 selected=[]
 for family,states in families.items():
  states.sort(key=lambda s:s['metadata']['cutoff_step']);s=states[len(states)//2];m=s['metadata'];pub=s['public'];static=np.asarray(pub['static_logits']);wrong=np.linalg.norm(world_grid(pub)[np.argsort(static,kind='stable')[-1]]-plan['labels'][m['sample_id']]['target_world'])>20
  # Type preference comes only from the supplied query's initial clause.
  clause=re.split(r'\b(?:beside|behind|next to|with|near)\b',m['instruction'],flags=re.I)[0].lower();noun=re.search(r'\b(building|house|road|parking|car|vehicle|truck|van)\b',clause)
  if wrong and noun and noun[0] in ['building','house','road','parking']:selected.append(s)
 selected.sort(key=lambda s:hashlib.sha256(('P3-error-probe:'+s['metadata']['family_id']).encode()).hexdigest());selected=selected[:8]
 labels={s['metadata']['sample_id']:plan['labels'][s['metadata']['sample_id']] for s in selected};torch.save({'states':selected,'views':{},'labels':labels},root/'plan.pt');digest=hashlib.sha256((root/'plan.pt').read_bytes()).hexdigest()
 manifest={'complete':True,'split':'internal_dev','states':len(selected),'families':len(selected),'files':{'plan.pt':digest},'source_plan_sha256':hashlib.sha256((source/'plan.pt').read_bytes()).hexdigest(),'GT_usage':'offline static-error evaluation cohort only','camera_selection':'fixed middle cached prefix and recorded yaw; no target-conditioned angle or position','source_sample_ids':[s['metadata']['sample_id'] for s in selected]};(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':run()
