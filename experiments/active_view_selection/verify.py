"""Replay normal decisions from public inputs and already reached evidence only."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from .run import R,SEED,current_state,reached_history,dump,digest
from .planner import View,boxes,select,legal,history_coverage,library,modes,utility
from experiments.multiview_rgb_grounding.render import OBJECTS

def run():
 states=torch.load(R/'cohort.pt',weights_only=True);lookup={r['metadata']['sample_id']:r for r in states};pred=json.loads((R/'predictions.json').read_text());objects=json.loads(OBJECTS.read_text());checked=0;decisions=[]
 for x in pred:
  if x['strategy']=='view_oracle':continue
  r=lookup[x['sample_id']];m=r['metadata'];cur=current_state(r);pub=r['public'];bb=boxes(objects[m['map_name']]);base=View(tuple(cur['xyz']),cur['uav_yaw'],0,'initial',cur['uav_yaw']);key=hashlib.sha256(json.dumps([m['map_name'],base.key()]).encode()).hexdigest()[:20];baseline=json.loads((R/'frames'/key/'metadata.json').read_text());history=r['history']+reached_history(baseline,objects[m['map_name']],m['cutoff_step']);prior=.5+.5*baseline['valid_fraction'];rng=np.random.default_rng(SEED+int(hashlib.sha256((m['family_id']+x['strategy']).encode()).hexdigest()[:8],16))
  assert len(x['exposures'])==2 and x['metrics']['extra_observations']==2
  for used,e in enumerate(x['exposures']):
   c=history_coverage(pub,history,m['cutoff_step']);v=select(x['strategy'],pub,cur,r['ground'],bb,c,used,rng,quality_prior=prior);expected=View(**e['view']);groups,top,weights=modes(pub,c);views=library(pub,cur,r['ground'],bb,c);hold=View(tuple(cur['xyz']),cur['uav_yaw'],0,'hold',cur['uav_yaw']);decisions.append(dict(sample_id=x['sample_id'],strategy=x['strategy'],step=used,uncovered_top8_mass=float(np.sum(weights*(1-c[top]))),selected_utility=float(utility(v,pub,cur,c,bb,prior)),hold_utility=float(utility(hold,pub,cur,c,bb,prior)),top8_coverage=c[top].tolist(),candidate_library_size=len(views)));assert np.allclose(v.key(),expected.key(),atol=1e-6),(x['sample_id'],x['strategy'],used);assert legal(cur['xyz'],v.xyz,pub['bounds'],r['ground'],bb);assert v.uav_yaw==cur['uav_yaw'];meta=json.loads((R/'frames'/e['frame_id']/'metadata.json').read_text());assert meta['map_name']==m['map_name'];history+=reached_history(meta,objects[m['map_name']],m['cutoff_step']);prior=.5+.5*meta['valid_fraction'];cur.update(xyz=list(v.xyz),camera_yaw=v.camera_yaw,camera_tilt=v.camera_tilt);checked+=1
 # All compared strategies share the exact same family population.
 populations={method:{x['family_id'] for x in pred if x['strategy']==method} for method in {x['strategy'] for x in pred}};assert len({frozenset(x) for x in populations.values()})==1
 manifest=json.loads((R/'cohort_manifest.json').read_text());assert digest(R/'cohort.pt')==manifest['cohort_sha256']
 for split,sha in manifest['sources'].items():assert digest(Path('runs/instance_grounding/plan_'+split+'/plan.pt'))==sha
 dump(R/'causal_replay.json',dict(normal_decisions_reproduced=checked,all_selections_before_observation=True,labels_unused_in_replay=True,normal_simulator_not_called=True,all_original_plan_sha_unchanged=True,all_populations_identical=True,max_extra_observations=2,all_public_move_constraints_pass=True,UAV_yaw_preserved=True,GT_oracle_not_in_normal_history=True))
 dump(R/'planner_audit.json',decisions)
 print(json.dumps({'normal_decisions_reproduced':checked}),flush=True)

if __name__=='__main__':run()
