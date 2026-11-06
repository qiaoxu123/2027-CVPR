"""Freeze all predefined arms before any formal Val evaluation; no best-arm selection."""
import argparse,json
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,file_sha256,provenance


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original-v2',type=Path,required=True);a=p.parse_args();r=a.run
 original=torch.load(a.original_v2,map_location='cpu',weights_only=True);arms={};init=[]
 for name,alpha in [('predicate',0.),('angle01',.1),('angle03',.3)]:
  folder=r/name;s=torch.load(folder/'last.pt',map_location='cpu',weights_only=True);m=json.loads((folder/'manifest.json').read_text())
  assert s['completed_epochs']==s['planned_epochs']==5 and not s['smoke'] and s['updates']==13675
  assert all(h['observed']==21878 for h in s['history']) and m['angle_supervision_weight']==alpha and m['seed']==42
  assert m['training_code_sha256']==file_sha256('scripts/train_unified_full_angular.py') and m['angular_module_sha256']==file_sha256('graphnav/unified_direct_bearing_loss.py')
  assert all(torch.equal(v,original['model'][k]) for k,v in s['model'].items() if not k.startswith(('geometry.','full_relation.')))
  assert json.loads((r/f'runtime_smoke_{name}/smoke.json').read_text())['interface_passed']
  initializer=json.loads((folder/'initialization.json').read_text())['model_state_SHA256'];init.append(initializer)
  arms[name]=dict(checkpoint=str(folder/'last.pt'),SHA256=file_sha256(folder/'last.pt'),alpha=alpha,source=m['source_sha256'],initialization_SHA256=initializer,frozen_other_modules=True)
 assert len(set(init))==1
 files=['graphnav/unified_pgvb.py','graphnav/full_experiment.py','graphnav/unified_direct_bearing_loss.py','graphnav/unified_direct_bearing_runtime.py','graphnav/unified_full_relation_runtime.py','graphnav/unified_text_ablation.py','graphnav/unified_text_ablation_runtime.py','graphnav/a3_runtime.py','graphnav/stable_navigation_goal.py','graphnav/trajectory_policy.py','scripts/evaluate_closedloop.py']
 f=dict(arms=arms,code={x:file_sha256(x) for x in files},selection='all fixed epoch5 arms, no Val checkpoint selection',scope='3xall2470 Val-Seen; no Test; no thresholds changed',inference_GT_angle=False,concurrency_training=3,concurrency_evaluation=1,provenance=provenance())
 dest=r/'FROZEN_BEFORE_VAL.json'
 if dest.exists():assert json.loads(dest.read_text())==f
 else:atomic_json(f,dest)
 print(json.dumps(f,indent=2))
if __name__=='__main__':main()
