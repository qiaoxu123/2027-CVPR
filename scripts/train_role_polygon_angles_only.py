"""Matched supervision ablation: only8-bin target-angle labels, no instance IDs.

The exact same public features, shape/matching/role architecture and optimizer
are used. This wrapper changes only the TRAIN objective and output directory.
"""
import argparse,json,sys
from pathlib import Path
import torch
import scripts.train_role_polygon_bearing as base
from graphnav.checkpointing import file_sha256


def angle_only(out,labels,anchor_valid):
 y=torch.stack([v['bins'] for v in labels]).to(anchor_valid.device);valid=y>=0;p=out['probabilities']
 loss=base.query_mean(-p.clamp_min(1e-9).log().gather(-1,y.clamp_min(0)[...,None]).squeeze(-1),valid)
 return loss,dict(angular=float(loss.detach()),object=0.,coverage=0.)


def main():
 p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--arm',choices=['matching','relations'],required=True);a,rest=p.parse_known_args();root=a.source_run/'angle_only';root.mkdir(exist_ok=True)
 for file in ['public_inputs.pt','instruction_tokens.pt','PRIVATE_fit_labels.pt','PRIVATE_dev_labels.pt','features.json']:
  path=root/file
  if not path.exists():path.symlink_to(a.source_run/file)
 if not (root/'data').exists():(root/'data').symlink_to(a.source_run/'data',target_is_directory=True)
 manifest=dict(supervision='8-bin angle labels ONLY; no target-ID loss or coverage labels',wrapper_SHA=file_sha256(__file__),source_run=str(a.source_run),GT_instance_ID_accessed_by_objective=False)
 lock=root/'OBJECTIVE.json'
 if lock.exists():assert json.loads(lock.read_text())==manifest
 else:lock.write_text(json.dumps(manifest,indent=2))
 base.objective=angle_only;sys.argv=['angles_only','--run',str(root),'--arm',a.arm,*rest];base.main()
if __name__=='__main__':main()
