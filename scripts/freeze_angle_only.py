"""Seal predicted-only final epoch before any Val rollout; Oracle remains TRAIN-only."""
import argparse,json
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,file_sha256,provenance


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run
 s=torch.load(r/'full/last.pt',weights_only=True,map_location='cpu');report=json.loads((r/'full/report.json').read_text());smoke=json.loads((r/'runtime_smoke/smoke.json').read_text())
 assert s['epoch']==s['manifest']['epochs']==3 and s['manifest']['max_batches']==0
 assert all(x['fit_seen']==21878 for x in s['history']) and report['frozen_core_bitwise_unchanged'] and smoke['interface_passed']
 for key,path in [('model_code_sha256','graphnav/unified_angle_only_belief.py'),('train_code_sha256','scripts/train_unified_angle_only.py')]:assert s['manifest'][key]==file_sha256(path)
 files=['graphnav/unified_angle_only_belief.py','graphnav/unified_angle_only_runtime.py','graphnav/unified_pgvb.py','graphnav/unified_full_relation_runtime.py','graphnav/unified_text_ablation.py','graphnav/unified_text_ablation_runtime.py','scripts/evaluate_closedloop.py']
 result=dict(angle_head_sha256=file_sha256(r/'full/last.pt'),baseline_sha256=s['manifest']['baseline_sha256'],public_catalog_sha256=s['manifest']['public_catalog_sha256'],code={x:file_sha256(x) for x in files},selection='fixed epoch3, no Val checkpoint selection or tuning',scope='all2470 Val-Seen, predicted bearings only; no Test',GT_oracle_at_runtime=False,provenance=provenance())
 dest=r/'FROZEN_BEFORE_VAL.json'
 if dest.exists():assert json.loads(dest.read_text())==result
 else:atomic_json(result,dest)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
