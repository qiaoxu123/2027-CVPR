"""Frozen gallery physical-view audit; no retrieval labels or predictions changed."""
import json,argparse
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-root',default='runs/visual_grounding_benchmark/data_audit');a=p.parse_args();root=Path(a.audit_root);result={}
 for split in ['train','seen','unseen']:
  frozen=root/f'frozen_{split}'
  if not (frozen/'frozen.json').exists():continue
  idx={r['sample_id']:r for r in map(json.loads,(frozen/'index.jsonl').open())};flag=[];total=0
  for pair in map(json.loads,(frozen/'retrieval.jsonl').open()):
   total+=1;a=np.asarray(idx[pair['query_sample_id']]['footprint_world']);b=np.asarray(idx[pair['gallery_sample_id']]['footprint_world']);center=float(np.linalg.norm(a.mean(0)-b.mean(0)));sidea=np.linalg.norm(a[1]-a[0]);sideb=np.linalg.norm(b[1]-b[0]);extent=abs(float(sidea-sideb))
   if center<.01 and extent<.01:flag.append(dict(pair,center_displacement_m=center,footprint_side_change_m=extent))
  result[split]={'retrieval_pairs':total,'same_physical_center_extent_under1cm':len(flag),'mask':flag}
 payload={'status':'post-freeze geometry audit only; no metric filtering or rerun','criterion':'XY camera center and FOV side differ by <1 cm; rotated footprints alone do not establish independent physical viewpoints','splits':result};(root/'view_independence.json').write_text(json.dumps(payload,indent=2));print(json.dumps({s:{k:v for k,v in r.items() if k!='mask'} for s,r in result.items()}))
if __name__=='__main__':main()
