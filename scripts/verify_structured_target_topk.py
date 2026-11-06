"""Verify real prediction files, coordinate restoration and reported Top-K metrics."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--report',required=True);a=p.parse_args();src=Path(a.run);out=Path(a.report);summary=json.load(open(src/'summary.json'));counts={}
 for arm,s in summary['metrics'].items():
  ar=np.load(src/f'{arm}_predictions.npz');l=ar['logits'];g=ar['grid'];gt=ar['gt'];assert l.shape==(7619,784);assert np.isfinite(l).all() and np.isfinite(g).all() and np.isfinite(gt).all()
  ids=np.argsort(-l,axis=-1);dist=np.linalg.norm(g-gt[:,None],axis=-1);ordered=np.take_along_axis(dist,ids,axis=-1)
  for k in [1,5,10,20,50]:assert abs(np.mean(np.min(ordered[:,:k],axis=-1)<=20)-s[f'R{k}_20m'])<1e-6
  counts[arm]=len(l)
 records=[json.loads(x) for x in open(src/'full.jsonl')];ar=np.load(src/'target_type_relation_predictions.npz');world_grid=ar['grid'];exports=[json.loads(x) for x in open(out/'target_top50_world.jsonl')];assert len(exports)==7619
 for i,r in enumerate(exports):
  expect=world_grid[i,r['grid_ids']]+np.array(records[i]['world_bounds'][:2]);assert np.max(np.abs(expect-r['world_xy']))<.001
  assert len(r['world_xy'])==50 and 'target_xy' not in r
 for x in json.load(open(out/'cases.json')):assert (out/x['figure']).is_file()
 result=dict(status='passed',rows_per_arm=counts,grid_points=784,topk_recomputed=True,world_origin_restored=True,GT_not_exported_with_candidates=True)
 (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
