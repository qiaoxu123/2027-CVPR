"""Posthoc target conflicts under identical public inputs; never used for prediction."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.previous);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);result={};group_reports={}
 ar=np.load(src/'full_predictions.npz');grid=ar['grid'];gt=ar['gt']
 for arm in ['full','target_type_relation']:
  records=[json.loads(x) for x in open(src/f'{arm}.jsonl')];groups=defaultdict(list)
  for i,r in enumerate(records):
   signature=json.dumps({k:r.get(k) for k in ['map_name','world_bounds','landmarks','instruction']},sort_keys=True);groups[hashlib.sha256(signature.encode()).hexdigest()].append(i)
  conflicts=[];upper=0;multi=0
  for key,ids in groups.items():
   if len(ids)>1:multi+=1
   pos=gt[ids];diam=float(np.linalg.norm(pos[:,None]-pos[None,:],axis=-1).max());best=int((np.linalg.norm(grid[ids[0]][:,None]-pos[None],axis=-1)<=20).sum(-1).max());upper+=best
   if diam>40:conflicts.append(dict(signature=key,indices=ids,diameter_m=diam,max_same_goal_hits=best,text=records[ids[0]]['instruction']))
  result[arm]=dict(N=len(records),unique_input_signatures=len(groups),repeated_groups=multi,conflict_groups_over40m=len(conflicts),queries_in_conflict_groups=sum(len(r['indices']) for r in conflicts),posthoc_identical_input_top1_grid_upperbound=upper/len(records),minimum_errors_from_identical_input_conflicts=len(records)-upper)
  group_reports[arm]=conflicts
 (out/'input_collision_summary.json').write_text(json.dumps(result,indent=2));(out/'input_collision_groups.json').write_text(json.dumps(group_reports,ensure_ascii=False,separators=(',',':')));print(result)
if __name__=='__main__':main()
