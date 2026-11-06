"""Frozen branch attribution after predictions were sealed; GT only scores outputs."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from graphnav.relation_bearing_network import RolePolygonBearing
from graphnav.checkpointing import file_sha256
from scripts.train_role_polygon_bearing import PublicBatch,predict

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;o=Path('reports/relation_bearing/large_errors');rows=[json.loads(s) for s in (o/'all_large_errors.jsonl').read_text().splitlines()];groups={}
 for x in rows:groups.setdefault(x['index'],[]).append(x)
 spec=json.load(open(a.run/'FROZEN_BEFORE_CONFIRM.json'))['models']['relations_angle'];assert file_sha256(spec['path'])==spec['SHA256'];model=RolePolygonBearing().cuda().eval();model.load_state_dict(torch.load(spec['path'],map_location='cpu',weights_only=True)['model']);cache=PublicBatch(a.run);ids=sorted(groups);result=[]
 with torch.no_grad():
  for start in range(0,len(ids),16):
   batchids=ids[start:start+16];x,old=cache.build(batchids);out=predict(model,x,old,'relations');out={k:v.detach().cpu() for k,v in out.items()};kernels=x['angle_kernel'].float().cpu()
   for k,i in enumerate(batchids):
    for row in groups[i]:
     j=row['anchor'];p8=out['probabilities'][k,j].numpy();assert np.max(abs(p8-np.array(row['probabilities'])))<1e-5,'Replay differs from frozen results';alpha=float(out['coverage_logits'][k,j].sigmoid());truth=row['GT_bearing_deg'];d={key:int(out[key][k,j].argmax()) for key in ['geometric_probabilities','free_probabilities']}
     d={key:dict(pred_bin=v,error_deg=float(abs((v*45-truth+180)%360-180))) for key,v in d.items()};ti=row['target_index_offline'];op=out['object_probabilities'][k,j];result.append(dict(index=i,anchor=j,alpha_geometric=alpha,target_present=ti>=0,top_object_matches_registered_target=ti>=0 and int(op.argmax())==ti,registered_target_object_mass=float(op[ti]) if ti>=0 else None,**d))
 stats=dict(N=len(result),mean_geometric_mixing_weight=float(np.mean([x['alpha_geometric'] for x in result])),geometric_branch_alone_over90=sum(x['geometric_probabilities']['error_deg']>90 for x in result),free_branch_alone_over90=sum(x['free_probabilities']['error_deg']>90 for x in result),both_branches_over90=sum(x['geometric_probabilities']['error_deg']>90 and x['free_probabilities']['error_deg']>90 for x in result),top_object_matches_registered_target=sum(x['top_object_matches_registered_target'] for x in result),interpretation='Branch scores only; this is not a new prediction method or retuned confirmation.')
 (o/'branch_replay.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in result));(o/'branch_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
