"""Preregistered frozen controls after primary confirmation; no model selection."""
import argparse,json
from pathlib import Path
import torch
from scripts.train_role_polygon_bearing import PublicBatch
from scripts.role_bearing_evaluation import evaluate_cpu_metrics
from scripts.role_bearing_controls import ControlledPublicBatch,public_type_prior
from graphnav.relation_bearing_network import RolePolygonBearing
from graphnav.checkpointing import file_sha256,atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--partition',choices=['dev','confirm'],required=True);a=p.parse_args();torch.set_num_threads(4)
 if a.partition=='confirm':assert (a.run/'confirmation/summary.json').exists(),'Finish frozen primary evaluation first'
 frozen=json.load(open(a.run/'FROZEN_BEFORE_CONFIRM.json'));spec=frozen['models']['relations_angle'];assert file_sha256(spec['path'])==spec['SHA256']
 model=RolePolygonBearing().cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(spec['path'],weights_only=True,map_location='cpu')['model']);base=PublicBatch(a.run)
 ids=json.load(open(a.run/'data/split.json'))['indices'][a.partition];labels=torch.load(a.run/f'PRIVATE_{a.partition}_labels.pt',weights_only=True,map_location='cpu');out=a.run/('controls_'+a.partition);out.mkdir(exist_ok=False);result={}
 for mode in ['language_roles_shuffle','path_sentences_shuffle','target_only','shape_topology_off','angle_projection_off']:
  cache=ControlledPublicBatch(base,ids,mode);summary,rows=evaluate_cpu_metrics(model,cache,ids,labels,'relations',16);result[mode]=summary
  with (out/(mode+'_rows.jsonl')).open('w') as f:
   for q in rows:f.write(json.dumps(q)+'\n')
  print(mode,json.dumps(summary),flush=True)
 summary,rows=evaluate_cpu_metrics(None,base,ids,labels,'relations',16,predict_fn=lambda x,old,ix:public_type_prior(base,ix,prepared=(x,old)))
 result['public_type_prior']=summary
 with (out/'public_type_prior_rows.jsonl').open('w') as f:
  for q in rows:f.write(json.dumps(q)+'\n')
 print('public_type_prior',json.dumps(summary),flush=True)
 atomic_json(dict(frozen_checkpoint_SHA=spec['SHA256'],partition=a.partition,results=result,controls_not_for_selection=True),out/'summary.json')
if __name__=='__main__':main()
