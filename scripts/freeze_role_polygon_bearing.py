"""Seal dev-selected heads before opening confirmation labels."""
import argparse,json,subprocess
from pathlib import Path
import torch
from graphnav.checkpointing import file_sha256,atomic_json

ARMS={'old':('old','old'),'matching_angle':('angle_only/matching','matching'),'relations_angle':('angle_only/relations','relations'),'matching_instance':('matching','matching'),'relations_instance':('relations','relations')}


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();models={};initial=[];split=json.loads((a.run/'data/split.json').read_text());expected_updates=12*((len(split['indices']['fit'])+15)//16)
 for name,(folder,arm) in ARMS.items():
  d=a.run/folder;history=json.loads((d/'history.json').read_text());last=torch.load(d/'last.pt',weights_only=True,map_location='cpu');best=torch.load(d/'best.pt',weights_only=True,map_location='cpu');config=last['config']
  assert last['epoch']==config['epochs']==12 and config['max_updates']==0
  assert last['updates']==expected_updates and len(history)==12
  assert config['features_SHA']==file_sha256(a.run/'public_inputs.pt') and config['split_SHA']==file_sha256(a.run/'data/split.json')
  selected=sorted(history,key=lambda v:(-v['dev']['macro_Top1'],v['dev']['NLL']))[0];assert best['epoch']==selected['epoch']
  assert config['confirm_labels_accessed'] is False
  if name=='old' or name.endswith('_angle'):
   assert all(v['loss_parts']['object']==0. and v['loss_parts']['coverage']==0. for v in history), 'Primary comparison used instance supervision'
  models[name]=dict(path=str(d/'best.pt'),SHA256=file_sha256(d/'best.pt'),model_arm=arm,epoch=best['epoch'],dev=selected['dev'],supervision='angle_only' if name=='old' or name.endswith('_angle') else 'angle+TRAIN_instance_ID',parameters=config['parameters'])
  if arm!='old':initial.append(config['initial_digest'])
 assert len(set(initial))==1
 public_code=['graphnav/relation_bearing_network.py','graphnav/relation_bearing_scene.py','graphnav/role_relation_language.py','scripts/train_role_polygon_bearing.py','scripts/train_role_polygon_angles_only.py','scripts/relation_bearing_strata.py','graphnav/polygon_neighborhood_bearing.py','scripts/evaluate_role_polygon_bearing.py','scripts/role_bearing_evaluation.py','scripts/role_bearing_controls.py','scripts/evaluate_role_controls.py']
 result=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),models=models,code={p:file_sha256(p) for p in public_code},primary='relations_angle versus old, query-macroTop1',angle_only_objective_SHA=file_sha256(a.run/'angle_only/OBJECTIVE.json'),parser_grouping='per-anchor trustworthy directed path, not verified semantic truth',no_RGB_no_navigation=True,confirmation_results_read=False)
 atomic_json(result,a.run/'FROZEN_BEFORE_CONFIRM.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
