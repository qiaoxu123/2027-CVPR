"""Frozen-image, frozen-base language interventions; not rewritten-goal accuracy."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader,Subset
from graphnav.workspace_paths import PROJECT
from graphnav.stage1 import GraphNavStage1
from graphnav.stage1_data import Stage1Dataset,collate_stage1
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.calibrated_visual_residual import CalibratedVisualResidual
from graphnav.token_language import FrozenBERTTokens
from graphnav.relation_roles import parse_roles,role_batch
from graphnav.checkpointing import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--max-batches',type=int,default=32);p.add_argument('--float32',action='store_true');p.add_argument('--native-text',action='store_true');a=p.parse_args()
 out=Path(a.output)
 if out.exists():raise ValueError('Refusing overwrite')
 out.mkdir(parents=True);device='cuda'
 state=torch.load(PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt',map_location='cpu',weights_only=True)
 base_model=GraphNavStage1(new_model(SimpleNamespace(**state['a3_args']),device),residual_bound=.5).to(device)
 base_model.load_state_dict(state['model']);base_model.eval().requires_grad_(False)
 s=torch.load(a.checkpoint,map_location='cpu',weights_only=True);head=CalibratedVisualResidual(**s['config']['model_kwargs']).to(device);head.load_state_dict(s['model']);head.eval().requires_grad_(False)
 cache=PROJECT/'GraphNav/data/rebuilt_stage1_20261025';data=Stage1Dataset(cache/'frames',verify=False)
 dev=set(state['config']['dev_episodes']);episodes=data.manifest['protocol']['episode_indices'];indices=[i for i in range(len(data)) if episodes[i//4] in dev]
 bert=FrozenBERTTokens(state['a3_args']['text_backbone']);bert.load_cache(cache/'bert_tokens.pt')
 native=None
 if a.native_text:
  from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
  native=ObservedSigLIPMatchingEncoder()
 rows=[]
 for iteration,(static,public,labels) in enumerate(DataLoader(Subset(data,indices),batch_size=16,num_workers=2,collate_fn=collate_stage1)):
  if iteration>=a.max_batches:break
  parsed=[parse_roles(t,n) for t,n in zip(static['text'],static['node_names'])]
  eligible=[r['valid'] and r['opposite_text']!=t for r,t in zip(parsed,static['text'])]
  if not any(eligible):continue
  inputs,tokens,mask,names=_batch_inputs(static,bert,device,'geometry');public={k:v.to(device) for k,v in public.items()}
  cf_text=[r['opposite_text'] for r in parsed];cf_tokens,cf_mask=bert.encode_tokens(cf_text);cf_tokens=cf_tokens.to(device);cf_mask=cf_mask.to(device)
  if getattr(head,'structured_relation',False):
   roles,_=role_batch(static['text'],static['node_names'],bert,device);public.update(roles)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=not a.float32):
   base=base_model(static_inputs=inputs,tokens=tokens,token_mask=mask,node_emb=names,**{k:v for k,v in public.items() if not k.startswith('role_')},use_history=True,visual_control='no-belief')
   if native is not None:public['native_text']=native.texts(static['text']).detach().clone()
   real=head(base,public,inputs,tokens,mask,names)[0]
   cf_public=dict(public)
   if native is not None:cf_public['native_text']=native.texts(cf_text).detach().clone()
   if getattr(head,'structured_relation',False):cf_public.update(role_batch(cf_text,static['node_names'],bert,device)[0])
   changed=head(base,cf_public,inputs,cf_tokens,cf_mask,names)[0]
   relation=head.relation;head.relation=None
   without=head(base,public,inputs,tokens,mask,names)[0];cf_without=head(base,cf_public,inputs,cf_tokens,cf_mask,names)[0]
   head.relation=relation
  effect=(changed-real).float();isolated=effect-(cf_without-without).float()
  before=(base.static.belief_logits.float()+real).softmax(-1);after=(base.static.belief_logits.float()+changed).softmax(-1)
  xy=inputs['grid_xy'].float();movement=((after-before)[...,None]*xy).sum(1)
  for b,ok in enumerate(eligible):
   if not ok:continue
   direction=torch.tensor(parsed[b]['direction'],device=device)
   rows.append(dict(index=indices[iteration*16+b],relation=parsed[b]['relation'],text=static['text'][b],counterfactual=cf_text[b],max_abs_delta_change=float(effect[b].abs().max()),max_abs_relation_change=float(isolated[b].abs().max()),argmax_changed=bool((base.static.belief_logits[b]+real[b]).argmax()!=(base.static.belief_logits[b]+changed[b]).argmax()),opposite_compass_projection=float(-(movement[b]*direction).sum()) if direction.abs().any() else None))
 summary=dict(states=len(rows),mean_max_abs_delta_change=sum(r['max_abs_delta_change'] for r in rows)/max(1,len(rows)),mean_max_abs_relation_change=sum(r['max_abs_relation_change'] for r in rows)/max(1,len(rows)),argmax_changed=sum(r['argmax_changed'] for r in rows),native_text_reencoded=a.native_text,interpretation='Visual-head language sensitivity with images and frozen base held fixed; native text reencoded only when explicitly enabled. Not accuracy against a rewritten goal.')
 atomic_json(dict(summary=summary,rows=rows),out/'results.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
