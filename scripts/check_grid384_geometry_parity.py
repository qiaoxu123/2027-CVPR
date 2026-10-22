"""Check padded-node omission preserves dense geometry outputs and gradients."""
import copy,json,subprocess,types,torch
from pathlib import Path
from graphnav.full_experiment import new_model
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.workspace_paths import PROJECT
from types import SimpleNamespace
import graphnav.full_localization as module

def main():
 torch.set_num_threads(1);torch.manual_seed(17)
 ck=torch.load(PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt',weights_only=True,map_location='cpu');args=SimpleNamespace(**ck['a3_args'])
 model=new_model(args,'cuda').float();old=copy.deepcopy(model)
 scope=dict(module.__dict__);exec(compile(subprocess.check_output(['git','show','21c42f4c721104454fd03340ef180b2ffccd4058:graphnav/full_localization.py'],text=True),'<original>','exec'),scope)
 old.forward=types.MethodType(scope[type(model).__name__].forward,old)
 ds=FullCandidateDataset(PROJECT/'GraphNav/data/graphnav_stage1_frames/queries.jsonl',grid_size=96,max_landmarks=args.max_landmarks,max_roads=args.max_roads,max_relations=args.max_relations,prior_sigma_m=args.prior_sigma_m,rule_sigma_m=args.rule_sigma_m)
 batch=collate_full([ds[0]]);x={k:v.cuda() for k,v in batch.items() if torch.is_tensor(v)};tokens=torch.randn(1,8,768,device='cuda');mask=torch.ones(1,8,dtype=torch.bool,device='cuda');names=torch.randn(1,x['nodes'].shape[1],768,device='cuda')
 outs=[];grads=[]
 for m in [old,model]:
  torch.manual_seed(123)
  o=m(x,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names);outs.append(o.belief_logits.detach());(o.belief_logits.float().mean()+o.dense_context.float().square().mean()).backward();grads.append({n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None})
 err=float((outs[0]-outs[1]).abs().max());torch.testing.assert_close(outs[0],outs[1],rtol=1e-5,atol=1e-5)
 maxgrad=0.
 for n,g in grads[0].items():torch.testing.assert_close(g,grads[1][n],rtol=1e-4,atol=1e-6);maxgrad=max(maxgrad,float((g-grads[1][n]).abs().max()))
 r=dict(grid=96,candidates=9216,valid_entities=int(x['entity_valid'].sum()),capacity=x['entity_valid'].shape[1],logit_max_error=err,gradient_max_error=maxgrad,status='passed',note='Optimization triggers above8192; parameter architecture unchanged; tests FP32 outputs and backprop')
 Path('runs/grid384_20261022/geometry_parity.json').write_text(json.dumps(r,indent=2));print(r)
if __name__=='__main__':main()
