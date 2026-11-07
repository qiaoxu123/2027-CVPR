"""Small TRAIN-only local Goal probability refiner over frozen public Geometry."""
import json,argparse
from pathlib import Path
import numpy as np,torch
from torch import nn

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--objective',choices=['ce','coordinate'],default='ce');a=p.parse_args();torch.manual_seed(42);torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
 labels=json.loads(a.labels.read_text());data={}
 for split in ['fit','development']:
  values=[]
  for l in labels:
   if l['split']!=split:continue
   z=np.load(a.cache/'public_features'/f"{l['index']:05d}.npz");logits=z['base_logits_full'];world=z['base_world_grid'];top=int(logits.argmax());r,c=divmod(top,28);ids=np.array([(r+dr)*28+c+dc if 0<=r+dr<28 and 0<=c+dc<28 else top for dr in range(-2,3) for dc in range(-2,3)]);mask=np.array([0<=r+dr<28 and 0<=c+dc<28 for dr in range(-2,3) for dc in range(-2,3)]);scores=(logits[ids]-logits[top])/.5;scores[~mask]=-1e4;xy=world[ids];offset=(xy-world[top])/100
   public=np.concatenate([z['candidate_context'][0].astype('float32'),np.clip(scores,-20,0),mask.astype('float32'),offset.flatten()]);target=np.array(l['target_world']);dist=np.linalg.norm(xy-target,axis=1);values.append((public,scores,mask,xy,target,float(dist[mask].min()<=20)))
  data[split]=[torch.tensor(np.stack([v[j] for v in values]),dtype=torch.float32,device='cuda') for j in range(6)]
 model=nn.Sequential(nn.Linear(228,64),nn.GELU(),nn.Linear(64,25)).cuda();nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);history=[]
 def forward(d,idx=None):
  f,base,mask,xy,target,support=[v if idx is None else v[idx] for v in d];delta=4*model(f).tanh();logits=(base+delta).masked_fill(~mask.bool(),-1e4);prob=logits.softmax(-1);pred=(prob[:,:,None]*xy).sum(1);return logits,prob,pred
 def evaluate():
  with torch.no_grad():
   d=data['development'];_,_,pred=forward(d);base=(d[1].softmax(-1)[:,:,None]*d[3]).sum(1);err=(pred-d[4]).norm(dim=-1);old=(base-d[4]).norm(dim=-1);return dict(n=len(err),R20=float((err<=20).float().mean()),NE=float(err.mean()),corrected=int(((err<=20)&(old>20)).sum()),damaged=int(((err>20)&(old<=20)).sum()))
 baseline=evaluate();best=baseline['R20'];torch.save(dict(model=model.state_dict(),epoch=0,metrics=baseline),a.output/'best.pt')
 for epoch in range(5):
  losses=[];d=data['fit']
  for ids in torch.randperm(len(d[0]),device='cuda').split(128):
   opt.zero_grad();logits,prob,pred=forward(d,ids);f,base,mask,xy,target,support=[v[ids] for v in d];distance=(xy-target[:,None]).square().sum(-1);teacher=(-distance/200).masked_fill(~mask.bool(),-1e4).softmax(-1);original=base.softmax(-1);teacher=torch.where(support[:,None].bool(),teacher,original);ce=-(teacher*logits.log_softmax(-1)).sum(-1).mean();fidelity=(original*(original.clamp_min(1e-8).log()-logits.log_softmax(-1))).sum(-1).mean();loss=ce+.2*fidelity
   if a.objective=='coordinate':
    baseline_xy=(original[:,:,None]*xy).sum(1);preserve=((baseline_xy-target).norm(dim=-1)<=20)|(~support.bool());coordinate_target=torch.where(preserve[:,None],baseline_xy,target);loss=torch.nn.functional.smooth_l1_loss(pred/20,coordinate_target/20)+.2*fidelity
   if not torch.isfinite(loss):raise FloatingPointError('Loss')
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
  metrics=evaluate();history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),metrics=metrics));torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,metrics=metrics),a.output/'last.pt')
  if metrics['R20']>best:best=metrics['R20'];torch.save(dict(model=model.state_dict(),epoch=epoch+1,metrics=metrics),a.output/'best.pt')
  print(json.dumps(history[-1]),flush=True)
 (a.output/'summary.json').write_text(json.dumps(dict(seed=42,objective=a.objective,baseline=baseline,history=history,fit_n=len(data['fit'][0]),scope='Frozen Geometry public local features; GT loss/scoring only; confirmation untouched'),indent=2)+'\n')
if __name__=='__main__':main()
