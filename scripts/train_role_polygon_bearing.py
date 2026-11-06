"""TRAIN fit/dev only. Confirmation labels are not opened by this executable."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.polygon_neighborhood_bearing import NeighborhoodBearingHead
from graphnav.relation_bearing_network import RolePolygonBearing
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256


class PublicBatch:
 def __init__(self,run,device='cuda'):
  self.data=torch.load(run/'public_inputs.pt',map_location='cpu',weights_only=True);self.device=device
  self.tokens=torch.load(run/'instruction_tokens.pt',map_location='cpu',weights_only=True)['cache'];self.vec=self.data['phrase_vectors']
 def build(self,ids):
  qs=[self.data['queries'][i] for i in ids];sc=[self.data['scenes'][q['scene']] for q in qs];device=self.device
  def stack(rows,key):return torch.stack([r[key] for r in rows]).to(device)
  t=[self.tokens[q['text']] for q in qs];length=max(v[0].shape[0] for v in t);tok=torch.zeros(len(ids),length,768);mask=torch.zeros(len(ids),length,dtype=torch.bool)
  for j,(v,m) in enumerate(t):tok[j,:len(v)]=v;mask[j,:len(v)]=m
  names=self.vec[torch.stack([s['name_ids'] for s in sc])].to(device)
  inputs=dict(tokens=tok.to(device),token_mask=mask.to(device),node_names=names,node_shape=stack(sc,'shape'),node_types=stack(sc,'types'),node_valid=stack(sc,'valid'),edges=stack(sc,'edges'),anchor_indices=stack(qs,'anchor_indices'),anchor_valid=stack(qs,'anchor_valid'),entity_steps=self.vec[torch.stack([q['entity_ids'] for q in qs])].to(device),relation_steps=self.vec[torch.stack([q['relation_ids'] for q in qs])].to(device),step_valid=stack(qs,'step_valid'),angle_kernel=stack(qs,'angle_kernel'))
  old=dict(tokens=inputs['tokens'],token_mask=inputs['token_mask'],node_emb=self.vec[torch.stack([q['old_names'] for q in qs])].to(device),landmark_valid=inputs['anchor_valid'],anchor_geometry=stack(qs,'old_anchor'),neighbor_geometry=stack(qs,'old_neighbors'),neighbor_types=stack(qs,'old_types'),neighbor_valid=stack(qs,'old_neighbor_valid'))
  return inputs,old


def predict(model,batch,old,arm):
 if arm=='old':return dict(probabilities=model(**old,geometry_enabled=True).softmax(-1))
 return model(**batch,use_relations=arm=='relations')


def query_mean(value,valid):
 count=valid.sum(-1);chosen=count>0
 per=(value*valid).sum(-1)/count.clamp_min(1)
 return per[chosen].mean() if chosen.any() else value.sum()*0


def objective(out,labels,anchor_valid):
 y=torch.stack([v['bins'] for v in labels]).to(anchor_valid.device);valid=y>=0;p=out['probabilities']
 ce=-p.clamp_min(1e-9).log().gather(-1,y.clamp_min(0)[...,None]).squeeze(-1);angular=query_mean(ce,valid)
 total=angular;obj=angular*0;coverage=angular*0
 if 'object_logits' in out:
  target=torch.tensor([v['target_index'] for v in labels],device=y.device);present=target>=0
  objce=-out['object_probabilities'].clamp_min(1e-9).log().gather(-1,target.clamp_min(0)[:,None,None].expand(-1,y.shape[1],1)).squeeze(-1)
  obj=query_mean(objce,anchor_valid&present[:,None]);cov=F.binary_cross_entropy_with_logits(out['coverage_logits'],present[:,None].expand_as(y).float(),reduction='none');coverage=query_mean(cov,anchor_valid)
  total=angular+.5*obj+.1*coverage
 return total,dict(angular=float(angular.detach()),object=float(obj.detach()),coverage=float(coverage.detach()))


@torch.no_grad()
def evaluate(model,cache,ids,labels,arm,batch_size):
 model.eval();rows=[]
 for start in range(0,len(ids),batch_size):
  indices=ids[start:start+batch_size];batch,old=cache.build(indices);out=predict(model,batch,old,arm);p=out['probabilities'].float();pred=p.argmax(-1);best2=p.topk(2,-1).indices
  for k,i in enumerate(indices):
   q=cache.data['queries'][i];y=labels[i]['bins'];theta=labels[i]['theta'];valid=y>=0;error=(pred[k].cpu().float()*math.pi/4-theta);error=torch.atan2(error.sin(),error.cos()).abs()*180/math.pi
   for j in torch.nonzero(valid).flatten().tolist():
    rows.append(dict(index=i,anchor=j,true_bin=int(y[j]),pred=int(pred[k,j]),top1=int(pred[k,j]==y[j]),top2=int((best2[k,j]==y[j]).any()),error=float(error[j]),NLL=float(-p[k,j,y[j]].clamp_min(1e-9).log()),probabilities=p[k,j].cpu().tolist(),confidence=float(p[k,j].max()),entropy=float(-(p[k,j]*p[k,j].clamp_min(1e-9).log()).sum()),category=q['category'],n_refs=int(q['anchor_valid'].sum()),physical_id=labels[i]['physical_id']))
 groups={}
 for v in rows:groups.setdefault(v['index'],[]).append(v)
 summary=dict(queries=len(groups),anchor_pairs=len(rows),macro_Top1=float(np.mean([np.mean([x['top1'] for x in v]) for v in groups.values()])),Top1=float(np.mean([v['top1'] for v in rows])),Top2=float(np.mean([v['top2'] for v in rows])),MAE=float(np.mean([v['error'] for v in rows])),median=float(np.median([v['error'] for v in rows])),NLL=float(np.mean([v['NLL'] for v in rows])))
 return summary,rows


def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['old','matching','relations'],required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--max-updates',type=int,default=0);p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.backends.cuda.matmul.allow_tf32=False
 split=json.loads((a.run/'data/split.json').read_text());fit=split['indices']['fit'];dev=split['indices']['dev'];fitlabels=torch.load(a.run/'PRIVATE_fit_labels.pt',weights_only=True);devlabels=torch.load(a.run/'PRIVATE_dev_labels.pt',weights_only=True)
 # Do not load PRIVATE_confirm_labels.pt here.
 cache=PublicBatch(a.run);model=(NeighborhoodBearingHead() if a.arm=='old' else RolePolygonBearing()).cuda();initial=digest(model);optim=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
 outdir=a.run/('smoke_'+a.arm if a.max_updates else a.arm);outdir.mkdir(exist_ok=True);config=dict(arm=a.arm,epochs=a.epochs,batch=a.batch_size,lr=a.lr,seed=42,max_updates=a.max_updates,features_SHA=file_sha256(a.run/'public_inputs.pt'),split_SHA=file_sha256(a.run/'data/split.json'),model_SHA=file_sha256('graphnav/relation_bearing_network.py'),train_code_SHA=file_sha256(__file__),initial_digest=initial,parameters=sum(p.numel() for p in model.parameters()),confirm_labels_accessed=False)
 start=0;history=[];best=-1.;bestnll=1e9;updates=0;started=time.perf_counter()
 if a.resume and (outdir/'last.pt').exists():
  state=torch.load(outdir/'last.pt',weights_only=True,map_location='cpu');assert state['config']==config;model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);start=state['epoch'];history=state['history'];best=state['best'];bestnll=state['bestnll'];updates=state['updates'];torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda'])
 atomic_json(config,outdir/'config.json')
 for epoch in range(start,a.epochs):
  model.train();order=np.random.default_rng(42+epoch).permutation(fit).tolist();losses=[];part=[];tick=time.perf_counter()
  for j in range(0,len(order),a.batch_size):
   ix=order[j:j+a.batch_size];batch,old=cache.build(ix);out=predict(model,batch,old,a.arm);loss,parts=objective(out,[fitlabels[i] for i in ix],batch['anchor_valid'])
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   optim.zero_grad(set_to_none=True);loss.backward()
   if updates==0:
    grads={k:float(v.grad.abs().sum()) for k,v in model.named_parameters() if v.grad is not None};assert sum(grads.values())>0
    if a.arm=='relations':assert grads.get('relation.weight',0)>0 and grads.get('shape.weight',0)>0
    atomic_json(dict(gradient_L1=grads,parameters=config['parameters'],GT_only_in_loss=True),outdir/'gradients.json')
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optim.step();updates+=1;losses.append(float(loss.detach()));part.append(parts)
   if updates%100==0:print(json.dumps(dict(arm=a.arm,epoch=epoch+1,updates=updates,loss=float(loss),parts=parts)),flush=True)
   if a.max_updates and updates>=a.max_updates:break
  torch.cuda.synchronize();train_seconds=time.perf_counter()-tick
  summary,rows=evaluate(model,cache,dev if not a.max_updates else dev[:64],devlabels,a.arm,a.batch_size)
  entry=dict(epoch=epoch+1,loss=float(np.mean(losses)),loss_parts={k:float(np.mean([v[k] for v in part])) for k in part[0]},train_seconds=train_seconds,updates=updates,dev=summary,elapsed=time.perf_counter()-started);history.append(entry);improved=summary['macro_Top1']>best+1e-9 or (abs(summary['macro_Top1']-best)<=1e-9 and summary['NLL']<bestnll)
  if improved:best=summary['macro_Top1'];bestnll=summary['NLL']
  state=dict(model=model.state_dict(),optimizer=optim.state_dict(),epoch=epoch+1,history=history,config=config,best=best,bestnll=bestnll,updates=updates,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all())
  atomic_save(state,outdir/'last.pt')
  if improved:
   atomic_save(state,outdir/'best.pt')
   with (outdir/'best_dev_rows.jsonl').open('w') as f:
    for v in rows:f.write(json.dumps(v)+'\n')
  atomic_json(history,outdir/'history.json');print(json.dumps(entry),flush=True)
  if a.max_updates and updates>=a.max_updates:break
 print('Done',a.arm,flush=True)
if __name__=='__main__':main()
