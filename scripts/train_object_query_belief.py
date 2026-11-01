"""Train actual object-query ranking/null evidence and supported goal fusion."""
import argparse,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from graphnav.object_query_belief import ObjectQueryRanker,object_evidence_loss
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256


def load(directory):
 directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());assert m['complete'];rows=[]
 for chunk in m['chunks']:
  assert file_sha256(directory/chunk['file'])==chunk['sha256'],'Corrupt native object training chunk'
  rows.extend(torch.load(directory/chunk['file'],map_location='cpu',weights_only=True))
 return m,rows


def collate(rows):
 keys=['features','geometry','valid','query_world','base_logits','grid_world','kernel','observed']
 batch={k:torch.stack([r[k] for r in rows]) for k in keys}
 batch['text']=pad_sequence([r['text'] for r in rows],batch_first=True)
 batch['text_mask']=pad_sequence([r['text_mask'] for r in rows],batch_first=True,padding_value=False)
 batch['target_world']=torch.stack([r['labels']['target_world'] for r in rows])
 batch['group']=[r['group'] for r in rows];batch['episode']=[r['episode'] for r in rows];batch['step']=[r['step'] for r in rows]
 return batch


def main():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--dev',required=True);p.add_argument('--output',required=True);p.add_argument('--relation',action='store_true');p.add_argument('--train-relation',action='store_true');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=0);p.add_argument('--bound',type=float,default=2.);p.add_argument('--kernel-sigma',type=float,default=20.);p.add_argument('--gain-lr',type=float,default=2e-4);a=p.parse_args()
 out=Path(a.output)
 if out.exists():raise ValueError('Refusing overwrite')
 if a.train_relation and not a.relation:raise ValueError('Fine-tuning requires --relation')
 train_meta,train=load(a.train);dev_meta,dev=load(a.dev)
 assert train_meta['partition']=='train' and dev_meta['partition']=='dev' and train_meta['base_sha']==dev_meta['base_sha']
 assert not ({r['group'] for r in train}&{r['group'] for r in dev}),'map+instruction leakage'
 out.mkdir(parents=True);torch.manual_seed(a.seed);device='cuda';model=ObjectQueryRanker(relation=a.relation,train_relation=a.train_relation).to(device)
 head=[v for k,v in model.named_parameters() if not k.startswith('relation.') and k!='fusion_gain'];groups=[dict(params=head,lr=2e-4),dict(params=[model.fusion_gain],lr=a.gain_lr)]
 if a.train_relation:groups.append(dict(params=model.relation.parameters(),lr=1e-5))
 optimizer=torch.optim.AdamW(groups,weight_decay=.01)
 generator=torch.Generator().manual_seed(a.seed)
 train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,collate_fn=collate,generator=generator)
 dev_loader=DataLoader(dev,batch_size=a.batch_size,shuffle=False,collate_fn=collate)
 config=dict(vars(a),base_sha=train_meta['base_sha'],train_manifest_sha=file_sha256(Path(a.train)/'manifest.json'),dev_manifest_sha=file_sha256(Path(a.dev)/'manifest.json'),train_states=len(train),dev_states=len(dev),train_groups=len({r['group'] for r in train}),dev_groups=len({r['group'] for r in dev}),selection='All prepared grouped Dev128 states R@1 then NE; this is a subset of original Train Dev; no Test use',forward='Actual frozen official image/text features; public geometry only; no GT in selection/forward')
 atomic_json(config,out/'config.json');history=[];best=(-1.,-float('inf'))
 def run(loader,training=False,control='real'):
  model.train(training);total={};count=0;predictions=[];start=time.perf_counter()
  for b in loader:
   x={k:(v.to(device).float() if v.is_floating_point() else v.to(device)) for k,v in b.items() if torch.is_tensor(v)}
   if a.kernel_sigma!=train_meta['kernel_sigma_m']:
    x['kernel']=torch.exp(-torch.cdist(x['grid_world'],x['query_world']).square()/(2*a.kernel_sigma**2))*x['observed'][...,None]*x['valid'][:,None]
   if control=='zero':x['valid']=torch.zeros_like(x['valid'])
   elif control=='shuffle':
    for j in range(len(x['features'])):
     ids=x['valid'][j].nonzero().flatten();order=ids.flip(0);x['features'][j,ids]=x['features'][j,order].clone()
     native=(x['features'][j]@x['text'][j].T).masked_fill(~x['text_mask'][j][None],-1e4).amax(-1)
     x['geometry'][j,:,1]=native/5
   with torch.set_grad_enabled(training),torch.autocast('cuda',dtype=torch.bfloat16):
    q,null=model(x['features'],x['text'],x['text_mask'],x['geometry'],x['valid'])
   updated,delta=model.update_belief(x['base_logits'],q,null,x['kernel'],x['observed'],x['valid'],bound=a.bound)
   losses=object_evidence_loss(q,null,x['query_world'],x['target_world'],x['valid'],updated,x['grid_world'],x['base_logits'])
   if not torch.isfinite(losses['total']):raise ValueError('Nonfinite native object loss')
   if training:
    optimizer.zero_grad(set_to_none=True);losses['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    with torch.no_grad():model.fusion_gain.clamp_(0,3)
   with torch.no_grad():
    n=len(updated);count+=n;distance=(x['grid_world']-x['target_world'][:,None,:2]).norm(dim=-1)
    errors=distance.gather(1,updated.argmax(-1)[:,None])[:,0];old=distance.gather(1,x['base_logits'].argmax(-1)[:,None])[:,0]
    for name,value in dict(R1=(errors<=20).float().mean(),NE=errors.mean(),static_R1=(old<=20).float().mean(),static_NE=old.mean(),mean_abs_delta=delta.abs().mean(),**losses).items():total[name]=total.get(name,0.)+float(value)*n
    query_good=((x['query_world']-x['target_world'][:,None,:2]).norm(dim=-1)<=20)&x['valid']
    total['proposal_coverage']=total.get('proposal_coverage',0.)+float(query_good.any(-1).sum())
    for i,error in enumerate(errors.cpu().tolist()):predictions.append(dict(group=b['group'][i],episode=b['episode'][i],step=b['step'][i],error_m=error,static_error_m=float(old[i])))
  if not count:raise ValueError('Empty loader')
  metrics={k:v/count for k,v in total.items()};metrics.update(states=count,seconds=time.perf_counter()-start)
  return metrics,predictions
 for epoch in range(1,a.epochs+1):
  training,_=run(train_loader,True);validation,_=run(dev_loader);record=dict(epoch=epoch,train=training,dev=validation,gain=float(model.fusion_gain.detach()));history.append(record)
  key=(validation['R1'],-validation['NE']);payload=dict(kind='native-aerialvg-object-query-belief-v1',model=model.state_dict(),config=config,epoch=epoch,history=history)
  atomic_save(payload,out/'last.pt')
  if key>best:best=key;atomic_save(payload,out/'best.pt')
  atomic_json(history,out/'history.json');print(json.dumps(record),flush=True)
 selected=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(selected['model'])
 controls={}
 for control in ['real','zero','shuffle']:
  metrics,predictions=run(dev_loader,control=control);controls[control]=metrics;atomic_json(predictions,out/('predictions_'+control+'.json'))
 if model.relation is not None:
  relation=model.relation;model.relation=None;metrics,predictions=run(dev_loader);model.relation=relation
  controls['no_relation']=metrics;atomic_json(predictions,out/'predictions_no_relation.json')
 atomic_json(controls,out/'controls.json');atomic_json(dict(complete=True,selected_epoch=selected['epoch'],best=best),out/'complete.json')


if __name__=='__main__':main()
