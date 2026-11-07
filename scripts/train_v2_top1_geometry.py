"""Three matched TRAIN-only adaptations; select on development, not Val/Test.
Public binding aligned to deployment; frozen BERT cache reused. No new inference heads.
"""
import argparse,json,time,copy
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256,rng_state,restore_rng
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.goal_hit_training import goal_hit_loss
from scripts.unified_pgvb_v3_pilot import _load_base

class Indexed(FullCandidateDataset):
 def __getitem__(self,i):
  r=super().__getitem__(i);r['row_index']=i;return r


def main():
 p=argparse.ArgumentParser()
 for k in ['data','source-v2','trained-v2','bert-cache','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--modes',nargs='+',choices=['legacy','ce8','hit20'],default=['legacy','ce8','hit20']);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--smoke-updates',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--max-hours',type=float,default=4.)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(42);np.random.seed(42)
 manifest=json.load(open(a.data/'manifest.json'));labels=json.load(open(a.data/'labels_private.json'));base,cfg=_load_base(a.source_v2,'cuda');payload=torch.load(a.trained_v2,map_location='cpu',weights_only=True);state={k[len('geometry.'):]:v for k,v in payload['model'].items() if k.startswith('geometry.')};base.load_state_dict(state,strict=True);del base
 ds=Indexed(a.data/'public_train.jsonl',grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False)
 assert len(ds)==21878 and all(r['target_xy']==[0.,0.] for r in ds.records)
 encoder=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));encoder.load_cache(a.bert_cache)
 ids={s:[x['index'] for x in labels if x['split']==s] for s in ['fit','development','confirmation']};assert len({labels[i]['group'] for i in ids['fit']}&{labels[i]['group'] for i in ids['development']+ids['confirmation']})==0
 config=dict(seed=42,epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,modes=a.modes,smoke_updates=a.smoke_updates,source_v2_sha256=file_sha256(a.source_v2),trained_v2_sha256=file_sha256(a.trained_v2),data_manifest_sha256=file_sha256(a.data/'manifest.json'),bert_cache_sha256=file_sha256(a.bert_cache),precision='FP32 model/logits/loss; frozen BERT FP32 cached features; TF32 off',splits={k:len(v) for k,v in ids.items()},selection='best development R20, tie NE; confirmation not evaluated during training',caveat=manifest['prior_checkpoint_caveat'])
 mp=a.output/'config.json'
 if mp.exists() and json.load(open(mp))!=config:raise ValueError('Input/config mismatch on resume')
 atomic_json(config,mp);started=time.perf_counter();summary={}
 def tensors(batch):
  obs,tokens,mask,nodes=_batch_inputs(batch,encoder,'cuda','geometry');ii=batch['row_index'].tolist();bounds=torch.tensor([ds.records[i]['world_bounds'] for i in ii],device='cuda',dtype=torch.float32);target=torch.tensor([labels[i]['target_world'] for i in ii],device='cuda',dtype=torch.float32);normalized=torch.tensor([labels[i]['target_xy'] for i in ii],device='cuda',dtype=torch.float32)
  return obs,tokens,mask,nodes,bounds,target,normalized
 @torch.no_grad()
 def evaluate(model,indices):
  model.eval();errors=[]
  for batch in DataLoader(ds,batch_size=a.batch_size,sampler=indices,num_workers=0,collate_fn=collate_full):
   obs,tokens,mask,nodes,bounds,target,_=tensors(batch);z=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes).belief_logits.float();pred=obs['grid_xy'][torch.arange(len(z),device='cuda'),z.argmax(-1)];xy=bounds[:,:2]+pred*(bounds[:,2:]-bounds[:,:2]);errors.extend((xy-target).norm(dim=-1).cpu().tolist())
  return dict(n=len(errors),R20=float(np.mean(np.array(errors)<=20)),NE=float(np.mean(errors)))
 for mode in a.modes:
  out=a.output/mode;out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():summary[mode]=json.load(open(out/'complete.json'));continue
  torch.manual_seed(42);np.random.seed(42);model,_=_load_base(a.source_v2,'cuda');model.load_state_dict(state,strict=True);model.requires_grad_(True)
  if hasattr(model,'gat'):model.gat.requires_grad_(False)
  params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.001);epoch=0;offset=0;updates=0;history=[];best_score=(-1.,-float('inf'));baseline=None
  if (out/'last.pt').exists():
   if not a.resume:raise FileExistsError('Use --resume')
   s=torch.load(out/'last.pt',weights_only=True,map_location='cpu');model.load_state_dict(s['geometry']);opt.load_state_dict(s['optimizer']);epoch=s['epoch'];offset=s['offset'];updates=s['updates'];history=s['history'];best_score=tuple(s['best_score']);baseline=s['baseline'];restore_rng(s['rng'])
  if baseline is None:
   baseline=evaluate(model,ids['development'][:64] if a.smoke_updates else ids['development']);atomic_json(baseline,out/'baseline_development.json');best_score=(baseline['R20'],-baseline['NE'])
  def checkpoint(name):
   atomic_save(dict(geometry=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,offset=offset,updates=updates,history=history,best_score=best_score,baseline=baseline,rng=rng_state(),training_config=config),out/name)
  model.train();mode_started=time.perf_counter();losses=[];torch.cuda.reset_peak_memory_stats()
  with (out/'steps.jsonl').open('a') as log:
   while epoch<a.epochs:
    order=np.random.default_rng(42+epoch).permutation(ids['fit']).tolist()[offset:];model.train()
    for batch in DataLoader(ds,batch_size=a.batch_size,sampler=order,num_workers=0,collate_fn=collate_full):
     if time.perf_counter()-started>a.max_hours*3600:checkpoint('last.pt');atomic_json(dict(reason='global wall budget',updates=updates,complete=False),a.output/'budget_stop.json');return
     torch.cuda.synchronize();t=time.perf_counter();obs,tokens,mask,nodes,bounds,target,normalized=tensors(batch);opt.zero_grad(set_to_none=True);field=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes);loss,parts=goal_hit_loss(field.belief_logits,obs['grid_xy'],bounds,target,normalized,mode)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
     if updates<3 and float(norm)<=0:raise RuntimeError('Zero geometry gradient')
     opt.step();updates+=1;offset+=len(batch['row_index']);torch.cuda.synchronize();record=dict(mode=mode,epoch=epoch+1,update=updates,loss=float(loss.detach()),gradient_norm=float(norm),step_s=time.perf_counter()-t,peak_GiB=torch.cuda.max_memory_allocated()/2**30,**parts);losses.append(record['loss']);log.write(json.dumps(record)+'\n');log.flush()
     if updates<=3 or updates%100==0:print(json.dumps(record),flush=True)
     if updates%200==0:checkpoint('last.pt')
     if a.smoke_updates and updates>=a.smoke_updates:
      checkpoint('last.pt');atomic_json(dict(smoke_pass=True,updates=updates,last_step=record,seconds=time.perf_counter()-mode_started),out/'smoke.json');break
    if a.smoke_updates:break
    epoch+=1;offset=0;dev=evaluate(model,ids['development']);score=(dev['R20'],-dev['NE']);entry=dict(epoch=epoch,updates=updates,mean_loss=float(np.mean(losses)),development=dev,elapsed_s=time.perf_counter()-mode_started);history.append(entry);losses=[]
    if score>best_score:best_score=score;checkpoint('best.pt')
    checkpoint('last.pt');atomic_json(history,out/'history.json');print(json.dumps(dict(mode=mode,**entry)),flush=True)
   if not a.smoke_updates:
    result=dict(complete=True,epochs=epoch,updates=updates,baseline=baseline,best_development=dict(R20=best_score[0],NE=-best_score[1]),seconds=time.perf_counter()-mode_started,core_parameters=sum(p.numel() for p in params));atomic_json(result,out/'complete.json');summary[mode]=result
  del model,opt;torch.cuda.empty_cache()
 atomic_json(dict(smoke=bool(a.smoke_updates),arms=summary,seconds=time.perf_counter()-started,confirmation_evaluated=False),a.output/'summary.json')
if __name__=='__main__':main()
