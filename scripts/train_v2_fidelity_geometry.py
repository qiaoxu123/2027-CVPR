"""TRAIN-only head/core adaptation with source-correct belief preservation.
The public name resolver is frozen; no new inference heads, rules or GT gates.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256,rng_state,restore_rng
from graphnav.full_experiment import _batch_inputs
from graphnav.candidate_geometry import collate_full
from graphnav.token_language import FrozenBERTTokens
from graphnav.static_localization import cross_entropy_goal
from graphnav.goal_hit_training import fidelity_hit_loss
from scripts.train_v2_top1_geometry import Indexed
from scripts.unified_pgvb_v3_pilot import _load_base


def main():
 p=argparse.ArgumentParser()
 for k in ['data','source-v2','trained-v2','bert-cache','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--smoke-updates',type=int,default=0);p.add_argument('--resume',action='store_true');p.add_argument('--max-hours',type=float,default=4.)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(42);np.random.seed(42)
 dm=json.load(open(a.data/'manifest.json'));assert dm['parser_sha256']==file_sha256(Path('graphnav/public_landmark_mentions.py'));labels=json.load(open(a.data/'labels_private.json'));ids={s:[r['index'] for r in labels if r['split']==s] for s in ['fit','development']};assert len(ids['fit'])==11133 and len(ids['development'])==3126
 teacher,cfg=_load_base(a.source_v2,'cuda');state=torch.load(a.trained_v2,map_location='cpu',weights_only=True);weights={k[len('geometry.'):]:v for k,v in state['model'].items() if k.startswith('geometry.')};teacher.load_state_dict(weights,strict=True);teacher.eval().requires_grad_(False)
 ds=Indexed(a.data/'public_train.jsonl',grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False);assert len(ds)==21878 and all(q['target_xy']==[0.,0.] for q in ds.records);bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));bert.load_cache(a.bert_cache)
 recipes={'head_ce':dict(scope='heads',loss='legacy',lr=1e-4),'head_fidelity':dict(scope='heads',loss='fidelity',lr=1e-4),'core_fidelity':dict(scope='core',loss='fidelity',lr=1e-5)}
 config=dict(seed=42,epochs=a.epochs,batch_size=a.batch_size,recipes=recipes,smoke_updates=a.smoke_updates,data_manifest_sha256=file_sha256(a.data/'manifest.json'),source_v2_sha256=file_sha256(a.source_v2),trained_v2_sha256=file_sha256(a.trained_v2),parser_sha256=dm['parser_sha256'],precision='FP32 TF32off',dropout='eval mode in TRAIN, same deterministic features as inference; gradients still enabled',GT='TRAIN loss only: source correctness decides preservation/correction loss; never runtime inputs',selection='Cambridge TRAIN development only; Birmingham confirmation not reused; minimum+1pp dev R20 and no NE regression before considering final refit',source_sha256=file_sha256(__file__),loss_sha256=file_sha256(Path('graphnav/goal_hit_training.py')))
 cp=a.output/'config.json'
 if cp.exists() and json.load(open(cp))!=config:raise ValueError('Resume config mismatch')
 atomic_json(config,cp);start=time.perf_counter()
 def inputs(batch):
  obs,tokens,mask,nodes=_batch_inputs(batch,bert,'cuda','geometry');ii=batch['row_index'].tolist();bounds=torch.tensor([ds.records[i]['world_bounds'] for i in ii],device='cuda');target=torch.tensor([labels[i]['target_world'] for i in ii],device='cuda');target_xy=torch.tensor([labels[i]['target_xy'] for i in ii],device='cuda');return obs,tokens,mask,nodes,bounds,target,target_xy
 @torch.no_grad()
 def evaluate(model):
  rows=[];indices=ids['development'][:64] if a.smoke_updates else ids['development'];model.eval()
  for batch in DataLoader(ds,batch_size=a.batch_size,sampler=indices,num_workers=0,collate_fn=collate_full):
   obs,tokens,mask,nodes,bounds,target,_=inputs(batch);z=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes).belief_logits;xy=obs['grid_xy'][torch.arange(len(z),device='cuda'),z.argmax(-1)];world=bounds[:,:2]+xy*(bounds[:,2:]-bounds[:,:2]);errors=(world-target).norm(dim=-1).cpu().tolist();rows.extend(dict(index=i,NE=e,hit=e<=20) for i,e in zip(batch['row_index'].tolist(),errors))
  return dict(n=len(rows),R20=float(np.mean([r['hit'] for r in rows])),NE=float(np.mean([r['NE'] for r in rows]))),rows
 baseline,baseline_rows=evaluate(teacher);atomic_json(baseline,a.output/'baseline_development.json');atomic_json(baseline_rows,a.output/'baseline_development_rows.json');basehit={r['index']:r['hit'] for r in baseline_rows};summary={}
 for name,recipe in recipes.items():
  out=a.output/name;out.mkdir(exist_ok=True)
  if (out/'complete.json').exists():summary[name]=json.load(open(out/'complete.json'));continue
  torch.manual_seed(42);model,_=_load_base(a.source_v2,'cuda');model.load_state_dict(weights,strict=True);model.requires_grad_(recipe['scope']=='core')
  if recipe['scope']=='heads':
   for m in [model.token_head,model.geometry_head]:m.requires_grad_(True)
   model.token_gain.requires_grad_(True);model.geometry_gain.requires_grad_(True)
  model.eval();params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=recipe['lr'],weight_decay=.001);epoch=0;offset=0;updates=0;history=[];best=(baseline['R20'],-baseline['NE'])
  if (out/'last.pt').exists():
   if not a.resume:raise FileExistsError('Use resume')
   s=torch.load(out/'last.pt',weights_only=True,map_location='cpu');model.load_state_dict(s['geometry']);opt.load_state_dict(s['optimizer']);epoch=s['epoch'];offset=s['offset'];updates=s['updates'];history=s['history'];best=tuple(s['best']);restore_rng(s['rng'])
  def save(file):atomic_save(dict(geometry=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,offset=offset,updates=updates,history=history,best=best,rng=rng_state(),config=config,recipe=recipe),out/file)
  begun=time.perf_counter()
  with (out/'steps.jsonl').open('a') as stream:
   while epoch<a.epochs:
    order=np.random.default_rng(42+epoch).permutation(ids['fit']).tolist()[offset:];losses=[]
    for batch in DataLoader(ds,batch_size=a.batch_size,sampler=order,num_workers=0,collate_fn=collate_full):
     if time.perf_counter()-start>a.max_hours*3600:save('last.pt');atomic_json(dict(complete=False,reason='budget'),a.output/'budget_stop.json');return
     torch.cuda.synchronize();t=time.perf_counter();obs,tokens,mask,nodes,bounds,target,target_xy=inputs(batch);opt.zero_grad(set_to_none=True);z=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes).belief_logits
     if recipe['loss']=='legacy':loss=cross_entropy_goal(z,target_xy,grid_size=28);parts={}
     else:
      with torch.no_grad():base=teacher(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes).belief_logits
      loss,parts=fidelity_hit_loss(z,base,obs['grid_xy'],bounds,target)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
     if updates<3 and float(norm)<=0:raise RuntimeError('Zero gradient')
     opt.step();updates+=1;offset+=len(batch['row_index']);losses.append(float(loss.detach()));torch.cuda.synchronize();record=dict(arm=name,epoch=epoch+1,update=updates,loss=losses[-1],gradient_norm=float(norm),compute_s=time.perf_counter()-t,**parts);stream.write(json.dumps(record)+'\n');stream.flush()
     if updates<=3 or updates%100==0:print(json.dumps(record),flush=True)
     if updates%200==0:save('last.pt')
     if a.smoke_updates and updates>=a.smoke_updates:save('last.pt');atomic_json(dict(passed=True,updates=updates,last=record),out/'smoke.json');break
    if a.smoke_updates:break
    epoch+=1;offset=0;dev,rows=evaluate(model);corr=sum(r['hit'] and not basehit[r['index']] for r in rows);damage=sum(not r['hit'] and basehit[r['index']] for r in rows);entry=dict(epoch=epoch,updates=updates,loss=float(np.mean(losses)),development=dev,corrected=corr,damaged=damage,net=corr-damage,elapsed_s=time.perf_counter()-begun);history.append(entry);score=(dev['R20'],-dev['NE'])
    if score>best:best=score;save('best.pt');atomic_json(rows,out/'best_development_rows.json')
    save('last.pt');atomic_json(history,out/'history.json');print(json.dumps(dict(arm=name,**entry)),flush=True)
   if not a.smoke_updates:
    result=dict(complete=True,epochs=epoch,updates=updates,best_R20=best[0],best_NE=-best[1],baseline=baseline,trainable_parameters=sum(p.numel() for p in params),seconds=time.perf_counter()-begun);atomic_json(result,out/'complete.json');summary[name]=result
  torch.cuda.empty_cache()
 atomic_json(dict(arms=summary,smoke=bool(a.smoke_updates),baseline=baseline,seconds=time.perf_counter()-start,Val_Test_used=False,confirmation_reused=False),a.output/'summary.json')
if __name__=='__main__':main()
