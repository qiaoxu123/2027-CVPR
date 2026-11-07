"""Differentiable instruction BERT plus existing Geometry; TRAIN-only adaptation."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.trainable_instruction_encoder import TrainableInstructionBERT
from graphnav.full_experiment import _batch_inputs
from graphnav.candidate_geometry import collate_full
from graphnav.goal_hit_training import fidelity_hit_loss
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256,rng_state,restore_rng
from scripts.train_v2_top1_geometry import Indexed
from scripts.unified_pgvb_v3_pilot import _load_base

def main():
 p=argparse.ArgumentParser()
 for k in ['data','cache','source','trained','bert-cache','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--smoke-updates',type=int,default=8);p.add_argument('--epochs',type=int,default=3);p.add_argument('--resume',action='store_true');p.add_argument('--max-hours',type=float,default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.manual_seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 labels=json.loads((a.data/'labels_private.json').read_text());ids=[l['index'] for l in labels if l['split']=='fit'];assert len(ids)==11133
 model,cfg=_load_base(a.source,'cuda');s=torch.load(a.trained,map_location='cpu',weights_only=True);model.load_state_dict({k[9:]:v for k,v in s['model'].items() if k.startswith('geometry.')});model.requires_grad_(True)
 for m in [model.gnn.layer1,model.gnn.layer2]:m.requires_grad_(False)
 model.eval();bert=TrainableInstructionBERT(cfg.text_backbone,device='cuda',max_length=96);bert.load_cache(a.bert_cache);bert.model.eval()
 ds=Indexed(a.data/'public_train.jsonl',grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False)
 params=[p for p in model.parameters() if p.requires_grad];bp=[p for p in bert.model.parameters() if p.requires_grad];opt=torch.optim.AdamW([dict(params=params,lr=1e-5),dict(params=bp,lr=2e-6)],weight_decay=.001)
 dev_ids=[l['index'] for l in labels if l['split']=='development']
 assert not ({labels[i]['group'] for i in ids}&{labels[i]['group'] for i in dev_ids})
 @torch.no_grad()
 def evaluate():
  errors=[];source_errors=[]
  for batch in DataLoader(ds,batch_size=16,sampler=dev_ids,num_workers=0,collate_fn=collate_full):
   obs,tokens,mask,names=_batch_inputs(batch,bert,'cuda','geometry');ii=batch['row_index'].tolist();bounds=torch.tensor([ds.records[i]['world_bounds'] for i in ii],device='cuda');target=torch.tensor([labels[i]['target_world'] for i in ii],device='cuda');z=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names).belief_logits
   base=torch.tensor(np.stack([np.load(a.cache/'public_features'/f'{i:05d}.npz')['base_logits_full'] for i in ii]),device='cuda')
   for logits,store in [(z,errors),(base,source_errors)]:
    pred=obs['grid_xy'][torch.arange(len(ii),device='cuda'),logits.argmax(-1)];world=bounds[:,:2]+pred*(bounds[:,2:]-bounds[:,:2]);store.extend((world-target).norm(dim=-1).cpu().tolist())
  e=np.array(errors);s=np.array(source_errors)
  return dict(n=len(e),R20=float((e<=20).mean()),NE=float(e.mean()),source_R20=float((s<=20).mean()),source_NE=float(s.mean()),corrections=int(((e<=20)&(s>20)).sum()),damage=int(((e>20)&(s<=20)).sum()))
 fingerprint=dict(data=file_sha256(a.data/'manifest.json'),geometry=file_sha256(a.trained),bert_cache=file_sha256(a.bert_cache),script=file_sha256(__file__),encoder=file_sha256(__import__('graphnav.trainable_instruction_encoder',fromlist=['x']).__file__));epoch=0;offset=0;updates=0;history=[];evaluations=[];best_score=(-1.,-float('inf'))
 if a.resume:
  saved=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False)
  if saved['fingerprint']!=fingerprint:raise ValueError('Resume input/code mismatch')
  model.load_state_dict(saved['geometry']);bert.model.load_state_dict(saved['bert']);opt.load_state_dict(saved['optimizer']);epoch=saved['epoch'];offset=saved['offset'];updates=saved['updates'];history=saved['history'];evaluations=saved['evaluations'];best_score=tuple(saved['best_score']);restore_rng(saved['rng'])
 start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 def save(name='last.pt'):atomic_save(dict(geometry=model.state_dict(),bert=bert.model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,offset=offset,updates=updates,history=history,evaluations=evaluations,best_score=best_score,rng=rng_state(),fingerprint=fingerprint),a.output/name)
 while epoch<a.epochs:
  order=np.random.default_rng(42+epoch).permutation(ids).tolist()[offset:]
  for batch in DataLoader(ds,batch_size=16,sampler=order,num_workers=0,collate_fn=collate_full):
   torch.cuda.synchronize();t=time.perf_counter();obs,tokens,mask,names=_batch_inputs(batch,bert,'cuda','geometry');ii=batch['row_index'].tolist();bounds=torch.tensor([ds.records[i]['world_bounds'] for i in ii],device='cuda');target=torch.tensor([labels[i]['target_world'] for i in ii],device='cuda');base=torch.tensor(np.stack([np.load(a.cache/'public_features'/f'{i:05d}.npz')['base_logits_full'] for i in ii]),device='cuda');opt.zero_grad(set_to_none=True);z=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names).belief_logits;loss,parts=fidelity_hit_loss(z,base,obs['grid_xy'],bounds,target)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite joint loss')
   loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);bn=torch.nn.utils.clip_grad_norm_(bp,1.,error_if_nonfinite=True)
   if updates<3 and (float(gn)<=0 or float(bn)<=0):raise RuntimeError('Missing joint gradient')
   opt.step();torch.cuda.synchronize();updates+=1;offset+=len(ii);row=dict(update=updates,epoch=epoch+1,loss=float(loss.detach()),geometry_grad=float(gn),bert_grad=float(bn),seconds=time.perf_counter()-t,tokens=int(mask.shape[1]),**parts);history.append(row);print(json.dumps(row),flush=True)
   if updates%100==0:save()
   if (a.smoke_updates and updates>=a.smoke_updates) or time.perf_counter()-start>=a.max_hours*3600:
    save();atomic_json(dict(updates=updates,history=history,elapsed_s=time.perf_counter()-start,peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,geometry_parameters=sum(p.numel() for p in params),bert_parameters=sum(p.numel() for p in bp),fit_n=len(ids)),a.output/'summary.json');return
  epoch+=1;offset=0
  if not a.smoke_updates:
   dev=evaluate();evaluations.append(dict(epoch=epoch,**dev));score=(dev['R20'],-dev['NE'])
   if score>best_score:best_score=score;save('best.pt')
   atomic_json(evaluations,a.output/'development.json');print(json.dumps(dict(development=dev,epoch=epoch)),flush=True)
  save()
 atomic_json(dict(complete=True,epochs=epoch,updates=updates),a.output/'complete.json')
if __name__=='__main__':main()
