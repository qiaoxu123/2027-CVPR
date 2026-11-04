"""Real cached-TRAIN PGVB-head optimization inside UnifiedPGVB-v2 (not v1).

Geometry/BERT/SigLIP2/controller remain frozen. Joint/direction/target are off.
Checkpoints retain the complete integrated V2 model plus optimizer/sampler/RNG.
"""
import argparse,json,math,time,subprocess,random
from pathlib import Path
from collections import deque
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256,rng_state,restore_rng
from graphnav.unified_pgvb_training import load_unified,configure_visual_training,CachedLandmarkFrames,collate_cached_frames,move_batch,frame_order,weak_landmark_loss

def sync():torch.cuda.synchronize()
def main():
 p=argparse.ArgumentParser()
 for key in ['data','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--until-updates',type=int,required=True);p.add_argument('--stage',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--batch-frames',type=int,default=4);p.add_argument('--num-workers',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--save-every',type=int,default=100);a=p.parse_args();started=time.perf_counter();torch.set_num_threads(4);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 proof=json.loads((a.data/'manifest.json').read_text());t=time.perf_counter();assert file_sha256(a.data/'cache.pt')==proof['cache_sha256'];hash_seconds=time.perf_counter()-t;t=time.perf_counter();data=torch.load(a.data/'cache.pt',map_location='cpu',weights_only=True);load_seconds=time.perf_counter()-t;ds=CachedLandmarkFrames(data,'fit');steps_per_epoch=math.ceil(len(ds)/a.batch_frames)
 payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=load_unified(payload);names=configure_visual_training(model);opt=torch.optim.AdamW(model.pgvb.parameters(),lr=a.lr,weight_decay=.001);config=dict(seed=a.seed,lr=a.lr,weight_decay=.001,clip_norm=5.,batch_frames=a.batch_frames,num_workers=a.num_workers,precision='cuda_bfloat16; losses float32',trainable=names,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),source_checkpoint_sha256=file_sha256(a.checkpoint),data_manifest_sha256=file_sha256(a.data/'manifest.json'),data_cache_sha256=proof['cache_sha256'],steps_per_epoch=steps_per_epoch,fit_frames=len(ds),fit_queries=len([r for r in data['records'] if r['subset']=='fit']),code_sha256={str(x):file_sha256(x) for x in [Path(__file__),Path('graphnav/unified_pgvb_training.py'),Path('graphnav/unified_pgvb.py')]})
 a.output.mkdir(parents=True,exist_ok=True);step=0;epoch=0;offset=0;best=float('inf');history=[];recent=deque(maxlen=100);resume_proof=None
 if a.resume:
  state=torch.load(a.output/'last.pt',map_location='cpu',weights_only=True)
  if state['training_config']!=config:raise ValueError('Resume configuration/source/data changed')
  model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);step=state['steps'];epoch=state['epoch'];offset=state['offset'];best=state['best_fit_window'];history=state['history'];recent.extend(state['recent_losses']);restore_rng(state['rng']);resume_proof=dict(restored_steps=step,restored_epoch=epoch,restored_offset=offset,optimizer_states=len(opt.state),optimizer_step_min=min(float(v['step']) for v in opt.state.values()),previous_checkpoint_sha256=file_sha256(a.output/'last.pt'))
 elif (a.output/'last.pt').exists():raise FileExistsError('Use --resume; never overwrite a training run')
 assert step<a.until_updates
 start_step=step;atomic_json(config,a.output/'config.json');gradient=None;frozen={k:v.clone() for k,v in payload['model'].items() if not k.startswith('pgvb.')};torch.cuda.reset_peak_memory_stats();records=[];save_times=[];stage_start=time.perf_counter();valid_steps=0;epoch_query_count=0
 def save(path):
  sync();t=time.perf_counter();cfg=dict(payload['config']);cfg.update(joint_fusion_ready=False,direction_ready=False,target_ready=False,jointly_trained=False,pgvb_visual_head_trained=True)
  state=dict(format='unified-pgvb-v2',model=model.state_dict(),config=cfg,training_config=config,optimizer=opt.state_dict(),steps=step,epoch=epoch,offset=offset,best_fit_window=best,history=history,recent_losses=list(recent),rng=rng_state());atomic_save(state,path);sync();save_times.append(time.perf_counter()-t)
 with (a.output/f'{a.stage}_steps.jsonl').open('x') as stream:
  while step<a.until_updates:
   order=frame_order(ds.frame_ids,a.seed,epoch);remaining=order[offset:];batches=[remaining[j:j+a.batch_frames] for j in range(0,len(remaining),a.batch_frames)];loader=DataLoader(ds,batch_sampler=batches,num_workers=a.num_workers,pin_memory=True,collate_fn=collate_cached_frames);iterator=iter(loader)
   while step<a.until_updates:
    sync();wall=time.perf_counter();wait=wall
    try:batch=next(iterator)
    except StopIteration:break
    load_ms=(time.perf_counter()-wait)*1000;t=time.perf_counter();inputs,labels,fids,types=move_batch(batch,'cuda');sync();transfer=(time.perf_counter()-t)*1000;opt.zero_grad(set_to_none=True);t=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(**inputs)['landmark_scores']
    losses=weak_landmark_loss(scores,labels,fids,types);sync();forward=(time.perf_counter()-t)*1000
    if not torch.isfinite(losses['total']):raise FloatingPointError('Nonfinite TRAIN loss')
    t=time.perf_counter();losses['total'].backward();sync();backward=(time.perf_counter()-t)*1000
    if gradient is None:
     gradient={k:dict(nonzero=int(torch.count_nonzero(p.grad)) if p.grad is not None else 0,finite=bool(p.grad is not None and torch.isfinite(p.grad).all()),abs_sum=float(p.grad.abs().sum()) if p.grad is not None else None) for k,p in model.named_parameters() if p.requires_grad}
     if not all(v['finite'] for v in gradient.values()) or not any(v['nonzero'] for v in gradient.values()):raise RuntimeError('Trainable gradient check failed')
     assert all(p.grad is None for k,p in model.named_parameters() if not p.requires_grad)
    t=time.perf_counter();norm=torch.nn.utils.clip_grad_norm_(model.pgvb.parameters(),5.,error_if_nonfinite=True);opt.step();sync();optimizer=(time.perf_counter()-t)*1000;wall_ms=(time.perf_counter()-wall)*1000;step+=1;valid_steps+=1;offset+=len(batch['unique_images']);epoch_query_count+=len(labels);value=float(losses['total'].detach());recent.append(value)
    rec=dict(step=step,stage_update=valid_steps,epoch=epoch,offset=offset,frames=len(batch['unique_images']),queries=len(labels),loss=value,region=float(losses['region'].detach()),bce=float(losses['bce'].detach()),pair=float(losses['pair'].detach()),pairs=int(losses['pairs']),grad_norm=float(norm),data_wait_ms=load_ms,transfer_ms=transfer,forward_loss_ms=forward,backward_ms=backward,optimizer_ms=optimizer,wall_ms=wall_ms,warmup=valid_steps<=20,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30);records.append(rec);stream.write(json.dumps(rec)+'\n')
    if offset>=len(order):epoch+=1;offset=0
    if step%100==0 or step==a.until_updates:
     vals=records[-100:];log=dict(stage=a.stage,step=step,epoch=epoch,offset=offset,loss_mean100=float(np.mean([v['loss'] for v in vals])),wall_seconds100=sum(v['wall_ms'] for v in vals)/1000,queries=sum(v['queries'] for v in vals),gpu_telemetry=subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip());history.append(log);print(json.dumps(log),flush=True);stream.flush()
    if step%a.save_every==0 or step==a.until_updates:
     improved=np.mean(recent)<best;best=min(best,float(np.mean(recent)));save(a.output/'last.pt')
     if improved:save(a.output/'best.pt')
   if offset==0 and step<a.until_updates:continue
  stream.flush()
 state=model.state_dict();assert all(torch.equal(state[k].detach().cpu(),v) for k,v in frozen.items());delta=sum(float((state[k].detach().cpu()-v).square().sum()) for k,v in payload['model'].items() if k.startswith('pgvb.'));assert delta>0
 steady=[r for r in records if not r['warmup']];wall=sum(r['wall_ms'] for r in steady)/1000;summary=dict(stage=a.stage,start_step=start_step,end_step=step,updates=step-start_step,epoch=epoch,offset=offset,steps_per_epoch=steps_per_epoch,stage_seconds=time.perf_counter()-stage_start,total_process_seconds=time.perf_counter()-started,cache_hash_seconds=hash_seconds,cache_load_seconds=load_seconds,mean_step_ms=float(np.mean([r['wall_ms'] for r in steady])),queries_per_second=sum(r['queries'] for r in steady)/wall,frames_per_second=sum(r['frames'] for r in steady)/wall,component_mean_ms={k:float(np.mean([r[k] for r in steady])) for k in ['data_wait_ms','transfer_ms','forward_loss_ms','backward_ms','optimizer_ms']},save_seconds=save_times,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,loss_first20=float(np.mean([r['loss'] for r in records[:20]])),loss_last20=float(np.mean([r['loss'] for r in records[-20:]])),pgvb_parameter_squared_delta=delta,frozen_parameters_bitwise_unchanged=True,gradient_proof=gradient,resume=resume_proof,last_sha256=file_sha256(a.output/'last.pt'),best_sha256=file_sha256(a.output/'best.pt'),GPU_utilization_note='Device telemetry includes co-tenants, not this process alone',warmup_excluded=20)
 atomic_json(summary,a.output/f'{a.stage}_summary.json');atomic_json(history,a.output/'history.json');print(json.dumps({k:v for k,v in summary.items() if k not in ['gradient_proof','save_seconds']},indent=2),flush=True)
if __name__=='__main__':main()
