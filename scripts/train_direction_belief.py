"""Train only the 32-parameter direction transform; frozen visual/nav files untouched."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from graphnav.direction_belief import DirectionBelief
from graphnav.direction_belief_data import synthetic,point_distribution
from graphnav.checkpointing import atomic_save,file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--synthetic-steps',type=int,default=400);p.add_argument('--real-steps',type=int,default=240);a=p.parse_args()
 if a.output.exists() and not a.resume:raise FileExistsError(a.output)
 a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0);dev='cuda';torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 public=json.loads((a.data/'public.json').read_text());labels={r['index']:r for r in json.loads((a.data/'private_fit_labels.json').read_text())};fit=[r for r in public if r['subset']=='fit'];assert fit and set(labels)=={r['index'] for r in fit}
 anchor=torch.tensor(np.asarray([r['anchor'] for r in fit]),device=dev,dtype=torch.float32);span=torch.tensor([r['span'] for r in fit],device=dev);direction=torch.tensor([r['direction_id'] for r in fit],device=dev);bounds=torch.tensor([r['bounds'] for r in fit],device=dev);points=torch.tensor([labels[r['index']]['target_world'][:2] for r in fit],device=dev)-bounds[:,:2];target=point_distribution(points,span)
 real=dict(anchor=anchor,direction=direction,span=span,target=target,heading=torch.zeros(len(fit),device=dev));syn={k:v.to(dev) for k,v in synthetic(4096,123).items()};history=json.loads((a.output/'history.json').read_text()) if (a.output/'history.json').exists() else []
 config=dict(seed=0,batch_size=32,synthetic_steps=a.synthetic_steps,real_steps=a.real_steps,fit_pairs=len(fit),parameters=32,lr_synthetic=.015,lr_real=.003,labels_sigma_m=10.,public_sha256=file_sha256(a.data/'public.json'),private_labels_sha256=file_sha256(a.data/'private_fit_labels.json'),code_sha256={k:file_sha256(k) for k in ['graphnav/direction_belief.py','graphnav/direction_belief_data.py',__file__]},no_checkpoint_selection_from_dev=True,frozen_PGVB_and_navigator=True)
 (a.output/'config.json').write_text(json.dumps(config,indent=2))
 # Real CUDA finite forward/backward, isolated from fitted models.
 smoke=DirectionBelief().to(dev);optimizer=torch.optim.Adam(smoke.parameters(),lr=.015);t=time.perf_counter()
 for _ in range(2):
  pred=smoke(anchor[:2],direction[:2],span[:2]);loss=-(target[:2]*pred.clamp_min(1e-20).log()).sum((-1,-2)).mean();optimizer.zero_grad();loss.backward();assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in smoke.parameters());optimizer.step()
 torch.cuda.synchronize();(a.output/'smoke.json').write_text(json.dumps(dict(loss=float(loss),gradient_abs=float(smoke.parameters_by_word.weight.grad.abs().sum()),seconds=time.perf_counter()-t,device=torch.cuda.get_device_name(0),parameters=sum(p.numel() for p in smoke.parameters())),indent=2));del smoke,optimizer
 models={}
 for ignore,label in [(False,'direction'),(True,'nodirection')]:
  for phase,steps,lr,source in [('synthetic',a.synthetic_steps,.015,syn),('real',a.real_steps,.003,real)]:
   name=f'{label}_{phase}';path=a.output/(name+'.pt');progress=a.output/(name+'_resume.pt');model=DirectionBelief(ignore_direction=ignore).to(dev);optimizer=torch.optim.Adam(model.parameters(),lr=lr);start_step=0
   if phase=='real':model.load_state_dict(models[label])
   if path.exists() and a.resume:
    state=torch.load(path,weights_only=True);assert state['config']==config;models[label]=state['model'];continue
   if progress.exists() and a.resume:
    state=torch.load(progress,weights_only=True);assert state['config']==config;model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);start_step=state['step']
   for step in range(start_step,steps):
    ids=torch.from_numpy(np.random.default_rng(step+200).integers(0,len(source['anchor']),32)).to(dev);kw={k:source[k][ids] for k in ['anchor','direction','span']};prediction=model(**kw,reference_heading=source['heading'][ids]);loss=-(source['target'][ids]*prediction.clamp_min(1e-20).log()).sum((-1,-2)).mean();optimizer.zero_grad();loss.backward();assert torch.isfinite(loss);optimizer.step()
    if (step+1)%40==0:
     row=dict(model=name,step=step+1,loss=float(loss),seconds=time.perf_counter()-started);history.append(row);(a.output/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    if (step+1)%80==0:atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,config=config),progress)
   models[label]={k:v.detach().cpu() for k,v in model.state_dict().items()};atomic_save(dict(model=models[label],config=config,ignore_direction=ignore,phase=phase),path)
 (a.output/'history.json').write_text(json.dumps(history,indent=2));(a.output/'training_summary.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,checkpoints={p.name:file_sha256(p) for p in a.output.glob('*.pt')}),indent=2))
if __name__=='__main__':main()
