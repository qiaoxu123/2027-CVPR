"""Isolated, durable A3/B3/C5 training and frozen 3-split evaluation. Never touch old runs."""
import argparse,json,os,subprocess,sys,time,traceback,shutil
from pathlib import Path
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.workspace_paths import PROJECT

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='runs/grid384_20261022');p.add_argument('--resume',action='store_true');a=p.parse_args()
 root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
 statuspath=root/'workflow_status.json'
 if statuspath.exists() and not a.resume:raise ValueError('explicit resume required')
 status=json.loads(statuspath.read_text()) if statuspath.exists() else {'started':time.time(),'jobs':{}}
 old=status.get('pid')
 if old and Path(f'/proc/{old}/cmdline').exists() and 'run_grid384_training' in Path(f'/proc/{old}/cmdline').read_bytes().decode(errors='replace'):raise ValueError('workflow already running')
 child=status.get('child_pid')
 if child and Path(f'/proc/{child}/cmdline').exists() and str(root) in Path(f'/proc/{child}/cmdline').read_bytes().decode(errors='replace'):raise ValueError('child still running')
 status['pid']=os.getpid()
 env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
 source=PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source';data=PROJECT.parent/'DATA/sbfnav-data';config=source/'configs/sbfnav/main.yaml'
 base=['--sbf-source',str(source),'--data-root',str(data)]
 def run(name,cmd):
  status.update(status='running',stage=name);status['jobs'][name]={'command':cmd,'started':time.time()};atomic_json(status,statuspath)
  with (root/f'{name}.log').open('a') as f:
   process=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);status['child_pid']=process.pid;atomic_json(status,statuspath);code=process.wait()
  status['jobs'][name].update(exit_code=code,finished=time.time());status['child_pid']=None;atomic_json(status,statuspath)
  if code:raise RuntimeError(name+' failed; inspect retained log, no subsequent jobs run')
 try:
  previous=None
  for stage,epochs in [('A',3),('B',3),('C',5)]:
   dest=root/'stages'/stage
   if not (dest/'complete.json').exists():
    if dest.exists() and not (dest/'last.pt').exists():dest.rename(dest.with_name(stage+'_failed_'+str(int(time.time()))))
    cmd=[sys.executable,'-m','scripts.train_graphnav_grid384','--cache',str(PROJECT/'GraphNav/data/graphnav_stage1_frames'),'--feature-cache',str(root/'features'),'--sources',str(root/'initial_sources.json'),'--stage',stage,'--output',str(dest),'--epochs',str(epochs),'--batch-size','4','--workers','0','--accumulation','4','--seed','0','--sbf-config',str(config),*base]
    if previous:cmd+=['--init',str(previous)]
    if (dest/'last.pt').exists():cmd+=['--resume',str(dest/'last.pt')]
    run('train_'+stage,cmd)
   complete=json.loads((dest/'complete.json').read_text());assert complete['epochs']==epochs and not complete['smoke']
   previous=dest/'best.pt'
  # Predeclared C-stage selection guarantees the static geometry was actually trained at 384.
  checkpoint=root/'best.pt'
  if not checkpoint.exists():shutil.copy2(previous,checkpoint)
  assert file_sha256(checkpoint)==file_sha256(previous)
  state=torch.load(checkpoint,map_location='cpu',weights_only=True);assert state['a3_args']['grid_size']==384 and state['args']['stage']=='C'
  freeze=root/'frozen_final_candidate.json'
  manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),all_stages_complete=True,completed_epochs=dict(A=3,B=3,C=5),selected_stage='C',selected_epoch=state['epoch'],grid_size=384,training_grid_size=384,visual_control='real',history_control='checkpoint',stop_threshold=.3,active_observation=False,selection='C checkpoints: Seen128 SR, SPL, internal Train Dev belief R@1; no held-out selection',frozen=time.time())
  if not freeze.exists():atomic_json(manifest,freeze)
  else:assert json.loads(freeze.read_text())['checkpoint_sha256']==manifest['checkpoint_sha256']
  for split,count in [('val_seen',2470),('val_unseen',2697),('test_unseen',5281)]:
   dest=root/'evaluation'/split
   if not (dest/'metrics.json').exists():
    cmd=[sys.executable,'-m','scripts.evaluate_grid384','--model','graphnav-v1','--checkpoint',str(checkpoint),*base,'--config',str(config),'--split',split,'--frozen-final-candidate',str(freeze),'--stage1-save-fields','--output',str(dest)]
    if (dest/'predictions.jsonl').exists():cmd+=['--resume']
    run('evaluate_'+split,cmd)
   m=json.loads((dest/'metrics.json').read_text());assert m['checkpoint_sha256']==manifest['checkpoint_sha256']
   assert sum(1 for _ in (dest/'predictions.jsonl').open())==count
  for control in ['real','zero','shuffle','no-belief']:
   dest=root/'controls'/control
   if not (dest/'metrics.json').exists():run('control_'+control,[sys.executable,'-m','scripts.evaluate_grid384','--model','graphnav-v1','--checkpoint',str(checkpoint),*base,'--config',str(config),'--split','val_seen','--max-episodes','128','--stage1-visual-control',control,'--output',str(dest)])
  dest=root/'profile_seen16'
  if not (dest/'metrics.json').exists():run('profile',[sys.executable,'-m','scripts.evaluate_grid384','--model','graphnav-v1','--checkpoint',str(checkpoint),*base,'--config',str(config),'--split','val_seen','--max-episodes','16','--profile','--output',str(dest)])
  run('summarize',[sys.executable,'-m','scripts.summarize_grid384','--root',str(root)])
  status.update(status='complete',finished=time.time());atomic_json(status,statuspath)
 except BaseException:
  status.update(status='failed',error=traceback.format_exc(),failed=time.time());atomic_json(status,statuspath);raise
if __name__=='__main__':main()
