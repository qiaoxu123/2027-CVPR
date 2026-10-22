"""Run finite CUDA smoke checks before durable full training; never promote smoke weights."""
import json,subprocess,sys,os,time
from pathlib import Path
from graphnav.workspace_paths import PROJECT
from graphnav.checkpointing import atomic_json,file_sha256

def main():
 root=Path('runs/grid384_20261022').resolve();source=PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'
 env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 previous=None;proof=[]
 for stage in ['A','B','C']:
  dest=root/f'preflight_scaled_{stage}'
  command=[sys.executable,'-m','scripts.train_graphnav_grid384','--cache',str(PROJECT/'GraphNav/data/graphnav_stage1_frames'),'--feature-cache',str(root/'features'),'--sources',str(root/'initial_sources.json'),'--stage',stage,'--output',str(dest),'--epochs','1','--batch-size','4','--workers','0','--accumulation','2','--max-batches','2','--smoke','--sbf-source',str(source),'--sbf-config',str(source/'configs/sbfnav/main.yaml'),'--data-root',str(PROJECT.parent/'DATA/sbfnav-data')]
  if previous:command+=['--init',str(previous)]
  if not (dest/'complete.json').exists():
   with (root/f'preflight_scaled_{stage}.log').open('a') as f:subprocess.run(command,env=env,check=True,stdout=f,stderr=subprocess.STDOUT)
  complete=json.loads((dest/'complete.json').read_text());assert complete['smoke'];h=json.loads((dest/'history.json').read_text())[-1];assert h['train']['samples']==8
  proof.append(dict(stage=stage,command=command,train=h['train'],dev=h['dev'],checkpoint_sha256=file_sha256(dest/'best.pt')));previous=dest/'best.pt'
 atomic_json(dict(status='passed',stages=proof,smoke_weights_excluded_from_full_training=True,finished=time.time()),root/'PREFLIGHT.json')
 command=[sys.executable,'-m','scripts.run_grid384_training','--output',str(root)]
 if (root/'workflow_status.json').exists():command+=['--resume']
 subprocess.run(command,env=env,check=True)
if __name__=='__main__':main()
