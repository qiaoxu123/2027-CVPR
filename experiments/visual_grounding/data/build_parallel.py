"""Map-disjoint six-process full observed-region audit and deterministic merge."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
ROOT=Path(__file__).resolve().parents[3]

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='runs/visual_grounding_benchmark/data_audit/full');p.add_argument('--observations-root',default='runs/visual_grounding_benchmark/data_audit/full_observations');p.add_argument('--workers',type=int,default=6);a=p.parse_args();out=Path(a.output).resolve();obs=Path(a.observations_root).resolve();out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 while True:
  try:
   if 'val_unseen' in json.loads((obs/'render_stats.json').read_text()):break
  except (FileNotFoundError,json.JSONDecodeError):pass
  time.sleep(3)
 tasks=[]
 for split in ['val_seen','val_unseen','train_seen']:
  maps=sorted({json.loads(l)['map_name'] for l in (obs/f'observations_{split}.jsonl').open()})
  tasks += [(split,m) for m in maps]
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 def worker(task):
  split,m=task;dest=out/'maps'/split/m;dest.mkdir(parents=True,exist_ok=True)
  while not (dest/'stats.json').exists():
   active=subprocess.check_output(['ps','-eo','args'],text=True)
   if not any('experiments/visual_grounding/data/build_manifest.py' in line and f'--map {m} --observations-root' in line for line in active.splitlines()):break
   time.sleep(3)
  if not (dest/'stats.json').exists():
   for partial in ('regions.jsonl','manifest.jsonl'):
    old=dest/partial
    if old.exists():old.rename(dest/(partial+'.unexpected_session_termination_'+str(int(time.time()))))
   with (dest/'build.log').open('w') as log:
    subprocess.run([sys.executable,'experiments/visual_grounding/data/build_manifest.py','--types','Building','Car','Ground','Parking','--splits',split,'--map',m,'--observations-root',str(obs),'--output',str(dest)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  with (dest/'candidate.log').open('w') as log:subprocess.run([sys.executable,'experiments/visual_grounding/data/build_candidate_sets.py','--data',str(dest)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print(json.dumps({'completed_split':split,'map':m,'seconds':time.monotonic()-start}),flush=True);return split,m
 completed=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for result in as_completed([pool.submit(worker,t) for t in tasks]):completed.append(result.result())
 merged={};geometry=None
 for split in ['train_seen','val_seen','val_unseen']:
  subset=sorted(m for s,m in completed if s==split);stats=[]
  dest=out/split;dest.mkdir(exist_ok=True)
  for filename in ['regions.jsonl','manifest.jsonl','candidates.jsonl']:
   with (dest/filename).open('wb') as handle:
    for m in subset:
     with (out/'maps'/split/m/filename).open('rb') as source:shutil.copyfileobj(source,handle)
  for m in subset:
   s=json.loads((out/'maps'/split/m/'stats.json').read_text());stats.append(s['splits'][split]);geometry=s['geometry']
  summary={k:sum(s[k] for s in stats) for k in ['episodes_total','unique_instructions','unique_target_objects','observations','observed_episode_count','roi_objects','visible_target_objects','verified_attribute_samples']};summary['maps']=len(subset)
  for k in ['counts','roi_types','weak_attribute_roi_counts']:
   c=Counter()
   for s in stats:c.update(s[k])
   summary[k]=dict(c)
  merged[split]=summary;(dest/'stats.json').write_text(json.dumps({'splits':{split:summary}},indent=2))
  with (dest/'candidate_final.log').open('w') as log:subprocess.run([sys.executable,'experiments/visual_grounding/data/build_candidate_sets.py','--data',str(dest)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 for filename in ['regions.jsonl','manifest.jsonl','candidates.jsonl']:
  with (out/filename).open('wb') as handle:
   for split in ['train_seen','val_seen','val_unseen']:
    with (out/split/filename).open('rb') as source:shutil.copyfileobj(source,handle)
 shutil.copyfile(out/'maps'/completed[0][0]/completed[0][1]/'objects.json',out/'objects.json');(out/'stats.json').write_text(json.dumps({'splits':merged,'geometry':geometry,'seconds':time.monotonic()-start,'workers':a.workers,'projection_equivalence_checks':sum(json.loads((out/'maps'/s/m/'stats.json').read_text())['prevalidated_projection_equivalence_checks'] for s,m in completed)},indent=2));print(json.dumps(merged),flush=True)
if __name__=='__main__':main()
