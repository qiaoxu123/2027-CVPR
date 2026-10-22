"""Verify completed new 384 training/evaluation; aggregate official difficulty memberships."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from graphnav.workspace_paths import PROJECT
from graphnav.checkpointing import atomic_json,file_sha256

def eid(r):return (f"{r['area']}_block_{r['block']}",int(r['object_ids'][0]),int(r['ann_ids'][0]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=root/'summary';out.mkdir(exist_ok=True)
 freeze=json.loads((root/'frozen_final_candidate.json').read_text());rows=[];proof=[];belief=[]
 for split,count in [('val_seen',2470),('val_unseen',2697),('test_unseen',5281)]:
  d=root/'evaluation'/split;m=json.loads((d/'metrics.json').read_text());assert m['checkpoint_sha256']==freeze['checkpoint_sha256']
  dataset=PROJECT.parent/'DATA/sbfnav-data/processed_citynav'/f'citynav_{split}.json';data={eid(r):r for r in json.loads(dataset.read_text())};members={}
  for group in ['easy','medium','hard']:
   source=PROJECT.parent/'DATA/citynav/extracted/data/citynav'/f'citynav_{split}_{group}.json'
   for r in json.loads(source.read_text()):
    k=eid(r)
    if k in data:assert k not in members;members[k]=group
  assert set(members)==set(data)
  seen=set();values=[];h=hashlib.sha256()
  with (d/'predictions.jsonl').open('rb') as f:
   for line in f:
    h.update(line);r=json.loads(line);k=tuple(r['episode_id']);assert k not in seen;seen.add(k)
    path=np.asarray(r['poses'])[:,:2];gt=np.asarray(data[k]['target_positions'][-1])[:2];dist=np.linalg.norm(path-gt,axis=-1);length=np.linalg.norm(np.diff(path,axis=0),axis=-1).sum();sr=float(dist[-1]<=20)
    v=dict(NE=float(dist[-1]),SR=sr,OSR=float(dist.min()<=20),SPL=sr*dist[0]/max(length,dist[0],1e-9))
    for key in v:assert abs(v[key]-r['metrics'][key])<1e-5
    values.append(dict(v,group=members[k]))
  assert seen==set(data) and len(seen)==count
  for key in ['NE','SR','OSR','SPL']:assert abs(np.mean([v[key] for v in values])-m['metrics'][key])<1e-6
  for group in ['all','easy','medium','hard']:
   selected=[v for v in values if group=='all' or v['group']==group]
   rows.append(dict(split=split,group=group,episodes=len(selected),training_grid=384,inference_grid=384,checkpoint_sha256=freeze['checkpoint_sha256'],**{key:np.mean([v[key] for v in selected]) for key in ['NE','SR','OSR','SPL']},source=str(d/'metrics.json')))
  proof.append(dict(split=split,episodes=count,predictions_sha256=h.hexdigest(),dataset_sha256=file_sha256(dataset),metrics_sha256=file_sha256(d/'metrics.json')))
  for file in sorted((d/'fields').glob('*.npz')):
   z=np.load(file);assert z['world_xy'].shape==(147456,2)
   def softmax(v):
    v=v.astype(np.float64);v=np.exp(v-v.max());return v/v.sum()
   b0=softmax(z['static_logits']);bt=softmax(z['belief_logits']);diff=np.max(abs(z['static_logits']+z['visual_delta']-z['belief_logits']));assert diff<1e-4
   belief.append(dict(split=split,source=str(file),step=int(z['step']),L1_from_B0=float(abs(bt-b0).sum()),KL_Bt_B0=float((bt*(np.log(bt.clip(1e-300))-np.log(b0.clip(1e-300)))).sum()),top1_shift_m=float(np.linalg.norm(z['world_xy'][bt.argmax()]-z['world_xy'][b0.argmax()])),sha256=file_sha256(file)))
 for name,rs in [('RESULTS.csv',rows),('belief_changes.csv',belief)]:
  with (out/name).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 atomic_json(proof,out/'VERIFICATION.json')
 (out/'REPORT.md').write_text('# Completed 384 training experiment\n\nTrain grid 384, inference grid 384; initialized from the recorded 28-grid A3/head pretrained sources. Checkpoint frozen from C stage using Seen128 only. All 10,448 trajectories independently verified. RESULTS.csv rates are fractions. Raw RGB-containing fields are local-only. Old main results remain untouched.\n')
if __name__=='__main__':main()
