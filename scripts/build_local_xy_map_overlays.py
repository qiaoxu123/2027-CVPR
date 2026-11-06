"""Frozen TRAIN map overlays. Private targets are read only after public Geometry prediction."""
import json,glob,argparse
from pathlib import Path
import torch,numpy as np
from scripts.unified_pgvb_v3_pilot import _load_base,V3CacheDataset
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source-v2',type=Path,required=True);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
g,_=_load_base(a.source_v2,'cuda');ck=torch.load(a.run/'train/last.pt',map_location='cpu',weights_only=True);g.load_state_dict({k.removeprefix('geometry.'):v for k,v in ck['model'].items() if k.startswith('geometry.')},strict=True);g.eval();ds=V3CacheDataset(a.run/'cache');official=json.load(open(a.data/'processed_citynav/citynav_train_seen.json'));seen=set()
for f in sorted((a.diagnosis/'full/rows').glob('*.json')):
 r=json.load(open(f));m=r['map_name']
 if m in seen:continue
 public=torch.load(ds.root/'public'/f'{ds.keys[r["index"]]}.pt',weights_only=True);s={k:v[None].cuda() for k,v in public['static'].items()};s['rules'].zero_();s['relation_fields'].zero_()
 with torch.no_grad():z=g(s,variant='geometry',tokens=public['bert_tokens'].float()[None].cuda(),token_mask=public['bert_mask'][None].cuda(),node_emb=public['landmark_embeddings'].float()[None].cuda()).belief_logits[0].float().cpu()
 b=public['world_bounds'].numpy();order=torch.argsort(z,descending=True,stable=True).tolist();c=[]
 for i in order:
  xy=b[:2]+np.array([(i%28+.5)/28,(i//28+.5)/28])*(b[2:]-b[:2])
  if all(np.linalg.norm(xy-np.array(v['xy']))>=20 for v in c):c.append(dict(xy=xy.tolist(),logit=float(z[i]),index=i))
  if len(c)==20:break
 private=torch.load(ds.root/'private'/f'{ds.keys[r["index"]]}.pt',weights_only=True);target=(private['target_xy'].numpy()*(b[2:]-b[:2])+b[:2]).tolist();parts=r['episode_id'].split(':');obj,ann=int(parts[1]),int(parts[2]);match=[v for v in official if f"{v['area']}_block_{v['block']}"==m and obj in v['object_ids'] and ann in v['ann_ids']];trajectory=[v[:2] for v in match[0]['trajectory']] if match else []
 out=dict(instruction=r['instruction'],episode_id=r['episode_id'],heatmap=z.reshape(28,28).tolist(),candidates=c,nms_radius_m=20,trajectory=trajectory,trajectory_source='official TRAIN reference, diagnosis only',footprint=None,target=target,private_target_posthoc=True,geometry='V3 frozen Geometry: zero rule fields, not original V2 full pipeline')
 (a.output/(m+'.json')).write_text(json.dumps(out));seen.add(m);print(m,flush=True)
