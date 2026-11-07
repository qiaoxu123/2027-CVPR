"""Full Val-Seen V2 Geometry only. Public text binding, original parser/rules intact.
Private destination coordinates are evaluated only AFTER frozen model forward.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.a3_runtime import public_static_record
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from scripts.unified_pgvb_v3_pilot import _load_base
from scripts.analyze_instruction_landmark_association import PublicNames
from graphnav.checkpointing import atomic_json,file_sha256

def spatial_nms(order,xy,radius=20,k=20):
 kept=[]
 for idx in order:
  if not kept or np.linalg.norm(xy[kept]-xy[idx],axis=1).min()>=radius:kept.append(int(idx))
  if len(kept)==k:break
 return kept

def main():
 p=argparse.ArgumentParser()
 for name in ['source-v2','trained-v2','data','sbf-source','output']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
 from sbfnav.coordinates import MapTransform
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
 ds=CityNavDataset(a.data,'val_seen',version='processed_citynav');assert len(ds)==2470
 objects=json.loads((a.data/'cityrefer/objects.json').read_text());names=PublicNames(objects);catalog=SimpleNamespace(objects=objects);geom,cfg=_load_base(a.source_v2,'cuda');geom.eval().requires_grad_(False)
 records=[];private=[];identities=[]
 for ep in ds:
  t=MapTransform.from_raster(a.data/'rgbd'/(ep.map_name+'.tif'),canvas_size=224,field_size=28);mentions=names.match(ep.map_name,ep.instruction);refnames=tuple(dict.fromkeys(n for m in mentions for n in m['public_names']));pub=SimpleNamespace(map_name=ep.map_name,instruction=ep.instruction,reference_names=refnames,references=());records.append(public_static_record(pub,t,catalog,max_landmarks=cfg.max_landmarks));private.append(ep.target_position[:2]);identities.append(dict(episode=list(ep.episode_id),map_name=ep.map_name))
 # No real target coordinates/identities in the dataset passed to model packing.
 path=a.output/'public_queries.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records));assert all(r['target_xy']==[0.,0.] for r in records)
 data=FullCandidateDataset(path,grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False)
 bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));outputs={}
 for label,ckpt in [('original_v2',a.source_v2),('trained_v2',a.trained_v2)]:
  if label=='trained_v2':
   state=torch.load(ckpt,map_location='cpu',weights_only=True);geo={k[len('geometry.'):]:v for k,v in state['model'].items() if k.startswith('geometry.')};geom.load_state_dict(geo,strict=True)
  rows=[];all_logits=[];all_xy=[];start=time.time();offset=0
  for batch in DataLoader(data,batch_size=16,shuffle=False,num_workers=0,collate_fn=collate_full):
   with torch.no_grad():
    obs,tokens,mask,node=_batch_inputs(batch,bert,'cuda','geometry')
    with torch.autocast('cuda',dtype=torch.bfloat16):field=geom(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=node)
   logits=field.belief_logits.float().cpu().numpy();grid=batch['grid_xy'].numpy();b=np.array([r['world_bounds'] for r in records[offset:offset+len(logits)]]);xy=b[:,None,:2]+grid*(b[:,None,2:]-b[:,None,:2]);all_logits.append(logits);all_xy.append(xy)
   for j,z in enumerate(logits):
    i=offset+j
    if not np.isfinite(z).all():raise ValueError('Nonfinite Geometry logits')
    # Retain V2 top_k prefix tie ordering; extend without changing first candidates.
    prefix=torch.tensor(z).topk(int(cfg.top_k)).indices.tolist();order=prefix+[int(x) for x in np.argsort(-z,kind='stable') if x not in prefix];nms=spatial_nms(order,xy[j]);target=np.asarray(private[i]);dist=np.linalg.norm(xy[j]-target,axis=1);rank=next((k+1 for k,idx in enumerate(order) if dist[idx]<=20),None)
    rows.append(dict(index=i,**identities[i],NE=float(dist[order[0]]),raw_hits={str(k):bool(dist[order[:k]].min()<=20) for k in [1,5,10,20]},nms_hits={str(k):bool(dist[nms[:k]].min()<=20) if nms else False for k in [1,5,10,20]},first_correct_rank=rank,top20_independent_count=len(spatial_nms(order[:20],xy[j])),raw_top20=order[:20],nms_top20=nms,public_anchor_n=len(records[i]['landmarks'])))
   offset+=len(logits)
   if offset%160==0:print(label,offset,flush=True)
  assert len(rows)==2470;atomic_json(rows,a.output/(label+'_rows.json'));np.savez_compressed(a.output/(label+'_public_fields.npz'),logits=np.concatenate(all_logits),xy=np.concatenate(all_xy));outputs[label]=dict(n=2470,NE=float(np.mean([r['NE'] for r in rows])),raw={k:sum(r['raw_hits'][k] for r in rows)/2470 for k in ['1','5','10','20']},spatial_nms={k:sum(r['nms_hits'][k] for r in rows)/2470 for k in ['1','5','10','20']},mean_distinct_positions_in_raw20=float(np.mean([r['top20_independent_count'] for r in rows])),no_public_landmarks=sum(not r['public_anchor_n'] for r in rows),elapsed_s=time.time()-start)
  print(label,json.dumps(outputs[label]),flush=True)
 atomic_json(dict(metrics=outputs,grid_size=28,nms_radius_m=20,success_radius_m=20,seed=42,source_v2_sha256=file_sha256(a.source_v2),trained_v2_sha256=file_sha256(a.trained_v2),annotation_sha256=file_sha256(ds.annotation_path),catalog_sha256=file_sha256(a.data/'cityrefer/objects.json'),input='Original instruction/public names only; original V2 rules+relation fields; no OSM, RGB, reference annotation IDs, target or trajectory in forward',historical_protocol_difference='Original runtime used annotation reference-name metadata; current public blind binding is explicit',data_identity='local Revised CityNav; HETT official identity verification pending',scope='offline initial localization; not navigation SR'),a.output/'summary.json')
if __name__=='__main__':main()
