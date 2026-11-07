"""One full Val-Seen test of a TRAIN-selected public name binding fix.
No training, GT binding, loss/NMS tuning, Test or closed loop in this runner.
"""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.a3_runtime import public_static_record
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.public_landmark_mentions import PublicLandmarkMentions,add_mention_aliases
from scripts.unified_pgvb_v3_pilot import _load_base
from scripts.evaluate_v2_public_candidate_recall import spatial_nms


def main():
 p=argparse.ArgumentParser()
 for k in ['data-root','sbf-source','initial-eval','source-v2','trained-v2','development-baseline','development-selected','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dev0=json.load(open(a.development_baseline))['arms']['original_public'];dev=json.load(open(a.development_selected));newdev=dev['arms']['name_variants'];parser_sha=file_sha256(Path('graphnav/public_landmark_mentions.py'))
 if dev['parser_source_sha256']!=parser_sha:raise ValueError('Selected parser changed since TRAIN evaluation')
 if newdev['R20']<=dev0['R20'] or newdev['NE']>=dev0['NE']:raise ValueError('TRAIN development did not improve R20 and NE')
 old=json.load(open(a.initial_eval/'summary.json'));seal=dict(objective='Full2470 Top1 within20m strictly over50%',checkpoint_sha256=file_sha256(a.trained_v2),source_v2_sha256=file_sha256(a.source_v2),parser_sha256=parser_sha,runner_sha256=file_sha256(__file__),development_baseline_sha256=file_sha256(a.development_baseline),development_selected_sha256=file_sha256(a.development_selected),selected_before_Val=True,precision='FP32 batch1 TF32off, identical for both arms',source_annotations_sha256=old['annotation_sha256'],catalog_sha256=old['catalog_sha256'],seed=42,grid=28,success_m=20,nms_m=20,expected=2470,training=False,weights_modified=False,GT_usage='Scoring AFTER frozen forward only; no ID/coordinate in public binding',fuzzy=dict(threshold=.92,margin=.05),scope='Original public binding vs scoped public name variants+literal aliases; original instruction unchanged')
 assert seal['checkpoint_sha256']==old['trained_v2_sha256']
 dest=a.output/'FROZEN_BEFORE_VAL.json'
 if dest.exists() and json.load(open(dest))!=seal:raise ValueError('Refuse changed experiment after Val consumed')
 atomic_json(seal,dest)
 if (a.output/'metrics.json').exists():print('Frozen evaluation already complete');return
 source=[json.loads(s) for s in (a.initial_eval/'public_queries.jsonl').read_text().splitlines()];assert len(source)==2470 and all(r['target_xy']==[0.,0.] for r in source)
 objects=json.load(open(a.data_root/'cityrefer/objects.json'));assert file_sha256(a.data_root/'cityrefer/objects.json')==seal['catalog_sha256'];catalog=SimpleNamespace(objects=objects);matcher=PublicLandmarkMentions(objects)
 sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
 ds=CityNavDataset(a.data_root,'val_seen',version='processed_citynav');assert len(ds)==2470 and file_sha256(ds.annotation_path)==seal['source_annotations_sha256']
 torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');model,cfg=_load_base(a.source_v2,'cuda');state=torch.load(a.trained_v2,map_location='cpu',weights_only=True);model.load_state_dict({k[len('geometry.'):]:v for k,v in state['model'].items() if k.startswith('geometry.')},strict=True);model.eval().requires_grad_(False);bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));results={}
 for arm in ['baseline','scoped_public_names']:
  path=a.output/(arm+'_public.jsonl');queries=[];traces=[]
  for i,q in enumerate(source):
   if arm=='baseline':r=q;mentions=[]
   else:
    mentions=matcher.match(q['map_name'],q['instruction']);refs=tuple(dict.fromkeys(n for m in mentions for n in m['public_names']));b=q['world_bounds'];t=SimpleNamespace(x_min=b[0],y_min=b[1],x_max=b[2],y_max=b[3]);ep=SimpleNamespace(map_name=q['map_name'],instruction=q['instruction'],reference_names=refs,references=());r=add_mention_aliases(public_static_record(ep,t,catalog,max_landmarks=cfg.max_landmarks),mentions)
   assert r['target_xy']==[0.,0.];queries.append(r);traces.append(dict(index=i,map_name=r['map_name'],instruction=r['instruction'],mentions=mentions,landmarks=r['landmarks']))
  path.write_text(''.join(json.dumps(r)+'\n' for r in queries));data=FullCandidateDataset(path,grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False);rows=[];start=time.perf_counter()
  with torch.no_grad():
   for i,batch in enumerate(DataLoader(data,batch_size=1,shuffle=False,num_workers=0,collate_fn=collate_full)):
    obs,tokens,mask,nodes=_batch_inputs(batch,bert,'cuda','geometry');field=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes);z=field.belief_logits.float()[0].cpu().numpy();bounds=np.array(queries[i]['world_bounds']);world=bounds[:2]+batch['grid_xy'][0].numpy()*(bounds[2:]-bounds[:2]);prefix=field.candidate_ids[0].cpu().tolist();order=prefix+[int(x) for x in np.argsort(-z,kind='stable') if x not in prefix];nms=spatial_nms(order,world,radius=20,k=20)
    # Only now score against the private endpoint. No private object IDs or
    # endpoint information was used in references, aliases, ranking or NMS.
    target=np.array(ds[i].target_position[:2]);d=np.linalg.norm(world-target,axis=1);rows.append(dict(index=i,episode=list(ds[i].episode_id),map_name=queries[i]['map_name'],NE=float(d[order[0]]),raw={str(k):bool(d[order[:k]].min()<=20) for k in [1,5,10,20]},nms={str(k):bool(d[nms[:k]].min()<=20) for k in [1,5,10,20]},top20_world=world[order[:20]].tolist(),top20_logits=z[order[:20]].tolist(),no_anchor=not queries[i]['landmarks']))
    if (i+1)%200==0:print(json.dumps(dict(arm=arm,completed=i+1,expected=2470)),flush=True)
  assert len(rows)==2470;atomic_json(rows,a.output/(arm+'_rows.json'));atomic_json(traces,a.output/(arm+'_traces.json'));results[arm]=dict(n=2470,NE=float(np.mean([r['NE'] for r in rows])),raw={str(k):sum(r['raw'][str(k)] for r in rows)/2470 for k in [1,5,10,20]},nms={str(k):sum(r['nms'][str(k)] for r in rows)/2470 for k in [1,5,10,20]},no_anchor=sum(r['no_anchor'] for r in rows),wall_s=time.perf_counter()-start);print(arm,json.dumps(results[arm]),flush=True);del data
 b=json.load(open(a.output/'baseline_rows.json'));n=json.load(open(a.output/'scoped_public_names_rows.json'));corr=sum(y['raw']['1'] and not x['raw']['1'] for x,y in zip(b,n));damage=sum(x['raw']['1'] and not y['raw']['1'] for x,y in zip(b,n));atomic_json(dict(coverage=dict(completed=2470,expected=2470),arms=results,corrected=corr,damaged=damage,net=corr-damage,goal_verified=results['scoped_public_names']['raw']['1']>.5,freeze=seal),a.output/'metrics.json')
if __name__=='__main__':main()
