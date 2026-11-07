"""TRAIN development only: frozen Geometry, exact/alias/name-variant binding.
Birmingham confirmation is NOT rescored for this new parser experiment.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.a3_runtime import public_static_record
from graphnav.candidate_geometry import collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.public_landmark_mentions import PublicLandmarkMentions,add_mention_aliases
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.train_v2_top1_geometry import Indexed
from scripts.unified_pgvb_v3_pilot import _load_base


def main():
 p=argparse.ArgumentParser()
 for k in ['data','catalog','old-train','source-v2','trained-v2','bert-cache','output']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--arms',nargs='+',choices=['original_public','literal_aliases','name_variants'],default=['original_public','literal_aliases','name_variants']);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);labels=json.load(open(a.data/'labels_private.json'));indices=[x['index'] for x in labels if x['split']=='development'];assert len(indices)==3126 and all(labels[i]['city']=='cambridge' for i in indices);source=[json.loads(s) for s in (a.data/'public_train.jsonl').read_text().splitlines()];old=[json.loads(s) for s in a.old_train.read_text().splitlines()];objects=json.load(open(a.catalog));catalog=SimpleNamespace(objects=objects);matchers={'literal_aliases':PublicNames(objects),'name_variants':PublicLandmarkMentions(objects)}
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model,cfg=_load_base(a.source_v2,'cuda');state=torch.load(a.trained_v2,map_location='cpu',weights_only=True);model.load_state_dict({k[len('geometry.'):]:v for k,v in state['model'].items() if k.startswith('geometry.')},strict=True);model.eval().requires_grad_(False);bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));bert.load_cache(a.bert_cache);outputs={}
 for arm in a.arms:
  queries=[];traces=[]
  for i in indices:
   base=source[i];r=json.loads(json.dumps(base));mentions=[]
   if arm!='original_public':
    mentions=matchers[arm].match(base['map_name'],base['instruction']);refs=tuple(dict.fromkeys(n for m in mentions for n in m['public_names']));bounds=base['world_bounds'];transform=SimpleNamespace(x_min=bounds[0],y_min=bounds[1],x_max=bounds[2],y_max=bounds[3]);episode=SimpleNamespace(map_name=base['map_name'],instruction=base['instruction'],reference_names=refs,references=());r=public_static_record(episode,transform,catalog,max_landmarks=cfg.max_landmarks);r=add_mention_aliases(r,mentions)
   queries.append(r);traces.append(dict(index=i,instruction=r['instruction'],map_name=r['map_name'],landmarks=r['landmarks'],mentions=mentions))
  path=a.output/(arm+'_public.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in queries));ds=Indexed(path,grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False);rows=[]
  with torch.no_grad():
   for batch in DataLoader(ds,batch_size=32,shuffle=False,num_workers=0,collate_fn=collate_full):
    obs,tokens,mask,nodes=_batch_inputs(batch,bert,'cuda','geometry');z=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes).belief_logits.float();pos=batch['row_index'].tolist();ii=[indices[j] for j in pos];bounds=torch.tensor([source[i]['world_bounds'] for i in ii],device='cuda');target=torch.tensor([labels[i]['target_world'] for i in ii],device='cuda');pred=obs['grid_xy'][torch.arange(len(z),device='cuda'),z.argmax(-1)];world=bounds[:,:2]+pred*(bounds[:,2:]-bounds[:,:2]);dist=(world-target).norm(dim=-1).cpu().tolist();rows.extend(dict(index=i,NE=e,hit20=e<=20,predicted_world=xy) for i,e,xy in zip(ii,dist,world.cpu().tolist()))
  tp=fp=fn=0
  for i,q in zip(indices,queries):
   names={x['name'] for x in q['landmarks']};weak={x['name'] for x in old[i]['landmarks']};tp+=len(names&weak);fp+=len(names-weak);fn+=len(weak-names)
  outputs[arm]=dict(n=len(rows),R20=float(np.mean([r['hit20'] for r in rows])),NE=float(np.mean([r['NE'] for r in rows])),no_anchor=sum(not q['landmarks'] for q in queries),weak_metadata_name_precision=tp/max(tp+fp,1),weak_metadata_name_recall=tp/max(tp+fn,1),name_tp=tp,name_fp=fp,name_fn=fn)
  atomic_json(rows,a.output/(arm+'_rows.json'));atomic_json(traces,a.output/(arm+'_traces.json'));print(arm,json.dumps(outputs[arm]),flush=True);del ds
 atomic_json(dict(arms=outputs,split='Cambridge TRAIN development only; no new Birmingham/Val/Test scoring',frozen_model_sha256=file_sha256(a.trained_v2),fuzzy=dict(threshold=.92,margin=.05),parser_source_sha256=file_sha256(Path('graphnav/public_landmark_mentions.py')),lexical_note='Reference metadata is weak TRAIN comparison, not manually verified parser GT'),a.output/'metrics.json')
if __name__=='__main__':main()
