"""One-shot Birmingham TRAIN confirmation after freezing development-selected model."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.candidate_geometry import collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from scripts.unified_pgvb_v3_pilot import _load_base
from scripts.train_v2_top1_geometry import Indexed


def main():
 p=argparse.ArgumentParser()
 for k in ['data','source-v2','trained-v2','bert-cache','training','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
 complete=json.load(open(a.training/'summary.json'));assert not complete['smoke'] and len(complete['arms'])==3
 mode=max(complete['arms'],key=lambda k:(complete['arms'][k]['best_development']['R20'],-complete['arms'][k]['best_development']['NE']));result=complete['arms'][mode];baseline=result['baseline'];candidate=a.training/mode/'best.pt'
 if not candidate.exists() or result['best_development']['R20']<=baseline['R20']:
  atomic_json(dict(expand=False,reason='No development R20 improvement; confirmation not consumed',development=complete),a.output/'negative_development.json');return
 seal=dict(mode=mode,checkpoint_sha256=file_sha256(candidate),training_summary_sha256=file_sha256(a.training/'summary.json'),selection_metric='Cambridge development only',confirmation_not_seen_at_selection=True)
 sealpath=a.output/'selected_before_confirmation.json'
 if sealpath.exists():
  if json.load(open(sealpath))!=seal:raise ValueError('Refuse reselection after confirmation')
  if (a.output/'metrics.json').exists():print('Confirmation already sealed and scored');return
 atomic_json(seal,sealpath)
 model,cfg=_load_base(a.source_v2,'cuda');source=torch.load(a.trained_v2,map_location='cpu',weights_only=True);base={k[len('geometry.'):]:v for k,v in source['model'].items() if k.startswith('geometry.')};adapt=torch.load(candidate,map_location='cpu',weights_only=True)['geometry'];labels=json.load(open(a.data/'labels_private.json'));ids=[r['index'] for r in labels if r['split']=='confirmation'];assert len(ids)==7619 and all(labels[i]['city']=='birmingham' for i in ids)
 ds=Indexed(a.data/'public_train.jsonl',grid_size=28,max_landmarks=cfg.max_landmarks,max_roads=cfg.max_roads,max_relations=cfg.max_relations,prior_sigma_m=cfg.prior_sigma_m,rule_sigma_m=cfg.rule_sigma_m,deduplicate_queries=False);bert=FrozenBERTTokens(cfg.text_backbone,device='cuda',max_length=getattr(cfg,'max_text_length',96));bert.load_cache(a.bert_cache);outputs={}
 with torch.no_grad():
  for arm,weights in [('baseline',base),('selected',adapt)]:
   model.load_state_dict(weights,strict=True);model.eval().requires_grad_(False);rows=[]
   for batch in DataLoader(ds,batch_size=32,sampler=ids,num_workers=0,collate_fn=collate_full):
    obs,tokens,mask,nodes=_batch_inputs(batch,bert,'cuda','geometry');z=model(obs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=nodes).belief_logits.float();ii=batch['row_index'].tolist();b=torch.tensor([ds.records[i]['world_bounds'] for i in ii],device='cuda');target=torch.tensor([labels[i]['target_world'] for i in ii],device='cuda');points=obs['grid_xy'][torch.arange(len(z),device='cuda'),z.argmax(-1)];xy=b[:,:2]+points*(b[:,2:]-b[:,:2]);error=(xy-target).norm(dim=-1).cpu().tolist();rows.extend(dict(index=i,NE=e,hit20=e<=20) for i,e in zip(ii,error))
   atomic_json(rows,a.output/(arm+'_rows.json'));outputs[arm]=dict(n=len(rows),R20=float(np.mean([r['hit20'] for r in rows])),NE=float(np.mean([r['NE'] for r in rows])))
 baseline_rows=json.load(open(a.output/'baseline_rows.json'));selected_rows=json.load(open(a.output/'selected_rows.json'));corrected=sum(b['hit20'] and not r['hit20'] for r,b in zip(baseline_rows,selected_rows));damaged=sum(r['hit20'] and not b['hit20'] for r,b in zip(baseline_rows,selected_rows));summary=dict(arms=outputs,corrected=corrected,damaged=damaged,net=corrected-damaged,selection=seal,scope='TRAIN adaptation confirmation; not formal Val/Test; base pretrained on official TRAIN',new_training_warranted=outputs['selected']['R20']>outputs['baseline']['R20'] and outputs['selected']['NE']<=outputs['baseline']['NE']);atomic_json(summary,a.output/'metrics.json');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
