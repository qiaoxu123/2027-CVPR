"""Real TRAIN text-selected angular gradient check, independent of random smoke coverage."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from graphnav.unified_direct_bearing_loss import DirectBearingDataset,angular_consistency_loss
from graphnav.candidate_geometry import collate_full
from graphnav.full_relation_training import load_full_relation_model,full_relation_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.static_localization import cross_entropy_goal
from graphnav.checkpointing import atomic_json,file_sha256


def main():
 p=argparse.ArgumentParser()
 for n in ('source','original-v2','public-objects','bert-cache','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 state=torch.load(a.original_v2,map_location='cpu',weights_only=True);c=state['config']['geometry_args']
 ds=DirectBearingDataset(a.source,public_objects=a.public_objects,deduplicate_queries=False,grid_size=28,max_landmarks=c['max_landmarks'],max_roads=c['max_roads'],max_relations=c['max_relations'],prior_sigma_m=c['prior_sigma_m'],rule_sigma_m=c['rule_sigma_m'])
 ids=[i for i,(_,j) in ds._predicates.items() if j>=0][:2]
 if not ids:raise RuntimeError('No public-text eligible TRAIN queries')
 model=load_full_relation_model(state,'cuda').train();model.requires_grad_(False);model.geometry.requires_grad_(True);model.full_relation.requires_grad_(True)
 bert=FrozenBERTTokens(c['text_backbone'],device='cuda',max_length=c['max_text_length']);bert.load_cache(a.bert_cache)
 opt=torch.optim.AdamW([dict(params=model.geometry.parameters(),lr=1e-5),dict(params=model.full_relation.parameters(),lr=1e-4)],weight_decay=.001)
 batch=collate_full([ds[i] for i in ids]);results=[]
 for iteration in range(2):
  model.reset_episode();x=full_relation_inputs(batch,bert,'cuda');assert 'target_xy' not in x['static_inputs'] and 'angular_anchor_index' not in x['static_inputs']
  out=model.forward_full_relation(**x);z=out['field'].belief_logits
  loss=angular_consistency_loss(z,x['static_inputs']['pair_geometry'],batch['target_xy'].cuda(),x['static_inputs']['nodes'],batch['map_sizes'].cuda(),batch['angular_anchor_index'].cuda())
  if not loss['eligible_count']:raise RuntimeError('Text-selected TRAIN angles undefined')
  stats={}
  for name,module in [('Geometry',model.geometry),('Relation',model.full_relation)]:
   grads=torch.autograd.grad(loss['loss'],list(module.parameters()),retain_graph=True,allow_unused=True);v=sum(float(g.abs().sum()) for g in grads if g is not None);assert np.isfinite(v) and v>0;stats[name]=v
  predgrad=torch.autograd.grad(loss['loss'],model.full_relation.predicate.weight,retain_graph=True,allow_unused=True)[0]
  stats['predicate_embedding_gradient_L1']=float(predgrad.abs().sum()) if predgrad is not None else 0.
  opt.zero_grad(set_to_none=True);total=cross_entropy_goal(z,batch['target_xy'].cuda())+.1*loss['loss'];total.backward();torch.nn.utils.clip_grad_norm_(list(model.geometry.parameters())+list(model.full_relation.parameters()),5.,error_if_nonfinite=True);opt.step()
  results.append(dict(iteration=iteration,angular_loss=float(loss['loss'].detach()),angular_only_gradients=stats,eligible=int(loss['eligible_count'])))
 assert results[-1]['angular_only_gradients']['predicate_embedding_gradient_L1']>0
 atomic_json(dict(TRAIN_indices=ids,selection='first two PUBLIC TEXT eligible, no GT filtering',rows=results,original_checkpoint_unchanged_sha256=file_sha256(a.original_v2),note='Predicate gradient may start zero because the original relation score is zero-initialized; second update must reach it.'),a.output)
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
