"""Batch-transfer evaluation: identical prediction/metric definitions, no GPU scalar sync loop."""
import math
import numpy as np
import torch
from scripts.train_role_polygon_bearing import predict


@torch.no_grad()
def evaluate_cpu_metrics(model,cache,ids,labels,arm,batch_size=16,predict_fn=None):
 if model is not None:model.eval()
 rows=[]
 for start in range(0,len(ids),batch_size):
  indices=ids[start:start+batch_size];batch,old=cache.build(indices);out=None if predict_fn is not None else predict(model,batch,old,arm);p=(predict_fn(batch,old,indices) if predict_fn is not None else out['probabilities']).float().cpu();obj=(out['object_probabilities'].argmax(-1).cpu() if out is not None and 'object_probabilities' in out else None);pred=p.argmax(-1);best2=p.topk(2,-1).indices
  confidence=p.max(-1).values;entropy=-(p*p.clamp_min(1e-9).log()).sum(-1)
  for k,i in enumerate(indices):
   q=cache.data['queries'][i];y=labels[i]['bins'];theta=labels[i]['theta'];error=pred[k].float()*math.pi/4-theta;error=torch.atan2(error.sin(),error.cos()).abs()*180/math.pi
   for j in torch.nonzero(y>=0).flatten().tolist():
    rows.append(dict(index=i,anchor=j,true_bin=int(y[j]),pred=int(pred[k,j]),top1=int(pred[k,j]==y[j]),top2=int((best2[k,j]==y[j]).any()),error=float(error[j]),NLL=float(-p[k,j,y[j]].clamp_min(1e-9).log()),probabilities=p[k,j].tolist(),confidence=float(confidence[k,j]),entropy=float(entropy[k,j]),object_pred=int(obj[k,j]) if obj is not None else None,target_index_offline=labels[i]['target_index'],category=q['category'],n_refs=int(q['anchor_valid'].sum()),physical_id=labels[i]['physical_id']))
 groups={}
 for v in rows:groups.setdefault(v['index'],[]).append(v)
 summary=dict(queries=len(groups),anchor_pairs=len(rows),macro_Top1=float(np.mean([np.mean([x['top1'] for x in v]) for v in groups.values()])),Top1=float(np.mean([v['top1'] for v in rows])),Top2=float(np.mean([v['top2'] for v in rows])),MAE=float(np.mean([v['error'] for v in rows])),median=float(np.median([v['error'] for v in rows])),NLL=float(np.mean([v['NLL'] for v in rows])))
 return summary,rows
