"""Frozen Stage1 appearance residual probe; quality gates disabled for equal-scale comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from graphnav.appearance_matching import MultiScaleAppearanceMatcher

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 state=torch.load(a.stage1,map_location='cpu',weights_only=True);model=MultiScaleAppearanceMatcher().cuda().eval();model.load_state_dict({k[8:]:v for k,v in state['model'].items() if k.startswith('matcher.')})
 labels=json.loads(a.labels.read_text());rows=[]
 with torch.no_grad():
  for path in sorted((a.cache/'public_features').glob('*.npz')):
   i=int(path.stem);z=np.load(path);valid=z['valid'];known=valid.any(-1);prediction={}
   for res in range(2):
    images=torch.tensor(z['features'][res].astype('float32'),device='cuda');v=torch.tensor(valid,device='cuda');text=torch.tensor(z['text_features'].astype('float32'),device='cuda')
    for q in range(2):
     language=torch.nn.functional.normalize(text[q]+.1*model.text_residual(text[q]),dim=-1)
     visual=torch.nn.functional.normalize(images+.1*model.image_residual(images),dim=-1)
     score=((visual*language).sum(-1)*v).sum(-1)/v.sum(-1).clamp_min(1)
     score=score.masked_fill(~v.any(-1),-torch.inf);prediction[f'{res}_{q}']=int(score.argmax()) if known.any() else None
   # GT consulted only after model scoring.
   dist=np.linalg.norm(z['candidate_world']-np.array(labels[i]['target_world']),axis=1)
   if known.sum()>=2 and (known&(dist<=20)).any():
    first=int(np.where(known)[0][0]);rows.append(dict(index=i,geometry_hit=bool(dist[first]<=20),adapter_hits={k:bool(dist[s]<=20) for k,s in prediction.items()}))
 out=dict(n=len(rows),geometry_hits=sum(r['geometry_hit'] for r in rows),adapter_hits={k:sum(r['adapter_hits'][k] for r in rows) for k in ['0_0','0_1','1_0','1_1']},rows=rows,scope='TRAIN 20m location proxy; frozen learned residuals only; equal scale weights, quality gate disabled; not deployed Stage1 or instance accuracy')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
