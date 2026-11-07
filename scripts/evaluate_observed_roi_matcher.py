"""Complete TRAIN split evaluation with sensor-only abstention and shuffled RGB control."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from graphnav.appearance_matching import MultiScaleAppearanceMatcher
from scripts.train_observed_roi_matcher import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['development','confirmation'],default='development');a=p.parse_args();torch.set_num_threads(4)
 manifest=json.loads((a.cache/'manifest.json').read_text());summary=json.loads((a.cache/'summary.json').read_text())
 if manifest['n']!=21878 or summary['n']!=21878:raise ValueError('Incomplete cache')
 payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=MultiScaleAppearanceMatcher().cuda().eval();model.load_state_dict(payload['model']);labels=json.loads(a.labels.read_text());records=[]
 with torch.no_grad():
  for label in labels:
   if label['split']!=a.split:continue
   i=label['index'];z=np.load(a.cache/'public_features'/f'{i:05d}.npz');valid=torch.tensor(z['valid']);known=valid.any(-1);choices={'geometry':0,'rgb':0,'shuffle':0}
   # No GT determines eligibility, candidate selection, or shuffle.
   if bool(known[0]) and int(known.sum())>=2:
    images=torch.tensor(z['features'][0].astype('float32'));row=dict(images=images,valid=valid,text=torch.tensor(z['text_features'].astype('float32')).mean(0))
    choices['rgb']=int(score(model,row,'cuda').masked_fill(~known.cuda(),-torch.inf).argmax())
    ids=known.nonzero().flatten();rng=np.random.default_rng(42+i);permuted=ids[torch.tensor(rng.permutation(len(ids)))];shuffled=images.clone();shuffled[ids]=images[permuted];row['images']=shuffled
    choices['shuffle']=int(score(model,row,'cuda').masked_fill(~known.cuda(),-torch.inf).argmax())
   # All private scoring happens after public decisions have been sealed.
   distance=np.linalg.norm(z['candidate_world']-label['target_world'],axis=1)
   records.append(dict(index=i,supported=int(known.sum()),choice=choices,NE={k:float(distance[v]) for k,v in choices.items()},hit={k:bool(distance[v]<=20) for k,v in choices.items()}))
 result={}
 for arm in ['geometry','rgb','shuffle']:
  result[arm]=dict(n=len(records),R20=float(np.mean([r['hit'][arm] for r in records])),NE=float(np.mean([r['NE'][arm] for r in records])),corrected=sum(r['hit'][arm] and not r['hit']['geometry'] for r in records),damaged=sum(r['hit']['geometry'] and not r['hit'][arm] for r in records),switches=sum(r['choice'][arm]!=0 for r in records))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(split=a.split,metrics=result,rows=records,scope='Complete TRAIN split, NMS candidate-centre 20m location proxies; no instance or navigation SR claim'),indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
