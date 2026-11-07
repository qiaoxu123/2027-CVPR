"""TRAIN-only weak location supervision for observed ROI matching, never instance gold."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from graphnav.appearance_matching import MultiScaleAppearanceMatcher

def load_rows(cache,labels,split):
 rows=[]
 for record in labels:
  if record['split']!=split:continue
  path=cache/'public_features'/f"{record['index']:05d}.npz"
  if not path.exists():continue
  z=np.load(path);valid=z['valid'];known=valid.any(-1)
  if known.sum()<2:continue
  distance=np.linalg.norm(z['candidate_world']-record['target_world'],axis=-1)
  # Ambiguous 20-40m band is excluded from this weak location loss.
  positive=known&(distance<=20);negative=known&(distance>=40)
  if not positive.any() or not negative.any():continue
  rows.append(dict(index=record['index'],images=torch.tensor(z['features'][0].astype('float32')),valid=torch.tensor(valid),text=torch.tensor(z['text_features'].astype('float32')).mean(0),positive=torch.tensor(positive),eligible=torch.tensor(positive|negative)))
 return rows

def score(model,row,device):
 text=row['text'].to(device);image=row['images'].to(device);valid=row['valid'].to(device)
 language=torch.nn.functional.normalize(text+.1*model.text_residual(text),dim=-1)
 image=torch.nn.functional.normalize(image+.1*model.image_residual(image),dim=-1)
 return ((image*language).sum(-1)*valid).sum(-1)/valid.sum(-1).clamp_min(1)*model.log_scale.exp().clamp(max=100)

def evaluate(model,rows,device):
 records=[]
 with torch.no_grad():
  for r in rows:
   s=score(model,r,device).masked_fill(~r['valid'].any(-1).to(device),-torch.inf);j=int(s.argmax());records.append(dict(index=r['index'],hit=bool(r['positive'][j]),selected=j))
 return dict(n=len(records),correct=sum(r['hit'] for r in records),accuracy=sum(r['hit'] for r in records)/max(1,len(records)),rows=records)

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=3);a=p.parse_args()
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4);device='cuda';a.output.mkdir(parents=True,exist_ok=True)
 labels=json.loads(a.labels.read_text());fit=load_rows(a.cache,labels,'fit');dev=load_rows(a.cache,labels,'development')
 if not fit or not dev:raise ValueError('No supported positive/negative fit or development rows')
 model=MultiScaleAppearanceMatcher().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);history=[];best=-1
 baseline=evaluate(model,dev,device)
 for epoch in range(a.epochs):
  model.train();random.shuffle(fit);losses=[]
  for r in fit:
   optimizer.zero_grad();scores=score(model,r,device).masked_fill(~r['eligible'].to(device),-1e4);target=r['positive'].to(device).float();target=target/target.sum();loss=-(target*scores.log_softmax(-1)).sum()
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
  model.eval();metrics=evaluate(model,dev,device);history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),development=metrics));payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history,seed=42,scope='weak 20m location proxy; 20-40m ignored; current384 real start observations')
  torch.save(payload,a.output/'last.pt')
  if metrics['accuracy']>best:best=metrics['accuracy'];torch.save(payload,a.output/'best.pt')
  print(json.dumps(dict(epoch=epoch+1,fit=len(fit),dev=len(dev),loss=history[-1]['loss'],accuracy=metrics['accuracy'])),flush=True)
 (a.output/'development.json').write_text(json.dumps(dict(fit=len(fit),baseline=baseline,history=history),indent=2)+'\n')
if __name__=='__main__':main()
