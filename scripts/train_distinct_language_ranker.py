"""Paired context-only/full-token ranking on frozen spatial NMS candidates."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from graphnav.distinct_candidate_language import DistinctCandidateLanguageRanker

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--bert-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
 labels=json.loads((a.root/'scoped_data/labels_private.json').read_text());public=[json.loads(s) for s in (a.root/'scoped_data/public_train.jsonl').read_text().splitlines()];bert=torch.load(a.bert_cache,map_location='cpu',weights_only=True);parts={}
 for split in ['fit','development']:
  rows=[]
  for l in labels:
   if l['split']!=split:continue
   i=l['index'];z=np.load(a.cache/'public_features'/f'{i:05d}.npz');world=z['candidate_world'];grid=z['base_world_grid'];xy=(world-grid.min(0))/(grid.max(0)-grid.min(0));t,m=bert['cache'][public[i]['instruction']];tokens=torch.zeros(96,768,dtype=torch.float16);mask=torch.zeros(96,dtype=torch.bool);tokens[:len(t)]=t.half();mask[:len(m)]=m;dist=np.linalg.norm(world-l['target_world'],axis=1);target=(dist<=20).astype('float32')
   if target[0] or not target.any():target[:]=0;target[0]=1
   else:target/=target.sum()
   rows.append((z['candidate_context'].astype('float32'),xy,z['geometry_logits'],tokens,mask,dist,target))
  parts[split]=dict(c=torch.tensor(np.stack([r[0] for r in rows]),device='cuda'),xy=torch.tensor(np.stack([r[1] for r in rows]),device='cuda'),g=torch.tensor(np.stack([r[2] for r in rows]),device='cuda'),t=torch.stack([r[3] for r in rows]),m=torch.stack([r[4] for r in rows]),dist=torch.tensor(np.stack([r[5] for r in rows]),device='cuda'),y=torch.tensor(np.stack([r[6] for r in rows]),device='cuda'))
 del bert
 def infer(model,d,ids,arm):
  tokens=d['t'][ids.cpu()].to('cuda').float();mask=d['m'][ids.cpu()].cuda()
  if arm=='context':tokens=torch.zeros_like(tokens)
  return model(d['c'][ids],d['xy'][ids],d['g'][ids],tokens,mask)
 def evaluate(model,arm):
  d=parts['development'];chosen=[]
  with torch.no_grad():
   for ids in torch.arange(len(d['g']),device='cuda').split(128):chosen.append(infer(model,d,ids,arm).argmax(-1))
  chosen=torch.cat(chosen);err=d['dist'].gather(1,chosen[:,None]).squeeze(1);old=d['dist'][:,0];return dict(n=len(err),R20=float((err<=20).float().mean()),NE=float(err.mean()),corrected=int(((err<=20)&(old>20)).sum()),damaged=int(((err>20)&(old<=20)).sum()))
 results={};start=time.perf_counter()
 for arm in ['context','language']:
  torch.manual_seed(42);model=DistinctCandidateLanguageRanker().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);baseline=evaluate(model,arm);best=baseline['R20'];history=[];torch.save(dict(model=model.state_dict(),epoch=0,metrics=baseline),a.output/f'{arm}_best.pt')
  for epoch in range(5):
   d=parts['fit'];losses=[]
   for ids in torch.randperm(len(d['g']),device='cuda').split(128):
    opt.zero_grad();logits=infer(model,d,ids,arm);loss=-(d['y'][ids]*logits.log_softmax(-1)).sum(-1).mean()+.005*(logits-d['g'][ids]).square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
   metrics=evaluate(model,arm);history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),metrics=metrics));torch.save(dict(model=model.state_dict(),epoch=epoch+1,metrics=metrics),a.output/f'{arm}_last.pt')
   if metrics['R20']>best:best=metrics['R20'];torch.save(dict(model=model.state_dict(),epoch=epoch+1,metrics=metrics),a.output/f'{arm}_best.pt')
   print(json.dumps(dict(arm=arm,**history[-1])),flush=True)
  results[arm]=dict(baseline=baseline,history=history)
 (a.output/'summary.json').write_text(json.dumps(dict(seed=42,fit_n=len(parts['fit']['g']),results=results,seconds=time.perf_counter()-start,scope='TRAIN location ranking, no RGB, confirmation unused'),indent=2)+'\n')
if __name__=='__main__':main()
