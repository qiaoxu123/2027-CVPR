"""TRAIN split paired candidate ranking followed by the fixed local decoder."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from graphnav.distinct_candidate_language import DistinctCandidateLanguageRanker

def decode(v,xy,top):
 rr,cc=np.divmod(np.arange(784),28);a,b=divmod(int(top),28);ids=np.where((abs(rr-a)<=2)&(abs(cc-b)<=2))[0];w=np.exp((v[ids]-v[ids].max())/.5);return (xy[ids]*w[:,None]).sum(0)/w.sum()

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['development','confirmation'],default='development');a=p.parse_args();torch.set_num_threads(4)
 model=DistinctCandidateLanguageRanker().cuda().eval();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model']);labels=json.loads(a.labels.read_text());rows=[]
 with torch.no_grad():
  for l in labels:
   if l['split']!=a.split:continue
   z=np.load(a.cache/'public_features'/f"{l['index']:05d}.npz");grid=z['base_world_grid'];world=z['candidate_world'];xy=(world-grid.min(0))/(grid.max(0)-grid.min(0));logits=model(torch.tensor(z['candidate_context'].astype('float32'),device='cuda')[None],torch.tensor(xy,device='cuda')[None],torch.tensor(z['geometry_logits'],device='cuda')[None],torch.zeros(1,1,768,device='cuda'),torch.ones(1,1,dtype=torch.bool,device='cuda'));slot=int(logits.argmax());v=z['base_logits_full'];ids=z['candidate_grid_ids'];base=decode(v,grid,ids[0]);new=decode(v,grid,ids[slot]);target=np.array(l['target_world']);rows.append(dict(index=l['index'],slot=slot,base_NE=float(np.linalg.norm(base-target)),ranked_NE=float(np.linalg.norm(new-target))))
 summary=dict(n=len(rows),baseline_R20=float(np.mean([r['base_NE']<=20 for r in rows])),ranked_R20=float(np.mean([r['ranked_NE']<=20 for r in rows])),baseline_NE=float(np.mean([r['base_NE'] for r in rows])),ranked_NE=float(np.mean([r['ranked_NE'] for r in rows])),corrected=sum(r['base_NE']>20 and r['ranked_NE']<=20 for r in rows),damaged=sum(r['base_NE']<=20 and r['ranked_NE']>20 for r in rows),switches=sum(r['slot']!=0 for r in rows));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
