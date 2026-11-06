"""Untrained, GT-free noun-type angular histogram reference, development only by default."""
import argparse,json,math,re
from pathlib import Path
import torch,numpy as np


def prior(data,i):
 q=data['queries'][i];s=data['scenes'][q['scene']];j=next((j for j in range(16) if q['anchor_valid'][j]),0);steps=max(int(q['step_valid'][j].sum()),1);text=data['phrases'][int(q['entity_ids'][j,steps-1])].casefold()
 if re.search(r'\b(car|van|truck|vehicle|bus|suv|sedan)\b',text):kind=[7,9]
 elif re.search(r'\b(building|house|home|warehouse|church|roof|hall|complex)\b',text):kind=[1]
 elif re.search(r'\b(parking|lot|ground|grass|field|area|garden)\b',text):kind=[5,6]
 else:kind=list(range(10))
 valid=s['valid']&torch.isin(s['types'],torch.tensor(kind))
 if not valid.any():valid=s['valid']
 p=torch.einsum('n,rnk->rk',valid.float()/valid.sum(),q['angle_kernel'].float());return p/p.sum(-1,keepdim=True).clamp_min(1e-9)


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
 data=torch.load(a.run/'public_inputs.pt',weights_only=True,mmap=True);labels=torch.load(a.run/'PRIVATE_dev_labels.pt',weights_only=True);ids=json.load(open(a.run/'data/split.json'))['indices']['dev'];byq=[];byanchor=[]
 for i in ids:
  prob=prior(data,i);y=labels[i]['bins'];v=y>=0;hits=prob.argmax(-1)==y
  if v.any():byq.append(float(hits[v].float().mean()));byanchor+=hits[v].tolist()
 out=dict(reference='untrained noun-type public angular histogram, no direction/frame rule',dev_macro_Top1=float(np.mean(byq)),dev_anchor_Top1=float(np.mean(byanchor)),queries=len(byq),GT_only_for_scoring=True)
 (a.run/'public_type_prior_dev.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
