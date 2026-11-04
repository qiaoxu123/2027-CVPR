"""Frozen 32-parameter direction forwards. No labels are loaded in this process."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from graphnav.direction_belief import DirectionBelief
from graphnav.checkpointing import file_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--models',type=Path,required=True);a=p.parse_args();o=a.cohort;assert not (o/'probabilities.npz').exists();seal=json.loads((o/'selection_seal.json').read_text());assert file_sha256(o/'public.json')==seal['public_sha256'];rows=json.loads((o/'public.json').read_text());assert all(not {'target_world','target_ids','target_type'}&set(r) for r in rows)
 torch.set_num_threads(4);anchor=torch.tensor([r['anchor'] for r in rows]);ids=torch.tensor([r['direction_id'] for r in rows]);span=torch.tensor([r['span'] for r in rows]);pred={'anchor':anchor.numpy()};sha={};timing={};loaded={}
 def forward(m,ids):
  with torch.inference_mode():return torch.cat([m(anchor[i:i+8],ids[i:i+8],span[i:i+8]) for i in range(0,len(rows),8)]).numpy()
 for name in ['nodirection_real','direction_real','direction_synthetic']:
  path=a.models/f'{name}.pt';sha[name]=file_sha256(path);state=torch.load(path,map_location='cpu',weights_only=True);m=DirectionBelief(state['ignore_direction']).eval().requires_grad_(False);m.load_state_dict(state['model'],strict=True);assert sum(x.numel() for x in m.parameters())==32;loaded[name]=m;t=time.perf_counter();pred[name]=forward(m,ids);timing[name]=(time.perf_counter()-t)*1000/len(rows)
 cf=np.stack([forward(loaded['direction_real'],torch.full_like(ids,k)) for k in range(4)],axis=1);pred['counterfactual']=cf;pred['direction_marginal']=cf.mean(1)
 np.savez_compressed(o/'probabilities.npz',**pred);assert all(file_sha256(a.models/f'{k}.pt')==v for k,v in sha.items());(o/'prediction_seal.json').write_text(json.dumps(dict(target_labels_read=False,cohort_sha256=file_sha256(o/'public.json'),selection_sha256=file_sha256(o/'selection.json'),prediction_sha256=file_sha256(o/'probabilities.npz'),model_sha256=sha,device='cpu',torch_threads=4,mean_cpu_forward_ms=timing,code_sha256=file_sha256(__file__),scope='Public Polygon-derived Anchor, word ID and public map span only; no RGB/PGVB forward, target, target ID or trajectory'),indent=2));print(json.dumps(timing))
if __name__=='__main__':main()
