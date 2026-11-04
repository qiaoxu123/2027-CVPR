"""Seal public-input direction predictions BEFORE opening any real Dev labels."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from graphnav.checkpointing import file_sha256
from graphnav.direction_belief import DirectionBelief
from graphnav.direction_belief_data import synthetic


def predict(model,a,ids,span,heading=None):
 pieces=[]
 with torch.inference_mode():
  for off in range(0,len(a),32):pieces.append(model(a[off:off+32],ids[off:off+32],span[off:off+32],reference_heading=heading[off:off+32] if heading is not None else None).cpu().numpy())
 return np.concatenate(pieces) if pieces else np.empty((0,28,28),np.float32)


def main():
 p=argparse.ArgumentParser()
 for k in ['public','models','pgvb','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();public=[r for r in json.loads(a.public.read_text()) if r['subset']=='dev'];assert all(not {'target_world','target_ids','target_type'}&set(r) for r in public)
 anchor=torch.tensor([r['anchor'] for r in public],device='cuda');ids=torch.tensor([r['direction_id'] for r in public],device='cuda');span=torch.tensor([r['span'] for r in public],device='cuda');syn={k:v.to('cuda') for k,v in synthetic(1024,881).items()};real_result=dict(anchor=anchor.cpu().numpy());syn_result=dict(anchor=syn['anchor'].cpu().numpy());latency={};models={};modelsha={}
 for name in ['nodirection_synthetic','direction_synthetic','nodirection_real','direction_real']:
  path=a.models/(name+'.pt');state=torch.load(path,map_location='cpu',weights_only=True);assert all(file_sha256(p)==v for p,v in state['config']['code_sha256'].items());m=DirectionBelief(ignore_direction=state['ignore_direction']).to('cuda').eval().requires_grad_(False);m.load_state_dict(state['model'],strict=True);models[name]=m;modelsha[name]=file_sha256(path)
  real_result[name]=predict(m,anchor,ids,span);syn_result[name]=predict(m,syn['anchor'],syn['direction'],syn['span'],syn['heading'])
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode():
   for _ in range(20):m(anchor[:1],ids[:1],span[:1])
  torch.cuda.synchronize();latency[name]=(time.perf_counter()-t)/20*1000
 real_result['opposite_direction']=predict(models['direction_real'],anchor,ids^1,span)
 pp=np.load(a.pgvb/'anchors.npz');available=[j for j,r in enumerate(public) if str(r['index']) in pp.files];pa=torch.tensor(np.asarray([pp[str(public[j]['index'])] for j in available]),device='cuda') if available else torch.empty(0,28,28,device='cuda');real_result['pgvb_indices']=np.asarray(available);real_result['pgvb_anchor']=pa.cpu().numpy()
 for name in ['nodirection_real','direction_real']:real_result['pgvb_'+name]=predict(models[name],pa,ids[available],span[available])
 np.savez_compressed(a.output/'real_probabilities.npz',**real_result);np.savez_compressed(a.output/'synthetic_probabilities.npz',**syn_result);np.savez_compressed(a.output/'synthetic_truth.npz',**{k:v.cpu().numpy() for k,v in syn.items() if k not in ['anchor','target']});(a.output/'public_dev.json').write_text(json.dumps(public,indent=2))
 (a.output/'seal_before_scoring.json').write_text(json.dumps(dict(real_targets_read=False,model_sha256=modelsha,public_sha256=file_sha256(a.public),prediction_sha256=file_sha256(a.output/'real_probabilities.npz'),synthetic_prediction_sha256=file_sha256(a.output/'synthetic_probabilities.npz'),synthetic_eval_seed=881,synthetic_train_seed=123,real_pairs=len(public),pgvb_supported=len(available),single_forward_ms=latency,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,producer_sha256=file_sha256(__file__)),indent=2));print('Sealed real predictions without reading real Dev targets:',len(public),'PGVB',len(available))
if __name__=='__main__':main()
