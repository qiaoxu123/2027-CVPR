"""Measure actual frozen SigLIP2 input/patch shapes and isolated process VRAM."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from graphnav.pgvb_observation import encode_landmark_observation
from graphnav.checkpointing import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4);data=torch.load(a.data/'cache.pt',map_location='cpu',weights_only=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();native=ObservedSigLIPMatchingEncoder('google/siglip2-large-patch16-384',device='cuda');torch.cuda.synchronize();cold=time.perf_counter()-t;shapes=[]
 def before_hook(module,args,kwargs):shapes.append(dict(pixel_values=list(kwargs['pixel_values'].shape)))
 def after_hook(module,args,result):shapes[-1]['native_patch_map']=list(result.last_hidden_state.shape)
 native.model.vision_model.register_forward_pre_hook(before_hook,with_kwargs=True);native.model.vision_model.register_forward_hook(after_hook);chosen=[];used=set()
 for r in data['records']:
  if r['frame_id'] in used:continue
  chosen.append(r);used.add(r['frame_id'])
  if len(chosen)==9:break
 times=[]
 for r in chosen:
  f=data['frames'][r['frame_id']];q=data['queries'][r['query_key']];image=Path(f['image_path']);depth=image.parent/Path(f['frame']['depth']).name
  with np.load(depth) as z:valid=z['depth_valid'].copy()
  obs=dict(bgr=cv2.imread(str(image)),depth_valid=valid,footprint_world=np.array(f['frame']['footprint_world']));native.texts([q['name']]);native.texts([q['type']]);torch.cuda.synchronize();t=time.perf_counter();inputs,reason=encode_landmark_observation(obs,dict(name=q['name'],typ=q['type'],contours=q['contours']),dict(world_bounds=f['bounds']),native,torch.device('cuda'));torch.cuda.synchronize();assert inputs is not None,reason;times.append(time.perf_counter()-t)
 result=dict(cold_model_load_seconds=cold,first_observed_forward_seconds=times[0],warm_forward_plus_pool_seconds_mean=float(np.mean(times[1:])),warm_samples=len(times)-1,siglip2_forwards=len(shapes),shapes=shapes,pooled_feature_shape=list(inputs['image_features'].shape),peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,backbone_has_gradient=any(p.grad is not None for p in native.model.parameters()),scope='Separate latency/memory diagnostic on 9 distinct real TRAIN observations; not a full-dataset time substitute. Full cache generation is measured independently.');atomic_json(result,a.output);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
