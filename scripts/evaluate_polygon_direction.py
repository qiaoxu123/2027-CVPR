"""Freeze all public-input predictions before separate post-hoc target scoring."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from graphnav.checkpointing import file_sha256
from graphnav.direction_belief import DirectionBelief
from graphnav.polygon_direction_region import polygon_direction_region,validate_public_record

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--models',type=Path,required=True);a=p.parse_args();o=a.directory;assert not (o/'predictions.npz').exists();rows=json.loads((o/'public.json').read_text());split=json.loads((o/'split_seal.json').read_text());assert file_sha256(o/'public.json')==split['source_sha256'][str(o/'public.json')]
 for row in rows:validate_public_record(row)
 torch.set_num_threads(4);anchor=torch.tensor([r['anchor'] for r in rows]);span=torch.tensor([r['span'] for r in rows]);ids=torch.tensor([r['direction_id'] for r in rows]);results={'anchor':anchor.numpy()};timing={};sha={};start=time.perf_counter();models={}
 for method,name in [('original32','direction_real'),('no_direction','nodirection_real')]:
  path=a.models/f'{name}.pt';state=torch.load(path,map_location='cpu',weights_only=True);m=DirectionBelief(state['ignore_direction']).eval().requires_grad_(False);m.load_state_dict(state['model']);models[method]=m;sha[name]=file_sha256(path);t=time.perf_counter()
  with torch.inference_mode():results[method]=torch.cat([m(anchor[j:j+8],ids[j:j+8],span[j:j+8]) for j in range(0,len(rows),8)]).numpy()
  timing[method]=(time.perf_counter()-t)*1000/len(rows)
 # Explicit centroid simplification is an additional control. The actual old
 # 32-param module already shifts a full Polygon-derived Anchor distribution.
 axis=(np.arange(28)+.5)/28;yy,xx=np.meshgrid(axis,axis,indexing='ij');mean=np.stack([(results['anchor']*xx).sum((1,2)),(results['anchor']*yy).sum((1,2))],-1);cells=np.clip((mean*28).astype(int),0,27);point=torch.zeros_like(anchor);point[torch.arange(len(rows)),cells[:,1],cells[:,0]]=1
 with torch.inference_mode():results['centroid32']=torch.cat([models['original32'](point[j:j+8],ids[j:j+8],span[j:j+8]) for j in range(0,len(rows),8)]).numpy()
 telemetry=[];cache={}
 for mode in ['ray','cone']:
  t=time.perf_counter();prob=[];support=[]
  for r in rows:
   key=(r['map_name'],r['landmark_name'],r['word'],mode)
   if key not in cache:cache[key]=polygon_direction_region(r['public_polygons'],r['bounds'],r['word'],mode)
   out=cache[key];prob.append(out['probabilities'] if out['available'] else np.asarray(r['anchor'],np.float32));support.append(out['support']);telemetry.append(dict(index=r['index'],candidate_id=r['candidate_id'],mode=mode,available=out['available'],reason=out['reason'],support_cells=int(out['support'].sum()),scale_m=out['scale_m'],fallback='none' if out['available'] else 'unchanged public Anchor (abstain)'))
  results['polygon_'+mode]=np.asarray(prob);results['support_'+mode]=np.asarray(support);timing['polygon_'+mode]=(time.perf_counter()-t)*1000/len(rows)
 np.savez_compressed(o/'predictions.npz',**results);(o/'region_telemetry.json').write_text(json.dumps(telemetry,indent=2));seal=dict(public_sha256=file_sha256(o/'public.json'),prediction_sha256=file_sha256(o/'predictions.npz'),model_sha256=sha,target_coordinates_used=False,target_labels_loaded=False,device='cpu',torch_threads=4,mean_ms_per_state=timing,total_prediction_seconds=time.perf_counter()-start,code_sha256={str(p):file_sha256(p) for p in [Path(__file__),Path('graphnav/polygon_direction_region.py')]},primary_method='polygon_cone',primary_training_gate='On NEW dev only: GT20 mass > original32 AND R20 >= original32 AND NE <= original32; otherwise no new training',configuration=dict(grid=28,cone_half_angle_degrees=45,front_samples_per_fragment=64,distance_weight='cosine(angle)^2 / (1 + distance^2 / scale^2)',scale='max(20, sqrt(public clipped union area)/2) meters',combine='maximum front-envelope kernel; all fragments; map clip',normal_inputs=['public Polygon','public map bounds','cardinal direction'],centroid_control='Existing original32 applied to point at Polygon grid mass mean; diagnostic only'))
 for name,hashvalue in sha.items():assert file_sha256(a.models/f'{name}.pt')==hashvalue
 (o/'prediction_seal.json').write_text(json.dumps(seal,indent=2));print(json.dumps({k:v for k,v in seal.items() if k not in ['configuration','code_sha256']},indent=2))
if __name__=='__main__':main()
