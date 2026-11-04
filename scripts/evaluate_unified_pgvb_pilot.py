"""Independent TRAIN-city evaluation of V2 visual-head learning, no target GT."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.unified_pgvb_training import load_unified,CachedLandmarkFrames,collate_cached_frames,move_batch,weak_landmark_loss
from graphnav.landmark_visual_only import visual_only_world_belief


def main():
 p=argparse.ArgumentParser()
 for k in ['data','before','after','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();torch.set_num_threads(4);data=torch.load(a.data/'cache.pt',map_location='cpu',weights_only=True);ds=CachedLandmarkFrames(data,'dev');batches=[ds.frame_ids[j:j+4] for j in range(0,len(ds),4)];outputs={};selected=[]
 for method,path,zero in [('before',a.before,False),('after',a.after,False),('after_zero_rgb',a.after,True)]:
  model=load_unified(torch.load(path,map_location='cpu',weights_only=True)).eval().requires_grad_(False);result=[]
  loader=DataLoader(ds,batch_sampler=batches,num_workers=0,collate_fn=collate_cached_frames)
  with torch.inference_mode():
   for order,batch in zip(batches,loader):
    inputs,labels,fids,types=move_batch(batch,'cuda')
    if zero:inputs['image_features'].zero_()
    with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(**inputs)['landmark_scores']
    loss=weak_landmark_loss(scores,labels,fids,types);prob=scores.float().masked_fill(~inputs['valid'],float('-inf')).softmax(-1);known=labels>=0;known_prob=scores.float().masked_fill(~known,float('-inf')).softmax(-1);rows=[data['records'][i] for fid in order for i in ds.by_frame[fid]];uv=torch.stack([data['frames'][r['frame_id']]['local_uv'] for r in rows]).cuda();visual=visual_only_world_belief(local_uv=uv,scores=scores.float(),valid=inputs['valid']);fraction=torch.stack([r['area_fraction'] for r in rows]).cuda();mass=(visual['probabilities']*fraction).sum(-1);ids=visual['probabilities'].argmax(-1);dist=torch.stack([r['distance'] for r in rows]).cuda().gather(1,ids[:,None])[:,0]
    for j,r in enumerate(rows):result.append(dict(frame_id=r['frame_id'],query_key=r['query_key'],type=r['type'],loss_batch=float(loss['total']),valid_positive_mass=float((prob[j]*(labels[j]==1)).sum()),known_positive_mass=float((known_prob[j]*(labels[j]==1)).sum()),visual_world_polygon_mass=float(mass[j]),visual_world_R20=bool(dist[j]<=20),visual_world_polygon_NE=float(dist[j]),positive_only=bool(not (labels[j]==0).any())))
  outputs[method]=dict(n=len(result),all={key:float(np.mean([r[key] for r in result])) for key in ['loss_batch','valid_positive_mass','known_positive_mass','visual_world_polygon_mass','visual_world_R20','visual_world_polygon_NE']},by_type={typ:dict(n=sum(r['type']==typ for r in result),**{key:float(np.mean([r[key] for r in result if r['type']==typ])) for key in ['loss_batch','valid_positive_mass','visual_world_polygon_mass','visual_world_R20','visual_world_polygon_NE']}) for typ in ['Building','Road']},rows=result)
  del model;torch.cuda.empty_cache()
 atomic_json(dict(scope='Full independent TRAIN Birmingham Dev; public Polygon projection weak proxies, not human pixel identity or CityNav original-target SR',data_sha256=file_sha256(a.data/'cache.pt'),before_sha256=file_sha256(a.before),after_sha256=file_sha256(a.after),checkpoint_selection='Fixed final Pilot checkpoint, never selected on this Dev',methods=outputs),a.output);print(json.dumps({k:{j:v[j] for j in ['n','all','by_type']} for k,v in outputs.items()},indent=2))
if __name__=='__main__':main()
