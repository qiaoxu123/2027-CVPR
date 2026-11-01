"""Build grouped, observed native-query training rows. GT lives only in labels."""
import argparse,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np,cv2
from graphnav.workspace_paths import PROJECT,WORKSPACE
from graphnav.stage1 import GraphNavStage1
from graphnav.stage1_data import Stage1Dataset
from graphnav.candidate_geometry import collate_full
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.token_language import FrozenBERTTokens
from graphnav.object_proposals import select_proposals
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256


def main():
 p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise ValueError('Refusing overwrite')
 manifests=[json.loads((Path(s)/'manifest.json').read_text()) for s in a.sources]
 if any(m['status']!='complete' for m in manifests):raise ValueError('Sources must be complete')
 partitions={m['partition'] for m in manifests}
 if len(partitions)!=1 or not partitions<={'train','dev'}:raise ValueError('Explicit single Train/Dev partition required')
 out.mkdir(parents=True);device='cuda'
 checkpoint=PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 model=GraphNavStage1(new_model(SimpleNamespace(**state['a3_args']),device),residual_bound=.5).to(device);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
 data=Stage1Dataset(PROJECT/'GraphNav/data/rebuilt_stage1_20261025/frames',verify=False);lookup={(q['map_name'],q['instruction']):i for i,q in enumerate(data.static.records)}
 bert=FrozenBERTTokens(state['a3_args']['text_backbone']);bert.load_cache(PROJECT/'GraphNav/data/rebuilt_stage1_20261025/bert_tokens.pt')
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data','train_seen',version='processed_citynav')
 seen=set();fields={};chunk=[];chunks=[];metadata=[]
 def save_chunk():
  nonlocal chunk
  if not chunk:return
  name=f'rows_{len(chunks):04d}.pt';atomic_save(chunk,out/name);chunks.append(dict(file=name,rows=len(chunk),sha256=file_sha256(out/name)));chunk=[]
 for directory,manifest in zip(map(Path,a.sources),manifests):
  for entry in manifest['frames']:
   identity=(entry['episode'],entry['step'])
   if identity in seen:continue
   seen.add(identity);d=torch.load(directory/(entry['key']+'.pt'),map_location='cpu',weights_only=True);key=(d['map_name'],d['instruction'])
   if key not in fields:
    record=data.static.records[lookup[key]];batch=collate_full([data.static[lookup[key]]]);inputs,tokens,mask,names=_batch_inputs(batch,bert,device,'geometry')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):field=model.a3(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
    bounds=torch.tensor(record['world_bounds'],device=device);world=bounds[:2]+inputs['grid_xy'][0]*(bounds[2:]-bounds[:2]);fields[key]=(world,field.belief_logits[0].float(),bounds)
   world,base,bounds=fields[key];features=d['all_query_features'].to(device).float();features[d['query_ids'].to(device)]=d['query_features'].to(device).float()
   text=d['encoded_text'][0].to(device).float();text_mask=d['text_token_mask'][0].to(device).bool();native=(features@text.T)[:,text_mask].sigmoid().amax(-1)
   centers=d['all_world_centers'].to(device).float();quality=d['all_valid_pixel_fraction'].to(device).float();boxes=d['all_boxes_cxcywh'].to(device).float()
   nearest=torch.cdist(world,centers).argmin(0);prior=base[nearest]
   ids=select_proposals(boxes,native,quality,15,prior,True)
   f=torch.zeros(15,256,device=device);g=torch.zeros(15,8,device=device);qworld=torch.zeros(15,2,device=device);valid=torch.zeros(15,device=device,dtype=torch.bool)
   count=len(ids);f[:count]=features[ids];qworld[:count]=centers[ids];valid[:count]=True
   relative=(prior[ids]-base.amax()).clamp(-20,0)/5
   g[:count]=torch.stack((relative,torch.logit(native[ids].clamp(1e-5,1-1e-5))/5,quality[ids],boxes[ids,2],boxes[ids,3],boxes[ids,0],boxes[ids,1],boxes[ids,2]*boxes[ids,3]),-1)
   homography=cv2.getPerspectiveTransform(d['footprint_world'].numpy().astype(np.float32),np.array([[0,0],[383,0],[383,383],[0,383]],np.float32))
   pixels=cv2.perspectiveTransform(world.cpu().numpy().astype(np.float32)[None],homography)[0];inside=((pixels>=0)&(pixels<=383)).all(-1)
   rounded=np.clip(np.rint(pixels).astype(int),0,383);support=torch.tensor(inside&d['depth_valid'].numpy()[rounded[:,1],rounded[:,0]],device=device)
   kernel=torch.exp(-torch.cdist(world,qworld).square()/(2*20.**2))*support[:,None]*valid[None]
   # All selections and forward tensors above are computed without target coordinates.
   target=torch.tensor(source[d['episode_index']].target_position,dtype=torch.float32)
   group=hashlib.sha256(json.dumps(key).encode()).hexdigest()
   row=dict(features=f.cpu().half(),text=text.cpu().half(),text_mask=text_mask.cpu(),geometry=g.cpu(),valid=valid.cpu(),query_world=qworld.cpu(),base_logits=base.cpu(),grid_world=world.cpu(),kernel=kernel.cpu().half(),observed=support.cpu(),labels=dict(target_world=target),episode=d['episode_index'],step=d['step'],group=group,instruction=d['instruction'],map_name=d['map_name'])
   chunk.append(row);metadata.append(dict(episode=d['episode_index'],step=d['step'],group=group,map=d['map_name'],queries=count,observed_cells=int(support.sum())))
   if len(chunk)>=32:save_chunk()
   if len(metadata)%100==0:print(json.dumps(dict(rows=len(metadata),partition=list(partitions)[0])),flush=True)
 save_chunk();atomic_json(dict(complete=True,partition=list(partitions)[0],rows=metadata,chunks=chunks,sources=[dict(path=str(Path(s).resolve()),manifest_sha=file_sha256(Path(s)/'manifest.json')) for s in a.sources],base_sha=file_sha256(checkpoint),selector='NMS0.5; geometry_weight0.5; spatial_suppression1.5 sigma0.12; budget15; validity0.6; no GT selection',kernel_sigma_m=20.),out/'manifest.json')


if __name__=='__main__':main()
