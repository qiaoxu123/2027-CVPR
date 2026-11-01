"""Extract actual official decoder queries from causal CityNav rendered observations."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
import cv2
from graphnav.workspace_paths import PROJECT,WORKSPACE
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.object_proposals import box_valid_fraction


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=12);p.add_argument('--partition',choices=['all','train','dev'],default='all');p.add_argument('--steps',choices=['probe','stage1','initial'],default='probe');a=p.parse_args();out=Path(a.output)
 if out.exists():raise ValueError('Refusing to overwrite probe')
 out.mkdir(parents=True)
 repo=WORKSPACE/'third_party/AerialVG';sys.path.insert(0,str(repo));sys.path.insert(0,str(WORKSPACE/'third_party/aerialvg_validation'))
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 from sbfnav.mapdata import GROUND_LEVEL
 from run_official_inference import TorchDeformableFallback
 from util.slconfig import SLConfig
 from util.utils import clean_state_dict
 from model import build_model
 from model.AerialVG import ms_deform_attn
 from demo import load_image
 ms_deform_attn.MultiScaleDeformableAttnFunction=TorchDeformableFallback
 args=SLConfig.fromfile(str(repo/'config/config_cfg.py'));args.device='cuda'
 args.text_encoder_type='/home/20261024_1/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
 model=build_model(args)
 with torch.serialization.safe_globals([argparse.Namespace]):state=torch.load(WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth',map_location='cpu',weights_only=True)
 model.load_state_dict(clean_state_dict(state['model']),strict=True);model.to('cuda').eval().requires_grad_(False);del state
 captured={}
 def queries(module,inputs):captured['all']=inputs[0].detach()
 def pairs(module,inputs):captured['selected']=inputs[0].detach();captured['text']=inputs[1]
 handles=[model.class_embed[-1].register_forward_pre_hook(queries),model.relation_transformer.register_forward_pre_hook(pairs)]
 source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data','train_seen',version='processed_citynav')
 split_state=torch.load(PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt',map_location='cpu',weights_only=True)
 dev=set(split_state['config']['dev_episodes']);dev_keys={(source[i].map_name,source[i].instruction) for i in dev};del split_state
 # Select by public map/instruction only, round-robin over maps; no GT visibility filtering.
 groups={}
 for index in range(len(source)):
  r=source[index];is_dev=(r.map_name,r.instruction) in dev_keys
  if a.partition=='train' and is_dev:continue
  if a.partition=='dev' and not is_dev:continue
  groups.setdefault(r.map_name,[]).append((hashlib.sha256(r.instruction.encode()).hexdigest(),index))
 for name in groups:groups[name].sort()
 selected=[]
 for offset in range(max(map(len,groups.values()))):
  for name in sorted(groups):
   if offset<len(groups[name]):selected.append(groups[name][offset][1])
  if len(selected)>=a.episodes:break
 selected=selected[:a.episodes]
 renderer=CityNavObservedViewRenderer(WORKSPACE/'DATA/sbfnav-data/rgbd',out,image_size=384);rows=[]
 metadata=dict(episode_indices=selected,partition=a.partition,steps=a.steps,policy='public map/instruction hash, grouped Train/Dev partition; no GT proposal selection',official_checkpoint_sha=file_sha256(WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth'))
 atomic_json(dict(metadata,status='running'),out/'manifest.json')
 for index in selected:
  record=source[index]
  from scripts.prepare_joint_active_data import prefix_indices
  steps=prefix_indices(len(record.trajectory)) if a.steps=='stage1' else [0,len(record.trajectory)//2]
  if a.steps=='initial':steps=[0]
  for step in sorted(set(steps)):
   pose=record.trajectory[step].astype(np.float32)
   if pose[2]<=GROUND_LEVEL[record.map_name]:continue
   observed=renderer.current_rgbd(record.map_name,pose,GROUND_LEVEL[record.map_name]);key=f'{index:05d}_{step:03d}';image_path=out/(key+'.png');cv2.imwrite(str(image_path),observed['bgr'])
   _,image=load_image(str(image_path));captured.clear();torch.cuda.synchronize();started=time.perf_counter()
   with torch.inference_mode():prediction=model(image[None].to('cuda'),captions=[record.instruction.lower().strip().rstrip('.')+'.'])
   torch.cuda.synchronize();duration=time.perf_counter()-started
   chosen=captured['selected'][0];all_queries=captured['all'][0]
   ids=torch.cdist(chosen.float(),all_queries.float()).argmin(-1)
   assert float((all_queries[ids]-chosen).abs().max())<1e-4,'Decoder query/box association mismatch'
   boxes=prediction['pred_boxes'][0,ids].detach().cpu().numpy();scores=prediction['pred_logits'][0,ids][:,prediction['text_mask'][0]].sigmoid().amax(-1).cpu()
   corners=np.array([[0,0],[383,0],[383,383],[0,383]],np.float32)
   homography=cv2.getPerspectiveTransform(corners,np.asarray(observed['footprint_world'],np.float32))
   centers=cv2.perspectiveTransform((boxes[:,:2]*383).astype(np.float32)[None],homography)[0]
   xyxy=np.concatenate((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),-1)
   quality=[]
   for box in xyxy:
    x0,y0,x1,y1=np.clip(np.round(box*383).astype(int),0,383)
    support=observed['depth_valid'][y0:y1+1,x0:x1+1]
    quality.append(float(support.mean()) if support.size else 0.)
   all_boxes=prediction['pred_boxes'][0].cpu().numpy()
   all_centers=cv2.perspectiveTransform((all_boxes[:,:2]*383).astype(np.float32)[None],homography)[0]
   all_scores=prediction['pred_logits'][0][:,prediction['text_mask'][0]].sigmoid().amax(-1).cpu()
   all_quality=box_valid_fraction(all_boxes,observed['depth_valid'])
   atomic_save(dict(query_features=chosen.cpu().clone(),query_ids=ids.cpu(),boxes_cxcywh=torch.from_numpy(boxes),world_centers=torch.from_numpy(centers),scores=scores,valid_pixel_fraction=torch.tensor(quality),all_query_features=all_queries.cpu().half().clone(),all_boxes_cxcywh=torch.from_numpy(all_boxes),all_world_centers=torch.from_numpy(all_centers),all_scores=all_scores,all_valid_pixel_fraction=torch.from_numpy(all_quality),depth_valid=torch.from_numpy(observed['depth_valid']),encoded_text=captured['text']['encoded_text'].cpu().clone(),text_token_mask=captured['text']['text_token_mask'].cpu().clone(),pose=torch.from_numpy(pose),footprint_world=torch.tensor(observed['footprint_world']),episode_index=index,step=step,map_name=record.map_name,instruction=record.instruction),out/(key+'.pt'))
   row=dict(key=key,episode=index,step=step,map=record.map_name,queries=len(ids),unique_queries=int(ids.unique().numel()),valid_queries=int((np.array(quality)>=.6).sum()),max_score=float(scores.max()),seconds=duration)
   rows.append(row);atomic_json(dict(status='running',frames=rows),out/'progress.json');print(json.dumps(row),flush=True)
 for h in handles:h.remove()
 atomic_json(dict(metadata,status='complete',frames=rows,source_sha=file_sha256(source.annotation_path),limitations='Orthographic footprint projection, vertical-depth proxy; not perspective occlusion geometry. 384px actual observations resized by official demo preprocessing; no extra image detail. Offline teacher-prefix observations, not policy rollout.'),out/'manifest.json')


if __name__=='__main__':main()
