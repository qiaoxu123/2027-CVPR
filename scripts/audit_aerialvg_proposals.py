"""Equal-validity proposal audit using frozen public geometric belief; labels after selection."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np,cv2
from graphnav.workspace_paths import PROJECT,WORKSPACE
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.stage1 import GraphNavStage1
from graphnav.full_experiment import new_model,_batch_inputs
from graphnav.stage1_data import Stage1Dataset
from graphnav.candidate_geometry import collate_full
from graphnav.token_language import FrozenBERTTokens
from graphnav.object_proposals import box_valid_fraction,select_proposals
from graphnav.aerialvg_pair_residual import official_relation_class
from graphnav.checkpointing import atomic_json,atomic_save


def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise ValueError('Refusing overwrite')
 out.mkdir(parents=True);root=Path(a.probe);manifest=json.loads((root/'manifest.json').read_text());assert manifest['status']=='complete'
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 from sbfnav.mapdata import GROUND_LEVEL
 source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data','train_seen',version='processed_citynav')
 state=torch.load(PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt',map_location='cpu',weights_only=True);device='cuda'
 model=GraphNavStage1(new_model(SimpleNamespace(**state['a3_args']),device),residual_bound=.5).to(device);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
 data=Stage1Dataset(PROJECT/'GraphNav/data/rebuilt_stage1_20261025/frames',verify=False)
 lookup={(q['map_name'],q['instruction']):i for i,q in enumerate(data.static.records)}
 bert=FrozenBERTTokens(state['a3_args']['text_backbone']);bert.load_cache(PROJECT/'GraphNav/data/rebuilt_stage1_20261025/bert_tokens.pt')
 pairs=official_relation_class()().to(device).eval()
 with torch.serialization.safe_globals([argparse.Namespace]):official=torch.load(WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth',map_location='cpu',weights_only=True)
 pairs.load_state_dict({k.removeprefix('relation_transformer.'):v for k,v in official['model'].items() if k.startswith('relation_transformer.')},strict=True);pairs.requires_grad_(False);del official
 renderer=CityNavObservedViewRenderer(WORKSPACE/'DATA/sbfnav-data/rgbd',out,image_size=384);fields={};rows=[]
 for frame in manifest['frames']:
  d=torch.load(root/(frame['key']+'.pt'),map_location='cpu',weights_only=True);key=(d['map_name'],d['instruction'])
  if key not in fields:
   batch=collate_full([data.static[lookup[key]]]);inputs,tokens,mask,names=_batch_inputs(batch,bert,device,'geometry')
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):base=model.a3(inputs,variant='geometry',tokens=tokens,token_mask=mask,node_emb=names)
   bounds=torch.tensor(data.static.records[lookup[key]]['world_bounds'],device=device)
   world=bounds[:2]+inputs['grid_xy'][0]*(bounds[2:]-bounds[:2]);fields[key]=(world,base.belief_logits[0].float())
  world,belief=fields[key];observed=renderer.current_rgbd(d['map_name'],d['pose'].numpy(),GROUND_LEVEL[d['map_name']])
  quality=torch.from_numpy(box_valid_fraction(d['all_boxes_cxcywh'].numpy(),observed['depth_valid'])).to(device)
  boxes=d['all_boxes_cxcywh'].to(device);centers=d['all_world_centers'].to(device)
  features=d['all_query_features'].to(device).float();features[d['query_ids'].to(device)]=d['query_features'].to(device).float()
  text=d['encoded_text'].to(device).float();text_mask=d['text_token_mask'].to(device).bool()
  native_logits=features@text[0].T;native_scores=native_logits[:,text_mask[0]].sigmoid().amax(-1)
  nearest=torch.cdist(world.float(),centers.float()).argmin(0);geometry=belief[nearest]
  official_ids=d['query_ids'].to(device);official_ids=official_ids[quality[official_ids]>=.6]
  ids={'official15':official_ids,'nms15':select_proposals(boxes,native_scores,quality,15),'nms64':select_proposals(boxes,native_scores,quality,64),'geometry15':select_proposals(boxes,native_scores,quality,15,geometry),'geometry64':select_proposals(boxes,native_scores,quality,64,geometry),'geometry_spatial15':select_proposals(boxes,native_scores,quality,15,geometry,True)}
  winners={}
  for strategy,chosen in ids.items():
   if not len(chosen):winners[strategy]=None;continue
   if len(chosen)<=15:
    with torch.no_grad():relation=pairs(features[chosen][None],dict(encoded_text=text,text_token_mask=text_mask))[0,:,:text.shape[1]]
    fused=.6*native_logits[chosen]+.4*relation
    rank=fused[:,text_mask[0]].sigmoid().amax(-1)
   else:rank=native_scores[chosen]
   winners[strategy]=int(chosen[rank.argmax()])
  # Save inference inputs/choices before accessing target coordinates.
  atomic_save(dict(d,all_valid_pixel_fraction=quality.cpu(),native_scores=native_scores.cpu(),static_query_logits=geometry.cpu(),proposal_ids={k:v.cpu() for k,v in ids.items()},strategy_winners=winners,depth_valid=torch.from_numpy(observed['depth_valid'])),out/(frame['key']+'.pt'))
  record=source[d['episode_index']];target=np.array(record.target_position[:2],np.float32)
  homography=cv2.getPerspectiveTransform(np.asarray(observed['footprint_world'],np.float32),np.array([[0,0],[383,0],[383,383],[0,383]],np.float32))
  pixel=cv2.perspectiveTransform(target[None,None],homography)[0,0];inside=bool(((pixel>=0)&(pixel<=383)).all());target_valid=inside and bool(observed['depth_valid'][int(round(pixel[1])),int(round(pixel[0]))])
  distance=np.linalg.norm(d['all_world_centers'].numpy()-target,axis=-1)
  row=dict(key=frame['key'],map=d['map_name'],target_in_footprint=inside,target_pixel_valid=target_valid,strategies={})
  for strategy,chosen in ids.items():
   selected=chosen.cpu().numpy();winner=winners[strategy]
   row['strategies'][strategy]=dict(count=len(selected),oracle20=bool((distance[selected]<=20).any()),top1_20=bool(distance[winner]<=20) if winner is not None else False,error_m=float(distance[winner]) if winner is not None else None)
  rows.append(row);print(json.dumps(row),flush=True)
 summary={}
 for subset,valid_rows in [('all',rows),('target_pixel_valid',[r for r in rows if r['target_pixel_valid']])]:
  summary[subset]={name:dict(states=len(valid_rows),oracle20=sum(r['strategies'][name]['oracle20'] for r in valid_rows),top1_20=sum(r['strategies'][name]['top1_20'] for r in valid_rows)) for name in ids}
 atomic_json(dict(status='complete',rows=rows,summary=summary,protocol='Same bbox retained-area and depth-valid support>=0.6 for every strategy; GT only post-selection audit. Geometry is frozen A3; fixed lambda0.5, NMS0.5, spatial suppression1.5, radius0.12. Proposal centers, not target masks/boxes.'),out/'audit.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
