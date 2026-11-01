"""Post-extraction label audit. GT is used only here, never for proposal extraction."""
import argparse,json,sys
from pathlib import Path
import torch,numpy as np,cv2
from graphnav.workspace_paths import PROJECT,WORKSPACE
from graphnav.checkpointing import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',required=True);a=p.parse_args();root=Path(a.probe)
 manifest=json.loads((root/'manifest.json').read_text());assert manifest['status']=='complete'
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data','train_seen',version='processed_citynav');rows=[]
 for entry in manifest['frames']:
  d=torch.load(root/(entry['key']+'.pt'),map_location='cpu',weights_only=True);record=source[d['episode_index']]
  footprint=d['footprint_world'].numpy().astype(np.float32);pixel=np.array([[0,0],[383,0],[383,383],[0,383]],np.float32)
  world_to_pixel=cv2.getPerspectiveTransform(footprint,pixel)
  target=np.array(record.target_position[:2],np.float32);target_pixel=cv2.perspectiveTransform(target[None,None],world_to_pixel)[0,0]
  in_footprint=bool(((target_pixel>=0)&(target_pixel<=383)).all())
  # Independently round-trip predicted world centers to verify projection wiring.
  back=cv2.perspectiveTransform(d['world_centers'].numpy().astype(np.float32)[None],world_to_pixel)[0]
  roundtrip=float(np.abs(back-d['boxes_cxcywh'].numpy()[:,:2]*383).max())
  distance=np.linalg.norm(d['world_centers'].numpy()-target,axis=-1);valid=d['valid_pixel_fraction'].numpy()>=.6
  score=d['scores'].numpy();eligible=np.flatnonzero(valid);winner=eligible[score[eligible].argmax()] if len(eligible) else None
  x=torch.nn.functional.normalize(d['query_features'].float(),dim=-1);sim=x@x.T;off=~torch.eye(len(x),dtype=torch.bool)
  rows.append(dict(key=entry['key'],map=d['map_name'],target_in_footprint=in_footprint,top1_error_m=float(distance[winner]) if winner is not None else None,top1_within20=bool(distance[winner]<=20) if winner is not None else False,oracle15_within20=bool((distance[valid]<=20).any()),query_mean_off_diagonal_cosine=float(sim[off].mean()),roundtrip_pixel_max_error=roundtrip))
  if 'all_world_centers' in d:
   from torchvision.ops import nms
   all_distance=np.linalg.norm(d['all_world_centers'].numpy()-target,axis=-1)
   boxes=d['all_boxes_cxcywh'].float();xyxy=torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),-1)
   kept=nms(xyxy,d['all_scores'].float(),.5)[:64].numpy()
   rows[-1].update(unfiltered900_oracle20=bool((all_distance<=20).any()),nms64_oracle20=bool((all_distance[kept]<=20).any()),nms64_count=len(kept))
 visible=[r for r in rows if r['target_in_footprint']]
 summary=dict(frames=len(rows),maps=len({r['map'] for r in rows}),cities=sorted({r['map'].split('_')[0] for r in rows}),target_in_footprint_frames=len(visible),visible_top1_within20=sum(r['top1_within20'] for r in visible),visible_oracle15_within20=sum(r['oracle15_within20'] for r in visible),max_roundtrip_pixel_error=max(r['roundtrip_pixel_max_error'] for r in rows),mean_query_cosine=float(np.mean([r['query_mean_off_diagonal_cosine'] for r in rows])),caveats='Footprint membership does not prove target pixels are valid/visible; query centers are not boxes/regions. Diagnostic teacher-prefix samples, not policy navigation accuracy.')
 if rows and 'unfiltered900_oracle20' in rows[0]:
  summary.update(visible_unfiltered900_oracle20=sum(r['unfiltered900_oracle20'] for r in visible),visible_nms64_oracle20=sum(r['nms64_oracle20'] for r in visible),proposal_ceiling_caveat='900/NMS64 ceiling is not depth-validity filtered; not a directly comparable valid-support metric or deployable result.')
 atomic_json(dict(summary=summary,rows=rows),root/'label_audit.json');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
