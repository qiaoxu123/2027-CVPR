"""Map-stratified paired current-frame polygon overlay and crop diagnostics."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from graphnav.landmark_projection import world_to_frame

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='runs/visual_grounding_benchmark/data_audit/full');p.add_argument('--output');p.add_argument('--count',type=int,default=150);a=p.parse_args();root=Path(a.data);out=Path(a.output) if a.output else root/'cases_paired';out.mkdir(parents=True,exist_ok=True);objects=json.loads((root/'objects.json').read_text());group=defaultdict(list);rng=np.random.default_rng(20261010)
 for line in (root/'manifest.jsonl').open():
  r=json.loads(line)
  if r['split']=='val_unseen' or r['object_type'] not in ('Building','Car') or not r.get('roi_path') or not r.get('query_attributes') or not r['roi_quality']['attribute_resolution_eligible']:continue
  group[r['map_name']].append(r)
 # One observation per object, taking strongest native pixel resolution; maps balanced.
 for m,rows in group.items():
  dedup={}
  for r in rows:
   key=r['object_id']
   if key not in dedup or min(r['roi_pixel_size'])>min(dedup[key]['roi_pixel_size']):dedup[key]=r
  rows=list(dedup.values());rng.shuffle(rows);group[m]=rows
 selected=[];maps=sorted(group)
 while len(selected)<a.count and any(group.values()):
  for m in maps:
   if group[m] and len(selected)<a.count:selected.append(group[m].pop())
 panels=[]
 for i,r in enumerate(selected):
  source=cv2.imread(r['roi_source']);poly=objects[r['map_name']][r['object_id']]['contour'];px=world_to_frame(poly,r['footprint_world'],source.shape);cv2.polylines(source,[np.rint(px).astype(np.int32)],True,(0,255,0),2);x0,y0,x1,y1=r['roi_bbox'];cv2.rectangle(source,(x0,y0),(x1-1,y1-1),(0,255,0),1)
  panel=Image.new('RGB',(800,330),'white');src=Image.fromarray(cv2.cvtColor(source,cv2.COLOR_BGR2RGB));src.thumbnail((295,295));panel.paste(src,(0,0));crop=Image.open(r['roi_path']).convert('RGB');crop.thumbnail((295,295));panel.paste(crop,(300,0));draw=ImageDraw.Draw(panel)
  phrase=r['attribute_phrase'];text=f"CASE {i}\n{r['split']}\n{r['map_name']}\nobject {r['object_id']}\n{r['object_type']}\nnative px {r['roi_pixel_size']}\nepisode {r['source_episode_index']}\nstep / cutoff\n{r['observed_step']} / {r['observation_cutoff_step']}"
  draw.text((602,5),text,fill='black');draw.text((3,298),phrase[:105],fill='black');draw.text((3,313),phrase[105:210],fill='black');panel.save(out/f'case_{i:03}.jpg',quality=95);panels.append(panel)
 for base in range(0,len(panels),6):
  sheet=Image.new('RGB',(1600,990),'white')
  for j,panel in enumerate(panels[base:base+6]):sheet.paste(panel,(j%2*800,j//2*330))
  sheet.save(out/f'contact_{base//6:02}.jpg',quality=95)
 (out/'cases.json').write_text(json.dumps(selected,indent=2));(out/'review_policy.json').write_text(json.dumps({'cases':len(selected),'sampling':'map balanced one highest-resolution observation per object, target phrases only; Train/Seen; seed20261010','annotation':'green geometry projected polygon/bbox is weak region; diagnostic reviewer labels are not independent human GT','human_verified_attribute_labels':0},indent=2));print(json.dumps({'cases':len(selected),'maps':len(maps),'output':str(out)}))
if __name__=='__main__':main()
