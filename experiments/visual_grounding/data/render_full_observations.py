"""All-episode fixed human-prefix observations; no target-selected camera poses."""
import argparse,json,sys,time
from pathlib import Path
from collections import Counter
import cv2
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from graphnav.renderer import CityNavObservedViewRenderer

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',default='/home/rental/20260922_1/Workspace/DATA/sbfnav-data');p.add_argument('--sbf-source',default='/home/rental/20260922_1/Workspace/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source');p.add_argument('--output',default='runs/visual_grounding_benchmark/data_audit/full_observations');p.add_argument('--splits',nargs='+',default=['train_seen','val_seen','val_unseen']);a=p.parse_args();sys.path.insert(0,a.sbf_source)
 from sbfnav.mapdata import GROUND_LEVEL
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);start=time.monotonic();cv2.setNumThreads(1)
 stats={}
 for split in a.splits:
  raw=json.loads((Path(a.data_root)/f'processed_citynav/citynav_{split}.json').read_text());renderer=CityNavObservedViewRenderer(Path(a.data_root)/'rgbd',out/'frames'/split,image_size=384);count=Counter()
  indices=sorted(range(len(raw)),key=lambda i:(raw[i]['area'],raw[i]['block'],i))
  with (out/f'observations_{split}.jsonl').open('w') as handle:
   for j,i in enumerate(indices):
    r=raw[i];m=f"{r['area']}_block_{r['block']}";n=len(r['trajectory']);steps=sorted({0,round(.5*(n-1)),round(.8*(n-1))})
    for step in steps:
     pose=r['trajectory'][step]
     if pose[2]-GROUND_LEVEL[m]<=0:count['nonpositive_altitude']+=1;continue
     path=out/'frames'/split/m/f'source_{i}'/f'{step:03}.png'
     frame=renderer.render(m,f'source_{i}',step,pose,GROUND_LEVEL[m]);handle.write(json.dumps({'dataset_version':'revised','map_name':m,'source_episode_index':i,'observation_cutoff_step':step,'rgb_observations':[frame]})+'\n');count['frames']+=1
    if j%500==0:print(json.dumps({'split':split,'episodes':j,'frames':count['frames'],'seconds':time.monotonic()-start}),flush=True)
  stats[split]=dict(count,episodes=len(raw));(out/'render_stats.json').write_text(json.dumps(stats,indent=2));del renderer
 print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
