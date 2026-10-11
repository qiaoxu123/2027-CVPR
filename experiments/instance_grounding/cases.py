"""Actual causally observed RGB examples, selected after fixed evaluation."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from graphnav.renderer import CityNavObservedViewRenderer
from experiments.bidirectional_core.cache import DEFAULT_DATA
from .dataset import city_rows
from .data import write


def run(a):
    out=Path(a.output)
    if out.exists():raise ValueError('no overwrite')
    out.mkdir(parents=True);cv2.setNumThreads(1);rows={r['metadata']['sample_id']:r for r in city_rows(a.features)};preds=[json.loads(x) for x in Path(a.predictions).read_text().splitlines()];groups={}
    for p in preds:
        if p['target_rank'] is None:continue
        groups.setdefault(p['family_id'],p)
    chosen=sorted(groups.values(),key=lambda p:(p['target_rank']==1,-p['target_rank'],p['sample_id']))[:12];renderer=CityNavObservedViewRenderer(Path(DEFAULT_DATA)/'rgbd',out/'environment_private',image_size=384);records=[]
    for i,p in enumerate(chosen):
        r=rows[p['sample_id']];rg=r['metadata']['roi_objects'];target=p['candidate_object_ids'].index(p['target_object_id']);best=int(np.argmax(p['scores']));ref=rg[target];obs=renderer.current_rgbd(p['map_name'],np.array(ref['actual_camera_pose']),ref['ground_level']);canvas=Image.new('RGB',(1000,480),'white');draw=ImageDraw.Draw(canvas);canvas.paste(Image.fromarray(cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB)),(0,65));draw.text((8,8),p['sample_id'][:85],fill='black');draw.text((8,27),'WEAK original identity; GT only for offline illustration',fill='red')
        for col,(idx,title) in enumerate([(target,'original target'),(best,'visual top-1')]):
            v=rg[idx];o=renderer.current_rgbd(p['map_name'],np.array(v['actual_camera_pose']),v['ground_level']);x,y,c,d=v['bbox'];crop=o['bgr'][y:d,x:c];im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((270,300));base=400+col*295;canvas.paste(im,(base,100));draw.text((base,65),title+' id='+v['object_id'],fill='black');draw.text((base,410),f"step {v['source_step']} <= {p['cutoff_step']}, native side {v['native_min_side']}",fill='black')
        dest=out/f'case_{i+1:02}.png';canvas.save(dest);records.append(dict(sample_id=p['sample_id'],family_id=p['family_id'],instruction=p['instruction'],target_rank=p['target_rank'],native_target_side=ref['native_min_side'],image=dest.name,source_step=ref['source_step'],cutoff=p['cutoff_step'],status='real source RGB, geometric crop, language identity weak not human checked'))
    write(out/'cases.json',records)

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);run(p.parse_args())
if __name__=='__main__':main()
