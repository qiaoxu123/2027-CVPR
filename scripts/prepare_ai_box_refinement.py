"""TRAIN-only current-RGB pixel-coordinate panels for AI weak-box refinement.

Approximate target centers and contours are annotation hints ONLY; never model
inference inputs or verified pixel truth. Group targets remain groups.
"""
import argparse
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_projection import project_landmark,world_to_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weak-boxes',type=Path)
    for k in ('root','review','catalog','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    reviewed=json.loads(a.review.read_text());selected=[x for x in reviewed if x['assessment'] in ('candidate','group_target')];keys={x['frame_key'] for x in selected}
    frames={};truth={}
    for obs in ('fixed_yaw_train400_s1','scoped_between_train294_observations_v1'):
        for x in map(json.loads,(a.root/obs/'public_frames.jsonl').read_text().splitlines()):
            if x['frame_key'] in keys:frames[x['frame_key']]=(x,a.root/obs)
        for x in map(json.loads,(a.root/obs/'labels_private.jsonl').read_text().splitlines()):
            if x.get('frame_key') in keys:truth[x['frame_key']]=x
    boxes={x['frame_key']:x for x in json.loads(a.weak_boxes.read_text())} if a.weak_boxes else {}
    catalog=json.loads(a.catalog.read_text());a.output.mkdir(parents=True);output=[]
    for start in range(0,len(selected),9):
        fig,axes=plt.subplots(3,3,figsize=(15,15))
        for ax in axes.flat:ax.axis('off')
        for ax,x in zip(axes.flat,selected[start:start+9]):
            key=x['frame_key'];f,obs=frames[key];label=truth[key];mp,oid=label['physical_group'];obj=catalog[mp][str(oid)]
            path=obs/f['image'];assert file_sha256(path)==f['image_sha256'];bgr=cv2.imread(str(path))
            roi=project_landmark({**obj,'id':str(oid)},f['footprint_world'],bgr.shape,min_visible=0,min_pixels=1)
            x0,y0,x1,y1=roi.bbox_xyxy;pad=max(8,round(max(x1-x0,y1-y0)*.25));x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(384,x1+pad);y1=min(384,y1+pad)
            hint=world_to_frame(np.asarray([label['target_world'][:2]]),f['footprint_world'],bgr.shape)[0]
            ax.axis('on');ax.imshow(cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),extent=[x0,x1,y1,y0],interpolation='nearest')
            ax.scatter(*hint,marker='+',c='yellow',s=50,label='TRAIN registration hint')
            for box in boxes.get(key,{}).get('boxes_px',[]):
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor='cyan',linewidth=1.5))
            group=boxes.get(key,{}).get('group_region_bbox_px')
            if group:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((group[0],group[1]),group[2]-group[0],group[3]-group[1],fill=False,edgecolor='lime',linestyle='--'))
            ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
            tick=max(10,int(np.ceil(max(x1-x0,y1-y0)/70))*10)
            ax.set_xticks(np.arange(int(np.ceil(x0/tick)*tick),x1+1,tick));ax.set_yticks(np.arange(int(np.ceil(y0/tick)*tick),y1+1,tick));ax.grid(alpha=.35);ax.tick_params(labelsize=9);ax.set_title(str(x['index'])+' '+key,fontsize=10)
            output.append(dict(index=x['index'],frame_key=key,image_sha256=f['image_sha256'],crop_xyxy=[x0,y0,x1,y1],approximate_registration_hint_px=hint.tolist(),image_shape=[384,384],teacher_hint_only=True,human_verified=False,pixel_gold=False))
        fig.suptitle('Native current RGB: pixel coordinates; yellow + is approximate TRAIN annotation hint, NOT pixel truth',fontsize=12);fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(a.output/f'coordinate_sheet_{start//9}.png',dpi=120);plt.close(fig)
    atomic_json(output,a.output/'coordinate_metadata.json');atomic_json(dict(cases=len(output),annotation_only=True,producer_sha256=file_sha256(__file__),review_sha256=file_sha256(a.review)),a.output/'manifest.json')

if __name__=='__main__':main()
