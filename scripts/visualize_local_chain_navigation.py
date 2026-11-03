"""Post-hoc real TRAIN trajectory cases; RGB rebuilt only at recorded decision poses."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import textwrap

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.landmark_projection import world_to_frame
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','data-root','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior visualizations')
    a.output.mkdir(parents=True);(a.output/'frames').mkdir()
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav');catalog=CityReferCatalog(a.data_root/'cityrefer')
    methods=('original','real','zero','shuffle')
    rows={m:{r['index']:r for r in map(json.loads,(a.run/m/'predictions.jsonl').read_text().splitlines())} for m in methods}
    picked=[]
    # Post-hoc case selection includes improvements, failures and a successful but inefficient rollout.
    picked += [i for i in rows['real'] if rows['real'][i]['metrics']['SR'] and not rows['original'][i]['metrics']['SR']]
    picked += [i for i in rows['real'] if not rows['real'][i]['metrics']['SR'] and rows['real'][i]['posthoc_goal_analysis']['ever_correct']][:2]
    car_fail=[i for i in rows['real'] if not rows['real'][i]['metrics']['SR'] and catalog.objects[ds[i].map_name][ds[i].object_id]['object_type']=='Car']
    picked += car_fail[:1]
    picked += [i for i in rows['real'] if rows['real'][i]['metrics']['SR'] and rows['original'][i]['metrics']['SR'] and rows['real'][i]['termination']=='horizon'][:1]
    picked=list(dict.fromkeys(picked))
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384),0.)
    metadata=[];cards=[]
    for index in picked:
        record=ds[index];fig,axes=plt.subplots(1,4,figsize=(16,4))
        for lm in catalog.referenced_landmarks(record):
            poly=np.array(lm.contour);axes[0].plot(*np.vstack([poly,poly[0]]).T,color='gray',alpha=.4)
        for method,color in zip(methods,('tab:blue','tab:red','gray','tab:orange')):
            xy=np.array(rows[method][index]['poses'])[:,:2];axes[0].plot(xy[:,0],xy[:,1],'-o',markersize=2,color=color,label=method)
        axes[0].scatter(*record.target_position[:2],marker='*',s=90,c='green',label='GT (posthoc only)')
        axes[0].set_aspect('equal');axes[0].legend(fontsize=7);axes[0].set_title('Real closed-loop world XY')
        row=rows['real'][index];diagnostics=row['step_diagnostics']
        steps=np.unique(np.array([0,len(diagnostics)//2,len(diagnostics)-1],int))
        for ax,step in zip(axes[1:],steps):
            pose=row['poses'][step];obs=renderer.current_rgbd(record.map_name,pose,GROUND_LEVEL[record.map_name])
            image_path=a.output/'frames'/f'{index}_{step}.png'
            if not cv2.imwrite(str(image_path),obs['bgr']):raise IOError(image_path)
            ax.imshow(cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB))
            for point,color,marker in [(record.target_position[:2],'lime','+'),(diagnostics[step]['selected_goal_xy'],'red','x')]:
                pixel=world_to_frame(np.array([point]),obs['footprint_world'],obs['bgr'].shape)[0]
                if ((pixel>=0)&(pixel<=383)).all():ax.scatter(pixel[0],pixel[1],c=color,marker=marker,s=65)
            ax.set_title(f'Real decision {step}; camera yaw=0');ax.axis('off')
            metadata.append(dict(index=index,step=int(step),pose=pose,camera_yaw_rad=0.,image=str(image_path.name),image_sha256=file_sha256(image_path),
                provenance='Deterministic reconstruction at actual visited decision pose. No original runtime pixel SHA was recorded.',
                gt_marker='world point projection only, not a verified pixel-instance label'))
        fig.suptitle('\n'.join(textwrap.wrap(record.instruction,130)),fontsize=9);fig.tight_layout()
        path=a.output/f'episode_{index}.png';fig.savefig(path,dpi=140);plt.close(fig)
        cards.append(f'<article><h2>TRAIN {index}</h2><p>{html.escape(record.instruction)}</p><a href="{path.name}"><img src="{path.name}" width="1400"></a></article>')
    atomic_json(metadata,a.output/'reconstructed_frames.json')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TRAIN闭环案例</title><style>body{font:16px sans-serif;max-width:1450px;margin:auto}img{max-width:100%}article{padding:18px 0;border-bottom:1px solid #aaa}</style><h1>冻结TRAIN闭环：改善、失败与低效案例</h1><p>绿色是事后GT世界点投影，不是人工像素框；红叉是当时Goal。RGB由真实决策位姿确定性重建，相机yaw=0，未使用教师轨迹。采样仅限实际发生观测的决策时刻，不包括horizon后未决策的最终位置。没有将这些GT标记输入模型。</p>'+''.join(cards))
    print(json.dumps(dict(cases=len(picked),indices=picked)))


if __name__=='__main__':main()
