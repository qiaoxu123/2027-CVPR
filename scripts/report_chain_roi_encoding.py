"""Posthoc proxy metrics and real RGB cases for frozen ROI encoding audit."""
import argparse,html,json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon,box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.landmark_projection import world_to_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','observations','labels','catalog','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior report')
    e=json.loads((a.run/'evaluation.json').read_text());pred={x['frame_key']:x for x in json.loads((a.run/'blind_predictions.json').read_text())};pub={x['frame_key']:x for x in map(json.loads,(a.observations/'public_frames.jsonl').read_text().splitlines())};labels={x['frame_key']:x for x in map(json.loads,a.labels.read_text().splitlines())};objects=json.loads(a.catalog.read_text())
    a.output.mkdir(parents=True);(a.output/'cases').mkdir();stats={};pixel=[];cards=[]
    methods=list(e['rows'][0]['methods'])
    for category in ['all','Car','Building','Ground','Parking']:
        rows=[x for x in e['rows'] if category=='all' or x['category']==category];stats[category]=dict(frames=len(rows),valid_observation_frames=sum(x['nearest_world_ne_m'] is not None for x in rows),support5=sum(x['nearest_world_ne_m'] is not None and x['nearest_world_ne_m']<=5 for x in rows),support20=sum(x['nearest_world_ne_m'] is not None and x['nearest_world_ne_m']<=20 for x in rows),methods={})
        for m in methods:
            stats[category]['methods'][m]={f'top{k}_{radius}m':sum(x['methods'][m][f'top{k}_ne_m'] is not None and x['methods'][m][f'top{k}_ne_m']<=radius for x in rows) for k in (1,5) for radius in (5,20)}
    gy,gx=np.meshgrid(np.arange(1,24,2),np.arange(1,24,2),indexing='ij');centers=(np.stack((gx.ravel(),gy.ravel()),-1)+.5)*16-.5
    for row in e['rows']:
        key=row['frame_key'];frame=pub[key];label=labels[key];oid=label['physical_group'][1];obj=objects[frame['map_name']][str(oid)];coords=world_to_frame(np.asarray(obj['contour']),np.asarray(frame['footprint_world']),(384,384,3));poly=Polygon(coords)
        clipped=poly.intersection(box(0,0,383,383)) if poly.is_valid else None
        bbox=poly.bounds;pixel.append(dict(frame_key=key,category=row['category'],observation_world_span_m=float(np.linalg.norm(np.asarray(frame['footprint_world'])[1]-np.asarray(frame['footprint_world'])[0])),geometry_proxy_bbox_width_px=bbox[2]-bbox[0],geometry_proxy_bbox_height_px=bbox[3]-bbox[1],geometry_proxy_area_px2=poly.area,geometry_proxy_in_view_area_px2=clipped.area if clipped is not None else None,visibility='unknown; no human pixel labels'))
        rgb=cv2.imread(str(a.observations/frame['image']));fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB));axes[0].plot(*np.vstack([coords,coords[0]]).T,color='lime',linewidth=1);axes[0].set_xlim(0,383);axes[0].set_ylim(383,0);axes[0].set_title('Geometry proxy; NOT pixel GT')
        for ax,m in zip(axes[1:],['native_pool_1','crop_112']):
            ax.imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB));idx=pred[key]['rankings'][m][0] if pred[key]['rankings'][m] else None
            if idx is not None:ax.scatter(*centers[idx],c='red',marker='x',s=70)
            ax.set_title(f'{m}: world NE {row["methods"][m]["top1_ne_m"]:.1f}m' if idx is not None else 'No valid candidate')
        for ax in axes:ax.axis('off')
        fig.suptitle(frame['instruction'],fontsize=8,wrap=True);fig.tight_layout();name='cases/'+key+'.png';fig.savefig(a.output/name,dpi=120);plt.close(fig)
        cards.append(f'<article><h3>{html.escape(key)}</h3><p>{html.escape(frame["instruction"])}</p><a href="{name}"><img width="1300" src="{name}"></a></article>')
    summary={k:v for k,v in e.items() if k!='rows'};summary.update(statistics=stats,pixel_geometry_proxies=pixel)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>img{max-width:100%}body{font:16px sans-serif;max-width:1400px;margin:auto}</style><h1>固定SigLIP2：全图patch与真实观测ROI编码</h1><p>绿色仅Contour几何代理，不是像素金标准；红叉为盲排序Top1。24个TRAIN拟合目标末参考帧，不是闭环或独立确认。</p>'+''.join(cards))
    print(json.dumps(stats))


if __name__=='__main__':main()
