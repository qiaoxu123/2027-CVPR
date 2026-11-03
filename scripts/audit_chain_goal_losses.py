"""Posthoc loss-of-correct-goal cases; GT never enters policy or rendering poses."""
import argparse, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.fixed_yaw_observation import FixedYawObservation
from graphnav.landmark_projection import world_to_frame
from graphnav.local_chain_observation import local_patch_regions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','data-root','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--case-type',choices=('harms','corrections'),default='harms')
    p.add_argument('--split',choices=('train_seen','val_seen'),default='train_seen')
    p.add_argument('--max-cases',type=int)
    a=p.parse_args()
    if a.max_cases is not None and a.max_cases<1:raise ValueError('positive visualization count required')
    source_manifest=json.loads((a.run/'run_manifest.json').read_text())
    if source_manifest.get('split','train_seen')!=a.split:raise ValueError('analysis split differs from actual rollout')
    if a.output.exists():raise FileExistsError('preserve previous audit')
    a.output.mkdir(parents=True);(a.output/'frames').mkdir()
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    ds=CityNavDataset(a.data_root,a.split,version='processed_citynav')
    catalog=CityReferCatalog(a.data_root/'cityrefer')
    rows={m:{x['index']:x for x in map(json.loads,(a.run/m/'predictions.jsonl').read_text().splitlines())} for m in ('original','real')}
    selected=[i for i,x in rows['original'].items() if (bool(x['metrics']['SR']) and not rows['real'][i]['metrics']['SR'])] if a.case_type=='harms' else [i for i,x in rows['real'].items() if x['metrics']['SR'] and not rows['original'][i]['metrics']['SR']]
    if a.max_cases is not None:selected=selected[:a.max_cases]
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'unused',image_size=384),0.)
    records=[];links=[]
    for index in selected:
        record=ds[index];r=rows['real'][index];gt=record.target_position[:2];diags=r['step_diagnostics']
        obj=catalog.objects[record.map_name][record.object_id]
        polygon=Polygon(obj['contour'])
        distances=[float(np.linalg.norm(np.array(d['selected_goal_xy'])-gt)) for d in diags]
        loss_steps=[s for s in range(1,len(diags)) if distances[s-1]<=20<distances[s]]
        steps=sorted(set([0,len(diags)-1]+loss_steps))
        fig,axes=plt.subplots(1,len(steps)+1,figsize=(4*(len(steps)+1),4))
        for m,color in [('original','blue'),('real','red')]:
            xy=np.asarray(rows[m][index]['poses'])[:,:2];axes[0].plot(*xy.T,'-o',color=color,markersize=2,label=m)
        axes[0].scatter(*gt,c='green',marker='*',s=70);axes[0].set_aspect('equal');axes[0].legend();axes[0].set_title(f'{a.split} {index}; posthoc GT')
        per_frame=[]
        for ax,s in zip(axes[1:],steps):
            obs=renderer.current_rgbd(record.map_name,r['poses'][s],GROUND_LEVEL[record.map_name]);im=obs['bgr']
            h=hashlib.sha256(str(im.shape).encode());h.update(np.ascontiguousarray(im).tobytes())
            if h.hexdigest()!=diags[s]['observed_rgb_sha256']:raise ValueError('reconstructed pixels differ from actual inference')
            cv2.imwrite(str(a.output/'frames'/f'{index}_{s}.png'),im)
            world,_,valid,_=local_patch_regions(obs)
            nearest=float(np.linalg.norm(world[valid]-gt,axis=-1).min()) if valid.any() else None
            ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.axis('off')
            for point,color,marker in [(gt,'lime','+'),(diags[s]['selected_goal_xy'],'red','x')]:
                uv=world_to_frame(np.array([point]),obs['footprint_world'],im.shape)[0]
                if ((uv>=0)&(uv<=383)).all():ax.scatter(*uv,c=color,marker=marker,s=80)
            mass=diags[s]['local_probability_mass'];ax.set_title(f'Step{s}: GoalNE {distances[s]:.1f}m\nlocal mass {mass:.3f}')
            per_frame.append(dict(step=s,goal_ne_m=distances[s],local_probability_mass=mass,nearest_valid_world_hypothesis_m=nearest,pixel_sha_verified=True,footprint_world=np.asarray(obs['footprint_world']).tolist()))
            overlap=polygon.intersection(Polygon(obs['footprint_world'])).area
            per_frame[-1].update(target_contour_intersection_m2=overlap,target_contour_covered_fraction=overlap/max(polygon.area,1e-9))
        fig.suptitle(record.instruction,fontsize=8,wrap=True);fig.tight_layout();fig.savefig(a.output/f'case_{index}.png',dpi=130);plt.close(fig)
        records.append(dict(split=a.split,index=index,instruction=record.instruction,loss_steps=loss_steps,frames=per_frame,
                            target_category=obj['object_type'],target_polygon_area_m2=polygon.area,
                            target_bounds=polygon.bounds,target_point_in_polygon=polygon.covers(Point(gt))))
        links.append(f'<h2>{a.split} {index}</h2><a href="case_{index}.png"><img width="1400" src="case_{index}.png"></a>')
    (a.output/'cases.json').write_text(json.dumps(records,indent=2))
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>配对闭环案例：实际输入像素SHA核验</h1><p>绿色仅是事后GT世界点投影，不是人工实例框；红色是当时Goal。不可见性仍未知。所有图像来自实际决策位姿。</p>'+''.join(links))
    print(json.dumps(records))


if __name__=='__main__':main()
