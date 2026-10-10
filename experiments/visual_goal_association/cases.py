"""Actual observed RGB plus geometry/visual residual failure figures."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import project_landmark
from .evaluate import load_prepared,observed_candidates,public_prediction,write_json


def run(prepared,evaluation,output,objects,rgbd):
    output=Path(output)
    if output.exists():raise ValueError('no overwrite')
    output.mkdir(parents=True);data=load_prepared(prepared);public=data['public'];labels=data['labels'];state={r['sample_id']:r for r in public};raw=json.loads(Path(objects).read_text());metrics=json.loads((Path(evaluation)/'metrics.json').read_text());selected=[]
    def rows(name):return [json.loads(x) for x in (Path(evaluation)/(name+'_predictions.jsonl')).read_text().splitlines()]
    selection={}
    for method in ['polygon_overlap','distance_soft','instance']:
        for trigger in ['recovery','damage']:
            options=[r for r in rows(method+'_selected_real') if r['metrics'][trigger]]
            if options:selection[method+'_'+trigger]=(method,options[0])
    centroid={r['sample_id']:r for r in rows('centroid_oracle_current_all_cached_rois')};poly=rows('polygon_overlap_oracle_current_all_cached_rois')
    differences=[r for r in poly if r['metrics']['r1']<centroid[r['sample_id']]['metrics']['r1']]
    if differences:selection['polygon_identity_still_wrong']=('polygon_overlap',differences[0])
    renderer=CityNavObservedViewRenderer(rgbd,output/'environment_private',image_size=384)
    memory_by_method={}
    for method in ['polygon_overlap','distance_soft','instance']:
        conf=metrics['config']['selected'][method]
        memory_by_method[method]={s['sample_id']:r for s,r in observed_candidates(public,conf['history'],'real')}
    for i,(reason,(method,pred)) in enumerate(selection.items()):
        sid=pred['sample_id'];s=state[sid];lab=labels[sid];conf=metrics['config']['selected'][method];regions=memory_by_method[method][sid]
        updated,known,residual,weights,values=public_prediction(s,regions,conf,'real');pose=s['pose'];fig,axes=plt.subplots(1,4,figsize=(16,4))
        if pose[2]-s['ground']>0:
            obs=renderer.current_rgbd(s['map_name'],pose,s['ground']);image=cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB)
            axes[0].imshow(image)
            for reg in s['all_regions']:
                oid=reg['object_id'];o=raw[s['map_name']][oid];roi=project_landmark(dict(id=oid,contour=o['contour']),obs['footprint_world'],image.shape,min_visible=.9,min_pixels=256)
                if roi is not None:
                    px=np.vstack([roi.contour_px,roi.contour_px[0]]);axes[0].plot(px[:,0],px[:,1],color='lime' if oid==lab['target_object_id'] else 'orange',lw=1)
            axes[0].set_xlim(0,383);axes[0].set_ylim(383,0)
        axes[0].set_title('Actual current RGB / weak polygons');axes[0].axis('off')
        bounds=s['bounds'];xy=bounds[:2]+s['grid_xy']*(bounds[2:]-bounds[:2]);target=lab['target_world']
        for ax,field,title in zip(axes[1:],[s['logits'],updated,residual],['Frozen Geometry','Frozen real RGB update','Residual (unknown = 0)']):
            ax.imshow(field.reshape(28,28),origin='lower',extent=[bounds[0],bounds[2],bounds[1],bounds[3]],cmap='coolwarm' if title.startswith('Residual') else 'viridis')
            ax.scatter(*target,marker='*',color='red',s=80,label='GT evaluation only');ax.scatter(*pose[:2],marker='x',color='white',s=45)
            ax.scatter(*xy[int(np.argmax(s['logits']))],marker='o',facecolors='none',edgecolors='cyan',s=60);ax.scatter(*xy[int(np.argmax(updated))],marker='s',facecolors='none',edgecolors='white',s=60)
            ax.set_title(title);ax.set_xlabel('World X / m');ax.set_ylabel('World Y / m')
        fig.suptitle(reason+' | '+sid+' | '+method,fontsize=9);fig.tight_layout();name=f'case_{i:02}.png';fig.savefig(output/name,dpi=130);plt.close(fig)
        selected.append(dict(path=name,reason=reason,sample_id=sid,map_name=s['map_name'],mapping=method,config=conf,prediction=pred,interpretation='GT plotted only after public prediction; source RGB current only; polygon registration weak'))
    write_json(output/'cases.json',selected);print('Actual cases:',len(selected))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);p.add_argument('--objects',required=True);p.add_argument('--rgbd',required=True);a=p.parse_args();run(a.prepared,a.evaluation,a.output,a.objects,a.rgbd)
