"""Frozen standalone visual-belief audit. No geometry logits enter prediction."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shapely
import torch
from graphnav.checkpointing import file_sha256
from graphnav.geometry_guided_data import load_landmark_data,batch_landmarks
from graphnav.geometry_guided_visual import LandmarkVisualBelief
from graphnav.landmark_region_supervision import named_union
from graphnav.landmark_visual_only import visual_only_world_belief
from scripts.train_geometry_guided_visual import replacements


def prediction_index(prob,covered,world):
    # Uniform/tied maps use the tied cell nearest their centroid, for all methods.
    ids=np.flatnonzero(covered & np.isclose(prob,prob.max(),rtol=1e-5,atol=1e-10))
    if not len(ids):raise ValueError('No valid prediction')
    center=world[ids].mean(0)
    return int(ids[np.argmin(((world[ids]-center)**2).sum(-1))])


def summarize(rows):
    if not rows:return {'n':0}
    return dict(n=len(rows),groups=len({r['query_key'] for r in rows}),maps=len({r['map_name'] for r in rows}),
        correct=sum(r['success'] for r in rows),r20=float(np.mean([r['success'] for r in rows])),
        ne=float(np.mean([r['ne'] for r in rows])),region_mass=float(np.mean([r['region_mass'] for r in rows])),
        uniform_region_mass=float(np.mean([r['uniform_region_mass'] for r in rows])),
        conditional_G_to_V_KL=float(np.mean([r['teacher_kl'] for r in rows])),
        corrected_vs_geometry=sum(r['success'] and not r['geometry_success'] for r in rows),
        lost_vs_geometry=sum(not r['success'] and r['geometry_success'] for r in rows),
        weak_eligible=sum(r['weak_eligible'] for r in rows),weak_top1=sum(r['weak_top1'] for r in rows if r['weak_eligible']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','teacher','catalog','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new audit directory')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(0)
    data=load_landmark_data(a.data,a.teacher,a.catalog)
    trained=LandmarkVisualBelief().cuda().eval().requires_grad_(False)
    trained.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model'])
    native=LandmarkVisualBelief().cuda().eval().requires_grad_(False)
    region_cache={};rows=[];abstentions=[];fields_cache={};started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    controls=('native','trained','coverage_only','zero_rgb','shuffle_rgb','same_type_rgb','wrong_footprint')
    for subset in ('dev','confirm'):
        selected=[r for r in data if r['metadata']['subset']==subset]
        replace=replacements(selected,True)
        for start in range(0,len(selected),16):
            chosen=selected[start:start+16]
            for control in controls:
                current=[r for r in chosen if r['metadata']['sample_key'] in replace] if control=='same_type_rgb' else chosen
                if not current:continue
                b=batch_landmarks(current,control=control if control in ('zero_rgb','shuffle_rgb','same_type_rgb') else 'real',
                    replacement=[replace[r['metadata']['sample_key']] for r in current] if control=='same_type_rgb' else None)
                with torch.inference_mode():
                    s=(native if control=='native' else trained)(image_features=b['image_features'],name_text=b['name_text'],type_text=b['type_text'],valid=b['valid'])
                    if control=='coverage_only':s=torch.zeros_like(s)
                    uv=b['local_uv']
                    if control=='wrong_footprint':uv=1-uv # fixed public-coordinate corruption, no target-guided choice
                    v=visual_only_world_belief(local_uv=uv,scores=s,valid=b['valid'])
                for i,r in enumerate(current):
                    q=r['metadata']['query_key'];key=r['metadata']['sample_key']
                    if q not in region_cache:
                        bounds=r['teacher']['metadata']['world_bounds'];x0,y0,x1,y1=bounds
                        dx=(x1-x0)/28;dy=(y1-y0)/28
                        world=r['frame']['candidate_world'][:784].numpy()
                        cells=shapely.box(world[:,0]-dx/2,world[:,1]-dy/2,world[:,0]+dx/2,world[:,1]+dy/2)
                        frac=shapely.area(shapely.intersection(cells,named_union(r['query']['public_polygons'])))/(dx*dy)
                        region_cache[q]=(world,np.clip(frac,0,1))
                    world,fraction=region_cache[q]
                    if not bool(v['has_view'][i]):
                        abstentions.append(dict(subset=subset,control=control,sample_key=key,reason='no_valid_world_observation'))
                        continue
                    prob=v['probabilities'][i].cpu().numpy();covered=v['covered'][i].cpu().numpy()
                    idx=prediction_index(prob,covered,world);ne=float(r['region_distance'][idx])
                    # Teacher is accessed only after standalone prediction for comparison.
                    teacher=b['geometry_logits'][i].float().masked_fill(~v['covered'][i],float('-inf')).softmax(-1).cpu().numpy()
                    kl=float((teacher[covered]*(np.log(np.maximum(teacher[covered],1e-30))-np.log(np.maximum(prob[covered],1e-30)))).sum())
                    gi=int(r['teacher']['geometry'].argmax());gne=float(r['region_distance'][gi])
                    local=int(s[i].masked_fill(~b['valid'][i],float('-inf')).argmax())
                    rows.append(dict(subset=subset,control=control,sample_key=key,query_key=q,map_name=r['query']['map_name'],
                        type=r['metadata']['public_type'],projection_supported=r['projection_supported'],visibility='unknown',
                        human_pixel_gt=False,ne=ne,success=ne<=20,world_xy=world[idx].tolist(),geometry_ne=gne,geometry_success=gne<=20,
                        region_mass=float(prob@fraction),uniform_region_mass=float(fraction[covered].mean()),teacher_kl=kl,
                        weak_eligible=bool((r['labels']==1).any() and (r['labels']==0).any()),weak_top1=bool(r['labels'][local]==1)))
                    if subset=='dev' and control in ('native','trained','coverage_only'):
                        fields_cache[(key,control)]=dict(prob=prob,covered=covered)
    summaries={}
    for subset in ('dev','confirm'):
        summaries[subset]={}
        for control in controls:
            rr=[r for r in rows if r['subset']==subset and r['control']==control]
            summaries[subset][control]={stratum:summarize([r for r in rr if predicate(r)]) for stratum,predicate in
                [('all_views',lambda r:True),('supported',lambda r:r['projection_supported']),('unsupported',lambda r:not r['projection_supported'])]}
            summaries[subset][control]['categories']={t:summarize([r for r in rr if r['type']==t and r['projection_supported']]) for t in sorted({r['type'] for r in rr})}
    delta={}
    for subset in ('dev','confirm'):
        left={r['sample_key']:r for r in rows if r['subset']==subset and r['control']=='native' and r['projection_supported']}
        right={r['sample_key']:r for r in rows if r['subset']==subset and r['control']=='trained' and r['projection_supported']}
        delta[subset]=dict(corrected=sum(right[k]['success'] and not left[k]['success'] for k in left),
            lost=sum(not right[k]['success'] and left[k]['success'] for k in left))
    result=dict(summaries=summaries,trained_vs_native=delta,abstentions=abstentions,
        numerical_tie_rule='max probability relative tolerance 1e-5; nearest tied-cell centroid, no Polygon access',
        inputs={k:str(getattr(a,k)) for k in ('data','teacher','catalog','checkpoint')},
        artifact_sha256={str(path):file_sha256(path) for path in (a.data/'manifest.json',a.teacher/'manifest.json',a.catalog,Path(__file__),Path('graphnav/landmark_visual_only.py'))},seconds=time.perf_counter()-started,
        peak_allocated_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_sha256=file_sha256(a.checkpoint),no_geometry_logits_in_visual_prediction=True,
        no_training_or_navigation=True,actual_visibility_verified=False,physical_instance_independent=False,
        note='Repeated audit of existing exposed TRAIN splits, no new confirmation claim. Polygon-area mass assumes uniform density within each grid cell; not pixel instance accuracy.')
    (a.output/'metrics.json').write_text(json.dumps(result,indent=2));(a.output/'rows.json').write_text(json.dumps(rows,indent=2))
    cases=[];dev=[r for r in rows if r['subset']=='dev' and r['control']=='trained']
    for label,predicate in [('success',lambda r:r['projection_supported'] and r['success']),('failure',lambda r:r['projection_supported'] and not r['success']),('unsupported',lambda r:not r['projection_supported'])]:
        for typ in ('Building','Road'):
            match=next((r for r in dev if r['type']==typ and predicate(r)),None)
            if match:cases.append((label,match))
    lookup={r['metadata']['sample_key']:r for r in data}
    for j,(label,metric) in enumerate(cases):
        r=lookup[metric['sample_key']];frame=r['metadata']['frame'];image_path=Path(r['metadata']['observations'])/frame['image']
        if file_sha256(image_path)!=frame['image_sha256']:raise ValueError('RGB SHA mismatch')
        rgb=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        fig,axes=plt.subplots(1,4,figsize=(16,4));axes[0].imshow(rgb);axes[0].set_title('Actual RGB (no target overlay)');axes[0].axis('off')
        bounds=r['teacher']['metadata']['world_bounds'];extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
        for ax,c,title in zip(axes[1:],('coverage_only','native','trained'),('Coverage only','Untrained visual only','Trained visual only')):
            field=fields_cache[(metric['sample_key'],c)];values=np.where(field['covered'],field['prob'],np.nan).reshape(28,28)
            ax.imshow(values,origin='lower',extent=extent,cmap='magma')
            for poly in r['query']['public_polygons']:
                xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],color='cyan',lw=1)
            footprint=np.asarray(frame['footprint_world']);footprint=np.vstack([footprint,footprint[0]])
            ax.plot(footprint[:,0],footprint[:,1],color='lime',ls='--',lw=.8)
            rr=next(x for x in rows if x['sample_key']==metric['sample_key'] and x['control']==c)
            ax.scatter(*rr['world_xy'],c='red',s=20);ax.set_title(f'{title}\nNE={rr["ne"]:.1f}m, mass={rr["region_mass"]:.3f}',fontsize=9)
        fig.suptitle(f'{r["query"]["instruction"]} | {label} | polygon weak reference only')
        fig.tight_layout();fig.savefig(a.output/f'case_{j}.png',dpi=150);plt.close(fig)
        cv2.imwrite(str(a.output/f'case_{j}_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    html='<!doctype html><meta charset="utf-8"><title>独立Visual Belief</title><style>body{max-width:1600px;margin:auto;font-family:sans-serif}img{width:100%}</style><h1>独立Visual Belief，不融合几何logits</h1><p>仅TRAIN既有分区回归；真实可见性未核验，非像素GT。青色为公开Polygon事后参考，绿色为观测Footprint，红点为预测。观测域外未知。</p><a href="REPORT_zh.md">中文报告</a> | <a href="metrics.json">完整统计</a>'
    for j,(label,m) in enumerate(cases):html+=f'<h2>{label}: {m["query_key"]}</h2><a href="case_{j}.png"><img src="case_{j}.png"></a><a href="case_{j}_rgb.png">原始RGB</a>'
    (a.output/'index.html').write_text(html)
    print(json.dumps({'summaries':summaries,'trained_vs_native':delta},indent=2))


if __name__=='__main__':main()
