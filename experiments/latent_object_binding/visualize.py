"""Scientific comparison figures and actual causal RGB case illustrations."""
import collections,json,textwrap
import cv2,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,torch
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import project_landmark
from experiments.semantic_polygon_grounding.audit import DATA,save
from .resources import OUT,baseline,primary_features,regions
from .diagnose import read_rows
from .evaluate import materialize
from .inference import posterior


def geometry_lines(ax,graph,center=None,radius=180):
    for i,s in enumerate(graph.shapes):
        if center is not None and np.linalg.norm(graph.centers[i]-center)>radius:continue
        if s.geom_type=='Polygon':
            x,y=s.exterior.xy;ax.plot(x,y,color='.75',linewidth=.5)
    ax.set_aspect('equal');ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)')


def main():
    torch.set_num_threads(1);dest=OUT/'figures';dest.mkdir(exist_ok=True)
    full=json.loads((OUT/'full_dev_metrics.json').read_text());trained=json.loads((OUT/'trained_full_metrics.json').read_text());obs=json.loads((OUT/'observed_metrics.json').read_text());names=['A0','A1','A2','Hard','TrainedGeom','TrainedRGB'];m=[full['metrics'][k] for k in names[:4]]+[trained['results']['dev'][k]['metrics'] for k in ('geometry','rgb')]
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(names));ax.bar(x-.18,[100*a['object_r1']['mean'] for a in m],.35,label='Object R@1 (A0 shadow)');ax.bar(x+.18,[100*a['goal_r1']['mean'] for a in m],.35,label='Goal R@1');ax.set_xticks(x,names);ax.set_ylabel('%');ax.set_title('Full Internal Dev: 2,285 queries / 4 maps');ax.legend();fig.tight_layout();fig.savefig(dest/'full_dev_comparison.png',dpi=160);plt.close(fig)
    names=['A0','A2','A3_HSV','A3_SigLIP','A3_Both','MatchedGeom','A4_Real','A4_Black','A4_Shuffled'];m=obs['metrics']['dev'];fig,axs=plt.subplots(1,2,figsize=(13,4))
    for ax,key in zip(axs,('object_r1','goal_r1')):
        ax.bar(np.arange(len(names)),[100*m[n][key]['mean'] for n in names]);ax.set_xticks(np.arange(len(names)),names,rotation=55,ha='right');ax.set_ylabel('%');ax.set_title(key+' / 128 independent families')
    fig.tight_layout();fig.savefig(dest/'rgb_controls.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for name in ('geometry','rgb'):
        h=trained['heads'][name]['history'];axs[0].plot([r['epoch'] for r in h],[r['loss'] for r in h],'o-',label=name);axs[1].plot([r['epoch'] for r in h],[100*r['dev']['goal_r1']['mean'] for r in h],'o-',label=name)
    axs[0].set_title('Full Train loss');axs[1].set_title('Full Dev Goal R@1');[a.set_xlabel('epoch') for a in axs];[a.legend() for a in axs];fig.tight_layout();fig.savefig(dest/'training.png',dpi=160);plt.close(fig)
    a3,source,graphs,proof=baseline();maps=torch.load(OUT/'full_geometry/maps.pt',weights_only=True);features,states,manifest=primary_features('dev');states={s['metadata']['sample_id']:s for s in states};data=torch.load(OUT/'observed/dev_public.pt',weights_only=True,mmap=True);data={s['metadata']['sample_id']:s for s in data};labels={r['sample_id']:r for r in torch.load(OUT/'observed/dev_labels.pt',weights_only=True)};rows=read_rows(OUT/'observed_dev_A4_Real.jsonl.gz');base={r['sample_id']:r for r in read_rows(OUT/'observed_dev_A0.jsonl.gz')};weight=torch.load(OUT/'training/rgb/best.pt',weights_only=True)['weights'];selected=[]
    for predicate in [lambda r:r['metrics']['recovery']==1,lambda r:r['metrics']['damage']==1,lambda r:r['metrics']['goal_r1']==0 and r['metrics']['object_r1']==1,lambda r:r['metrics']['object_r1']==0 and labels[r['sample_id']]['target_object_id'] in regions(states[r['sample_id']]),lambda r:len(r['reference_assignments'])>1 and any(f['kind']=='between' for f in r['language']['factors'])]:
        candidate=next((r for r in rows if predicate(r) and r['sample_id'] not in selected),None)
        if candidate:selected.append(candidate['sample_id'])
    if len(selected)<4:
        for r in rows:
            if r['sample_id'] not in selected:selected.append(r['sample_id'])
            if len(selected)>=4:break
    renderer=CityNavObservedViewRenderer(DATA/'rgbd',OUT/'pixels/environment_private',image_size=384);case_manifest=[]
    for k,sid in enumerate(selected):
        d=data[sid];s=states[sid];lab=labels[sid];g=graphs[d['metadata']['map_name']];gm=maps[d['metadata']['map_name']];p=materialize(d['versions']['real'],gm,d['a0']);pred=posterior(p,weight);belief=pred['object_belief'].detach().numpy();top=np.argsort(-belief)[:8];xy=np.array(lab['target_world'])[:2];target=g.byid[lab['target_object_id']];visible=regions(s);ref=visible.get(lab['target_object_id']) or next(iter(visible.values()),None)
        fig,axs=plt.subplots(2,3,figsize=(13,8));axs=axs.flat
        geometry_lines(axs[0],g,xy);axs[0].scatter(g.centers[top,0],g.centers[top,1],s=30,c=belief[top],cmap='viridis');axs[0].scatter(*xy,marker='x',color='red',s=60);axs[0].set_xlim(xy[0]-180,xy[0]+180);axs[0].set_ylim(xy[1]-180,xy[1]+180);axs[0].set_title('Target marginal top-8 / GT offline only')
        # Reference beliefs are proposal distributions; no anonymous truth.
        geometry_lines(axs[1],g,xy)
        for n,(var,rp) in enumerate(d['versions']['real']['reference_posteriors'].items()):
            idx=np.array(rp['indices'],int)
            if len(idx):axs[1].scatter(g.centers[idx,0],g.centers[idx,1],s=20+300*np.array(rp['probabilities']),label='X'+var+' proposal')
        axs[1].scatter(*xy,marker='x',color='red');axs[1].set_xlim(xy[0]-180,xy[0]+180);axs[1].set_ylim(xy[1]-180,xy[1]+180);axs[1].set_title('Latent reference support, sizes = mass')
        if d['versions']['real']['reference_posteriors']:axs[1].legend(fontsize=7)
        axs[2].imshow(pred['heatmap'].detach().numpy().reshape(28,28),origin='lower');axs[2].set_title('28 x 28 goal belief, gate %.5f'%pred['gate'])
        rgbsource=None
        if ref:
            observation=renderer.current_rgbd(ref['map_name'],ref['actual_camera_pose'],ref['ground_level']);bgr=observation['bgr'];axs[3].imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));axs[3].set_title('Actual causal visited view step '+str(ref['source_step']));rgbsource={key:ref[key] for key in ('source_step','episode_id','actual_camera_pose','view_key')}
            roi=project_landmark(dict(g.objects[target],id=g.ids[target]),observation['footprint_world'],bgr.shape,min_visible=.0,min_pixels=1)
            if roi is not None:
                x,y,u,v=roi.bbox_xyxy;crop=bgr[y:v,x:u].copy();axs[4].imshow(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));mask=roi.mask[y:v,x:u].astype(bool);crop[~mask]=0;axs[5].imshow(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));axs[4].set_title('Projected target ROI, weak geometry');axs[5].set_title('Polygon foreground, not semantic truth')
            else:axs[4].text(.1,.5,'Target unobserved / projection unavailable');axs[5].text(.1,.5,'Unknown')
        else:
            for a in list(axs)[3:]:a.text(.1,.5,'No credible observed ROI')
        for a in (axs[3],axs[4],axs[5]):a.axis('off')
        variables='; '.join(e['variable']+': '+e['phrase'] for e in d['versions']['real']['language']['variables']);fig.suptitle('\n'.join(textwrap.wrap(d['metadata']['instruction'],110))+'\n'+'\n'.join(textwrap.wrap(variables,130)),fontsize=9);fig.tight_layout(rect=(0,0,1,.90));name=f'case_{k:02d}.png';fig.savefig(dest/name,dpi=130);plt.close(fig)
        rr=next(r for r in rows if r['sample_id']==sid);case_manifest.append(dict(sample_id=sid,map_name=rr['map_name'],instruction=rr['instruction'],variables=d['versions']['real']['language']['variables'],factors=d['versions']['real']['language']['factors'],reference_proposals=d['versions']['real']['reference_posteriors'],reference_assignments=d['versions']['real']['assignments'],target_top8=[g.ids[i] for i in top],target_top8_probability=belief[top].tolist(),metrics=rr['metrics'],baseline=base[sid]['metrics'],rgb_source=rgbsource,figure=name,offline_only='GT used to select and illustrate cases only, never model input'))
    save(dest/'cases.json',case_manifest)

if __name__=='__main__':main()
