"""Offline public relation plots with explicitly separate GT annotation layer."""
from common import *
import argparse
import textwrap
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame
from sbfnav.mapdata import GROUND_LEVEL
from schema import to_device
from fusion import local_mix


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--limit',type=int,default=6);p.add_argument('--diagnostic-alpha',type=float,default=0.)
    p.add_argument('--all-samples',action='store_true',help='Include target-unmatched or target-out-of-view samples instead of matched debugging examples')
    args=p.parse_args();torch.set_num_threads(2);manifest=json.loads((ROOT/'data/manifest.json').read_text())
    renderer=CityNavObservedViewRenderer(RGBD,ROOT/'runs/visualize_render',image_size=768)
    model=None
    if args.checkpoint:model,_=load_branch(args.checkpoint,'cpu')
    selected=[]
    for entry in manifest['entries']:
        r=torch.load(resolve_row_path(entry['path']),map_location='cpu',weights_only=False)
        eligible=args.all_samples or (bool(r['labels']['ranking'].sum()) and r['private']['target_visible'])
        if r['meta']['candidates'] and r['meta']['references'] and eligible:selected.append(r)
        if len(selected)>=args.limit:break
    for i,row in enumerate(selected):
        meta=row['meta'];map_name=row['target_instance'][0];ob=renderer.current_rgbd(map_name,np.asarray(meta['pose']),GROUND_LEVEL[map_name]);image=ob['bgr'][...,::-1]
        def pixels(xy):return world_to_frame(np.asarray(xy,np.float32).reshape(-1,2),ob['footprint_world'],ob['bgr'].shape)
        scores=np.zeros(len(meta['candidates']));s=row['public']['spatial'];updated=s['base_logits']
        with torch.no_grad():
            if model:
                out=model(**to_device(row['public'],'cpu'));scores=out['scores'].numpy()
                updated=local_mix(s['base_logits'],s['grid_world'],s['candidate_xy']+out['offset'],out['scores'],s['candidate_support'],args.diagnostic_alpha)['logits']
        fig,axes=plt.subplots(1,3,figsize=(18,6))
        for ax in axes:ax.imshow(image);ax.set_xlim(0,image.shape[1]);ax.set_ylim(image.shape[0],0);ax.axis('off')
        for j,c in enumerate(meta['candidates']):
            q=pixels(c['poly']);q=np.concatenate((q,q[:1]));axes[0].plot(q[:,0],q[:,1],lw=1,color='lime')
            point=pixels([c['xy']])[0];axes[0].text(*point,f'C{j}: {scores[j]:.2f}',color='yellow',fontsize=7,clip_on=True)
        for j,r in enumerate(meta['references']):
            q=pixels(r['poly']);q=np.concatenate((q,q[:1]));axes[0].plot(q[:,0],q[:,1],color='cyan');point=pixels([r['xy']])[0]
            axes[0].text(*point,f'R{j}: {r["name"]}',color='cyan',fontsize=7,clip_on=True)
        if len(scores):
            j=int(scores.argmax());a=pixels([meta['candidates'][j]['xy']])[0]
            for k,r in enumerate(meta['references'][:3]):
                b=pixels([r['xy']])[0];axes[0].plot([a[0],b[0]],[a[1],b[1]],color='orange',lw=1)
                d=np.asarray(meta['candidates'][j]['xy'])-np.asarray(r['xy']);angle=np.degrees(np.arctan2(d[1],d[0]))
                pair=row['public']['network']['pair'][j,k];height=f'{float(pair[8])*20:.1f}m' if pair[12]>.5 else 'unknown'
                axes[0].text(*((a+b)/2),f'{angle:.0f}deg / {np.linalg.norm(d):.1f}m / dz={height}',color='white',fontsize=7,clip_on=True)
        gp=pixels(s['grid_world'].numpy())
        for ax,logits,title in [(axes[1],s['base_logits'],'Original 28x28'),(axes[2],updated,f'DIAGNOSTIC local mix alpha={args.diagnostic_alpha}')]:
            ax.scatter(gp[:,0],gp[:,1],c=logits.softmax(0).numpy(),cmap='hot',s=35,alpha=.7);ax.set_title(title)
        gt=pixels(row['labels']['target_xy'].numpy()[None])[0]
        polygon=pixels(row['private']['target_polygon']);polygon=np.concatenate((polygon,polygon[:1]))
        axes[0].plot(polygon[:,0],polygon[:,1],color='magenta',lw=2)
        for ax in axes:ax.scatter(*gt,marker='*',s=150,color='magenta',edgecolor='white')
        axes[0].set_title('Public boxes / reference relations; magenta = offline GT')
        fig.suptitle(textwrap.fill(row['instruction'],140),fontsize=10,y=.98)
        fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.84,wspace=.04)
        path=ROOT/f'reports/example_{i:02d}.png';fig.savefig(path,dpi=150);plt.close(fig)
        record=dict(instruction=row['instruction'],meta=meta,private_annotation=row['private'],diagnostic_alpha=args.diagnostic_alpha,
            caveat='Diagnostic mixing is not calibrated inference and not evidence of improvement')
        record=json.loads(json.dumps(record,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
        save(record,path.with_suffix('.json'))
        print(path,flush=True)


if __name__=='__main__':main()
