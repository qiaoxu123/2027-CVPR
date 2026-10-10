"""Post-rollout diagnostic maps. GT and full-source backgrounds never enter inference."""
import argparse,json,sys
from pathlib import Path
import cv2
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','sbf-source','data-root','split','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--cases',type=int,default=12);args=p.parse_args();out=Path(args.output)
    if out.exists():raise ValueError('refusing diagnostic overwrite')
    out.mkdir(parents=True);sys.path.insert(0,args.sbf_source)
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.memory import ObservedRGBMemory
    dataset=CityNavDataset(args.data_root,args.split,version='processed_citynav')
    def records(path):return {r['index']:r for r in map(json.loads,(Path(path)/'predictions.jsonl').read_text().splitlines())}
    baseline,candidate=records(args.baseline),records(args.candidate)
    if baseline.keys()!=candidate.keys():raise ValueError('paired diagnostic episode sets differ')
    if (Path(args.baseline)/'episodes.json').read_bytes()!=(Path(args.candidate)/'episodes.json').read_bytes():raise ValueError('paired start/manifest differs')
    groups={'gained':[],'lost':[],'both_success':[],'both_failed':[]}
    for key in baseline:
        a,b=baseline[key]['metrics']['SR'],candidate[key]['metrics']['SR']
        group='gained' if b>a else ('lost' if a>b else ('both_success' if a else 'both_failed'))
        groups[group].append(key)
    selected=[]
    for i in range(args.cases):
        group=list(groups)[i%4]
        if groups[group]:selected.append((group,groups[group].pop(0)))
    background_name=None;background=None;manifest=[]
    for group,key in selected:
        source=dataset[key];map_name=source.map_name
        if map_name!=background_name:
            background=cv2.cvtColor(cv2.imread(str(Path(args.data_root)/'rgbd'/(map_name+'.png'))),cv2.COLOR_BGR2RGB)
            with rasterio.open(Path(args.data_root)/'rgbd'/(map_name+'.tif')) as raster:
                transform=raster.transform
            background_name=map_name
        target=source.target_position[:2];expert=np.asarray(source.trajectory)[:,:2]
        paths=[np.asarray(baseline[key]['poses']),np.asarray(candidate[key]['poses'])]
        points=np.vstack([target[None],expert,*[x[:,:2] for x in paths]])
        lo,hi=points.min(0)-30,points.max(0)+30
        inv=~transform;left,top=inv*(lo[0],hi[1]);right,bottom=inv*(hi[0],lo[1])
        left=max(0,int(left));top=max(0,int(top));right=min(background.shape[1],int(right)+1);bottom=min(background.shape[0],int(bottom)+1)
        crop=background[top:bottom,left:right];x0,y1=transform*(left,top);x1,y0=transform*(right,bottom)
        fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
        for ax,path,row,label in zip(axes,paths,[baseline[key],candidate[key]],['Baseline','Candidate']):
            ax.imshow(crop,extent=(x0,x1,y0,y1),origin='upper')
            ax.plot(expert[:,0],expert[:,1],'--',color='white',lw=1.5,label='Human reference (diagnostic)')
            ax.plot(path[:,0],path[:,1],'-o',color='#0077ff',ms=3,lw=2,label='Executed path')
            ax.scatter(*path[0,:2],marker='^',s=80,color='lime',edgecolor='black',label='Start')
            ax.scatter(*path[-1,:2],marker='s',s=60,color='cyan',edgecolor='black',label='Final')
            ax.scatter(*target,marker='*',s=150,color='red',edgecolor='white',label='GT target (diagnostic only)')
            ax.add_patch(plt.Circle(target,20,color='red',fill=False,lw=1.5))
            for pose,color in ((path[0],'lime'),(path[-1],'cyan')):
                footprint=ObservedRGBMemory.view_corners(pose[:2],pose[3],pose[2]-GROUND_LEVEL[map_name])
                footprint=np.vstack((footprint,footprint[0]));ax.plot(footprint[:,0],footprint[:,1],':',color=color,lw=1)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
            ax.set_title(f"{label}: SR={int(row['metrics']['SR'])}, NE={row['metrics']['NE']:.1f}m\n{row['termination']}, {row['metrics']['steps']} moves")
            ax.set_xlabel('World East X (m)');ax.set_ylabel('World North Y (m)')
        axes[0].legend(fontsize=7,loc='lower left')
        fig.suptitle(f'{group} | {map_name}, episode {key}\n'+source.instruction,fontsize=9,wrap=True)
        name=f'{group}_{key}.png';fig.savefig(out/name,dpi=150);plt.close(fig)
        manifest.append(dict(group=group,index=key,map_name=map_name,image=name,instruction=source.instruction,
            baseline=baseline[key]['metrics'],candidate=candidate[key]['metrics']))
    (out/'manifest.json').write_text(json.dumps(dict(args=vars(args),cases=manifest,
        note='Full orthophoto, GT target and human future shown only after completed inference for diagnostic visualization.'),indent=2))


if __name__=='__main__':main()
