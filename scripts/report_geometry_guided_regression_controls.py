"""Name-navigation RGB-zero control and actual fixed-yaw corrected trajectories."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.fixed_yaw_observation import FixedYawObservation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('real-run','zero-run','tasks','data-root','sbf-source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--original-run',type=Path)
    p.add_argument('--native-run',type=Path);p.add_argument('--shuffle-run',type=Path)
    a=p.parse_args();sys.path.insert(0,str(a.sbf_source))
    from sbfnav.mapdata import GROUND_LEVEL
    tasks=json.loads(a.tasks.read_text())
    real=[json.loads(x) for x in (a.real_run/'predictions.jsonl').read_text().splitlines()]
    zero=[json.loads(x) for x in (a.zero_run/'predictions.jsonl').read_text().splitlines()]
    if len(real)!=128 or len(zero)!=64:raise ValueError('Completed paired64 real and zero control required')
    original=[json.loads(x) for x in (a.original_run/'predictions.jsonl').read_text().splitlines()] if a.original_run else []
    if a.original_run and len(original)!=64:raise ValueError('Complete original RGB baseline required')
    native=[json.loads(x) for x in (a.native_run/'predictions.jsonl').read_text().splitlines()] if a.native_run else []
    shuffled=[json.loads(x) for x in (a.shuffle_run/'predictions.jsonl').read_text().splitlines()] if a.shuffle_run else []
    if a.native_run and len(native)!=64:raise ValueError('Native control incomplete')
    if a.shuffle_run and len(shuffled)!=64:raise ValueError('Shuffle control incomplete')
    paired={i:{r['method']:r for r in real+zero+original+native+shuffled if r['index']==i} for i in range(64)}
    for pair in paired.values():
        assert len({tuple(r['poses'][0]) for r in pair.values()})==1
        assert len({r['instruction'] for r in pair.values()})==1
    metrics={}
    methods=['Geometry','GeometryVisual','GeometryZeroVisual']+(['OriginalRGB'] if original else [])+(['GeometryNativeVisual'] if native else [])+(['GeometryShuffledVisual'] if shuffled else [])
    for m in methods:
        rows=[r for pair in paired.values() for name,r in pair.items() if name==m]
        metrics[m]=dict(episodes=len(rows),successes=int(sum(r['metrics']['SR'] for r in rows)),
            **{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('SR','OSR','NE')})
    corrected=[i for i,pair in paired.items() if pair['GeometryVisual']['metrics']['SR'] and not pair['Geometry']['metrics']['SR']]
    harmed=[i for i,pair in paired.items() if not pair['GeometryVisual']['metrics']['SR'] and pair['Geometry']['metrics']['SR']]
    real_vs_zero=dict(corrected=sum(p['GeometryVisual']['metrics']['SR'] and not p['GeometryZeroVisual']['metrics']['SR'] for p in paired.values()),
        harmed=sum(not p['GeometryVisual']['metrics']['SR'] and p['GeometryZeroVisual']['metrics']['SR'] for p in paired.values()))
    result=dict(metrics=metrics,corrected_indices=corrected,harmed_indices=harmed,real_vs_zero=real_vs_zero,
        McNemar_exact_two_sided=.5 if len(corrected)==2 and not harmed else None,
        warning='Fixed64 named-landmark diagnostic, not original CityNav task SR; two improved episodes do not establish statistical significance.')
    if original:
        result['real_vs_original']=dict(corrected=sum(p['GeometryVisual']['metrics']['SR'] and not p['OriginalRGB']['metrics']['SR'] for p in paired.values()),
            harmed=sum(not p['GeometryVisual']['metrics']['SR'] and p['OriginalRGB']['metrics']['SR'] for p in paired.values()))
        assert all(d['visual_control']=='real' for r in original for d in r['step_diagnostics'])
    for key,method in [('real_vs_native','GeometryNativeVisual'),('real_vs_shuffled','GeometryShuffledVisual')]:
        if method in metrics:
            result[key]=dict(corrected=sum(p['GeometryVisual']['metrics']['SR'] and not p[method]['metrics']['SR'] for p in paired.values()),
                harmed=sum(not p['GeometryVisual']['metrics']['SR'] and p[method]['metrics']['SR'] for p in paired.values()))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'regression_controls.json').write_text(json.dumps(result,indent=2))
    for name in ('metrics.json','run_manifest.json','predictions.jsonl'):shutil.copyfile(a.zero_run/name,a.output/('zero_'+name))
    if a.original_run:
        for name in ('metrics.json','run_manifest.json','predictions.jsonl'):shutil.copyfile(a.original_run/name,a.output/('original_rgb_'+name))
    for path,prefix in [(a.native_run,'native'),(a.shuffle_run,'shuffled')]:
        if path:
            for name in ('metrics.json','run_manifest.json','predictions.jsonl'):shutil.copyfile(path/name,a.output/(prefix+'_'+name))
    fig,axes=plt.subplots(1,2,figsize=(14,4))
    labels=[m.replace('Geometry','G+').replace('Visual','V') for m in methods]
    for ax,key in zip(axes,('SR','NE')):
        ax.bar(range(len(methods)),[metrics[m][key] for m in methods]);ax.set_xticks(range(len(methods)),labels,rotation=25,ha='right');ax.set_title(key+' (same fixed64)')
    fig.tight_layout();fig.savefig(a.output/'navigation_ablations.png',dpi=160);plt.close(fig)
    renderer=FixedYawObservation(CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'frames',image_size=384),0.)
    cases=[]
    for i in corrected:
        t=tasks[i];pair=paired[i];fig,axes=plt.subplots(1,4,figsize=(16,4))
        for poly in t['public_polygons']:
            xy=np.array(poly);axes[0].fill(xy[:,0],xy[:,1],color='gray',alpha=.25)
        plot_methods=[('Geometry','green'),('GeometryVisual','royalblue'),('GeometryZeroVisual','darkorange')]
        if native:plot_methods.append(('GeometryNativeVisual','purple'))
        if shuffled:plot_methods.append(('GeometryShuffledVisual','red'))
        for m,c in plot_methods:
            path=np.array(pair[m]['poses']);axes[0].plot(path[:,0],path[:,1],'.-',color=c,label=m)
        axes[0].scatter(*t['start_pose'][:2],marker='*',color='black');axes[0].set_aspect('equal');axes[0].legend(fontsize=6)
        for j,m in enumerate(('Geometry','GeometryVisual','GeometryZeroVisual'),1):
            view=renderer.current_rgbd(t['map_name'],pair[m]['poses'][-1],GROUND_LEVEL[t['map_name']])
            axes[j].imshow(cv2.cvtColor(view['bgr'],cv2.COLOR_BGR2RGB));axes[j].axis('off')
            axes[j].set_title(m+f" NE={pair[m]['metrics']['NE']:.2f}m",fontsize=9)
            cv2.imwrite(str(a.output/f'nav_case_{i}_{m}.png'),view['bgr'])
        fig.suptitle(t['instruction']+' | all camera yaw=0, real visited end poses');fig.tight_layout()
        file=f'nav_case_{i}.png';fig.savefig(a.output/file,dpi=150);plt.close(fig);cases.append(dict(index=i,file=file,name=t['landmark_name']))
    (a.output/'nav_cases.json').write_text(json.dumps(cases,indent=2))
    index=a.output/'index.html'
    if index.exists():
        page=index.read_text().split('<h1>真实闭环与RGB置零对照</h1>')[0]+'<h1>真实闭环与RGB置零对照</h1><img src="navigation_ablations.png">'
        for case in cases:page+=f'<h2>{case["name"]}</h2><a href="{case["file"]}"><img src="{case["file"]}"></a>'
        index.write_text(page)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
