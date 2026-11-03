"""Completed visual-only named-landmark results, with matched interface control."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import numpy as np
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.landmark_navigation_diagnostic import public_region
from graphnav.renderer import CityNavObservedViewRenderer
from scripts.report_named_landmark_navigation import summarize,failure_tags


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','original-run','selection','data-root','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();metadata=json.loads((a.run/'metrics.json').read_text())
    if not metadata['complete'] or metadata['geometry_branch_calls']!=0:raise ValueError('Complete zero-geometry64 pairs required')
    extra=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
    original=[json.loads(x) for x in (a.original_run/'predictions.jsonl').read_text().splitlines()]
    tasks=json.loads((a.selection/'tasks_public.json').read_text());rows=original+extra
    for method in ('E1','E2','E3_visual_only','E1_neutral_context'):
        group=[r for r in rows if r['method']==method]
        if len(group)!=64 or {r['index'] for r in group}!=set(range(64)):raise ValueError('Same task set required')
    for r in rows:
        region=public_region(tasks[r['index']]['public_polygons'],tasks[r['index']]['world_bounds'])
        r['strict_ne_m']=float(shapely.distance(shapely.Point(r['poses'][-1][:2]),region))
        r['failure_tags']=failure_tags(r)
    for r in extra:
        assert all(d['geometry_branch_calls']==0 for d in r['step_diagnostics'])
        assert all(d['candidate_context']=='neutral_zero_no_A3' for d in r['step_diagnostics'])
        assert all(d['oracle_control']==(r['method']=='E1_neutral_context') for d in r['step_diagnostics'])
        counterpart=next(x for x in original if x['method']=='E2' and x['index']==r['index'])
        assert counterpart['poses'][0]==r['poses'][0] and counterpart['instruction']==r['instruction']
    methods=('E1','E2','E3_visual_only','E1_neutral_context')
    by_method={m:summarize([r for r in rows if r['method']==m]) for m in methods}
    by_category={m:{c:summarize([r for r in rows if r['method']==m and r['category']==c]) for c in ('Building','Road','Other')} for m in methods}
    e3=[r for r in rows if r['method']=='E3_visual_only']
    paired={i:{r['method']:r for r in rows if r['index']==i} for i in range(64)}
    paired_changes=dict(visual_only_corrected_full=sum(not p['E2']['metrics']['SR'] and p['E3_visual_only']['metrics']['SR'] for p in paired.values()),
        visual_only_harmed_full=sum(p['E2']['metrics']['SR'] and not p['E3_visual_only']['metrics']['SR'] for p in paired.values()))
    from collections import Counter
    audit=dict(geometry_calls=0,all_candidate_context_neutral=True,no_geometry_query_written=not list((a.run/'unused_geometry_inputs').glob('*')),
        first_goal_correct=sum(r['goal_analysis']['initial_ne_m']<=20 for r in e3),ever_correct=sum(r['goal_analysis']['ever_correct'] for r in e3),
        final_goal_correct=sum(r['goal_analysis']['final_ne_m']<=20 for r in e3),goal_correct_lost=sum(r['goal_analysis']['lost_correct'] for r in e3),
        failures_nonexclusive=dict(Counter(tag for r in e3 for tag in r['failure_tags'] if tag!='success')),
        e3_tasks_with_zero_observed_first_frame=sum(r['step_diagnostics'][0]['observed_cells']==0 for r in e3),
        e3_total_decisions=sum(len(r['step_diagnostics']) for r in e3),
        e3_zero_observed_decisions=sum(d['observed_cells']==0 for r in e3 for d in r['step_diagnostics']),
        decision_median_ms={m:float(np.median([s for r in extra if r['method']==m for s in r['decision_seconds']])*1000) for m in methods[2:]})
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(by_method=by_method,by_category=by_category,paired_changes=paired_changes,audit=audit,runtime=metadata,
        caution='Original residual on uniform prior, neutral controller context; not an independently trained visual navigator. Name-only task, public polygon used only for E3 posthoc scoring.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    for file in ('predictions.jsonl','run_manifest.json','metrics.json'):shutil.copyfile(a.run/file,a.output/file)
    with (a.output/'per_task.csv').open('w') as f:
        fields=['index','method','map_name','landmark_name','category','SR','OSR','NE','termination','goal_initial_ne','goal_final_ne','failure_tags']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in extra:
            w.writerow(dict(index=r['index'],method=r['method'],map_name=r['map_name'],landmark_name=r['landmark_name'],category=r['category'],
                SR=r['metrics']['SR'],OSR=r['metrics']['OSR'],NE=r['metrics']['NE'],termination=r['termination'],
                goal_initial_ne=r['goal_analysis']['initial_ne_m'],goal_final_ne=r['goal_analysis']['final_ne_m'],failure_tags=';'.join(failure_tags(r))))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,k in zip(axes,('SR','OSR','NE')):
        ax.bar(range(4),[by_method[m][k] for m in methods]);ax.set_xticks(range(4),['Oracle','Full','Visual only','Oracle zero ctx'],rotation=20);ax.set_title(k)
    fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=160);plt.close(fig)
    choices=[]
    for predicate in [lambda p:p['E3_visual_only']['metrics']['SR'] and not p['E2']['metrics']['SR'],
        lambda p:not p['E3_visual_only']['metrics']['SR'] and p['E2']['metrics']['SR'] and p['E2']['category']=='Building',
        lambda p:p['E3_visual_only']['metrics']['OSR'] and not p['E3_visual_only']['metrics']['SR'],
        lambda p:p['E3_visual_only']['termination']=='horizon']:
        ids=[i for i,x in paired.items() if predicate(x) and i not in choices]
        if ids:choices.append(ids[0])
    renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'frames',image_size=384)
    from sbfnav.mapdata import GROUND_LEVEL
    import cv2
    cases=[]
    for i in choices:
        t=tasks[i];pair=paired[i];fig,axes=plt.subplots(1,3,figsize=(12,4))
        for poly in t['public_polygons']:
            pxy=np.array(poly);axes[0].fill(pxy[:,0],pxy[:,1],color='gray',alpha=.25)
        for method,color in [('E2','royalblue'),('E3_visual_only','darkorange')]:
            path=np.array(pair[method]['poses']);axes[0].plot(path[:,0],path[:,1],'.-',color=color,label=method)
            goals=np.array([d['selected_goal_xy'] for d in pair[method]['step_diagnostics']]);axes[0].scatter(goals[:,0],goals[:,1],color=color,marker='x',alpha=.4)
        axes[0].scatter(*t['start_pose'][:2],color='black',marker='*');axes[0].set_aspect('equal');axes[0].legend(fontsize=7)
        for j,method in enumerate(('E2','E3_visual_only'),1):
            obs=renderer.current_rgbd(t['map_name'],pair[method]['poses'][-1],GROUND_LEVEL[t['map_name']])
            axes[j].imshow(cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB));axes[j].axis('off')
            axes[j].set_title(f"{method}: NE={pair[method]['metrics']['NE']:.1f}m",fontsize=10)
            (a.output/'frames').mkdir(exist_ok=True);cv2.imwrite(str(a.output/'frames'/f'{i}_{method}.png'),obs['bgr'])
        file=f'case_{i}.png';fig.suptitle(t['instruction']);fig.tight_layout();fig.savefig(a.output/file,dpi=150);plt.close(fig)
        cases.append(dict(index=i,name=t['landmark_name'],file=file))
    (a.output/'cases.json').write_text(json.dumps(cases,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Visual only landmark navigation</title><style>body{max-width:1400px;margin:auto;font-family:sans-serif}img{width:100%}</style><h1>地标名称输入：完整模型 vs 仅视觉</h1><p>仅视觉不接收目标Polygon；图中Polygon只作事后分析。</p><p><a href="../VISUAL_ONLY_REPORT_zh.md">中文报告</a> | <a href="per_task.csv">CSV</a> | <a href="predictions.jsonl">全部轨迹</a></p><img src="comparison.png">'
    for case in cases:page+=f'<h2>{html.escape(case["name"])}</h2><a href="{case["file"]}"><img src="{case["file"]}"></a>'
    (a.output/'index.html').write_text(page)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
