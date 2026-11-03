"""Summarize completed paired landmark rollouts; render real visited RGB cases."""
import argparse
from collections import Counter,defaultdict
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
from graphnav.landmark_alias import resolve_references
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame


def summarize(rows):
    return dict(n=len(rows),successes=int(sum(r['metrics']['SR'] for r in rows)),
        **{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('SR','OSR','NE')},
        terminations=dict(Counter(r['termination'] for r in rows)),
        ever_correct_goal=sum(r['goal_analysis']['ever_correct'] for r in rows),
        first_goal_correct=sum(r['goal_analysis']['initial_ne_m']<=20 for r in rows),
        final_goal_correct=sum(r['goal_analysis']['final_ne_m']<=20 for r in rows),
        entered_polygon=sum(r['strict_ne_m']<=.001 for r in rows),
        near5m=sum(r['strict_ne_m']<=5 for r in rows))


def failure_tags(row):
    if row['metrics']['SR']:return ['success']
    tags=[]
    if not row['goal_analysis']['ever_correct']:tags.append('never_correct_landmark_goal')
    if row['goal_analysis']['lost_correct']:tags.append('correct_goal_lost')
    if row['metrics']['OSR']:tags.append('entered20m_but_failed_to_finish_there')
    if row['goal_analysis']['ever_correct'] and not row['metrics']['OSR']:tags.append('goal_once_correct_but_navigation_not_reached')
    if row['goal_analysis']['final_ne_m']<=20:tags.append('final_predicted_goal_correct_but_final_pose_wrong')
    if row['termination']=='policy_stop':tags.append('stopped_outside_landmark20m')
    if not tags:tags.append('unknown')
    return tags


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('selection','run','data-root','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tasks={t['index']:t for t in json.loads((a.selection/'tasks_public.json').read_text())}
    metadata=json.loads((a.run/'metrics.json').read_text())
    if metadata['complete'] is not True:raise ValueError('Complete64x2 needed')
    rows=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
    if {(r['method'],r['index']) for r in rows}!={(m,i) for m in ('E1','E2') for i in range(64)} or len(rows)!=128:
        raise ValueError('Paired episode completeness mismatch')
    objects=json.loads((a.data_root/'cityrefer/objects.json').read_text())
    binding=[]
    for t in tasks.values():
        resolved=resolve_references([t['landmark_name']],objects[t['map_name']],max_landmarks=16,match_mode='fuzzy')
        given=set(t['public_object_ids']);used={r['id'] for r in resolved}
        # Exact selected public geometry must equal what the actual catalog stores.
        for oid,poly in zip(t['public_object_ids'],t['public_polygons']):
            assert objects[t['map_name']][oid]['contour']==poly
        binding.append(dict(index=t['index'],name=t['landmark_name'],polygon_count=len(given),
            source_types=t['public_types'],components=t['polygon_components'],
            runtime_resolved_count=len(used),missing_public_ids=sorted(given-used),unexpected_public_ids=sorted(used-given)))
    for r in rows:
        t=tasks[r['index']];region=public_region(t['public_polygons'],t['world_bounds'])
        r['strict_ne_m']=float(shapely.distance(shapely.Point(r['poses'][-1][:2]),region))
        r['failure_tags']=failure_tags(r)
    summary={m:summarize([r for r in rows if r['method']==m]) for m in ('E1','E2')}
    stratified={m:{c:summarize([r for r in rows if r['method']==m and r['category']==c])
        for c in ('Building','Road','Other')} for m in ('E1','E2')}
    ambiguity={m:{kind:summarize([r for r in rows if r['method']==m and
        (len(tasks[r['index']]['public_object_ids'])>1)==(kind=='multi_polygon')])
        for kind in ('single_polygon','multi_polygon')} for m in ('E1','E2')}
    failure_counts={m:dict(Counter(tag for r in rows if r['method']==m for tag in r['failure_tags'] if tag!='success')) for m in ('E1','E2')}
    paired={}
    for i in range(64):
        pair={r['method']:r for r in rows if r['index']==i}
        assert pair['E1']['poses'][0]==pair['E2']['poses'][0]
        assert all(d['public_landmark_goal_distance_m']<1e-3 for d in pair['E1']['step_diagnostics'])
        assert all('pre_oracle_model_goal_xy' not in d for d in pair['E2']['step_diagnostics'])
        paired[i]=pair
    count_pair=dict(both_success=sum(p['E1']['metrics']['SR'] and p['E2']['metrics']['SR'] for p in paired.values()),
        only_E1_success=sum(p['E1']['metrics']['SR'] and not p['E2']['metrics']['SR'] for p in paired.values()),
        only_E2_success=sum(not p['E1']['metrics']['SR'] and p['E2']['metrics']['SR'] for p in paired.values()),
        both_fail=sum(not p['E1']['metrics']['SR'] and not p['E2']['metrics']['SR'] for p in paired.values()))
    stop_entry=[]
    for r in rows:
        k=r['metrics']['first_entry_step']
        if k is None:continue
        d=r['step_diagnostics'][k] if k<len(r['step_diagnostics']) else None
        stop_entry.append(dict(index=r['index'],method=r['method'],first_entry_step=k,SR=r['metrics']['SR'],
            stop_probability=d.get('stop_probability') if d else None,
            predicted_goal_distance=d.get('predicted_goal_distance') if d else None,
            altitude_guard=d.get('stop_altitude_guard') if d else None,
            note='Reached on final horizon action; no following decision' if d is None else 'Original Stop guard: probability>=0.3, predicted-point distance<=0.001m, legal altitude'))
    output=dict(summary=summary,by_category=stratified,by_ambiguity=ambiguity,
        failure_tags_nonexclusive=failure_counts,paired=count_pair,binding_audit=binding,stop_at_first_entry=stop_entry,
        runtime=metadata,paired_checks_passed=True,
        warnings=['Fixed64 public-name diagnostic, not official CityNav task SR.',
            'Multiple polygons are alternatives; reaching any fragment counts, not verified instance identity.',
            'No collision/3D visibility guarantee; original orthophoto navigation environment.',
            'Failure tags describe recorded behavior; do not independently identify visual vs language encoder causes.'])
    (a.output/'summary.json').write_text(json.dumps(output,indent=2))
    for src in ('predictions.jsonl','run_manifest.json','metrics.json'):
        shutil.copyfile(a.run/src,a.output/src)
    with (a.output/'per_task.csv').open('w') as f:
        columns=['index','method','map_name','landmark_name','category','SR','OSR','NE','termination','initial_goal_ne_m','final_goal_ne_m','failure_tags']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:
            w.writerow(dict(index=r['index'],method=r['method'],map_name=r['map_name'],landmark_name=r['landmark_name'],category=r['category'],
                SR=r['metrics']['SR'],OSR=r['metrics']['OSR'],NE=r['metrics']['NE'],termination=r['termination'],
                initial_goal_ne_m=r['goal_analysis']['initial_ne_m'],final_goal_ne_m=r['goal_analysis']['final_ne_m'],failure_tags=';'.join(r['failure_tags'])))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,metric in zip(axes,['SR','OSR','NE']):
        for j,m in enumerate(('E1','E2')):
            ax.bar(np.arange(3)+j*.36,[stratified[m][c][metric] for c in ('Building','Road','Other')],width=.36,label=m)
        ax.set_xticks(np.arange(3)+.18,['Building','Road','Other']);ax.set_title(metric);ax.legend()
    fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=160);plt.close(fig)
    chosen=[]
    predicates=[lambda p:p['E1']['metrics']['SR'] and not p['E2']['metrics']['SR'] and p['E2']['category']=='Building',
        lambda p:p['E1']['metrics']['SR'] and not p['E2']['metrics']['SR'] and p['E2']['category']=='Road',
        lambda p:p['E2']['metrics']['SR'] and p['E2']['category']=='Road',
        lambda p:p['E2']['metrics']['OSR'] and not p['E2']['metrics']['SR'],
        lambda p:p['E1']['termination']=='horizon',
        lambda p:not p['E1']['metrics']['SR'],lambda p:p['E2']['goal_analysis']['lost_correct']>0,
        lambda p:p['E2']['metrics']['SR'] and p['E2']['category']=='Building']
    for predicate in predicates:
        ids=[i for i,pair in paired.items() if predicate(pair) and i not in chosen]
        if ids:chosen.append(ids[0])
    renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'frames',image_size=384)
    import sys
    # Ground constants are data-environment conventions, not target labels.
    from sbfnav.mapdata import GROUND_LEVEL
    import cv2
    cases=[]
    for i in chosen:
        t=tasks[i];pair=paired[i];fig,axes=plt.subplots(1,4,figsize=(16,4))
        ax=axes[0]
        for poly in t['public_polygons']:
            p=np.asarray(poly);ax.fill(p[:,0],p[:,1],color='gray',alpha=.25)
        for method,color in [('E1','green'),('E2','royalblue')]:
            trace=np.asarray(pair[method]['poses']);ax.plot(trace[:,0],trace[:,1],'.-',color=color,label=method)
            ax.scatter(*trace[-1,:2],color=color,marker='s')
            goals=np.array([d['selected_goal_xy'] for d in pair[method]['step_diagnostics']]);ax.scatter(goals[:,0],goals[:,1],color=color,marker='x',alpha=.45)
        ax.scatter(*t['start_pose'][:2],color='black',marker='*',s=90,label='Start')
        ax.scatter(*t['oracle_goal_xy'],color='orange',marker='o',label='Public oracle')
        ax.set_aspect('equal');ax.legend(fontsize=7);ax.set_title(t['landmark_name'],fontsize=9)
        views=[('Start',t['start_pose']),('E1 final',pair['E1']['poses'][-1]),('E2 final',pair['E2']['poses'][-1])]
        for j,(label,pose) in enumerate(views,1):
            obs=renderer.current_rgbd(t['map_name'],pose,GROUND_LEVEL[t['map_name']]);rgb=cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB)
            filename=f'case_{i}_{j}.png';(a.output/'frames').mkdir(exist_ok=True)
            cv2.imwrite(str(a.output/'frames'/filename),obs['bgr'])
            ax=axes[j];ax.imshow(rgb)
            for contour in t['public_polygons']:
                px=world_to_frame(np.asarray(contour),obs['footprint_world'],obs['bgr'].shape)
                ax.plot(px[:,0],px[:,1],color='cyan',linestyle='--',linewidth=.7)
            ax.set_xlim(0,383);ax.set_ylim(383,0);ax.set_title(label+' (public projection, not visible mask)',fontsize=8);ax.axis('off')
        filename=f'case_{i}.png';fig.suptitle(f"{t['instruction']} | E1 NE={pair['E1']['metrics']['NE']:.2f}m; E2 NE={pair['E2']['metrics']['NE']:.2f}m",fontsize=11)
        fig.tight_layout();fig.savefig(a.output/filename,dpi=150);plt.close(fig)
        cases.append(dict(index=i,file=filename,name=t['landmark_name'],E2_failure_tags=pair['E2']['failure_tags']))
    (a.output/'cases.json').write_text(json.dumps(cases,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Named landmark E1/E2</title><style>body{max-width:1500px;margin:auto;font-family:sans-serif}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}</style><h1>公开具名地标 E1 / E2</h1><p>固定64任务；Polygon距离，非官方CityNav SR。青色RGB轮廓仅地图投影，不代表真实可见性。</p><p><a href="../REPORT_zh.md">中文报告</a> | <a href="per_task.csv">逐任务CSV</a> | <a href="predictions.jsonl">完整轨迹</a></p><img src="comparison.png">'
    for case in cases:
        page+=f'<h2>{html.escape(case["name"])}</h2><a href="{case["file"]}"><img src="{case["file"]}"></a>'
    page+='<h2>全部任务</h2><table><tr><th>任务</th><th>地标</th><th>类别</th><th>E1 SR/NE</th><th>E2 SR/NE</th></tr>'
    for i,t in tasks.items():
        pair=paired[i];page+=f'<tr><td>{i}</td><td>{html.escape(t["landmark_name"])}</td><td>{t["category"]}</td>'
        for m in ('E1','E2'):page+=f'<td>{int(pair[m]["metrics"]["SR"])}/{pair[m]["metrics"]["NE"]:.2f}m</td>'
        page+='</tr>'
    (a.output/'index.html').write_text(page+'</table>')
    print(json.dumps(dict(summary=summary,by_category=stratified,paired=count_pair,failure_tags=failure_counts),indent=2))


if __name__=='__main__':main()
