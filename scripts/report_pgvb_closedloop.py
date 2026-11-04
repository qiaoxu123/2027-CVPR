"""Summarize sealed 64x3 PGVB landmark rollouts; no policy or model selection."""
import argparse,csv,gzip,html,json,shutil
from pathlib import Path
from collections import Counter
import cv2,numpy as np,shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import file_sha256
from graphnav.landmark_navigation_diagnostic import public_region
from graphnav.landmark_projection import world_to_frame

METHODS=('E2_original','E4_PGVB','E5_PGVB_only')


def summarize(rows):
    ds=[d for r in rows for d in r['step_diagnostics']]
    timings=[t for r in rows for t in r['decision_seconds']]
    warm=[t for r in rows for t in r['decision_seconds'][1:]]
    active=[d for d in ds if d['pgvb_active']]
    jumps=[float(np.linalg.norm(np.asarray(b['selected_goal_xy'])-np.asarray(a['selected_goal_xy']))) for r in rows for a,b in zip(r['step_diagnostics'][:-1],r['step_diagnostics'][1:])]
    return dict(episodes=len(rows),successes=int(sum(r['metrics']['SR'] for r in rows)),
        **{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('SR','OSR','NE','steps','path_length_m','false_stop')},
        terminations=dict(Counter(r['termination'] for r in rows)),decisions=len(ds),
        active_decisions=len(active),active_fraction=len(active)/len(ds),active_episodes=sum(any(d['pgvb_active'] for d in r['step_diagnostics']) for r in rows),
        changed_top1_decisions=sum(d.get('pgvb_changed_top1',False) for d in ds),
        abstained_decisions=sum(d.get('abstained_no_visual_goal',False) for d in ds),
        fallback_reasons=dict(Counter(d.get('pgvb_reason','baseline') for d in ds)),
        latency_ms_mean=float(np.mean(timings)*1000),latency_ms_p50=float(np.median(timings)*1000),
        warm_latency_ms_mean=float(np.mean(warm)*1000) if warm else None,
        pgvb_ms_per_decision=float(np.mean([d.get('pgvb_seconds',0) for d in ds])*1000),
        pgvb_ms_active=float(np.mean([d.get('pgvb_seconds',0) for d in active])*1000) if active else None,
        goal_jumps_over20=sum(x>20 for x in jumps),goal_transition_count=len(jumps),
        goal_jump_mean_m=float(np.mean(jumps)) if jumps else 0.,
        correct_goal_losses=sum(r['goal_analysis']['lost_correct'] for r in rows),
        initial_goal_correct=sum(r['goal_analysis']['initial_ne_m']<=20 for r in rows),
        final_goal_correct=sum(r['goal_analysis']['final_ne_m']<=20 for r in rows),
        successes_without_policy_stop=sum(bool(r['metrics']['SR']) and r['termination']!='policy_stop' for r in rows))


def main():
    p=argparse.ArgumentParser()
    for k in ('run','selection','history','execution','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in (a.run/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==192 and {(r['index'],r['method']) for r in rows}=={(i,m) for i in range(64) for m in METHODS}
    assert json.loads((a.run/'metrics.json').read_text())['complete']
    tasks={x['index']:x for x in json.loads((a.selection/'tasks_public.json').read_text())};lookup={(r['index'],r['method']):r for r in rows}
    history={r['index']:r for r in map(json.loads,(a.history/'predictions.jsonl').read_text().splitlines()) if r['method']=='E2'}
    for i in range(64):
        b=lookup[(i,METHODS[0])];assert b['poses']==history[i]['poses'] and b['metrics']==history[i]['metrics']
        assert all(lookup[(i,m)]['poses'][0]==b['poses'][0] for m in METHODS)
    groups={m:[r for r in rows if r['method']==m] for m in METHODS}
    summary={m:summarize(groups[m]) for m in METHODS}
    stratified={m:{c:summarize([r for r in groups[m] if r['category']==c]) for c in ('Building','Road','Other')} for m in METHODS}
    paired=[];events=[];artifact_hashes=[]
    for i in range(64):
        b=lookup[(i,METHODS[0])];t=tasks[i];region=public_region(t['public_polygons'],t['world_bounds'])
        item=dict(index=i,name=t['landmark_name'],category=t['category'],map_name=t['map_name'])
        for m in METHODS:
            r=lookup[(i,m)];item[m]=dict(SR=r['metrics']['SR'],NE=r['metrics']['NE'],steps=r['metrics']['steps'],termination=r['termination'],active_decisions=sum(d['pgvb_active'] for d in r['step_diagnostics']))
            for d in r['step_diagnostics']:
                path=Path(d['audit_artifact']);z=np.load(path);assert np.isfinite(z['pose']).all()
                artifact_hashes.append(dict(file=str(path),sha256=file_sha256(path)))
                if 'final_candidate_xy' in z:
                    bounds=z['bounds'];xy=bounds[:2]+z['final_candidate_xy'][0]*(bounds[2:]-bounds[:2]);np.testing.assert_allclose(xy,d['selected_goal_xy'],atol=1e-4)
                if m=='E5_PGVB_only':assert d['geometry_branch_calls']==0
                if m=='E4_PGVB' and d['pgvb_active']:
                    delta=z['final_logits']-z['original_logits'];assert abs(delta).max()<=.50001;assert np.all(delta[~z['pgvb_covered']]==0)
                    bounds=z['bounds'];pre=bounds[:2]+z['original_candidate_xy'][0]*(bounds[2:]-bounds[:2]);post=np.asarray(d['selected_goal_xy']);pre_ne=float(shapely.distance(shapely.Point(pre),region));post_ne=float(shapely.distance(shapely.Point(post),region))
                    events.append(dict(index=i,step=d['observation_step'],pre_ne=pre_ne,post_ne=post_ne,corrected=pre_ne>20 and post_ne<=20,damaged=pre_ne<=20 and post_ne>20,goal_displacement_m=float(np.linalg.norm(post-pre)),delta_max=float(abs(delta).max())))
        paired.append(item)
    comparisons={m:dict(corrected=sum(not p[METHODS[0]]['SR'] and p[m]['SR'] for p in paired),damaged=sum(p[METHODS[0]]['SR'] and not p[m]['SR'] for p in paired),both_success=sum(p[METHODS[0]]['SR'] and p[m]['SR'] for p in paired),both_failure=sum(not p[METHODS[0]]['SR'] and not p[m]['SR'] for p in paired)) for m in METHODS[1:]}
    event_summary=dict(active=len(events),corrected_decisions=sum(x['corrected'] for x in events),damaged_decisions=sum(x['damaged'] for x in events),corrected_episodes=sorted({x['index'] for x in events if x['corrected']}),damaged_episodes=sorted({x['index'] for x in events if x['damaged']}),changed_goal_decisions=sum(x['goal_displacement_m']>1e-3 for x in events))
    failure=[]
    for r in rows:
        if r['metrics']['SR']:continue
        tag=[]
        if r['method']=='E5_PGVB_only' and not any(d['pgvb_active'] for d in r['step_diagnostics']):tag.append('no_supported_local_anchor_no_search_policy')
        if not r['metrics']['OSR']:tag.append('never_entered_public_region20m')
        else:tag.append('entered20m_but_final_outside')
        if not r['goal_analysis']['ever_correct']:tag.append('never_correct_goal')
        if r['goal_analysis']['lost_correct']:tag.append('correct_goal_lost')
        if r['termination']=='policy_stop':tag.append('stopped_outside20m')
        if r['goal_analysis']['final_ne_m']<=20:tag.append('final_goal_correct_navigation_or_stop_failed')
        failure.append(dict(index=r['index'],method=r['method'],tags=tag))
    # Descriptive paired map-cluster bootstrap; no hyperparameter/model selection.
    maps=sorted({t['map_name'] for t in tasks.values()});rng=np.random.default_rng(0);boot=[]
    for _ in range(2000):
        ids=[i for mp in rng.choice(maps,len(maps),replace=True) for i,t in tasks.items() if t['map_name']==mp]
        boot.append(np.mean([lookup[(i,'E4_PGVB')]['metrics']['SR']-lookup[(i,'E2_original')]['metrics']['SR'] for i in ids]))
    report=dict(summary=summary,stratified=stratified,paired=comparisons,within_E4_goal_changes=event_summary,
        uncertainty=dict(unit='paired episode clustered by map; fixed diagnostic set, not model selection',maps=len(maps),E4_minus_E2_SR_ci95=np.quantile(boot,[.025,.975]).tolist()),
        historical_E2_exact=True,annotations='public landmark Polygon region; no human pixel instance accuracy; Val-Seen sourced diagnostic tasks')
    for name,value in [('summary.json',report),('paired_tasks.json',paired),('goal_update_events.json',events),('failure_tags.json',failure),('artifact_sha256.json',artifact_hashes)]:
        (o/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    with gzip.open(o/'predictions.jsonl.gz','wb') as f:f.write((a.run/'predictions.jsonl').read_bytes())
    for name in ('run_manifest.json','metrics.json'):shutil.copyfile(a.run/name,o/name)
    shutil.copyfile(a.selection/'manifest.json',o/'selection_manifest.json');shutil.copyfile(a.selection/'tasks_public.json',o/'tasks_public.json')
    logs=o/'logs';logs.mkdir(exist_ok=True)
    for f in a.execution.iterdir():
        if f.suffix in ('.log','.exit','.sh'):shutil.copyfile(f,logs/f.name)
    with (o/'per_task.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['index','name','category','map','method','SR','OSR','NE','steps','termination','pgvb_active_decisions'])
        for r in rows:w.writerow([r['index'],r['landmark_name'],r['category'],r['map_name'],r['method'],r['metrics']['SR'],r['metrics']['OSR'],r['metrics']['NE'],r['metrics']['steps'],r['termination'],sum(d['pgvb_active'] for d in r['step_diagnostics'])])
    # Deterministic evidence-first case inclusion: every changed SR, E2 failures,
    # plus maximum PGVB activity in Building/Road and a standalone success.
    chosen=[]
    def add(i):
        if i not in chosen:chosen.append(i)
    for p0 in paired:
        if p0['E2_original']['SR']!=p0['E4_PGVB']['SR']:add(p0['index'])
    for c in ('Building','Road'):
        pool=[r for r in groups['E4_PGVB'] if r['category']==c];add(max(pool,key=lambda r:sum(d['pgvb_active'] for d in r['step_diagnostics']))['index'])
    for r in groups['E5_PGVB_only']:
        if r['metrics']['SR']:add(r['index']);break
    for r in groups['E2_original']:
        if not r['metrics']['SR']:add(r['index'])
    for i in range(64):
        if len(chosen)>=6:break
        add(i)
    cases=[]
    for i in chosen:
        t=tasks[i];fig,axes=plt.subplots(3,4,figsize=(16,11));case=o/'cases'/f'{i:02d}';case.mkdir(parents=True,exist_ok=True)
        for rowno,m in enumerate(METHODS):
            r=lookup[(i,m)];ds=r['step_diagnostics'];active=[k for k,d in enumerate(ds) if d['pgvb_active']];step=active[-1] if active else len(ds)-1;d=ds[step];z=np.load(d['audit_artifact']);png=Path(d['audit_artifact']).with_suffix('.png');rgb=cv2.cvtColor(cv2.imread(str(png)),cv2.COLOR_BGR2RGB)
            shutil.copyfile(png,case/(m+'.png'));shutil.copyfile(d['audit_artifact'],case/(m+'.npz'))
            ax=axes[rowno,0];ax.imshow(rgb)
            for poly in t['public_polygons']:
                xy=world_to_frame(np.asarray(poly),z['footprint_world'],rgb.shape);ax.plot(xy[:,0],xy[:,1],'c-',lw=.7)
            ax.set_xlim(0,383);ax.set_ylim(383,0);ax.set_title(f'{m}: step {step}, active={d["pgvb_active"]}')
            b=z['bounds'];extent=[b[0],b[2],b[1],b[3]]
            def draw(ax,values,title):
                if values is not None:ax.imshow(values.reshape(28,28),origin='lower',extent=extent,cmap='magma',vmin=0,vmax=max(float(np.max(values)),1e-8))
                else:ax.text(.5,.5,'No current evidence / no global field',ha='center',transform=ax.transAxes)
                for poly in t['public_polygons']:
                    xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],'c-',lw=.7)
                fp=np.vstack((z['footprint_world'],z['footprint_world'][0]));ax.plot(fp[:,0],fp[:,1],'w--' if values is not None else 'k--',lw=.6);ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3]);ax.set_title(title)
            draw(axes[rowno,1],z['pgvb_probability'] if 'pgvb_probability' in z else None,'PGVB conditional probability')
            logits=z['final_logits'] if 'final_logits' in z else None
            prob=None if logits is None else np.exp(logits-logits.max())/np.exp(logits-logits.max()).sum()
            draw(axes[rowno,2],prob,'Actual controller Goal Belief')
            axes[rowno,2].scatter(*d['selected_goal_xy'],c='red',s=20)
            ax=axes[rowno,3]
            for other,color in zip(METHODS,('gray','blue','orange')):
                poses=np.asarray(lookup[(i,other)]['poses']);ax.plot(poses[:,0],poses[:,1],'-o',color=color,lw=1,ms=2,label=other)
            for poly in t['public_polygons']:
                xy=np.asarray(poly);ax.plot(xy[:,0],xy[:,1],'c-',lw=1)
            ax.scatter(*r['poses'][0][:2],c='green',marker='*',s=60);ax.scatter(*r['poses'][-1][:2],c='red',s=20);ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'SR={r["metrics"]["SR"]:.0f}, NE={r["metrics"]["NE"]:.2f}m; {r["termination"]}');ax.legend(fontsize=6)
        fig.suptitle(f'{i}: {t["instruction"]} | public Polygon diagnostic; native384 RGB',fontsize=12);fig.tight_layout();fig.savefig(case/'comparison.png',dpi=130);plt.close(fig);cases.append(dict(index=i,name=t['landmark_name'],category=t['category'],file=f'cases/{i:02d}/comparison.png'))
    (o/'cases.json').write_text(json.dumps(cases,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(13,4));labels=['E2','E4 +PGVB','E5 PGVB only']
    for ax,key,title in zip(axs,('SR','NE','active_fraction'),('Landmark SR','Polygon NE (m)','PGVB active decisions')):ax.bar(labels,[summary[m][key] for m in METHODS]);ax.set_title(title)
    fig.tight_layout();fig.savefig(o/'summary.png',dpi=160);plt.close(fig)
    page='<meta charset="utf-8"><title>PGVB 64地标闭环</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}img{max-width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>64条具名地标：E2 / E4 / E5</h1><p>公开Polygon距离评价；不是原始CityNav目标任务或像素实例准确率。</p><a href="REPORT_zh.md">中文报告</a> | <a href="per_task.csv">逐任务CSV</a> | <a href="predictions.jsonl.gz">全部原始轨迹</a><img src="summary.png"><table><tr><th>Index</th><th>地标</th><th>类别</th><th>E2 SR</th><th>E4 SR</th><th>E5 SR</th></tr>'
    for x in paired:page+=f'<tr><td>{x["index"]}</td><td>{html.escape(x["name"])}</td><td>{x["category"]}</td>'+''.join(f'<td>{x[m]["SR"]:.0f}</td>' for m in METHODS)+'</tr>'
    page+='</table>'
    for c in cases:
        page+=f'<h2>{c["index"]}: {html.escape(c["name"])}</h2><a href="{c["file"]}"><img src="{c["file"]}"></a>'
        for m in METHODS:page+=f'<a href="cases/{c["index"]:02d}/{m}.png">{m} 原生RGB</a> | <a href="cases/{c["index"]:02d}/{m}.npz">原始Belief/位姿</a><br>'
    (o/'index.html').write_text(page)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
