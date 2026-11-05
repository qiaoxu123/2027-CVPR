"""Measured PR16 fixed56 paired results, effective fields and RGB case gallery."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_projection import world_to_frame


def main():
    p=argparse.ArgumentParser()
    for key in ('root','source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--run',default='fixed56')
    a=p.parse_args();r=a.root;s=a.source;o=a.output;o.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in (r/a.run/'predictions.jsonl').read_text().splitlines()]
    old=[json.loads(l) for l in (s/'closedloop/predictions.jsonl').read_text().splitlines()]
    tasks=json.loads((s/'val_seen_selection/tasks_public.json').read_text())
    labels=json.loads((s/'val_seen_selection/labels_private.json').read_text())
    train=json.loads((r/('train_audit_fp32' if 'fp32' in a.run else 'train_audit')/'summary.json').read_text());full=json.loads((r/a.run/'full_summary.json').read_text())
    assert len(rows)==168 and len({(q['index'],q['method']) for q in rows})==168
    assert file_sha256(s/'val_seen_selection/tasks_public.json')=='6dc11959e4a94ba14adc20984e9973ca9739face16d5388c2bc32f4e7c2b5c14'
    by={(q['index'],q['method']):q for q in rows};before={(q['index'],q['method']):q for q in old}
    methods=['Geometry','Relation','RGB'];comparison={};old_comparison={}
    for left,right in [('Geometry','Relation'),('Relation','RGB'),('Geometry','RGB')]:
        corrected=sum(bool(not by[i,left]['metrics']['SR'] and by[i,right]['metrics']['SR']) for i in range(56))
        damaged=sum(bool(by[i,left]['metrics']['SR'] and not by[i,right]['metrics']['SR']) for i in range(56))
        comparison[left+' -> '+right]=dict(corrected=corrected,damaged=damaged,net=corrected-damaged)
    for method in methods:
        corrected=sum(bool(not before[i,method]['metrics']['SR'] and by[i,method]['metrics']['SR']) for i in range(56))
        damaged=sum(bool(before[i,method]['metrics']['SR'] and not by[i,method]['metrics']['SR']) for i in range(56))
        old_comparison[method]=dict(corrected=corrected,damaged=damaged,net=corrected-damaged,
            max_abs_final_NE_difference_m=max(abs(before[i,method]['metrics']['NE']-by[i,method]['metrics']['NE']) for i in range(56)))
    shift=[];blocks=Counter();admissions=Counter();changes=[];binding_mismatches=[];case_priority=set()
    final_goal_metrics={};failures=Counter()
    for row in rows:
        i=row['index'];method=row['method'];task=tasks[i];target=np.array(labels[i]['target']);bounds=np.array(task['world_bounds'])
        world=lambda cell:bounds[:2]+np.array([(cell%28+.5)/28,(cell//28+.5)/28])*(bounds[2:]-bounds[:2])
        selected=[];dense=[]
        for step,d in enumerate(row['diagnostics']):
            key=dict(Geometry='geometry_top1',Relation='relation_top1',RGB='full_top1')[method]
            goal=d[key];base=d['geometry_top1'];ne=float(np.linalg.norm(world(goal)-target))
            d_selected=float(np.linalg.norm(np.array(d['selected_goal_xy'])-target))
            selected.append(d_selected);dense.append(ne)
            shift.append(dict(index=i,method=method,step=step,top1_shift_from_geometry_m=float(np.linalg.norm(world(goal)-world(base))),
                              Belief_NE=ne,selected_Goal_NE=d_selected,
                              public_binding_changed=any(q['contours_changed'] or q['public_IDs_changed'] for q in d['public_binding_audit'])))
            if any(q['contours_changed'] or q['public_IDs_changed'] for q in d['public_binding_audit']):binding_mismatches.append((i,method,step))
            if method=='RGB':
                blocks[d['stable_goal']['replan_block_reason']]+=1;admissions[d['RGB_admission_reason']]+=1
                if d['full_top1']!=d['relation_top1']:
                    source_ne=float(np.linalg.norm(world(d['relation_top1'])-target))
                    changes.append(dict(index=i,step=step,Relation_NE=source_ne,RGB_NE=ne,
                        corrected_location_proxy=source_ne>20 and ne<=20,damaged_location_proxy=source_ne<=20 and ne>20,
                        logit_advantage=d['stable_goal']['replan_logit_advantage'],
                        block_reason=d['stable_goal']['replan_block_reason']))
                    case_priority.add(i)
        final_goal_metrics[i,method]=dict(final_Belief_NE=dense[-1] if dense else None,
            final_selected_Goal_NE=selected[-1] if selected else None,
            ever_correct_Belief=any(x<=20 for x in dense),ever_correct_selected_Goal=any(x<=20 for x in selected))
        if not row['metrics']['SR']:
            if row['error']:reason='runtime_error'
            elif selected and selected[-1]>20:
                reason='stopped_or_ended_at_wrong_predicted_Goal'
            elif row['metrics']['OSR']:
                reason='entered_then_left_target_region'
            else:reason='correct_selected_Goal_but_navigation_failed_or_unknown'
            failures[method+'::'+reason]+=1
    ds=[d for q in rows if q['method']=='RGB' for d in q['diagnostics']]
    summary=dict(fixed_tasks=56,real_rollouts=len(rows),closedloop=full['all'],paired_changes=comparison,
        versus_previous_frozen_run=old_comparison,RGB_decisions=len(ds),RGB_admitted=sum(bool(d['RGB_admitted']) for d in ds),
        admission_reasons=dict(admissions),replan_block_reasons=dict(blocks),
        replan_events=sum(d['stable_goal']['event']=='replanned' for d in ds),
        RGB_source_reasons=dict(Counter(d['RGB_reason'] for d in ds)),
        raw_RGB_top1_changes=len(changes),raw_location_proxy_corrections=sum(q['corrected_location_proxy'] for q in changes),
        raw_location_proxy_damages=sum(q['damaged_location_proxy'] for q in changes),
        max_confirmed_visual_logit_advantage=max((d['stable_goal']['replan_logit_advantage'] for d in ds if d['RGB_admitted']),default=None),
        binding_mismatches=binding_mismatches,failure_causes=dict(failures),
        proxy_warning='20m position success is not independent target-instance identification',
        GPU_peak_GiB=max(q['peak_gpu_GiB'] for q in rows))
    paired=[]
    for i in range(56):
        paired.append(dict(index=i,instruction=tasks[i]['instruction'],map_name=tasks[i]['map_name'],
            **{m:dict(metrics=by[i,m]['metrics'],termination=by[i,m]['termination'],error=by[i,m]['error'],
                       **final_goal_metrics[i,m]) for m in methods}))
    for name,value in [('summary.json',summary),('paired_tasks.json',paired),('Belief_shifts.json',shift),('blocked_RGB_changes.json',changes)]:atomic_json(value,o/name)
    shutil.copy2(r/a.run/'predictions.jsonl',o/'all_168_trajectories.jsonl')
    shutil.copy2(r/a.run/'freeze_manifest.json',o/'freeze_manifest.json')
    shutil.copy2(r/'cpu_final.txt',o/'cpu_tests.txt')
    # Six tasks include changed/blocked proposals plus representative failures
    # and successes. Selection affects only visualization, never evaluation.
    selected=sorted(case_priority)[:5]
    for success in (False,True):
        for i in range(56):
            if bool(by[i,'RGB']['metrics']['SR'])==success and i not in selected:
                selected.append(i);break
    cardinal=next(i for i,t in enumerate(tasks) if t['anchors'][0]['predicate'] in ('north','south','east','west'))
    if cardinal not in selected:selected.append(cardinal)
    cases=[]
    for i in selected:
        task=tasks[i];row=by[i,'RGB'];target=labels[i]['target'];bounds=task['world_bounds']
        times=sorted(set([0,len(row['captures'])//2,len(row['captures'])-1]+
            [j for j,d in enumerate(row['diagnostics']) if d['full_top1']!=d['relation_top1']][:1]))
        pictures=[]
        for step in times:
            if step<0:continue
            path=Path(row['captures'][step]);z=np.load(path);rgb=cv2.cvtColor(cv2.imread(str(path.with_suffix('.png'))),cv2.COLOR_BGR2RGB)
            fig,axes=plt.subplots(1,5,figsize=(24,4));axes[0].imshow(rgb)
            for anchor in task['anchors']:
                for cs in anchor['contours']:
                    px=world_to_frame(np.array(cs),z['footprint'],rgb.shape);axes[0].plot(*np.vstack((px,px[0])).T,c='yellow',lw=.8)
            axes[0].set_xlim(0,383);axes[0].set_ylim(383,0);axes[0].set_title('Actual RGB / public Polygon (not pixel GT)')
            extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
            for ax,key,title in zip(axes[1:4],('geometry','relation','full'),('Geometry','Effective Relation','Admitted RGB')):
                log=z[key];prob=np.exp(log-log.max());prob/=prob.sum();ax.imshow(prob.reshape(28,28),origin='lower',extent=extent,cmap='magma')
                for anchor in task['anchors']:
                    for cs in anchor['contours']:ax.plot(*np.array(cs).T,c='cyan',lw=.6)
                fp=np.asarray(z['footprint']);ax.plot(*np.vstack((fp,fp[0])).T,'w--',lw=.6)
                ax.scatter(*target,c='lime',marker='x');ax.set_title(title)
            for m in methods:
                points=np.array(by[i,m]['poses']);axes[4].plot(points[:,0],points[:,1],label=m)
            axes[4].scatter(*target,c='red',marker='x');axes[4].legend(fontsize=7);axes[4].set_title('Actual paired trajectories')
            d=row['diagnostics'][step]
            fig.suptitle(f"Task {i}, step {step}: {task['instruction']}\nRGB admitted={d['RGB_admitted']}; reason={d['RGB_admission_reason']}; replan={d['stable_goal']['replan_block_reason']}",fontsize=8)
            fig.tight_layout();name=f'case_{i:02d}_step_{step:02d}.png';fig.savefig(o/name,dpi=120);plt.close(fig)
            rawname=f'RGB_{i:02d}_{step:02d}.png';shutil.copy2(path.with_suffix('.png'),o/rawname)
            pictures.append(dict(step=step,figure=name,raw_RGB=rawname))
        cases.append(dict(index=i,instruction=task['instruction'],pictures=pictures,metrics=by[i,'RGB']['metrics']))
    atomic_json(cases,o/'cases.json')
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    axes[0].bar(methods,[full['all'][m]['SR']*100 for m in methods]);axes[0].set_title('Fixed56 SR (%)')
    axes[1].bar(methods,[full['all'][m]['NE'] for m in methods]);axes[1].set_title('Navigation NE (m)')
    train_ms=['Geometry','RawRelation','RawRGB','SafeRGB'];axes[2].bar(train_ms,[train['raw_and_safe_state_metrics'][m]['R20']*100 for m in train_ms]);axes[2].tick_params(axis='x',rotation=20);axes[2].set_title('TRAIN position-proxy R20 (%)')
    fig.tight_layout();fig.savefig(o/'summary.png',dpi=140);plt.close(fig)
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PR16 evidence safety</title><style>body{font:16px sans-serif;max-width:1600px;margin:auto}img{width:100%}pre{white-space:pre-wrap}section{margin:30px 0}</style><h1>PR #16真实证据安全诊断</h1><p>固定56任务、168次真实闭环。黄色为公开Polygon投影；绿色叉为事后GT位置，不是人工像素Mask。</p><p><a href="REPORT_zh.md">中文报告</a> · <a href="paired_tasks.json">全部配对结果</a> · <a href="all_168_trajectories.jsonl">全部轨迹</a></p><img src="summary.png">'''
    for c in cases:
        ident=c['index'];first=c['pictures'][0]
        body+=f'<section><h2>Task {ident}</h2><p>{html.escape(c["instruction"])}</p><input type="range" min="0" max="{len(c["pictures"])-1}" value="0" oninput="show({ident},this.value)"><span id="step{ident}">step {first["step"]}</span><a id="raw{ident}" href="{first["raw_RGB"]}">原生RGB</a><a id="figlink{ident}" href="{first["figure"]}"><img loading="lazy" id="fig{ident}" src="{first["figure"]}"></a></section>'
    body+='<script>const cases='+json.dumps({c['index']:c['pictures'] for c in cases})+''';function show(id,n){const p=cases[id][n];document.getElementById('fig'+id).src=p.figure;document.getElementById('figlink'+id).href=p.figure;document.getElementById('raw'+id).href=p.raw_RGB;document.getElementById('step'+id).textContent='step '+p.step;}</script></html>'''
    (o/'index.html').write_text(body)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
