"""Post-hoc paired128 report; labels never used by navigation or task selection."""
import argparse,csv,html,json,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser()
    for n in ('run','previous','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--evaluation',choices=['pilot128','full_eval'],default='pilot128')
    a=p.parse_args();r=a.run;out=a.output;out.mkdir(parents=True,exist_ok=True)
    navdir=r/a.evaluation;N=128 if a.evaluation=='pilot128' else 2470;is_pilot=N==128
    metric=json.loads((navdir/'metrics.json').read_text());assert metric['coverage']['completed']==metric['coverage']['expected']==N
    assert metric['diagnostic_pilot'] is is_pilot
    new={x['index']:x for x in map(json.loads,(navdir/'predictions.jsonl').read_text().splitlines())};assert len(new)==N
    current_manifest=json.loads((navdir/'full_relation_manifest.json').read_text())
    if not is_pilot:
        full_freeze=json.loads((r/'FROZEN_BEFORE_FULL.json').read_text())
        assert full_freeze['checkpoint_SHA256']==metric['full_relation_checkpoint_sha256']
        assert full_freeze['evaluator_sha256']==file_sha256('scripts/evaluate_closedloop.py')
    ids=sorted(new);arms={'old_full':{},'compact':{},'target_features':new}
    for name,folder in [('old_full','eval_full_text'),('compact','eval_landmark_relation')]:
        old_manifest=json.loads((a.previous/folder/'full_relation_manifest.json').read_text())
        for key in ('stage1','original_v2','split','data_root','relation_control','relation_visual_control','geometry_source','model_code','relation_runtime_code','config','public_catalog','text_adapter','text_inputs','text_parser'):
            assert old_manifest[key]==current_manifest[key],f'Baseline protocol mismatch: {key}'
        rows={x['index']:x for x in map(json.loads,(a.previous/folder/'predictions.jsonl').read_text().splitlines())}
        for idx in ids:
            b=rows[idx];n=new[idx];assert b['episode_id']==n['episode_id'] and b['poses'][0]==n['poses'][0]
            prev=json.loads((a.previous/folder/f'belief_steps/{idx:05d}/public_record.json').read_text())
            curr=json.loads((navdir/f'belief_steps/{idx:05d}/public_record.json').read_text())
            assert prev['landmarks']==curr['landmarks'] and prev['world_bounds']==curr['world_bounds']
            if name=='old_full':assert prev['instruction']==curr['instruction']
            arms[name][idx]=b
    keys=['SR','OSR','SPL','NE','nDTW','steps','path_length','inference_ms']
    summary={k:{m:float(np.mean([row['metrics'][m] for row in rows.values()])) for m in keys} for k,rows in arms.items()}
    paired={};csvrows=[]
    for name in ('old_full','compact'):
        base=arms[name]
        fixes=sum(not base[i]['metrics']['SR'] and bool(new[i]['metrics']['SR']) for i in ids)
        damage=sum(bool(base[i]['metrics']['SR']) and not new[i]['metrics']['SR'] for i in ids)
        paired[name]=dict(fixes=int(fixes),damage=int(damage),net=int(fixes-damage))
    groups=defaultdict(list)
    for i in ids:groups[tuple(new[i]['episode_id'][:2])].append(i)
    rng=np.random.default_rng(42);g=list(groups.values());delta={}
    for name in ('old_full','compact'):
        samples=[]
        for _ in range(2000):
            selected=[i for k in rng.integers(0,len(g),len(g)) for i in g[k]]
            samples.append(np.mean([new[i]['metrics']['SR']-arms[name][i]['metrics']['SR'] for i in selected]))
        delta[name]=np.quantile(samples,[.025,.975]).tolist()
    for i in ids:
        row=dict(index=i,episode_id=json.dumps(new[i]['episode_id']))
        for name in arms:
            for m in keys:row[name+'_'+m]=arms[name][i]['metrics'][m]
            row[name+'_termination']=arms[name][i]['termination']
        csvrows.append(row)
    with (out/f'paired{N}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    s=torch.load(r/'train/last.pt',map_location='cpu',weights_only=True)
    frozen=json.loads((r/'FROZEN_BEFORE_PILOT.json').read_text());assert frozen['checkpoint_SHA256']==file_sha256(r/'train/last.pt')
    orig=torch.load(Path(r.parent)/'unified_pgvb_v2_pilot_s42/full_training/last.pt',map_location='cpu',weights_only=True)
    same=all(torch.equal(v,orig['model'][k]) for k,v in s['model'].items() if not k.startswith(('geometry.','full_relation.')))
    assert same,'Frozen PGVB module changed'
    report=dict(N=N,diagnostic_only=is_pilot,selection=('evenly spaced indices across all2470 Val-Seen, before scoring' if is_pilot else 'all2470 Val-Seen episodes; no filtering'),metrics=summary,paired=paired,physical_target_groups=len(groups),group_bootstrap_delta95CI=delta,single_seed=True,checkpoint_sha256=frozen['checkpoint_SHA256'],frozen_modules_bitwise_unchanged=same,training_seconds=s['elapsed_seconds'],training_updates=s['updates'],training_history=s['history'],inference_errors=metric['coverage']['inference_errors'],RGB_active=False,baseline_cache_reused=True,latency_caveat='Cached baseline timing collected previously; not a controlled contemporaneous speed comparison')
    raw_val=json.loads((Path(metric['args']['data_root'])/'processed_citynav/citynav_val_seen.json').read_text())
    failure_counts={}
    for name,rows in arms.items():
        counts=defaultdict(int)
        for idx,row in rows.items():
            target=np.asarray(raw_val[idx]['target_positions'][-1][:2])
            goals=np.asarray([d['selected_goal_xy'] for d in row['step_diagnostics']])
            correct_goal=bool(len(goals) and np.linalg.norm(goals[-1]-target)<=20)
            counts['failed_with_correct_final_goal' if correct_goal else 'failed_with_wrong_final_goal']+=int(not row['metrics']['SR'])
            counts['OSR_success_SR_failed']+=int(row['metrics']['OSR'] and not row['metrics']['SR'])
            counts['correct_policy_stop']+=int(row['termination']=='policy_stop' and row['metrics']['SR'])
            counts['goal_jumps_over20m']+=int((np.linalg.norm(np.diff(goals,axis=0),axis=-1)>20).sum()) if len(goals)>1 else 0
            counts[row['termination']]+=1
        failure_counts[name]=dict(counts)
    report['failure_attribution']=failure_counts
    atomic_json(report,out/'summary.json')
    for src,dest in [(r/'FROZEN_BEFORE_PILOT.json','FROZEN_BEFORE_PILOT.json'),(r/'inputs/manifest.json','data_manifest.json'),(r/'train/gradient_audit.json','gradient_audit.json'),(navdir/'episodes.json',f'episodes{N}.json'),(navdir/'metrics.json','new_navigation_metrics.json')]:shutil.copyfile(src,out/dest)
    fig,ax=plt.subplots(figsize=(6,3));ax.plot([x['epoch'] for x in s['history']],[x['train_loss'] for x in s['history']],'o-');ax.set(xlabel='Epoch',ylabel='TRAIN CE + 0.05 KL');fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for ax,key in zip(axs,['SR','NE']):
        ax.bar(list(summary),[summary[k][key]*(100 if key=='SR' else 1) for k in summary]);ax.set_ylabel(key+(' (%)' if key=='SR' else ' (m)'));ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(out/'navigation_comparison.png',dpi=150);plt.close(fig)
    selected=[i for i in ids if not arms['compact'][i]['metrics']['SR'] and new[i]['metrics']['SR']][:3]+[i for i in ids if arms['compact'][i]['metrics']['SR'] and not new[i]['metrics']['SR']][:3]
    selected=list(dict.fromkeys(selected));cases=[]
    for i in selected:
        rec=json.loads((navdir/f'belief_steps/{i:05d}/public_record.json').read_text());bounds=rec['world_bounds'];extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,(name,rows) in zip(axes,arms.items()):
            folder=(navdir) if name=='target_features' else a.previous/('eval_full_text' if name=='old_full' else 'eval_landmark_relation')
            z=np.load(folder/f'belief_steps/{i:05d}/00.npz')['full'];prob=torch.as_tensor(z).softmax(-1).reshape(28,28).numpy();ax.imshow(prob,origin='lower',extent=extent)
            for lm in rec['landmarks']:
                c=np.asarray(lm['contour'])*(np.asarray(bounds[2:])-bounds[:2])+bounds[:2];ax.plot(*np.vstack((c,c[0])).T,'c-',lw=.5)
            target=np.asarray(raw_val[i]['target_positions'][-1][:2]);ax.plot(*target,'r*',markersize=10)
            xy=np.asarray(rows[i]['poses'])[:,:2];ax.plot(xy[:,0],xy[:,1],'w.-',lw=.8);ax.plot(*xy[0],'go');ax.plot(*xy[-1],'rx');ax.set_title(f'{name}: SR={rows[i]["metrics"]["SR"]:.0f}, NE={rows[i]["metrics"]["NE"]:.1f}m',fontsize=8)
        fig.suptitle(rec['instruction'][:160],fontsize=8);fig.tight_layout();file=f'case_{i:05d}.png';fig.savefig(out/file,dpi=140);plt.close(fig);cases.append(dict(index=i,instruction=rec['instruction'],image=file))
    atomic_json(cases,out/'cases.json')
    table='\n'.join(f'|{name}|{m["SR"]*100:.2f}%|{m["OSR"]*100:.2f}%|{m["SPL"]:.4f}|{m["NE"]:.3f}|{m["nDTW"]:.4f}|{m["steps"]:.2f}|' for name,m in summary.items())
    scope_note=('仅诊断，不是2470条正式全量成绩。未运行新全量Val或Test。' if is_pilot else '全部2470/2470条Val-Seen真实闭环。Pilot后未改变模型或配置，不使用Val选择Checkpoint或调参，未运行Test。')
    selection_note=('从2470条Val-Seen按既有evenly_spaced_indices确定128条，不按成功或GT选择。' if is_pilot else '包含全部2470条Val-Seen，不按候选可用性、成功或GT过滤。')
    text=f'''# 保留目标＋特征＋地标＋关系：完整TRAIN训练及{N}条导航


{scope_note}未修改Stop/Goal/控制器。

## 输入与训练

全部21878条TRAIN文本本身不含可删除的开头套话，本轮实际完整保留原文，数据SHA与上一轮public-bound full_text完全相同。没有声称完成LLM重构、可靠四槽语义解析或新增外观网络；目标外观及实例关系仍由原BERT Token通道输入。

沿用原架构、原V2初始化、seed42、5轮、batch8、Geometry1e-5/Relation1e-4、CE+0.05KL、FP32。训练Geometry和Relation，PGVB/BERT/控制器冻结；静态文本消融RGB关闭，不能解释为验证了视觉识别。13675次更新，耗时{s['elapsed_seconds']:.2f}秒。PGVB等未训练模块逐张量核对与初始化完全相同。

与上一轮精简模型使用相同公开Polygon绑定和训练设置。本轮是补齐完整内容在新绑定下的训练对照；旧全文Checkpoint在旧绑定下训练，因此与它的差异也包含重新绑定后的训练适配。

## 固定{N}条配对闭环

{selection_note}新模型真实闭环；两个旧模型复用之前完整评测中完全相同的{N}条轨迹，逐条核对Episode、起点、公共Polygon及世界范围。无候选和失败均入分母。指标为Episode平均。

|方法|SR|OSR|SPL|NE(m)|nDTW|步数|
|---|---:|---:|---:|---:|---:|---:|
{table}

相对旧完整指令：{paired['old_full']}；相对精简：{paired['compact']}。物理目标组bootstrap95% SR差值：{delta}。一个训练种子；{'小样本Pilot不能作为全量SR。' if is_pilot else '本次全量Seen结果不能替代Unseen泛化。'}不能作为多种子泛化结论。

导航失败归因（GT仅事后使用）：{failure_counts}。

新增推理错误：{report['inference_errors']}。新模型平均单步推理{summary['target_features']['inference_ms']:.3f}ms；旧模型为历史缓存耗时，不是同一时段速度对照。

Checkpoint: {r/'train/last.pt'}
SHA256: {frozen['checkpoint_SHA256']}

原始训练日志、Checkpoint和逐步Belief/轨迹保留服务器{r}；GitHub包含配对CSV、固定任务清单、曲线及典型成功/失败热图轨迹。世界目标20m指标不证明真实实例匹配。公开地图同名Polygon统计见上级实验目录homonymous_polygons.json；道路分段不能直接解释为不同实例。
'''
    (out/'REPORT_zh.md').write_text(text)
    items=''.join(f'<article><h3>Episode index {x["index"]}</h3><p>{html.escape(x["instruction"])}</p><a href="{x["image"]}"><img width="100%" src="{x["image"]}"></a></article>' for x in cases)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Paired navigation comparison</title><body style="max-width:1400px;margin:auto"><h1>配对导航对照</h1><p>任务数量及范围见中文报告；RGB关闭、目标描述完整保留。白线为真实闭环轨迹，青色为公开Polygon，红星为事后评价GT。</p><a href="REPORT_zh.md">中文报告</a><img width="100%" src="navigation_comparison.png">'+items+'</body>')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
