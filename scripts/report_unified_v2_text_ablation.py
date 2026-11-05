"""PR23 complete TRAIN diagnosis and full paired Val-Seen artifacts."""
import argparse,json,csv,gzip,html,shutil,collections
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.semantic_geometry import canonical

ARMS=('full_text','landmark_relation')

def main():
    p=argparse.ArgumentParser()
    for k in ('run','output','data-root'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=a.run;a.output.mkdir(parents=True,exist_ok=True)
    dev=json.load(open(r/'dev_report/summary.json'));pred=torch.load(r/'dev_report/all_predictions.pt',map_location='cpu',weights_only=True)
    cf=json.load(open(r/'counterfactual/counterfactual.json'));amb=json.load(open(r/'input_ambiguity.json'));train=json.load(open(r/'train_landmark_relation/progress.json'));frozen=json.load(open(r/'FROZEN_BEFORE_VAL.json'))
    for k,v in frozen['code'].items():assert file_sha256(k)==v,'Frozen code changed'
    for v in frozen['sources'].values():assert file_sha256(v['path'])==v['SHA256'],'Frozen source changed'
    vals={};nav={}
    for arm in ARMS:
        summary=json.load(open(r/f'eval_{arm}/metrics.json'));rows=[json.loads(s) for s in (r/f'eval_{arm}/predictions.jsonl').read_text().splitlines()]
        assert len(rows)==len({x['index'] for x in rows})==2470
        assert summary['coverage']['completed']==2470;vals[arm]={x['index']:x for x in rows};nav[arm]=summary
    assert set(vals['full_text'])==set(vals['landmark_relation'])
    paired=[]
    for idx in sorted(vals['full_text']):
        b=vals['full_text'][idx];o=vals['landmark_relation'][idx];assert b['episode_id']==o['episode_id'] and b['poses'][0]==o['poses'][0]
        br=json.load(open(r/f'eval_full_text/belief_steps/{idx:05d}/public_record.json'));orr=json.load(open(r/f'eval_landmark_relation/belief_steps/{idx:05d}/public_record.json'))
        assert br['world_bounds']==orr['world_bounds'] and br['landmarks']==orr['landmarks']
        paired.append(dict(index=idx,map_name=b['map_name'],baseline_SR=b['metrics']['SR'],compact_SR=o['metrics']['SR'],baseline_NE=b['metrics']['NE'],compact_NE=o['metrics']['NE'],baseline_termination=b['termination'],compact_termination=o['termination']))
    fix=sum(bool(not x['baseline_SR'] and x['compact_SR']) for x in paired);damage=sum(bool(x['baseline_SR'] and not x['compact_SR']) for x in paired)
    raw_val=json.load(open(a.data_root/'processed_citynav/citynav_val_seen.json'));groups=collections.defaultdict(list)
    for j,x in enumerate(paired):groups[(x['map_name'],str(raw_val[x['index']]['object_ids'][0]))].append(j)
    rng=np.random.default_rng(42);keys=list(groups);dif=[]
    for _ in range(2000):
        ix=np.concatenate([groups[keys[j]] for j in rng.integers(len(keys),size=len(keys))]);dif.append(float(np.mean([paired[j]['compact_SR']-paired[j]['baseline_SR'] for j in ix])))
    result=dict(Val_Seen={k:v['metrics'] for k,v in nav.items()},paired=dict(fixes=fix,damage=damage,net=fix-damage),physical_target_group_bootstrap_SR_delta_95CI=np.quantile(dif,[.025,.975]).tolist(),physical_target_groups=len(groups),single_seed=True)
    failures={}
    for arm in ARMS:
        counts=collections.Counter()
        for idx,row in vals[arm].items():
            target=np.asarray(raw_val[idx]['target_positions'][-1][:2]);goals=np.asarray([d['selected_goal_xy'] for d in row['step_diagnostics']])
            good_goal=bool(len(goals) and np.linalg.norm(goals[-1]-target)<=20)
            if not row['metrics']['SR']:counts['goal_correct_navigation_failed' if good_goal else 'goal_wrong_navigation_failed']+=1
            counts['OSR_success_SR_failed']+=int(row['metrics']['OSR'] and not row['metrics']['SR'])
            counts['goal_jumps_over20m']+=int((np.linalg.norm(np.diff(goals,axis=0),axis=-1)>20).sum()) if len(goals)>1 else 0
            counts['correct_policy_stop']+=int(row['termination']=='policy_stop' and row['metrics']['SR'])
            counts[row['termination']]+=1
        failures[arm]=dict(counts)
    result['navigation_failure_attribution']=failures
    atomic_json(result,a.output/'navigation_summary.json')
    with (a.output/'navigation_paired.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    for src,name in [(r/'dev_report/summary.json','dev_summary.json'),(r/'counterfactual/counterfactual.json','counterfactual.json'),(r/'input_ambiguity.json','input_ambiguity.json'),(r/'token_counts.json','token_counts.json'),(r/'train_landmark_relation/progress.json','training_progress.json'),(r/'FROZEN_BEFORE_VAL.json','FROZEN_BEFORE_VAL.json')]:shutil.copyfile(src,a.output/name)
    audit=[json.loads(s) for s in (r/'paired_inputs/extraction_audit.jsonl').read_text().splitlines()]
    data={arm:[json.loads(s) for s in (r/'paired_inputs'/f'{arm}.jsonl').read_text().splitlines()] for arm in ARMS};ids=pred['dev_ids']['full_text'];outputs=pred['outputs']
    baseline=outputs['full_text']['r1'].numpy();ours=outputs['landmark_relation']['r1'].numpy();masks={}
    for status in ('paired_words','name_only','no_public_anchor'):masks[status]=np.array([audit[i]['status']==status for i in ids])
    masks['multiple_landmark_names']=np.array([len({canonical(x['name']) for x in data['full_text'][i]['landmarks']})>1 for i in ids])
    masks['same_name_multiple_Polygons']=np.array([len(data['full_text'][i]['landmarks'])>len({canonical(x['name']) for x in data['full_text'][i]['landmarks']}) for i in ids])
    subgroup={}
    for group,mask in masks.items():
        subgroup[group]={}
        for arm in ARMS:
            q=outputs[arm];good=q['r1'].numpy()
            subgroup[group][arm]=dict(N=int(mask.sum()),R20=float(good[mask].mean()) if mask.any() else None,NE_m=float(q['ne'].numpy()[mask].mean()) if mask.any() else None,fixes=int((~baseline&good&mask).sum()),damage=int((baseline&~good&mask).sum()))
    atomic_json(subgroup,a.output/'dev_subgroups.json');contribution={}
    for arm,q in outputs.items():
        g=q['geometry_r1'];z=q['r1']
        contribution[arm]=dict(corrections=int((~g&z).sum()),damage=int((g&~z).sum()),net=int(z.sum()-g.sum()),max_residual=float(q['relation_delta_max'].max()),median_residual=float(q['relation_delta_max'].median()),manual_rule_active_queries=int(q['rule_nonzero'].sum()))
    atomic_json(contribution,a.output/'relation_contribution.json')
    with gzip.open(a.output/'dev_paired.jsonl.gz','wt') as f:f.write((r/'dev_report/paired_tasks.jsonl').read_text())
    hist=train['history'];fig,ax=plt.subplots(figsize=(6,3));ax.plot([x['epoch'] for x in hist],[x['train_loss'] for x in hist],'o-');ax.set_xlabel('Epoch');ax.set_ylabel('Full TRAIN target CE + KL');fig.tight_layout();fig.savefig(a.output/'training_curve.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    for ax,metric,title in zip(axes,('SR','NE'),('Full Val-Seen SR (%)','Full Val-Seen NE (m)')):
        ax.bar(ARMS,[nav[k]['metrics'][metric]*(100 if metric=='SR' else 1) for k in ARMS]);ax.set_ylabel(title)
    fig.tight_layout();fig.savefig(a.output/'navigation_comparison.png',dpi=140);plt.close(fig)
    selected=list(np.flatnonzero(~baseline&ours)[:3])+list(np.flatnonzero(baseline&~ours)[:5])+list(np.flatnonzero(masks['same_name_multiple_Polygons'])[:2])+list(np.flatnonzero(masks['no_public_anchor'])[:2]);selected=list(dict.fromkeys(selected));cases=[]
    for j in selected:
        idx=ids[j];record=data['full_text'][idx];b=np.asarray(record['world_bounds']);target=b[:2]+np.asarray(record['target_xy'])*(b[2:]-b[:2])
        fields=[('Full text',outputs['full_text']['logits'][j]),('Compact Geometry',outputs['landmark_relation']['geometry_logits'][j]),('Compact + Relation',outputs['landmark_relation']['logits'][j])]
        fig,axes=plt.subplots(1,3,figsize=(12,4));vmax=max(float(z.softmax(-1).max()) for _,z in fields)
        for ax,(label,z) in zip(axes,fields):
            ax.imshow(z.softmax(-1).reshape(28,28).numpy(),origin='lower',extent=[b[0],b[2],b[1],b[3]],vmin=0,vmax=vmax)
            for lm in record['landmarks']:
                c=np.asarray(lm['contour'])*(b[2:]-b[:2])+b[:2];ax.plot(*np.vstack((c,c[0])).T,'c-',lw=.5)
            t=int(z.argmax());xy=b[:2]+np.array([t%28+.5,t//28+.5])/28*(b[2:]-b[:2]);ax.plot(*xy,'wx');ax.plot(*target,'r*');ax.set_aspect('equal');ax.set_title(label,fontsize=9)
        fig.suptitle(f'TRAIN {idx}: '+record['instruction'][:140],fontsize=8);fig.tight_layout();fname=f'dev_{idx:05d}.png';fig.savefig(a.output/fname,dpi=130);plt.close(fig)
        cases.append(dict(index=idx,original=record['instruction'],compact=data['landmark_relation'][idx]['instruction'],figure=fname))
    choose=[x['index'] for x in paired if not x['baseline_SR'] and x['compact_SR']][:3]+[x['index'] for x in paired if x['baseline_SR'] and not x['compact_SR']][:5]
    for idx in choose:
        record=json.load(open(r/f'eval_full_text/belief_steps/{idx:05d}/public_record.json'));b=np.asarray(record['world_bounds']);target=np.asarray(raw_val[idx]['target_positions'][-1][:2]);fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,arm in zip(axes,ARMS):
            trace=vals[arm][idx];xy=np.asarray(trace['poses'])[:,:2];field=np.load(r/f'eval_{arm}/belief_steps/{idx:05d}/00.npz')['full'];prob=torch.from_numpy(field).softmax(-1).reshape(28,28).numpy()
            ax.imshow(prob,origin='lower',extent=[b[0],b[2],b[1],b[3]])
            for lm in record['landmarks']:
                c=np.asarray(lm['contour'])*(b[2:]-b[:2])+b[:2];ax.plot(*np.vstack((c,c[0])).T,'c-',lw=.5)
            ax.plot(*xy.T,'w.-');ax.plot(*xy[0],'go');ax.plot(*xy[-1],'rx');ax.plot(*target,'r*');ax.set_title(f"{arm}: SR={trace['metrics']['SR']:.0f}, NE={trace['metrics']['NE']:.1f}m");ax.set_aspect('equal')
        fig.tight_layout();fname=f'nav_{idx:05d}.png';fig.savefig(a.output/fname,dpi=130);plt.close(fig);cases.append(dict(Val_index=idx,original=record['instruction'],compact=json.load(open(r/f'eval_landmark_relation/belief_steps/{idx:05d}/public_record.json'))['instruction'],figure=fname))
    cf_fields=torch.load(r/'counterfactual/fields.pt',weights_only=True,map_location='cpu');sample=cf['source_samples'][0];b=np.array(sample['world_bounds']);fig,axes=plt.subplots(6,4,figsize=(12,16))
    cfmax=max(float(cf_fields['landmark_relation/'+mode][head][:4].softmax(-1).max()) for mode in ('full','additive_off','manual_off') for head in ('geometry','relation'))
    for row,(mode,head) in enumerate((m,h) for m in ('full','additive_off','manual_off') for h in ('geometry','relation')):
        for col,word in enumerate(('north','south','east','west')):
            ax=axes[row,col];z=cf_fields['landmark_relation/'+mode][head][col];prob=z.softmax(-1).reshape(28,28);ax.imshow(prob,origin='lower',extent=[b[0],b[2],b[1],b[3]],vmin=0,vmax=cfmax)
            for lm in sample['landmarks']:
                c=np.array(lm['contour'])*(b[2:]-b[:2])+b[:2];ax.plot(*np.vstack((c,c[0])).T,'c-',lw=.5)
            ax.set_title(mode+' / '+head+' / '+word,fontsize=8)
    fig.suptitle('TRAIN direction flips: existing ENGINE axes, not geographic truth');fig.tight_layout();fig.savefig(a.output/'counterfactual_fields.png',dpi=130);plt.close(fig)
    atomic_json(cases,a.output/'cases.json')
    body='<meta charset="utf-8"><style>body{max-width:1400px;margin:auto;font:16px sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>PR23 Unified V2全文与地标关系短语</h1><p>TRAIN诊断与完整Val-Seen导航分开，无Test与GT推理。红星离线GT，青线公开Polygon。</p>'
    for f in ('training_curve.png','navigation_comparison.png','counterfactual_fields.png'):body+=f'<a href="{f}"><img src="{f}"></a>'
    for c in cases:body+=f'<h3>{c.get("index",c.get("Val_index"))}</h3><p>{html.escape(c.get("original",""))}</p><pre>{html.escape(c.get("compact",""))}</pre><a href="{c["figure"]}"><img src="{c["figure"]}"></a>'
    (a.output/'index.html').write_text(body)
    write_report(a,dev,nav,result,cf,amb,contribution,train)
    print(json.dumps(result,indent=2))

def write_report(a,dev,nav,result,cf,amb,contribution,train):
    r=a.run
    table='\n'.join(f"|{k}|{v['R20']*100:.3f}%|{v['R5_20m']*100:.3f}%|{v['R15_20m']*100:.3f}%|{v['NE_m']:.4f}|" for k,v in dev['arms'].items())
    navtable='\n'.join(f"|{k}|2470|{v['metrics']['SR']*100:.3f}%|{v['metrics']['OSR']*100:.3f}%|{v['metrics']['SPL']:.4f}|{v['metrics']['NE']:.4f}|{v['metrics']['nDTW']:.4f}|{v['metrics']['inference_ms']:.3f}|" for k,v in nav.items())
    rules='\n'.join(f"|{arm}|{mode}|{v['R20']*100:.3f}%|{v['NE_m']:.4f}|" for arm,group in dev['rule_interventions'].items() for mode,v in group.items())
    cft='\n'.join(f"|{k}|{v['relation']['NS_engine_direction_response']*100:.2f}%|{v['relation']['EW_engine_direction_response']*100:.2f}%|{v['relation']['NS_mass_response_rate']*100:.2f}%|{v['relation']['EW_mass_response_rate']*100:.2f}%|{v['relation']['mean_NS_shift_m']:.3f}|{v['relation']['mean_EW_shift_m']:.3f}|" for k,v in cf['methods'].items())
    delta=result['physical_target_group_bootstrap_SR_delta_95CI'];new=contribution['landmark_relation'];total=train['history'][-1]['wall_seconds'];scope=json.load(open(r/'paired_inputs/manifest.json'));tok=json.load(open(r/'token_counts.json'))
    delta_sr=nav['landmark_relation']['metrics']['SR']-nav['full_text']['metrics']['SR']
    verdict=('本轮完整Val-Seen的精简组SR点估计提高；仍需结合置信区间，不能把单seed结果或输入重绑定适应等同于空间语义提升。暂不外推Unseen/Test。' if delta_sr>0 else '本轮完整Val-Seen不支持用该精简方案替换全文。')
    md=f"""# PR23 Unified V2精简文本：完整训练、诊断及Val-Seen闭环

完成：只新训练landmark_relation；relation_only已取消，full_text只复用旧Checkpoint。未合并main，未运行Test。

## 配置和数据口径

沿用原Unified V2初始化/架构/Geometry+CandidateAnchorRelation/28×28网格及10D候选—Polygon几何；seed42、5轮、batch8、Geometry1e-5/Relation1e-4、原CE+0.05KL。PGVB/BERT/控制器冻结，RGB关闭以隔离语言；原Stable Goal、Trajectory/Stop及阈值不改。

全21,878条TRAIN每轮均拟合；Birmingham7,619条是TRAIN内拟合诊断，不是独立Dev泛化。5轮13,675更新，耗时{total:.2f}s。既有架构使用geometry节点投影路径，GAT消息传播未启用，与旧实验一致。原始权重保留。

公开名称只从原文匹配FULL CityRefer目录，同名合法片段全部保留；超16容量整行空Anchor回退，不据GT选择。统计：{scope['counts']}。精简JSONL直接替换instruction，BERT/Rule/关系字段/合成节点同步重算；地图节点仍保留地标名字。修复别名传递到Rule问题。词法邻接不等于Target作用域已验证，between等未覆盖结构可能丢失。

完整文本平均{tok['full_text']['mean']:.2f} WordPiece、精简{tok['landmark_relation']['mean']:.2f}（含CLS/SEP），max_length96不变；仅全文11条超过96。GT只用于损失和事后评分。

## Birmingham全部7619条

|同一公共Polygon绑定|R20|Top5/20m|Top15/20m|NE(m)|
|---|---:|---:|---:|---:|
{table}

新模型对公共绑定全文：{dev['paired']['existing_full_vs_landmark_relation']}。

旧全文原生绑定R20={dev['native_full_text']['R20']*100:.3f}%，NE={dev['native_full_text']['NE_m']:.4f}m；改公共绑定后64.195%，净破坏449条，确有输入分布变化，不能全归因删词。冻结全文直接使用精简文本R20={dev['frozen_full_on_compact_text']['R20']*100:.3f}%；训练适配回到32.734%，仍未恢复全文水平。没有重训匹配分布的全文控制，适配与文本改变不能完全分离。

新Relation分支对新Geometry纠正{new['corrections']}、破坏{new['damage']}、净{new['net']:+d}，最大残差{new['max_residual']:.3f}。分支有用不代表整体更好。TopK是世界网格位置召回，不是视觉实例识别。

## 手工Rule与网络

additive_off只取消直接加分；manual_off还清掉规则字段、关系节点和对应名字/连接，保留公共Polygon先验及网络。它们是冻结干预，不是另训模型。

|模型|干预|R20|NE(m)|
|---|---|---:|---:|
{rules}

64个公开单Polygon地标，无GT选样，方向反事实如下：

|模型/干预|N→S工程方向响应率|E→W响应率|N区域质量响应率|E区域质量响应率|N−S质心差(m)|E−W差(m)|
|---|---:|---:|---:|---:|---:|---:|
{cft}

north=+Y只是旧Rule工程约定；TIF没有CRS/地理配准，以上不是地理方向准确率，不据GT翻转轴。仅去掉直接加分后明显反向，显示规则特征学习与直接加分存在补偿/抵消。清除全部手工关系输入后，新模型方向响应未稳定吻合工程约定，不能据热图移动宣称学会语义。全文模型的合成简句也存在输入分布变化。

## 信息损失证据

Birmingham全文有{amb['full_text']['distinct_input_signatures']}种严格输入签名，精简仅{amb['landmark_relation']['distinct_input_signatures']}种。精简后{amb['landmark_relation']['GT_diameter_over40m_groups']}组相同输入对应GT跨度>40m，涉及{amb['landmark_relation']['queries_in_incompatible_target_groups']}条；全文对应仅{amb['full_text']['queries_in_incompatible_target_groups']}条。这是事后标签冲突分析，不是GT选Goal或准确率上界，也不说明模型已最优或6322条都必失败。相同输入无法同时区分这些目标，更多Epoch不能恢复被删的身份、距离和复杂约束信息。

## 完整2470条Val-Seen真实闭环

最终5轮Checkpoint及代码先冻结，见FROZEN_BEFORE_VAL.json，无Val调参。两组同任务、起点、公共Polygon和控制器；原目标GT只在rollout后评分，未以地标Polygon距离替换原20m目标中心指标。全部失败和无候选计入分母。

|方法|N|SR|OSR|SPL|NE(m)|nDTW|决策ms/步|
|---|---:|---:|---:|---:|---:|---:|---:|
{navtable}

配对纠正{result['paired']['fixes']}、破坏{result['paired']['damage']}、净{result['paired']['net']:+d}。\n\n停止/Goal事后归因：{result['navigation_failure_attribution']}。这是预测Goal与GT的事后对照，GT未进入Stop或导航。官方物理目标分组bootstrap的SR差95%区间[{delta[0]*100:.3f},{delta[1]*100:.3f}]个百分点。单seed，不宣称多种子稳定性。策略决策时延排除环境渲染，共享GPU，不是独占硬件成绩。

## 决策与交付

{verdict} 目标身份和约束丢失已经由标签冲突检查证实；后续优先保留这些必要信息、独立核对方向参考系和关系作用域。本轮不增加网络、重训全文或调Rule/Stop。

381项CPU测试通过，见CPU_tests.txt。GPU训练Smoke、4次TRAIN导航接口通过，真实公共记录与准备文件逐项一致。adversarial-review服务超时，未列为独立审查通过。两组均2470次Policy Stop、Goal大幅跳变0，正确Goal却导航失败0；本轮失败指向目标定位，而非Stop未触发。最终代码/权重SHA已再次核验。

最终Checkpoint：{r}/train_landmark_relation/last.pt，SHA256 0c441e20922c3e75914022cf85bf0131917c4ea7f57d964e09401bf80540cc1d。\n\n服务器根目录：{r}，保存模型、日志、全部轨迹/逐步RGB SHA/Belief及预测。GitHub保存汇总、逐任务CSV、真实热图和轨迹，[HTML图册](index.html)。
"""
    (a.output/'REPORT_zh.md').write_text(md)

if __name__=='__main__':main()
