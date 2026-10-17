"""Paired, map-cluster inference and honest partial/final Polygon reports."""
import csv
import json
from pathlib import Path
import hashlib
from collections import defaultdict
import numpy as np

from scripts.polygon_ablation_plan import HERE,OUT,BASE,FROZEN,TRAINED,canonical_job,init
from graphnav.checkpointing import atomic_json,file_sha256


def write_csv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status','phase','scope','representation','seed','split','samples']
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def table(rows,cols):
    return '\n'.join(['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']+
                     ['|'+'|'.join(str(r.get(k,'')).replace('|','\\|') for k in cols)+'|' for r in rows])+'\n'


def paired_ci(delta,maps,metric_names,seed=20261017,draws=10000):
    delta=np.asarray(delta,np.float64);maps=np.asarray(maps)
    if delta.ndim==1:delta=delta[:,None]
    n=len(delta);unique=np.unique(maps);m=len(unique)
    rng=np.random.default_rng(seed);episode=[]
    for start in range(0,draws,128):
        indices=rng.integers(0,n,size=(min(128,draws-start),n))
        episode.append(delta[indices].mean(1))
    episode=np.concatenate(episode)
    sums=np.stack([delta[maps==u].sum(0) for u in unique]);counts=np.array([(maps==u).sum() for u in unique])
    means=sums/counts[:,None]
    indices=rng.integers(0,m,size=(draws,m))
    cluster=sums[indices].sum(1)/counts[indices].sum(1)[:,None]
    equal=means[indices].mean(1)
    result=dict(samples=n,independent_maps=m,draws=draws,seed=seed,
                limitations='Pointwise percentile CIs; no multiple-comparison adjustment. Episode bootstrap is diagnostic only; map clusters primary. Few maps imply discrete, unstable intervals; more draws do not add maps.',metrics={},per_map=[],leave_one_map_out=[])
    for i,k in enumerate(metric_names):
        result['metrics'][k]=dict(delta=float(delta[:,i].mean()),episode_ci95=np.quantile(episode[:,i],[.025,.975]).tolist(),
                                   map_cluster_episode_weighted_ci95=np.quantile(cluster[:,i],[.025,.975]).tolist(),
                                   map_equal_delta=float(means[:,i].mean()),map_cluster_map_equal_ci95=np.quantile(equal[:,i],[.025,.975]).tolist(),
                                   improved=int((delta[:,i]>1e-10).sum()),degraded=int((delta[:,i]<-1e-10).sum()),unchanged=int((np.abs(delta[:,i])<=1e-10).sum()))
    for u in unique:
        selected=maps==u
        result['per_map'].append(dict(map_name=str(u),samples=int(selected.sum()),**dict(zip(metric_names,delta[selected].mean(0).tolist()))))
        if m>1:
            kept=maps!=u
            remaining=np.stack([delta[maps==v].mean(0) for v in unique if v!=u])
            result['leave_one_map_out'].append(dict(excluded_map=str(u),samples=int(kept.sum()),episode_weighted=dict(zip(metric_names,delta[kept].mean(0).tolist())),map_equal=dict(zip(metric_names,remaining.mean(0).tolist()))))
    return result


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def identity(row,endpoint):
    return row['query_id'] if endpoint=='localization' else tuple(row['episode_id'])


def paired_rows(a,b,endpoint,metrics):
    left={identity(r,endpoint):r for r in a};right={identity(r,endpoint):r for r in b}
    assert len(left)==len(a) and len(right)==len(b) and set(left)==set(right)
    output=[];maps=[];ids=[]
    for k,r in left.items():
        s=right[k];assert r['map_name']==s['map_name']
        x=r if endpoint=='localization' else r['metrics']
        y=s if endpoint=='localization' else s['metrics']
        output.append([x[v]-y[v] for v in metrics]);maps.append(r['map_name']);ids.append(k)
    return np.asarray(output),maps,ids


def summarize(plan=None,partial=False):
    plan=plan or init();local=[];navigation=[];data={};groups=[];maprows=[];stat={}
    training_done=list((OUT/'training').glob('*/complete.json'))
    execution=dict(status='Running' if partial else 'Done',completed_training_runs=len(training_done),required_training_runs=15,
                   trained_static_complete=(OUT/'all_static_complete.json').exists(),
                   active_progress=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else {})
    atomic_json(execution,OUT/'execution_status.json')
    episode_queries={}
    # Only completed metrics are eligible. Raw partial files are preserved but
    # never summarized as a complete split.
    for f in sorted((OUT/'localization').glob('*/*.json')):
        if f.name.endswith(('.metrics.json','.started.json')):continue
        d=json.loads(f.read_text())
        if d.get('status')!='Done':continue
        split=d['split'];rows=read_rows(f.parent/f'{split}.localization.jsonl')
        assert len(rows)==d['metrics']['samples']=={'val_seen':2426,'val_unseen':2679}[split]
        key=(d['phase'],d['scope'],d['representation'],d['seed'],split,'localization')
        data[key]=rows
        local.append(dict(status='Done',phase=d['phase'],scope=d['scope'],representation=d['representation'],seed=d['seed'],split=split,
                          **d['metrics'],checkpoint_sha256=d['checkpoint_sha256'],result_path=str(f),predictions=str(f.parent/f'{split}.localization.jsonl'),plan_sha256=d['plan_sha256']))
    for f in sorted((OUT/'navigation').glob('*/*/metrics.json')):
        if 'seen128' in f.parent.name:continue
        d=json.loads(f.read_text());s=d['signature'];parts=f.parent.parent.name.split('__')
        phase='frozen' if parts[0]=='frozen' else 'retrained'
        scope,rep=parts[1:3];seed=int(parts[3].replace('seed','')) if len(parts)>3 else None
        split=s['split'];rows=read_rows(f.parent/'predictions.jsonl')
        assert len(rows)==d['metrics']['episodes']==plan['navigation']['splits'][split]
        key=(phase,scope,rep,seed,split,'navigation');data[key]=rows
        navigation.append(dict(status='Done',phase=phase,scope=scope,representation=rep,seed=seed,split=split,samples=len(rows),
                               **d['metrics'],checkpoint_sha256=s['checkpoint_sha256'],head_sha256=s['head_sha256'],result_path=str(f),predictions=str(f.parent/'predictions.jsonl')))
    # Exact Full-input identity allows aliases without rerunning/choosing seeds.
    for split in ('val_seen','val_unseen'):
        for phase,seeds in [('frozen',[None]),('retrained',[0,1,2])]:
            for seed in seeds:
                for endpoint,rows in [('localization',local),('navigation',navigation)]:
                    source=(phase,'descriptor_only','full_polygon',seed,split,endpoint)
                    target=(phase,'end_to_end','full_polygon',seed,split,endpoint)
                    if target not in data and source in data:
                        data[target]=data[source]
                        original=next(r for r in rows if (r['phase'],r['scope'],r['representation'],r['seed'],r['split'])==source[:5])
                        rows.append(dict(original,scope='end_to_end',alias_of_scope='descriptor_only; identical Full Polygon inputs and checkpoint SHA'))
    if not partial:
        expected={(phase,scope,rep,seed,split,endpoint)
                  for phase,conditions in [('frozen',[(s,r,None) for s,r in FROZEN]),('retrained',TRAINED)]
                  for scope,rep,seed in conditions for split in ('val_seen','val_unseen')
                  for endpoint in ('localization','navigation')}
        assert expected <= data.keys(), f'Missing completed conditions: {expected-data.keys()}'
    for endpoint,rows in [('localization',local),('navigation',navigation)]:
        metrics=('R@1','R@4','R@8','NE') if endpoint=='localization' else ('SR','SPL','OSR','NE')
        for r in list(rows):
            raw=data[(r['phase'],r['scope'],r['representation'],r['seed'],r['split'],endpoint)]
            # Group membership defined before results from public map geometry.
            if endpoint=='navigation':
                source_rows=data.get((r['phase'],r['scope'],r['representation'],r['seed'],r['split'],'localization'),[])
                # Exact public query IDs cannot be recovered from episode IDs alone;
                # use the dataset's fixed published Episode -> map/instruction map.
                from scripts.polygon_ablation_navigation import resources
                if r['split'] not in episode_queries:
                    ds,*_=resources(plan,r['split'])
                    episode_queries[r['split']]={tuple(ds[i].episode_id):(ds[i].map_name,ds[i].instruction) for i in range(len(ds))}
                by_episode=episode_queries[r['split']]
                from scripts.polygon_ablation_plan import query_key
                qgroups={x['query_id']:x['groups'] for x in source_rows}
                memberships={identity(x,endpoint):qgroups.get(query_key(dict(zip(('map_name','instruction'),by_episode[tuple(x['episode_id'])]))),['group_unknown']) for x in raw}
            else:memberships={identity(x,endpoint):x['groups'] for x in raw}
            names=sorted({g for v in memberships.values() for g in v}|{'regular','irregular','compact','elongated','small','large','point_like','area_like','road_like','single_landmark_name','multiple_landmark_names','explicit_bound_relation','multiple_explicit_bound_relations'})
            for group in names:
                selected=[x for x in raw if group in memberships[identity(x,endpoint)]]
                vals=[x if endpoint=='localization' else x['metrics'] for x in selected]
                groups.append(dict(phase=r['phase'],scope=r['scope'],representation=r['representation'],seed=r['seed'],split=r['split'],endpoint=endpoint,group=group,
                                   samples=len(selected),independent_maps=len({x['map_name'] for x in selected}),
                                   **{k:float(np.mean([v[k] for v in vals])) if vals else None for k in metrics}))
            for map_name in sorted({x['map_name'] for x in raw}):
                selected=[x for x in raw if x['map_name']==map_name];vals=[x if endpoint=='localization' else x['metrics'] for x in selected]
                maprows.append(dict(phase=r['phase'],scope=r['scope'],representation=r['representation'],seed=r['seed'],split=r['split'],endpoint=endpoint,map_name=map_name,
                                    samples=len(selected),**{k:float(np.mean([v[k] for v in vals])) for k in metrics}))
    # Per-seed and fixed-seed-averaged paired effects. Seeds never count as maps.
    for endpoint,metrics in [('localization',('R@1','R@4','R@8')),('navigation',('SR','SPL'))]:
        for phase,seeds in [('frozen',[None]),('retrained',[0,1,2])]:
            for scope in ('descriptor_only','end_to_end'):
                for split in ('val_seen','val_unseen'):
                    for other in ('centroid','bounding_box'):
                        seed_deltas=[];common_maps=None;common_ids=None
                        for seed in seeds:
                            a=data.get((phase,scope,'full_polygon',seed,split,endpoint));b=data.get((phase,scope,other,seed,split,endpoint))
                            if a is None or b is None:continue
                            delta,maps,ids=paired_rows(a,b,endpoint,metrics)
                            label=f'{endpoint}/{phase}/{scope}/{split}/full_polygon-minus-{other}/seed{seed}'
                            stat[label]=paired_ci(delta,maps,metrics,draws=plan['bootstrap']['draws'],seed=plan['bootstrap']['seed'])
                            if common_ids is not None:assert ids==common_ids and maps==common_maps
                            common_ids,common_maps=ids,maps;seed_deltas.append(delta)
                        if phase=='retrained' and len(seed_deltas)==3:
                            label=f'{endpoint}/{phase}/{scope}/{split}/full_polygon-minus-{other}/seed_mean'
                            stat[label]=paired_ci(np.mean(seed_deltas,0),common_maps,metrics,draws=10000,seed=20261017)
    # Channel and bypass interventions: dependence, not fair representation claims.
    for split in ('val_seen','val_unseen'):
        for endpoint,metrics in [('localization',('R@1','R@4','R@8')),('navigation',('SR','SPL'))]:
            full=data.get(('frozen','descriptor_only','full_polygon',None,split,endpoint))
            if full is None:continue
            for scope,rep in FROZEN:
                if rep in ('full_polygon','centroid','bounding_box'):continue
                other=data.get(('frozen',scope,rep,None,split,endpoint))
                if other is None:continue
                delta,maps,_=paired_rows(full,other,endpoint,metrics)
                stat[f'{endpoint}/frozen_feature_dependence/{split}/full-minus-{rep}']=paired_ci(delta,maps,metrics,draws=10000,seed=20261017)
    summaries=[]
    for endpoint,rows in [('localization',local),('navigation',navigation)]:
        metric_names=('R@1','R@4','R@8','NE') if endpoint=='localization' else ('SR','SPL','OSR','NE','False_Stop_Rate')
        for scope in ('descriptor_only','end_to_end'):
            for rep in ('centroid','bounding_box','full_polygon'):
                for split in ('val_seen','val_unseen'):
                    rs=[r for r in rows if (r['phase'],r['scope'],r['representation'],r['split'])==('retrained',scope,rep,split)]
                    if len(rs)!=3:continue
                    summaries.append(dict(endpoint=endpoint,scope=scope,representation=rep,split=split,seeds=3,samples_per_seed=rs[0]['samples'],
                                          **{f'{k}_{s}':float(getattr(np,s)([r[k] for r in rs],**({'ddof':1} if s=='std' else {}))) for k in metric_names for s in ('mean','std') if all(r[k] is not None for r in rs)}))
    write_csv('localization_results.csv',local);write_csv('navigation_results.csv',navigation)
    write_csv('polygon_ablation_results.csv',[dict(r,endpoint='localization') for r in local]+[dict(r,endpoint='navigation') for r in navigation])
    write_csv('geometry_subgroup_results.csv',groups);write_csv('per_map_results.csv',maprows);write_csv('seed_summary.csv',summaries)
    subgroup_summary=[]
    for key in sorted({(r['endpoint'],r['split'],r['group']) for r in groups if r['phase']=='retrained' and r['scope']=='end_to_end'}):
        endpoint,split,group=key;metric='R@1' if endpoint=='localization' else 'SR'
        row=dict(endpoint=endpoint,split=split,group=group)
        for rep in ('centroid','bounding_box','full_polygon'):
            values=[r for r in groups if (r['phase'],r['scope'],r['representation'],r['endpoint'],r['split'],r['group'])==('retrained','end_to_end',rep,*key)]
            if len(values)==3:
                assert len({r['samples'] for r in values})==1
                row['samples_per_seed']=values[0]['samples'];row['independent_maps']=values[0]['independent_maps']
                row[f'{rep}_{metric}_mean']=float(np.mean([r[metric] for r in values])) if values[0]['samples'] else None
        for rep in ('centroid','bounding_box'):
            f=row.get(f'full_polygon_{metric}_mean');b=row.get(f'{rep}_{metric}_mean')
            row[f'full_minus_{rep}_pp']=100*(f-b) if f is not None and b is not None else None
        subgroup_summary.append(row)
    write_csv('geometry_subgroup_seed_summary.csv',subgroup_summary)
    atomic_json(dict(protocol=plan['bootstrap'],comparisons=stat),OUT/'paired_statistics.json')
    paired_summary=[]
    for comparison,result in stat.items():
        for metric,value in result['metrics'].items():
            lomo=[x['episode_weighted'][metric] for x in result['leave_one_map_out']]
            paired_summary.append(dict(comparison=comparison,metric=metric,samples=result['samples'],independent_maps=result['independent_maps'],
                                       **value,lomo_episode_weighted_min=min(lomo) if lomo else None,lomo_episode_weighted_max=max(lomo) if lomo else None))
    write_csv('paired_summary.csv',paired_summary)
    logs=defaultdict(list)
    if (OUT/'events.jsonl').exists():
        for line in (OUT/'events.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row.get('job'):logs[row['job']].append(line)
    (OUT/'logs').mkdir(exist_ok=True)
    for name,lines in logs.items():(OUT/'logs'/f'{name}.log').write_text('\n'.join(lines)+'\n')
    # A concise verdict is based on predeclared seed-averaged E2E map intervals.
    verdict=[]
    for endpoint,metrics in [('localization',('R@1','R@4','R@8')),('navigation',('SR','SPL'))]:
        for split in ('val_seen','val_unseen'):
            for other in ('centroid','bounding_box'):
                label=f'{endpoint}/retrained/end_to_end/{split}/full_polygon-minus-{other}/seed_mean'
                if label in stat:
                    for k in metrics:
                        v=stat[label]['metrics'][k];verdict.append(dict(endpoint=endpoint,split=split,control=other,metric=k,
                            delta_pp=100*v['delta'],episode_ci95_pp=[100*x for x in v['episode_ci95']],
                            map_ci95_pp=[100*x for x in v['map_cluster_episode_weighted_ci95']],
                            map_equal_delta_pp=100*v['map_equal_delta'],map_equal_ci95_pp=[100*x for x in v['map_cluster_map_equal_ci95']],
                            maps=stat[label]['independent_maps']))
    audit='[信息通路审计](POLYGON_FEATURE_AUDIT.md)'
    answers=[]
    for endpoint,metric in [('localization','R@1'),('navigation','SR'),('navigation','SPL')]:
        values=[v for v in verdict if v['endpoint']==endpoint and v['metric']==metric]
        if values:
            for v in values:
                lower,upper=v['map_ci95_pp']
                support='地图级正收益区间下界>0' if lower>0 and v['delta_pp']>0 else '未取得地图级明确正收益支持'
                answers.append(f"- {v['split']} {metric}，Full Polygon−{v['control']}：{v['delta_pp']:+.4f}pp，整地图95%CI [{lower:+.4f},{upper:+.4f}]pp；{support}。")
    if not answers:answers=['- 公平重训及完整闭环尚未全部完成，不能从冻结特征敏感性直接判断表示优劣。']
    dependence=[]
    for split in ('val_seen','val_unseen'):
        for rep in ('no_boundary','no_inside','no_inside_information','no_extent','prior_to_centroid','rules_to_centroid','nodes_to_centroid'):
            key=f'localization/frozen_feature_dependence/{split}/full-minus-{rep}'
            if key in stat:
                v=stat[key]['metrics']['R@1'];dependence.append(dict(split=split,intervention=rep,delta_R1_pp=100*v['delta'],map_ci95_pp=[100*x for x in v['map_cluster_episode_weighted_ci95']]))
    report=f'''# PolyBelief Polygon Geometry 独立贡献验证

状态：{'部分完成；未完成项目保留Pending' if partial else '所有预注册静态及导航条件已完成'}。{audit}与frozen_plan.json先于实验结果产生。
独立训练已完成{len(training_done)}/15；当前结果行数：静态{len(local)}、导航{len(navigation)}，包含明确标记的Full跨口径相同输入别名。最新运行阶段另见progress.json及workflow.log。
主对照是End-to-End独立重训；Descriptor Only仅解释描述符增量，冻结干预仅解释已训练权重的依赖。
相同10-D/模型容量、3个种子、15epoch、batch16、AdamW5e-5。训练样本按Train对象分组保留固定Dev，选择只用Dev R@1及NE；未用Unseen/Test选模或修改配置。
Full Polygon在两个口径的输入完全相同，每seed按同一SHA复用；共18个逻辑条件、15次独立训练。seed0预先指定优先闭环，并保留seed1/2所有结果。
所有静态评估完成标志all_static_complete.json先于任何闭环。静态Seen2426/Unseen2679，导航Seen2470/Unseen2697；原始定位/导航结果未冒充本实验结果。

## 三个最终问题

1. Full Polygon是否比Centroid/Bounding Box更好：以下End-to-End重训配对效应给出实际方向，不预设正收益；统计区间跨零或点估计为负时，不能写成普遍优势。
2. Seen/Unseen是否一致及统计支持：分别报告；4张Unseen地图仅为4个独立地图簇，主要依据整地图cluster CI和Leave-One-Map-Out，不能以Episode CI替代。
3. 来源是边界/Inside/Extent还是旁路：Descriptor Only的Full与BBox只改变边界/Inside，Extent相同；End-to-End还改变prior/rules/relation_fields及节点统计。冻结No Inside仍保留signed boundary符号；No Inside Information另清除符号。仅冻结敏感性不能证明某模块公平重训后的独立因果贡献。

'''+ '\n'.join(answers)+'''

冻结权重通路敏感性（Full−干预，负值表示干预后反而更好，不隐藏）：

'''+table(dependence,['split','intervention','delta_R1_pp','map_ci95_pp'])+'''

## 冻结权重完整静态结果

这些是当前权重的输入干预，不能代替公平重训结论。完整逐样本预测与配对变化保留。

'''+table([r for r in local if r['phase']=='frozen'],['scope','representation','split','samples','R@1','R@4','R@8','NE'])+'''

## 三Seed均值及样本标准差

比例列均为0–1，NE为米；delta_pp及map_ci95_pp明确使用百分点。每seed原始结果分别保留在两个results.csv中，不选择最佳seed。

'''+table(summaries,['endpoint','scope','representation','split','seeds','samples_per_seed','R@1_mean','R@1_std','R@4_mean','R@4_std','R@8_mean','R@8_std','SR_mean','SR_std','SPL_mean','SPL_std','NE_mean','NE_std'])+'''
## End-to-End主要配对效应（seed平均）

'''+table(verdict,['endpoint','split','control','metric','delta_pp','episode_ci95_pp','map_ci95_pp','map_equal_delta_pp','map_equal_ci95_pp','maps'])+'''

所有Seed、指标及对照的帮助/损害/不变数、逐地图效应和Leave-One-Map-Out结果见paired_summary.csv和paired_statistics.json。下表给出每个Seed的原始结果：

'''+table([dict(r,endpoint=e) for e,rs in [('localization',local),('navigation',navigation)] for r in rs if r['phase']=='retrained'],['endpoint','scope','representation','seed','split','samples','R@1','R@4','R@8','SR','SPL','OSR','NE','False_Stop_Rate'])+'''
## 几何复杂度与范围限制

geometry_subgroup_results.csv按预先固定的旋转最小包围矩形填充率0.85、长宽比3、最大地标面积1000m²、点状面积25m²及源类型/名称数分组；每组保留R@1/SR、NE和N，不剔除Polygon退化组。
多参照物的类型旗标可重叠，不能把组N相加当全样本数。关系组只使用明确解析且唯一绑定的公开参照，非GT目标–参照关系；不能强称所有复杂关系均得到验证。
完整分组、每图实值、paired_statistics.json中的help/hurt/unchanged、两种地图权重及Leave-One-Map-Out敏感性均随结果保存。百分位区间未作多重比较校正，分组分析属于探索性。

End-to-End三Seed平均分组结果（同一Query/Episode重复三个Seed不增加独立地图数；N为每Seed样本数）：

'''+table(subgroup_summary,['endpoint','split','group','samples_per_seed','independent_maps','centroid_R@1_mean','bounding_box_R@1_mean','full_polygon_R@1_mean','centroid_SR_mean','bounding_box_SR_mean','full_polygon_SR_mean','full_minus_centroid_pp','full_minus_bounding_box_pp'])+'''

## 控制器与可复现性

全部使用同一个原始预训练Trajectory/Stop Head，SHA为'''+plan['navigation']['head_sha256']+'''，不微调；28×28、stop0.3、20决策、水平≤50m、垂直≤10m，原1mm预测目标到达门控保留。
重训后的候选上下文与旧head的分布可能不匹配，因此导航差异同时含表示和固定控制器迁移响应；不将其隐瞒或分别微调head以提高结果。
本实验为几何/语言定位和控制器迁移，不启用RGB/历史，不把它与384或独立视觉残差的提升相加。
public_static_record只使用公开地标绑定/名称/轮廓，目标和未来轨迹仅在forward/rollout之后评分；渲染优化须先通过128条完整轨迹与原观测builder逐点一致性。
配置、源/权重SHA、Query清单、Train-Dev分组、初始化及每epoch数据顺序hash、训练断点、逐Query/episode预测和日志均原地保留。复现命令：`python -m scripts.run_polygon_ablation --resume`；统计重建：`python -m scripts.polygon_ablation_statistics`。
implementation_provenance.json及source_snapshot_manifest.json记录实现Commit与代码SHA；frozen_plan.json中的source_commit是实验开始前的基线Commit。旧架构args保留来源字段（如原continue_from/seed），实际独立随机初始化、Seed和预算以新config.json及顶层Checkpoint元数据为准，不执行旧continue_from。navigation_data_audit.json记录processed_citynav目录与revised静态记录的Query ID集合一致性及全部来源SHA，导航started.json另记录实际策略代码SHA。

若主要End-to-End Unseen效应未获地图级正下界支持，论文核心创新应强调“使用完整公开几何的可解释接口/表示设计”，而不能声称已证明Polygon跨地图优于点/框。若仅Descriptor Only有效，也只能声称候选–地标描述符的条件增量价值。
'''
    (OUT/'POLYGON_ABLATION_REPORT.md').write_text(report)
    # Append a bounded, replaceable update; do not regenerate/overwrite older tables.
    for path,title in [(HERE/'docs/PAPER_EXPERIMENT_TODO.md','Polygon独立贡献实验进度'),(HERE/'reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md','Polygon独立贡献补充')]:
        marker='\n<!-- POLYGON_ABLATION_UPDATE -->\n'
        text=path.read_text().split(marker)[0]
        if path.name=='PAPER_EXPERIMENT_TODO.md':
            lines=text.splitlines()
            for i,line in enumerate(lines):
                if line.startswith('|Polygon vs Centroid vs Bounding Box闭环|'):
                    cells=line.split('|');cells[2]='Running' if partial else 'Done';cells[4]='新实验：固定Train-Dev、3seed、独立初始化';cells[5]='reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md'
                    cells[6]='正在执行完整队列' if partial else '无；诊断与重训两口径分别报告，结果正负均保留'
                    lines[i]='|'.join(cells)
            text='\n'.join(lines)+'\n'
        update=f'\n## {title}\n\n状态：'+('Running' if partial else 'Done')+f'。新静态结果{len(local)}行，导航结果{len(navigation)}行。独立报告位于reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md；冻结干预与3seed公平重训、Descriptor Only与End-to-End严格分开。\n'
        path.write_text(text+marker+update)
    from scripts.polygon_ablation_index_update import update_index
    update_index(local,navigation,partial)
    if not partial:
        # The runner writes its final completion record after this function;
        # atexit keeps backup in the same live process after that record exists.
        import atexit
        from scripts.sync_polygon_ablation import sync_completed_experiment
        atexit.register(sync_completed_experiment)
    return dict(localization_rows=len(local),navigation_rows=len(navigation),comparisons=len(stat))


if __name__=='__main__':
    print(summarize(partial=not (OUT/'workflow_complete.json').exists()))
