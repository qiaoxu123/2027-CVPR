"""Episode-paired and map-cluster analysis of fixed residual validation.

All effects are Real minus control, paired WITHIN each split by episode_id.
Map bootstrap draws M whole maps with replacement from the M observed maps.
Episode-weighted ratios retain each sampled map's original episode count;
map-equal effects average the sampled map means. No outcome-dependent sampling.
"""
import ast
import csv
from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import PublicEpisode,aggregate
from graphnav.calibrated_visual_residual import CalibratedVisualResidual
from graphnav.stage1 import GraphNavStage1
from graphnav.full_experiment import _batch_inputs
from graphnav.workspace_paths import PROJECT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'visual_residual_unseen_validation'
CONTROLS=('real','no-belief','zero','shuffle')
REPLICATES=20000
SEED=20261017
BASE_SHA='00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a'
RESIDUAL_SHA='5fc1a3e115b71d14fe1e8fbaf3ed29c1863d8b2bc080455d89f7a4e503d25f11'


def paired_bootstrap(diff,maps,replicates=REPLICATES,seed=SEED):
    """Return percent/percentage-point statistics for columns e.g. SR and SPL."""
    diff=np.asarray(diff,np.float64)
    if diff.ndim==1:diff=diff[:,None]
    maps=np.asarray(maps)
    assert len(diff)==len(maps)>0 and replicates>=10000
    names,inverse=np.unique(maps,return_inverse=True)
    counts=np.bincount(inverse,minlength=len(names)).astype(np.float64)
    sums=np.stack([np.bincount(inverse,weights=diff[:,j],minlength=len(names)) for j in range(diff.shape[1])],1)
    means=sums/counts[:,None]
    rng=np.random.default_rng(seed)
    episode=[]
    for start in range(0,replicates,128):
        idx=rng.integers(len(diff),size=(min(128,replicates-start),len(diff)))
        episode.append(diff[idx].mean(1))
    episode=np.concatenate(episode)
    # Separate deterministic stream makes map samples invariant to episode count.
    sampled=np.random.default_rng(seed+1).integers(len(names),size=(replicates,len(names)))
    cluster_weighted=sums[sampled].sum(1)/counts[sampled].sum(1)[:,None]
    cluster_equal=means[sampled].mean(1)
    results=[]
    for j in range(diff.shape[1]):
        results.append(dict(episode_weighted_delta_pp=float(diff[:,j].mean()*100),
                            map_equal_delta_pp=float(means[:,j].mean()*100),
                            episode_paired_95ci_pp=(np.quantile(episode[:,j],[.025,.975])*100).tolist(),
                            map_cluster_episode_weighted_95ci_pp=(np.quantile(cluster_weighted[:,j],[.025,.975])*100).tolist(),
                            map_cluster_equal_weighted_95ci_pp=(np.quantile(cluster_equal[:,j],[.025,.975])*100).tolist(),
                            map_deltas_pp={str(n):float(means[i,j]*100) for i,n in enumerate(names)}))
    return dict(replicates=replicates,episode_seed=seed,map_seed=seed+1,maps=names.tolist(),map_count=len(names),
                map_episode_counts={str(n):int(counts[i]) for i,n in enumerate(names)},statistics=results)


def pair_ids(real,control):
    def index(rows):
        result={}
        for row in rows:
            key=tuple(row['episode_id'])
            if key in result:raise ValueError(f'Duplicate episode ID: {key}')
            result[key]=row
        return result
    a,b=index(real),index(control)
    if a.keys()!=b.keys():raise ValueError('Episode IDs differ between paired controls')
    result=[]
    for key,x in a.items():
        y=b[key]
        if x['map_name']!=y['map_name'] or x['index']!=y['index']:
            raise ValueError('Paired map/index mismatch')
        result.append((x,y))
    return result


def aggregate_rows(rows):
    return aggregate([r['metrics'] for r in rows])


def map_equal_metrics(rows):
    names=sorted({r['map_name'] for r in rows})
    values=[aggregate_rows([r for r in rows if r['map_name']==name]) for name in names]
    result={k:float(np.mean([m[k] for m in values])) for k in ('SR','SPL','OSR','NE','false_stop_episode_fraction')}
    rates=[v['False_Stop_Rate'] for v in values if v['False_Stop_Rate'] is not None]
    result.update(False_Stop_Rate=float(np.mean(rates)) if rates else None,maps_with_policy_stop=len(rates),maps=len(names))
    return result


def load_control(split,control):
    scope='full_seen' if split=='val_seen' else 'full_unseen'
    directory=ROOT/'visual_evaluations'/f'calibrated_residual_{scope}_{control}'
    report=json.loads((directory/'metrics.json').read_text())
    rows=[json.loads(line) for line in (directory/'predictions.jsonl').open()]
    expected=2470 if split=='val_seen' else 2697
    assert len(rows)==report['metrics']['episodes']==expected
    assert report['split']==split and report['checkpoint_sha256']==BASE_SHA
    assert report['calibrated_residual_sha256']==RESIDUAL_SHA
    assert report['args']['stage1_visual_control']==control
    assert report['args']['stage1_history_control']=='checkpoint'
    assert report['stage1_inference_controls']['grid_size']==28
    assert report['stage1_inference_controls']['stop_threshold']==.3
    assert report['protocol']['horizon']==20 and report['protocol']['success_radius_m']==20
    assert report['protocol']['horizontal_step_m']==50 and report['protocol']['vertical_step_m']==10
    assert report['protocol']['min_altitude_m']==10 and report['protocol']['max_altitude_m']==100
    assert not report['inference_target_access']
    assert [r['index'] for r in rows]==list(range(expected))
    assert len({tuple(r['episode_id']) for r in rows})==expected
    for row in rows:assert row['map_name']==row['episode_id'][0]
    assert file_sha256(directory/'episodes.json')==report['episode_manifest_sha256']
    recomputed=aggregate_rows(rows)
    for key in ('SR','SPL','OSR','NE','False_Stop_Rate','false_stop_episode_fraction'):
        if recomputed[key] is None:assert report['metrics'][key] is None
        else:assert abs(recomputed[key]-report['metrics'][key])<1e-9,(split,control,key)
    provenance=dict(directory=str(directory),metrics_sha256=file_sha256(directory/'metrics.json'),
                    predictions_sha256=file_sha256(directory/'predictions.jsonl'),
                    episode_manifest_sha256=report['episode_manifest_sha256'],
                    evaluator_sha256=report['evaluator_sha256'],commit=report['graphnav_commit'],
                    implementation_sha256=report['implementation_sha256'],input_sha256=report['input_sha256'])
    return report,rows,provenance


def behavior_equality(a,b):
    pairs=pair_ids(a,b);differences=[];max_pose=0.;max_probability=0.
    for x,y in pairs:
        changed=[]
        px,py=np.asarray(x['poses']),np.asarray(y['poses'])
        if px.shape!=py.shape:changed.append('pose_shape')
        else:
            value=float(np.max(np.abs(px-py)));max_pose=max(max_pose,value)
            if value!=0:changed.append('poses')
        if x['termination']!=y['termination']:changed.append('termination')
        for k in ('SR','SPL','OSR','NE','steps','path_length','policy_stop'):
            if abs(x['metrics'][k]-y['metrics'][k])>1e-10:changed.append(k)
        dx,dy=x['step_diagnostics'],y['step_diagnostics']
        if len(dx)!=len(dy):changed.append('decision_count')
        else:
            for q,r in zip(dx,dy):
                max_probability=max(max_probability,abs(q['stop_probability']-r['stop_probability']))
                if q['stop_probability']!=r['stop_probability']:changed.append('stop_probability')
                for k in ('selected_goal_xy','belief_top16_world_xy'):
                    if not np.array_equal(q[k],r[k]):changed.append(k)
        if changed:differences.append(dict(episode_id=x['episode_id'],map_name=x['map_name'],fields=sorted(set(changed))))
    return dict(episodes=len(pairs),identical_behavior=len(differences)==0,differing_episodes=len(differences),
                max_pose_coordinate_difference_m=max_pose,max_stop_probability_difference=max_probability,
                differences=differences,excluded='Wall-clock/inference latency and appearance-only diagnostics are not behavior equality criteria.')


def public_input_audit():
    forbidden={'target_position','target_world','target_xy','teacher_world','labels','future','trajectory'}
    checks={}
    for label,function in [('residual_forward',CalibratedVisualResidual.forward),('stage1_forward',GraphNavStage1.forward),('public_batch_filter',_batch_inputs)]:
        source=inspect.getsource(function)
        import textwrap
        tree=ast.parse(textwrap.dedent(source))
        keys={node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
        args=set(inspect.signature(function).parameters)
        bad=forbidden&(keys|args)
        assert not bad,(label,bad)
        checks[label]=dict(forbidden_fields_found=[],source_sha256=hashlib.sha256(source.encode()).hexdigest())
    names=[f.name for f in fields(PublicEpisode)]
    assert not forbidden&set(names)
    checks['public_episode_fields']=names
    checks['manual_path_audit']=[
        'Evaluator builds PublicEpisode(map_name, instruction, start_pose, reference polygons/names); policy does not receive the label-bearing dataset record.',
        'Real target and human future trajectory enter score_trace only after rollout. The public packing target_xy=[0,0] scaffold is omitted by _batch_inputs.',
        'Residual forward accepts only current/past ROI features, validity/quality, text, pose, public bounds, static logits/context and grid geometry.',
        'Native observation builder supplies the actual current footprint. CandidateMemory accepts strictly increasing observed steps; unobserved pixels are not a negative/future gallery.',
        'GT is used here only for post-hoc metrics/case descriptions, never supplied to the inference networks.'
    ]
    return checks


def csv_write(path,rows):
    assert rows
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def fmt_ci(value):return f'[{value[0]:+.3f}, {value[1]:+.3f}]'


def main():
    OUT.mkdir(exist_ok=True)
    freeze=json.loads((OUT/'frozen_validation.json').read_text())
    assert file_sha256(ROOT/'calibrated_residual/best.pt')==RESIDUAL_SHA
    assert file_sha256(PROJECT/'GraphNav/runs/graphnav_v1_stage1/best.pt')==BASE_SHA
    import torch
    trained=torch.load(ROOT/'calibrated_residual/best.pt',map_location='cpu',weights_only=True)
    assert trained['kind']=='calibrated-visual-residual-v2' and trained['base_sha']==BASE_SHA
    assert not trained['config']['smoke']
    training_provenance=dict(selected_epoch=trained['epoch'],config=trained['config'],
                             cache_manifest_sha256=trained['config']['cache_manifest_sha'],
                             completion=json.loads((ROOT/'calibrated_residual/complete.json').read_text()),
                             effective_model=dict(context_dim=128,vision_dim=1024,hidden=64,bound=2.,centered=True,use_pose=True))
    cache_manifest=PROJECT/'GraphNav/data/graphnav_stage1_frames/manifest.json'
    assert file_sha256(cache_manifest)==training_provenance['cache_manifest_sha256']
    cache_protocol=json.loads(cache_manifest.read_text())['protocol']
    assert cache_protocol['split']=='train_seen' and cache_protocol['causal_observations']
    training_provenance['training_cache_split']=cache_protocol['split']
    training_provenance['training_source_sha256']=cache_protocol['source_sha256']
    del trained
    data={};metrics={};bootstrap={};map_rows=[];paired_rows=[];examples=[];equality={};rgb_sensitivity={};versions={}
    for split in ('val_seen','val_unseen'):
        data[split]={};metrics[split]={};versions[split]={};reports={}
        for control in CONTROLS:
            report,rows,provenance=load_control(split,control)
            data[split][control]=rows;reports[control]=report;versions[split][control]=provenance
            metrics[split][control]=dict(episode_weighted=aggregate_rows(rows),map_equal=map_equal_metrics(rows))
        assert len({reports[c]['episode_manifest_sha256'] for c in CONTROLS})==1
        protocols=[json.dumps(reports[c]['protocol'],sort_keys=True) for c in CONTROLS]
        assert len(set(protocols))==1
        assert len({json.dumps(reports[c]['input_sha256'],sort_keys=True) for c in CONTROLS})==1
        equality[split]=behavior_equality(data[split]['no-belief'],data[split]['zero'])
        rgb_sensitivity[split]={c:behavior_equality(data[split]['real'],data[split][c]) for c in ('zero','shuffle')}
        maps=sorted({r['map_name'] for r in data[split]['real']})
        for name in maps:
            ref=[r for r in data[split]['real'] if r['map_name']==name];rm=aggregate_rows(ref)
            for control in CONTROLS:
                subset=[r for r in data[split][control] if r['map_name']==name];m=aggregate_rows(subset)
                assert len(subset)==len(ref)
                map_rows.append(dict(split=split,map_name=name,control=control,episodes=len(subset),
                                     episode_weight=len(subset)/len(data[split]['real']),map_equal_weight=1/len(maps),
                                     SR_percent=m['SR']*100,SPL_percent=m['SPL']*100,OSR_percent=m['OSR']*100,NE_m=m['NE'],
                                     false_stop_rate_percent=m['False_Stop_Rate']*100 if m['False_Stop_Rate'] is not None else None,
                                     real_minus_control_SR_pp=(rm['SR']-m['SR'])*100,real_minus_control_SPL_pp=(rm['SPL']-m['SPL'])*100))
        bootstrap[split]={}
        for control in ('no-belief','zero','shuffle'):
            pairs=pair_ids(data[split]['real'],data[split][control])
            diff=np.asarray([[a['metrics'][k]-b['metrics'][k] for k in ('SR','SPL')] for a,b in pairs])
            boot=paired_bootstrap(diff,[a['map_name'] for a,b in pairs])
            boot['effects']={key:value for key,value in zip(('SR','SPL'),boot.pop('statistics'))}
            for j,key in enumerate(('SR','SPL')):
                values=diff[:,j];tol=0 if key=='SR' else 1e-12
                boot['effects'][key]['episode_outcomes']=dict(improved=int((values>tol).sum()),degraded=int((values<-tol).sum()),unchanged=int((np.abs(values)<=tol).sum()))
            bootstrap[split][control]=boot
            for a,b in pairs:
                delta=a['metrics']['SR']-b['metrics']['SR']
                paired_rows.append(dict(split=split,map_name=a['map_name'],episode_id=json.dumps(a['episode_id']),index=a['index'],control=control,
                                        real_SR=a['metrics']['SR'],control_SR=b['metrics']['SR'],delta_SR=delta,
                                        real_SPL=a['metrics']['SPL'],control_SPL=b['metrics']['SPL'],delta_SPL=a['metrics']['SPL']-b['metrics']['SPL'],
                                        real_NE_m=a['metrics']['NE'],control_NE_m=b['metrics']['NE'],
                                        outcome='improved' if delta>0 else 'degraded' if delta<0 else 'unchanged',
                                        real_termination=a['termination'],control_termination=b['termination']))
            if control=='no-belief':
                for sign,label in [(1,'improved'),(-1,'degraded')]:
                    candidates=[(a,b) for a,b in pairs if (a['metrics']['SR']-b['metrics']['SR'])*sign>0]
                    candidates.sort(key=lambda p:(-abs(p[0]['metrics']['NE']-p[1]['metrics']['NE']),tuple(p[0]['episode_id'])))
                    chosen=[];used=set()
                    for a,b in candidates:
                        if a['map_name'] in used:continue
                        chosen.append((a,b));used.add(a['map_name'])
                        if len(chosen)==2:break
                    for a,b in chosen:
                        examples.append(dict(split=split,outcome=label,episode_id=a['episode_id'],index=a['index'],map_name=a['map_name'],
                                             real={k:a[k] for k in ('metrics','termination')},no_belief={k:b[k] for k in ('metrics','termination')},
                                             start_pose=a['poses'][0],real_final_pose=a['poses'][-1],no_belief_final_pose=b['poses'][-1],
                                             real_final_predicted_goal=a['step_diagnostics'][-1]['selected_goal_xy'],
                                             no_belief_final_predicted_goal=b['step_diagnostics'][-1]['selected_goal_xy'],
                                             real_max_abs_residual=max(x['residual_max_abs'] for x in a['step_diagnostics'])))
    primary=bootstrap['val_unseen']['no-belief']['effects']['SR']
    accepted=primary['episode_weighted_delta_pp']>0 and primary['map_cluster_episode_weighted_95ci_pp'][0]>0
    inputs=public_input_audit()
    output=dict(configuration=freeze,training_provenance=training_provenance,units='SR/SPL/OSR/False_Stop_Rate are fractions in metrics.json; NE meters. CSV display rates percent; bootstrap effects percentage points.',
                metrics=metrics,map_counts={s:len({r['map_name'] for r in data[s]['real']}) for s in data},
                zero_no_belief_behavior=equality,real_behavior_sensitivity=rgb_sensitivity,
                public_input_audit=inputs,versions=versions,typical_episodes=examples,
                acceptance=dict(reference='Real minus No-belief, full Val-Unseen',positive_point_and_positive_map_cluster_lower=accepted,
                                criterion='episode-weighted Unseen delta_SR > 0 and paired map-cluster bootstrap 95% CI lower > 0',
                                no_retraining_or_unseen_selection=True))
    atomic_json(output,OUT/'metrics.json')
    atomic_json(dict(replicates=REPLICATES,episode_seed=SEED,map_seed=SEED+1,
                     pairing='Within each split, identical episode_id and map_name; Seen and Unseen are not paired with each other.',
                     sampling='Resample whole maps M times with replacement. Retain all episodes in each sampled map. Ratio-of-sums for episode weighting; mean of map means for equal-map weighting.',
                     multiple_comparisons='Pointwise 95% percentile intervals, not multiplicity-adjusted; primary predeclared comparison Real vs No-belief on Unseen.',
                     results=bootstrap),OUT/'bootstrap_results.json')
    csv_write(OUT/'paired_results.csv',paired_rows);csv_write(OUT/'map_results.csv',map_rows)
    atomic_json(inputs,OUT/'public_input_audit.json')
    build_report(output,bootstrap,map_rows,examples)
    source=Path(__file__);(OUT/'analysis_source_snapshot.py').write_bytes(source.read_bytes())
    atomic_json(dict(source_sha256=file_sha256(source),replicates=REPLICATES,seed=SEED,finished=time.time()),OUT/'analysis_version.json')
    print('Validation report:',ROOT/'VISUAL_RESIDUAL_UNSEEN_VALIDATION.md',flush=True)


def build_report(output,bootstrap,map_rows,examples):
    counts=output['map_counts'];lines=['# Calibrated Visual Residual：Val-Unseen 泛化与地图级统计验证','',
        '本报告只分析既有 Calibrated Residual，不重新训练、不重新选模型或阈值，不包含384×384实验。','',
        '## 配置与复现','',f'- 原 GraphNav-V1 C Epoch1 SHA256：`{BASE_SHA}`。',f'- Calibrated Residual SHA256：`{RESIDUAL_SHA}`。',
        '- 目标网格28×28；校正Stop分数阈值0.3；原1毫米预测目标到达门控与高度门控保持不变；原引用地标输入。',
        '- 动作预算20次决策，水平每步≤50m、垂直≤10m；SR按最终XY距离≤20m，停止/超时/停滞均依相同最终位置计分。',
        '- 原几何、视觉编码器、匹配器、历史模块、轨迹/Stop及新残差头全部冻结；Unseen启动时逐参数检查requires_grad=False。',
        '- 残差结构：视觉/文本1024→32投影、静态上下文128→16投影、106→64→2 MLP；中心化修正×置信度，有界幅度2.0，保留原use_pose=True。原训练3轮，固定复用Train Dev选出的第1轮best.pt。',
        '- 训练仅使用原Train缓存及Train Dev整组划分选出的权重。Seen四组完整2470条原样复用，Unseen每组完整2697条。',
        '- 四组均运行同一残差适配器：Real真实当前/历史特征；No-belief将残差置零；Zero将当前与历史视觉特征置零；Shuffle按固定种子0在每个有效尺度内打乱视觉向量与候选位置的对应，位置/有效性/质量掩码不变。',
        '- 旧evaluator仅允许Seen残差、且held-out只允许real；本次最小修复加入独立冻结manifest，约束固定权重、完整Unseen清单和四控制。原正式模型/Test检查未放宽。Seen算法和已有输出未改变。',
        '- 完整命令、PID、运行时间与返回码：`visual_residual_unseen_validation/queue.json`；日志为同目录各control.log。',
        '- 逐episode原始输出：`visual_evaluations/calibrated_residual_full_seen_{control}/`及`calibrated_residual_full_unseen_{control}/`。保留predictions.jsonl、poses、map_name、metrics、termination、episodes.json。',
        '- 版本与输入SHA：`launch_record.json`、`source_sha256.json`、`source_snapshot/`、`analysis_version.json`及本报告的metrics.json。',
        '- 复现：在本worktree使用Baselines/.reproduction-env/bin/python运行`-m scripts.validate_visual_residual_unseen`；完成项验证后跳过，未完成项使用--resume。独立统计重建：`-m scripts.analyze_visual_residual_unseen`。','',
        '## 四组完整结果（Episode加权）','',
        '|Split|地图簇数|控制|Episodes|SR%|SPL%|OSR%|NE(m)|False-Stop/Stop%|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for split in ('val_seen','val_unseen'):
        for c in CONTROLS:
            m=output['metrics'][split][c]['episode_weighted'];fs='NA' if m['False_Stop_Rate'] is None else f"{m['False_Stop_Rate']*100:.4f}"
            lines.append(f"|{split}|{counts[split]}|{c}|{m['episodes']}|{m['SR']*100:.4f}|{m['SPL']*100:.4f}|{m['OSR']*100:.4f}|{m['NE']:.4f}|{fs}|")
    lines+=['','False-Stop Rate = 失败且主动Stop的episode数 / 主动Stop的episode数；分母为0时NA。所有差值使用同一残差适配器中的No-belief对照，不替换成原正式完整模型的指标。','',
            '## 地图等权结果','', '|Split|控制|地图数|SR%|SPL%|OSR%|NE(m)|平均地图False-Stop/Stop%|','|---|---|---:|---:|---:|---:|---:|---:|']
    for split in ('val_seen','val_unseen'):
        for c in CONTROLS:
            m=output['metrics'][split][c]['map_equal'];fs='NA' if m['False_Stop_Rate'] is None else f"{m['False_Stop_Rate']*100:.4f}"
            lines.append(f"|{split}|{c}|{m['maps']}|{m['SR']*100:.4f}|{m['SPL']*100:.4f}|{m['OSR']*100:.4f}|{m['NE']:.4f}|{fs}|")
    lines+=['','地图等权对各地图指标取算术平均，地图大小不影响权重；False-Stop率只平均具有主动Stop的地图，maps_with_policy_stop见metrics.json。','',
            '## 配对效应与置信区间','',
            f'每个split内部按episode_id严格配对，校验map_name、索引、清单及权重SHA；Seen与Unseen之间不配对。每项bootstrap {REPLICATES:,}次，episode随机种子{SEED}，地图种子{SEED+1}。区间为点态percentile 95% CI，未作多重比较校正，预先声明的主要比较为Unseen Real−No-belief。',
            '', 'Episode bootstrap仅作为诊断，假设episode独立过强；主要泛化判断采用整地图重采样。每次抽取M张地图（有放回），地图内保留全部episode，重复抽中的地图整体重复。Episode加权使用抽样地图效应总和/episode总数；地图等权使用抽样地图均值的均值。']
    for metric in ('SR','SPL'):
        lines+=['',f'### Δ{metric}（Real−控制，单位pp）','',
                '|Split|控制|Episode加权Δ|Episode bootstrap CI|地图cluster CI（Episode加权）|地图等权Δ|地图cluster CI（地图等权）|','|---|---|---:|---|---|---:|---|']
        for split in ('val_seen','val_unseen'):
            for c in ('no-belief','zero','shuffle'):
                d=bootstrap[split][c]['effects'][metric]
                lines.append(f"|{split}|{c}|{d['episode_weighted_delta_pp']:+.4f}|{fmt_ci(d['episode_paired_95ci_pp'])}|{fmt_ci(d['map_cluster_episode_weighted_95ci_pp'])}|{d['map_equal_delta_pp']:+.4f}|{fmt_ci(d['map_cluster_equal_weighted_95ci_pp'])}|")
    lines+=['','### 改善/退化/不变的episode数量','',
            '|Split|控制|SR改善|SR退化|SR不变|SPL改善|SPL退化|SPL不变|','|---|---|---:|---:|---:|---:|---:|---:|']
    for split in ('val_seen','val_unseen'):
        for c in ('no-belief','zero','shuffle'):
            a=bootstrap[split][c]['effects']['SR']['episode_outcomes'];b=bootstrap[split][c]['effects']['SPL']['episode_outcomes']
            lines.append(f"|{split}|{c}|{a['improved']}|{a['degraded']}|{a['unchanged']}|{b['improved']}|{b['degraded']}|{b['unchanged']}|")
    lines+=['','## 逐地图统计','',
            '|Split|地图|Episodes|Real SR%|No-belief SR%|Zero SR%|Shuffle SR%|ΔSR vs No-belief pp|ΔSR vs Zero pp|ΔSR vs Shuffle pp|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('val_seen','val_unseen'):
        for name in sorted({r['map_name'] for r in map_rows if r['split']==split}):
            d={r['control']:r for r in map_rows if r['split']==split and r['map_name']==name}
            lines.append(f"|{split}|{name}|{d['real']['episodes']}|{d['real']['SR_percent']:.3f}|{d['no-belief']['SR_percent']:.3f}|{d['zero']['SR_percent']:.3f}|{d['shuffle']['SR_percent']:.3f}|{d['no-belief']['real_minus_control_SR_pp']:+.3f}|{d['zero']['real_minus_control_SR_pp']:+.3f}|{d['shuffle']['real_minus_control_SR_pp']:+.3f}|")
    lines+=['',f"Val-Seen有{counts['val_seen']}个地图簇；Val-Unseen只有{counts['val_unseen']}个地图簇（两个城市中的地图块）。这是重采样单元数，不是2697个独立泛化单元，也不保证同城地图彼此完全独立。Unseen地图数过少，cluster区间非常离散、对单张地图敏感，无法覆盖未观测地图分布的不确定性。4张地图每次抽4张，实际只有35种地图计数组合；增加bootstrap次数不会增加独立地图信息。",'',
            '## Zero / No-belief 行为验收','']
    for split,d in output['zero_no_belief_behavior'].items():
        lines.append(f"- {split}：{d['episodes']}条，行为差异{d['differing_episodes']}条，最大位姿坐标差{d['max_pose_coordinate_difference_m']:.9g}m，最大Stop分数差{d['max_stop_probability_difference']:.9g}。")
    lines+=['','比较全部逐episode位姿数组、终止原因、SR/SPL/OSR/NE/步数/路径长度，以及逐决策目标坐标、信念候选和Stop分数；不比较受运行负载影响的计时或仅外观诊断值。若出现差异，原样列于metrics.json，不修改预测以强制一致。Zero无有效非零视觉向量，因此修正为零；No-belief直接将修正置零，二者预期回退到同一静态信念。','',
            '## 改善与损害案例','',
            '从Real相对No-belief的成功翻转样本中，按|ΔNE|降序，每种方向优先选不同地图的最多两条；这是可复现的代表性诊断，不代替总体统计。','',
            '|Split|方向|Episode ID|Real SR/SPL%/NE(m)|No-belief SR/SPL%/NE(m)|Real/控制终止|','|---|---|---|---|---|---|']
    for e in examples:
        a,b=e['real']['metrics'],e['no_belief']['metrics']
        lines.append(f"|{e['split']}|{e['outcome']}|`{e['episode_id']}`|{a['SR']:.0f}/{a['SPL']*100:.2f}/{a['NE']:.2f}|{b['SR']:.0f}/{b['SPL']*100:.2f}/{b['NE']:.2f}|{e['real']['termination']}/{e['no_belief']['termination']}|")
    lines+=['','案例起点、终点、最终预测目标与最大残差值见metrics.json的typical_episodes；完整决策和轨迹均在对应predictions.jsonl，按episode_id/index定位。','',
            '## 私有信息隔离','',
            '已核查PublicEpisode字段、_batch_inputs白名单、Stage1.forward和残差forward参数/字典键：目标位置、目标标签、人类未来轨迹未进入推理网络；源代码检查摘要见public_input_audit.json。运行时所有网络参数冻结，残差加载检查base_sha。真实目标仅在rollout完成后的指标计算及本报告事后案例分析中使用。当前/历史视觉由实际观测与严格递增的历史记忆构建，没有未来图像库。','',
            '## 结论','']
    for split in ('val_seen','val_unseen'):
        a=bootstrap[split]['no-belief']['effects']['SR'];b=bootstrap[split]['no-belief']['effects']['SPL']
        lines.append(f"- {split}：Real相对No-belief，Episode加权ΔSR {a['episode_weighted_delta_pp']:+.4f}pp、ΔSPL {b['episode_weighted_delta_pp']:+.4f}pp；地图cluster 95% CI分别为{fmt_ci(a['map_cluster_episode_weighted_95ci_pp'])}pp、{fmt_ci(b['map_cluster_episode_weighted_95ci_pp'])}pp。地图等权结果及CI见上表。")
    sens=output['real_behavior_sensitivity']['val_unseen']
    shuffle=bootstrap['val_unseen']['shuffle']['effects']['SR']
    lines.append(f"- RGB与空间对应：Unseen Real相对Zero有{sens['zero']['differing_episodes']}条行为不同，相对Shuffle有{sens['shuffle']['differing_episodes']}条不同；Real−Shuffle ΔSR {shuffle['episode_weighted_delta_pp']:+.4f}pp，地图cluster CI {fmt_ci(shuffle['map_cluster_episode_weighted_95ci_pp'])}pp。这证明是否对视觉干预敏感；只有真实RGB优于打乱且区间支持正效应时，才支持其对应关系带来有用信息。不能仅凭分数变化断言完整语义理解。")
    if sens['zero']['differing_episodes'] and sens['shuffle']['differing_episodes']:
        lines.append('- 行为层面的RGB利用：是，非零视觉向量及其地图对应会改变闭环行为，不能解释成完全忽略RGB的静态旁路；这不等价于Unseen任务收益，也不单独证明目标外观语义理解。')
    else:
        lines.append('- 行为层面的RGB利用证据有限：至少一种视觉干预未改变闭环行为，不能仅凭模块存在就断言RGB被有效使用。')
    if shuffle['episode_weighted_delta_pp']>0 and shuffle['map_cluster_episode_weighted_95ci_pp'][0]>0:
        lines.append('- 真实视觉对应优于打乱：Unseen ΔSR及其地图cluster区间支持正效应；仍需与关闭残差对照分别判断，打乱更差不等于优于静态基线。')
    else:
        lines.append('- 真实视觉对应优于打乱：Unseen尚无地图cluster区间下界>0的正收益支持，不能把干预敏感性直接解释成有效视觉泛化。')
    unseen=bootstrap['val_unseen']['no-belief']['effects']
    signs={k:('提高' if unseen[k]['episode_weighted_delta_pp']>0 else '降低' if unseen[k]['episode_weighted_delta_pp']<0 else '不变') for k in ('SR','SPL')}
    lines.append(f"- Unseen任务点估计：相对No-belief，SR{signs['SR']}、SPL{signs['SPL']}；是否具有地图级统计支持按上述cluster区间判断。")
    if output['acceptance']['positive_point_and_positive_map_cluster_lower']:
        lines.append('- 按预先规定门槛，Unseen Real−No-belief ΔSR>0且Episode加权地图cluster CI下界>0，可报告本次四地图范围内有地图级正收益支持；必须同时披露地图等权区间与仅4地图的限制，不外推为普遍跨城市泛化。')
    else:
        lines.append('- 未满足“Unseen Real−No-belief ΔSR>0且地图cluster 95% CI下界>0”的验收门槛，不能报告“跨地图视觉残差有明确正收益”。即使episode bootstrap显著或点估计为正，也不能替代地图级支持。')
    lines+=['- 所有负结果、跨零区间、帮助与损害样本原样保留，没有据此重选权重、训练或调整Stop阈值。','',
            '## 输出文件','',
            '- metrics.json：完整配置、逐split/控制的两种加权指标、SHA/版本/输入检查、Zero一致性及案例。',
            '- paired_results.csv：Real相对每个控制、逐episode配对指标与结局。',
            '- map_results.csv：每张地图、每控制的指标、权重及Real差值。',
            '- bootstrap_results.json：20000次episode与地图聚类bootstrap、两种地图加权结果及区间。','']
    text='\n'.join(lines)
    (OUT/'VISUAL_RESIDUAL_UNSEEN_VALIDATION.md').write_text(text)
    (ROOT/'VISUAL_RESIDUAL_UNSEEN_VALIDATION.md').write_text(text)


if __name__=='__main__':main()
