"""Deliver frozen Stage1 checkpoints, paired metrics, plots and a Chinese report."""
import argparse
import csv
import json
from pathlib import Path
from graphnav.workspace_paths import PROJECT
import shutil
import sys
import time
import numpy as np
import torch
from graphnav.checkpointing import atomic_json, file_sha256, provenance
from scripts.summarize_closedloop import read_run, paired_stats


def table(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/graphnav_v1_stage1')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    audit_path = root/'audit_status.json'
    while audit_path.exists() and json.loads(audit_path.read_text())['status'] not in ('complete', 'failed'):
        time.sleep(5)
    if not audit_path.exists() or json.loads(audit_path.read_text())['status'] != 'complete':
        raise ValueError('supplementary Stage A/B and Seen128 parity audits must complete')
    freeze = json.loads((root/'frozen_final_candidate.json').read_text())
    if freeze['checkpoint_sha256'] != file_sha256(root/'best.pt'):
        raise ValueError('final frozen checkpoint changed')
    histories = {stage: json.loads((root/f'stages/{stage}/history.json').read_text()) for stage in ('A', 'B', 'C')}
    expected = dict(A=3, B=3, C=5)
    for stage, history in histories.items():
        if [row['epoch'] for row in history] != list(range(1, expected[stage]+1)):
            raise ValueError('complete planned epoch history required')
        if any(row['train']['samples'] != 83140 or row['dev']['samples'] != 4372 for row in history):
            raise ValueError('partial epoch cannot satisfy Stage1 delivery')
    state = torch.load(root/'best.pt', map_location='cpu', weights_only=True)
    import yaml
    configuration = yaml.safe_load((root/f"stages/{freeze['selected_stage']}/config.yaml").read_text())
    configuration['complete_architecture'] = dict(grid_size=28, candidate_context_hidden=state['a3_args']['hidden'],
        top_k=state['a3_args']['top_k'], vision_feature_dimension=1024, appearance_adapter_hidden=64,
        observed_roi_extents_m=[32,64,128], history_gate='Linear(6,1); quality-weighted causal candidate memory',
        trajectory_modes=3, trajectory_waypoints=8, residual_bound=.5, active_observation=False)
    configuration['stage_epochs'] = expected
    configuration['stage_C_learning_rates'] = dict(a3=1e-5,visual=1e-4,history=1e-4,trajectory_stop=2e-5)
    (root/'config.yaml').write_text(yaml.safe_dump(configuration,allow_unicode=True,sort_keys=False))
    rows = []
    for stage, history in histories.items():
        for record in history:
            rows.append(dict(stage=stage, epoch=record['epoch'], **{split+'_'+k: v
                for split in ('train', 'dev', 'seen128') for k, v in record[split].items() if isinstance(v, (int, float))}))
    table(root/'training.csv', rows)
    names = ['baseline_seen128', 'initial_seen128', 'selected_seen128', 'zero_rgb_seen128', 'shuffle_rgb_seen128', 'no_belief_seen128', 'current_only_seen128',
        'stage_A_real_seen128', 'stage_A_zero_seen128', 'stage_A_shuffle_seen128', 'stage_A_no-belief_seen128', 'stage_B_current_seen128', 'stage_B_history_seen128',
        'baseline_full_seen', 'full_seen', 'baseline_full_unseen', 'full_unseen']
    runs = {name: read_run(root/name) for name in names}
    full_rows = []
    visual_rows = []
    history_rows = []
    paired = {}
    for name, (report, predictions, digest) in runs.items():
        row = dict(name=name, split=report['split'], **{k:v for k,v in report['metrics'].items() if isinstance(v,(int,float))},
            seconds=report['seconds'], gpu_peak_allocated_gib=report['gpu_peak_allocated_gib'], checkpoint_sha256=report['checkpoint_sha256'])
        if 'seen128' in name:
            if digest != runs['baseline_seen128'][2] or report['protocol'] != runs['baseline_seen128'][0]['protocol']:
                raise ValueError('Seen128 comparison protocol or episodes differ: '+name)
            paired[name] = paired_stats(runs['baseline_seen128'][1], predictions)
            full_rows.append(row)
        else:
            full_rows.append(row)
        if name in ('selected_seen128', 'zero_rgb_seen128', 'shuffle_rgb_seen128', 'no_belief_seen128') or name.startswith('stage_A_'):
            visual_rows.append(row)
        if name in ('selected_seen128','current_only_seen128') or name.startswith('stage_B_'):
            history_rows.append(row)
    for split in ('seen','unseen'):
        baseline, selected = runs['baseline_full_'+split], runs['full_'+split]
        if baseline[2] != selected[2] or baseline[0]['protocol'] != selected[0]['protocol']:
            raise ValueError('full baseline/selected protocol differs: '+split)
        paired['full_'+split] = paired_stats(baseline[1], selected[1])
    causal_pairs = {
        'A_real_minus_zero': ('stage_A_zero_seen128', 'stage_A_real_seen128'),
        'A_real_minus_shuffle': ('stage_A_shuffle_seen128', 'stage_A_real_seen128'),
        'A_real_minus_no_belief': ('stage_A_no-belief_seen128', 'stage_A_real_seen128'),
        'B_history_minus_current': ('stage_B_current_seen128', 'stage_B_history_seen128'),
        'selected_real_minus_zero': ('zero_rgb_seen128', 'selected_seen128'),
        'selected_real_minus_shuffle': ('shuffle_rgb_seen128', 'selected_seen128'),
        'selected_real_minus_no_belief': ('no_belief_seen128', 'selected_seen128'),
        'selected_history_minus_current': ('current_only_seen128', 'selected_seen128'),
    }
    causal = {name: paired_stats(runs[base][1], runs[candidate][1]) for name,(base,candidate) in causal_pairs.items()}
    epoch_comparisons = [row for row in full_rows if 'seen128' in row['name']]
    for stage, history in histories.items():
        for record in history:
            epoch_comparisons.append(dict(name=f"{stage}_epoch{record['epoch']:02d}", **record['seen128']))
    table(root/'seen128_comparison.csv', epoch_comparisons)
    table(root/'full_comparison.csv', [row for row in full_rows if 'full_' in row['name']])
    table(root/'visual_ablation.csv', visual_rows)
    table(root/'history_ablation.csv', history_rows)
    table(root/'goal_retrieval.csv', [dict(name=name, **report['goal_retrieval']) for name,(report,_,_) in runs.items() if 'goal_retrieval' in report])
    # Read saved outputs only: action and history audits never enter inference.
    sys.path.insert(0, runs['full_seen'][0]['args']['sbf_source'])
    from sbfnav.mapdata import GROUND_LEVEL
    map_bounds = {}
    for split in ('seen','unseen'):
        for line in (PROJECT/f'Ablations/GraphNav/2027-CVPR-full-token-rgb/data/revised_static/graphnav_val_{split}.jsonl').read_text().splitlines():
            record = json.loads(line)
            map_bounds[record['map_name']] = np.asarray(record['world_bounds'])
    action_audits = {}
    for name in ('full_seen','full_unseen'):
        predictions = runs[name][1]
        steps = np.concatenate([np.diff(np.asarray(row['poses']),axis=0) for row in predictions.values()])
        diagnostics = [d for row in predictions.values() for d in row['step_diagnostics']]
        invalid_starts = 0; altitude_violations = 0; boundary_violations = 0; illegal_ends = 0
        for row in predictions.values():
            poses = np.asarray(row['poses']); height = poses[:,2]-GROUND_LEVEL[row['map_name']]
            invalid_starts += int(height[0]<10.-.001 or height[0]>100.+.001)
            # Preserve source starts; inherited violations may only approach the legal range.
            altitude_violations += int(((height[1:]<np.minimum(10.,height[:-1])-.001)|
                                       (height[1:]>np.maximum(100.,height[:-1])+.001)).sum())
            bounds = map_bounds[row['map_name']]
            boundary_violations += int(((poses[1:,:2]<bounds[:2]-.001)|(poses[1:,:2]>bounds[2:]+.001)).any(-1).sum())
            illegal_ends += int(height[-1]<10.-.001 or height[-1]>100.+.001)
        action_audits[name] = dict(episodes=len(predictions), executed_movements=len(steps),
            horizontal_over_50m=int((np.linalg.norm(steps[:,:2],axis=-1)>50.001).sum()),
            vertical_over_10m=int((np.abs(steps[:,2])>10.001).sum()),
            horizon_over_20=int(sum(len(row['poses'])-1>20 for row in predictions.values())),
            inherited_invalid_altitude_starts=invalid_starts,new_or_worsened_altitude_violations=altitude_violations,
            executed_xy_boundary_violations=boundary_violations,illegal_final_altitudes=illegal_ends,
            residual_max_abs=max(d['residual_max_abs'] for d in diagnostics),
            active_observation_actions=sum(d.get('active_observation', False) for d in diagnostics),
            history_note='Past observations only; memory updates after prediction. Expert prefixes are the training distribution; no on-policy augmentation was used.')
    parameters = dict(state['config']['parameter_counts'])
    full_report = runs['full_seen'][0]
    parameters.update(frozen_bert=full_report['bert_backbone_parameters'], frozen_siglip2=full_report['visual_backbone_parameters'],
        trainable_at_selected_training_stage=state['config']['actual_trainable_parameters'], inference_trainable=0)
    parameters['complete_system'] = sum(state['config']['parameter_counts'].values()) + 1 + parameters['frozen_bert'] + parameters['frozen_siglip2']
    manifest = dict(provenance=provenance(), frozen_selection=freeze, sources=json.loads((root/'initial_sources.json').read_text()),
        checkpoints={str(path.relative_to(root)):dict(sha256=file_sha256(path),bytes=path.stat().st_size) for path in
            [root/'best.pt',root/'last.pt',root/'initial_complete.pt']+[root/f'stages/{s}/{k}.pt' for s in histories for k in ('best','last')]},
        inputs=state['input_fingerprint'], protocol=state['protocol'], parameters=parameters, seed=state['args']['seed'],
        all_stages_completed_epochs=expected, output_sha256={name+'/metrics.json':file_sha256(root/name/'metrics.json') for name in names},
        note='Best is selected complete B/C by Seen128 SR then SPL then Train Dev Recall; Last is final C epoch with optimizer/RNG. All stage checkpoints retained. Frozen backbones are externally resolved by model name and immutable revision.')
    atomic_json(manifest,root/'checkpoint_manifest.json')
    summary = dict(provenance=provenance(), parameters=parameters, paired_vs_B0=paired, paired_causal_controls=causal, action_audits=action_audits,
        initial_parity=json.loads((root/'initial_seen128_parity.json').read_text()), historical_baseline_audit=json.loads((root/'historical_baseline_audit.json').read_text()), selected_stage=freeze['selected_stage'], selected_epoch=state['epoch'],
        performance_improved_on_full_seen=runs['full_seen'][0]['metrics']['SR']>runs['baseline_full_seen'][0]['metrics']['SR'],
        training_seconds=sum(record['train']['seconds'] for history in histories.values() for record in history),
        measurement_note='Training time sums successfully checkpointed full epochs; interrupted partial work and feature preparation are separate costs. Supplementary controls overlapped training. Final rollouts ran serially. No isolated speedup claim.')
    atomic_json(summary, root/'analysis.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    xs=np.arange(1,len(rows)+1)
    for key in ('goal','visual','belief','rank','trajectory','stop'):
        axes[0,0].plot(xs,[row['train_'+key] for row in rows],label=key)
    axes[0,0].set_title('Full-epoch component losses');axes[0,0].legend()
    for key in ('static_R@1/20m','belief_R@1/20m'):
        axes[0,1].plot(xs,[row['dev_'+key] for row in rows],label=key)
    axes[0,1].set_title('Internal Train Dev Recall');axes[0,1].legend()
    for key in ('SR','SPL','OSR'):
        axes[1,0].plot(xs,[row['seen128_'+key] for row in rows],label=key)
    axes[1,0].set_title('Seen128 closed-loop fractions');axes[1,0].legend()
    axes[1,1].plot(xs,[row['train_seconds'] for row in rows],label='train seconds')
    axes[1,1].set_title('Complete training epoch time');axes[1,1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.set_xlabel('Stage A1-3, B1-3, C1-5')
    fig.tight_layout();fig.savefig(root/'training_curves.png',dpi=160);plt.close(fig)
    sbf_root=(PROJECT/'Ablations/GraphNav/2027-CVPR-closedloop-sbf-parity/runs/closedloop_baseline')
    sbf={split:json.loads((sbf_root/f'sbf_val_{split}_full/metrics.json').read_text()) for split in ('seen','unseen')}
    def navigation_row(name,report):
        m=report['metrics']
        return '|'+name+'|'+ '|'.join(f'{100*m[k]:.4f}' for k in ('SR','OSR','SPL'))+f"|{m['NE']:.4f}|"+'|'.join(f'{100*m[k]:.4f}' for k in ('nDTW','sDTW'))+'|'+('NA' if m.get('False_Stop_Rate') is None else f"{100*m['False_Stop_Rate']:.4f}")+'|'
    lines=['# GraphNav-V1 第一阶段完整训练报告','',f"完整执行 A3轮、B3轮、C5轮；最终完整模型选择阶段 {freeze['selected_stage']} 第 {state['epoch']} 轮，选择仅使用 Seen128 SR、SPL 及内部 Train Dev。所有模型和推理参数在 Unseen 前冻结。",'',
        '## 初始化及强基线','',
        'A3 Seed1 Epoch14、原 Trajectory/Stop、原控制器参考 A3 三份 SHA 与任务要求一致。初始完整模型固定 Seen128 的轨迹、候选、模式、Stop 概率与 static-transfer 逐项核验；结果见 initial_seen128_parity.json。原基线和阶段权重独立保留。',
        '历史报告 SPL47.4130% 为四位小数显示，本次独立 B0 Seen128 为47.4129893%；该微小数值差异不隐去。与历史E1 Seed1独立重评的128条轨迹完全一致，候选、模式、Stop和协议也一致（historical_baseline_audit.json）。完整模型以本次保存的原始轨迹为精确重放参考。','',
        '## 全量导航指标','',
        '|模型/分割|SR%|OSR%|SPL%|NE m|nDTW%|sDTW%|False Stop%|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('seen','unseen'):
        lines += [navigation_row('SBF '+split,sbf[split]),navigation_row('B0 '+split,runs['baseline_full_'+split][0]),navigation_row('GraphNav-V1 '+split,runs['full_'+split][0])]
    lines += ['', 'False Stop 以 policy Stop 为分母；错误 Stop Episode 占比保留在 full_comparison.csv。SBF 没有学习 Stop，NA不能解释为零错误。Seen2470、Unseen2697；不将 Seen128 与全量基线直接比较。', '',
        '## 定位、视觉和 History 证据','',
        '静态唯一 map+instruction 查询指标分别保存 static_B0_seen/unseen.json、static_selected_seen/unseen.json；实际闭环的初始静态、第一次观测信念、最终观测信念 R@1/4/8/16 单独记录在 goal_retrieval.csv。最终观测时间随 Episode 不同。',
        '阶段 A 的 Real/Zero/Shuffle/No-Belief、阶段 B 的 Current/History、最终候选同类消融均使用固定 Seen128。visual_ablation.csv、history_ablation.csv、analysis.json 保留指标及配对 Episode/地图 Bootstrap。消融在模型固定后执行，没有根据结果再调参。Loss或梯度非零不构成视觉收益证据。',
        '训练使用实际 Train Seen 专家前缀的87512状态，内部查询组隔离83140训练/4372开发；不是87512个独立Episode。当前帧严格位于cutoff，历史严格早于当前帧；目标和专家轨迹只用于损失。无on-policy补充，训练与闭环状态分布差异仍存在。',
        'History按可见比例和有效支持质量加权，并通过小型质量门融合当前/历史；位置关联为28×28网格，重复视角通过相邻已接收位姿2m阈值去重。它不是完整长期回访去重模型。',
        '视觉使用实际384px观测的冻结SigLIP2上下文Patch池化32/64/128m区域；Depth用于可信区域/支持掩码，没有另行训练深度Backbone。所有区域共享当前帧Transformer上下文，不是独立重编码高分辨率ROI。困难负样本通过最高匹配分数选择，并排除可能覆盖GT的模糊区域；没有额外的逐物体颜色/类别真值，不能据此宣称完成同色同类物体识别基准。',
        '视觉残差绝对值上限0.5，未观测候选残差为0，无有效证据回退原静态场。没有主动观察、新GAT或关系辅助损失。', '',
        '目标Selector采用更新后信念Top-K中的第一名；Guarded-Blend关闭多候选探索，覆盖控制器联合候选选择，因此训练的rank分支不参与执行目标选择。所选目标的轨迹模式和Stop由预训练网络预测，C阶段联合微调；不能将本次结果解释为学习多候选Selector的收益。', '',
        '## 训练、参数和效率','',f"完成轮次训练时间合计 {summary['training_seconds']:.2f}s（只计成功保存完整轮次；中断部分另计）。各轮吞吐、峰值显存、模块Loss和排名提升诊断见training.csv/history.json。单步延迟按冻结评测器的policy预测计时，包含原生视觉编码及模型推理，不包含prediction之前的观测构建/渲染；全量总时间另列。",'',
        '|模块|参数量|','|---|---:|']
    lines += [f'|{key}|{value}|' for key,value in parameters.items()]
    lines += ['', 'A/B冻结A3和控制器；C学习率A3=1e-5、Visual/History=1e-4、Trajectory/Stop=2e-5，5%Warmup、余弦衰减、梯度裁剪1。BERT/SigLIP2全程冻结；原A3几何模式不使用的视觉/GNN分支也保持冻结。参数组和损失权重见config.yaml。', '',
        '|评估|时间s|单步ms|峰值分配GiB|','|---|---:|---:|---:|']
    for name in ('baseline_full_seen','full_seen','baseline_full_unseen','full_unseen'):
        report=runs[name][0];lines.append(f"|{name}|{report['seconds']:.2f}|{report['metrics']['inference_ms']:.3f}|{report['gpu_peak_allocated_gib']:.3f}|")
    localization_lines = ['', '|静态模型/分割|唯一查询数|R@1%|R@4%|R@8%|R@16%|NE m|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split in ('seen','unseen'):
        for model in ('B0','selected'):
            static = json.loads((root/f'static_{model}_{split}.json').read_text())
            values = static['metrics']
            localization_lines.append(f"|{model}/{split}|{static['samples']}|"+
                '|'.join(f"{100*values[f'R@{k}/20m']:.4f}" for k in (1,4,8,16))+f"|{values['NE_m']:.4f}|")
    localization_lines += ['', '|实际闭环观测时刻/分割|唯一查询数|R@1%|R@4%|R@8%|R@16%|NE m|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split in ('seen','unseen'):
        values = runs['full_'+split][0]['goal_retrieval']
        for moment in ('initial_static','dynamic_first','dynamic_final'):
            localization_lines.append(f"|{moment}/{split}|{values['unique_queries']}|"+
                '|'.join(f"{100*values[f'{moment}_R@{k}/20m']:.4f}" for k in (1,4,8,16))+f"|{values[moment+'_NE_m']:.4f}|")
    localization_lines += ['', '|固定Seen128因果对照|SR变化百分点|Episode配对95%区间|', '|---|---:|---|']
    for name, statistics in causal.items():
        effect = statistics['episode_weighted']['SR']
        localization_lines.append(f"|{name}|{100*effect['delta']:.4f}|{[round(100*x,4) for x in effect['ci95']]}|")
    insertion = lines.index('## 训练、参数和效率')
    lines[insertion:insertion] = localization_lines+['']
    lines += ['',summary['measurement_note'],'', '## 结果解释和发布候选','',
        '完整系统是否优于强基线以全量同协议对照和配对不确定性为准，所有失败或下降结果保留。'+('完整Seen SR提升；仍需结合SPL、Unseen和配对区间判断。' if summary['performance_improved_on_full_seen'] else '完整Seen SR未超过B0；不宣称视觉或History已带来稳定收益，B0作为独立发布候选保留。'),
        'best.pt是Seen选出的完整模型；last.pt是C最终轮，包含优化器、调度器及RNG，可继续显式恢复；stages/A/B/C各有独立Best/Last。Backbone通过明确模型名和不可变revision解析，本地离线缓存必须存在；不能将小型完整训练状态文件称为包含所有Backbone张量的单文件离线包。',
        '基准使用正射RGB和表面高度导出的垂直Depth代理，不代表真实透视相机或三维遮挡；不宣称真实飞行安全或扫掠体碰撞验证。历史SBF权重曾由Unseen proxy选出，该历史差异保留。', '',
        '## 可复现运行','', '```bash',
        '../Ablations/GraphNav/2027-CVPR-static-geometry/.venv/bin/python -m scripts.run_graphnav_stage1 --resume',
        '../Ablations/GraphNav/2027-CVPR-static-geometry/.venv/bin/python -m scripts.audit_graphnav_stage1',
        '```', '', '所有评估均为已有scripts.evaluate_closedloop.py，具体命令保存在workflow_status.json及audit_status.json。检查点和输入SHA在checkpoint_manifest.json；大文件不提交Git。']
    report='\n'.join(lines)+'\n'
    (root/'GRAPHNAV_V1_STAGE1_REPORT.md').write_text(report)
    archive=Path('reports/graphnav_v1_stage1');archive.mkdir(parents=True,exist_ok=True)
    for name in ('training.csv','seen128_comparison.csv','full_comparison.csv','visual_ablation.csv','history_ablation.csv','goal_retrieval.csv','analysis.json','checkpoint_manifest.json','initial_seen128_parity.json','historical_baseline_audit.json','verification_preflight.json','training_curves.png','GRAPHNAV_V1_STAGE1_REPORT.md','frozen_final_candidate.json','static_B0_seen.json','static_B0_unseen.json','static_selected_seen.json','static_selected_unseen.json'):
        shutil.copy2(root/name,archive/name)
    print(json.dumps(dict(output=str(root),selected_stage=freeze['selected_stage'],selected_epoch=state['epoch'],complete=True)),flush=True)


if __name__ == '__main__':
    main()
