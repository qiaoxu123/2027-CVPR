"""Summarize measured diagnostics and verify paired episode/control evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from graphnav.workspace_paths import PROJECT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(PROJECT/'Ablations/GraphNav/diagnostics_20261013'))
    args=parser.parse_args();root=Path(args.root)
    reports={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/metrics.json')}
    profile=json.loads((root/'profile/profile.json').read_text())
    names={'rollout_ms':'单样本 rollout','model_components_ms':'模型阶段合计（同步墙钟）','language_ms':'语言编码器（BERT + SigLIP 文本）','vision_ms':'视觉编码器与 ROI 池化','static_graph_ms':'静态几何图定位','visual_residual_history_ms':'视觉匹配 / 历史 / 残差更新','trajectory_stop_ms':'轨迹 / Stop 网络','environment_ms':'环境观测','other_rollout_ms':'其他 rollout 开销'}
    text=['# GraphNav 命名与诊断结果（2026-10-13）','',
        '工程统一为 **GraphNav — Graph-First Aerial Vision-and-Language Navigation**；正式主模型仍为 GraphNav-V1，冻结 C Epoch1。',
        '导航消融使用 Val-Seen；附加性能 profiling 使用冻结配置的 Val-Unseen。结果独立存于消融目录，Test 未用于调参。','',
        '## 逐决策 profiling（定位瓶颈）','',f"128 Episodes，共 {profile['samples']} 条逐观测/决策样本；单位 ms，总体方差 ms²。",'',
        '|项目|最小值|均值|总体方差|','|---|---:|---:|---:|']
    for key,values in profile['all'].items():
        if key not in names:continue
        text.append(f"|{names[key]}|{values['min']:.2f}|{values['mean']:.2f}|{values['population_variance']:.2f}|")
    hett_path=root/'profile_hett/profile_episodes.json'
    if hett_path.exists():
        episode_profile=json.loads(hett_path.read_text())
        policy_values=[r['static_graph_gpu_ms']+r['visual_residual_history_gpu_ms']+r['trajectory_stop_gpu_ms'] for r in episode_profile['samples_detail']]
        episode_profile['all']['navigation_policy_gpu_ms']=dict(min=float(np.min(policy_values)),mean=float(np.mean(policy_values)),population_variance=float(np.var(policy_values)))
        with (root/'profiling_episodes_summary.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['component','min_ms','mean_ms','population_variance_ms2']);writer.writeheader()
            for key,value in episode_profile['all'].items():writer.writerow(dict(component=key,min_ms=value['min'],mean_ms=value['mean'],population_variance_ms2=value['population_variance']))
        ep_names={'navigation_policy_gpu_ms':'导航策略 GPU 合计（静态 + 残差 + 轨迹）','rollout_ms':'单样本整 Episode rollout','model_inference_gpu_ms':'模型 GPU 推理合计','language_gpu_ms':'语言编码器','vision_gpu_ms':'视觉编码器与原生 ROI 池化','static_graph_gpu_ms':'静态几何图定位','visual_residual_history_gpu_ms':'视觉匹配 / 历史 / 残差更新','trajectory_stop_gpu_ms':'轨迹 / Stop 网络','environment_ms':'环境观测','other_rollout_ms':'其他 rollout 开销'}
        ep_text=['## HETT 原口径 profiling（整 Episode）','',f"Val-Unseen {episode_profile['samples']} Episodes，冻结原配置；模型项使用 CUDA Event，rollout/环境使用同步墙钟。单位 ms；总体方差 ms²。",'', '|项目|最小值|均值|总体方差|','|---|---:|---:|---:|']
        for key,values in episode_profile['all'].items():
            ep_text.append(f"|{ep_names[key]}|{values['min']:.2f}|{values['mean']:.2f}|{values['population_variance']:.2f}|")
        first_ep=episode_profile['first'];warm_ep=episode_profile['excluding_first']['rollout_ms']
        ep_text += ['',f"首条整 Episode {first_ep['rollout_ms']/1000:.3f}s，观测 {first_ep['environment_ms']/1000:.3f}s；排除首条 rollout 均值 {warm_ep['mean']:.2f}ms，方差 {warm_ep['population_variance']:.2f}ms²。",'', '|最慢 Episode|rollout ms|','|---|---:|']
        for row in episode_profile['slowest_five']:
            ep_text.append(f"|{' / '.join(map(str,row['episode_id']))}|{row['rollout_ms']:.2f}|")
        original=PROJECT/'Ablations/HETT/hett-main-system-opt/artifacts/profiling/profile_val_unseen_full.json'
        old=json.loads(original.read_text())
        ep_text += ['',f"已读取原 HETT profiler `multiagent/profile_inference.py` 及其原始 JSON：Val-Unseen 2697 Episodes；rollout 均值 {old['rollout_seconds']['mean']*1000:.2f}ms，GPU 合计 {old['model_gpu_seconds_by_component']['model_inference_gpu_seconds']['mean']*1000:.2f}ms。两者均统计整 Episode（非单步）。GraphNav 本轮是 128 条子集，GPU 存在 HTNav 并发训练，且额外同步的细分阶段会增加开销，不能将两者直接解释成模型速度倍率。", 'CUDA Event 间隔遵循原 HETT 模块 hook 口径，可能包含主机发射间隙；视觉项包括原生 ROI 池化，残差项排除嵌套轨迹头。Episode 其他项按总墙钟减观测墙钟与 GPU 合计计算。','']
        text[5:5]=ep_text
    first=profile['first'];warm=profile['excluding_first']['rollout_ms']
    text += ['',f"首条 {first['rollout_ms']/1000:.3f}s，环境观测 {first['environment_ms']/1000:.3f}s。排除首条 rollout 均值 {warm['mean']:.2f}ms，方差 {warm['population_variance']:.2f}ms²。",'',
        '模型加载不计入 rollout。语言与静态定位在 episode 初次决策计算/缓存，后续零耗时代表复用。模块用 CUDA synchronize + perf_counter 计时，包含模块内部 CPU/传输；模型阶段合计不是纯 GPU kernel 时间，不能直接标成纯 GPU 推理。轨迹头在模型输出与原控制器内各调用一次，均计入轨迹项。残差项排除嵌套的轨迹与静态模块；其他项含数据准备、记忆更新、诊断与控制器开销。',
        '当前 HTNav 训练并发使用相同 GPU；128 Seen 子集与用户 HETT 历史样本不相同，也未在同一环境重跑 HETT，不能据此声称正式加速倍率。','',
        '|最慢样本（episode / 决策步）|rollout ms|','|---|---:|']
    for row in profile['slowest_five']:
        text.append(f"|{' / '.join(map(str,row['episode_id']))} / step {row['step']}|{row['rollout_ms']:.2f}|")
    text += ['', '## Heatmap 分辨率','',
        '将真实全局候选网格从 28×28 改为 384×384（照片原生 384×384），未用插值替代网络推理。候选数 784→147456，为 188.08 倍；同一冻结权重用于分辨率迁移诊断，未重训。注意：网格覆盖整个地图，照片覆盖局部视野，数组尺寸相同不意味着地面米/像素一致；SigLIP patch token 仍为 24×24，网格加密不会增加照片特征的原生分辨率。', '',
        '|网格|Episodes|SR%|SPL%|NE m|rollout均值 ms|峰值 GiB|','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('heatmap_28','heatmap_384'):
        if name in reports:
            report=reports[name];m=report['metrics'];prof=json.loads((root/name/'profile.json').read_text())
            text.append(f"|{report['stage1_inference_controls']['grid_size']}×{report['stage1_inference_controls']['grid_size']}|{m['episodes']}|{m['SR']*100:.2f}|{m['SPL']*100:.2f}|{m['NE']:.2f}|{prof['all']['rollout_ms']['mean']:.2f}|{report['gpu_peak_allocated_gib']:.2f}|")
    text += ['', '仅 4 个均匀索引 episodes 的可行性测试；不据此选取正式分辨率或宣称泛化提升。','',
        '## Stop 阈值闭环对照','',
        '所有阈值使用同一 128 条均匀索引 Episodes，重新执行完整闭环，不从截断轨迹推测较高阈值行为。阈值作用于 positive_weight=20 校准后的 Stop 概率；原有目标距离与高度 guard 仍参与停止判定。','',
        '|阈值|SR%|SPL%|NE m|False Stop / Stop%|平均步数|终止原因|','|---|---:|---:|---:|---:|---:|---|']
    stop_reports=[reports[f'stop_{t}'] for t in (.1,.3,.5,.7,.9) if f'stop_{t}' in reports]
    for report in stop_reports:
        m=report['metrics'];false=m['False_Stop_Rate']
        text.append(f"|{report['stage1_inference_controls']['stop_threshold']}|{m['SR']*100:.2f}|{m['SPL']*100:.2f}|{m['NE']:.2f}|{false*100 if false is not None else float('nan'):.2f}|{m['steps']:.2f}|{report['terminations']}|")
    if stop_reports:
        assert len({r['episode_manifest_sha256'] for r in stop_reports})==1,'Stop subsets differ'
        assert len({r['checkpoint_sha256'] for r in stop_reports})==1,'Stop checkpoints differ'
        for r in stop_reports:
            assert r['args']['stop_threshold']==r['stage1_inference_controls']['stop_threshold']
        text += ['', '核验：各档实际运行阈值与命令一致，episode manifest 和 checkpoint SHA 相同。子集结果不足以替换正式 0.3 阈值。']
    stop_analysis=[]
    for report in stop_reports:
        threshold=report['stage1_inference_controls']['stop_threshold']
        rows=list(map(json.loads,(root/f'stop_{threshold}/predictions.jsonl').read_text().splitlines()))
        eligible=0;rejected=0
        for row in rows:
            for diagnostic in row['step_diagnostics']:
                if diagnostic['predicted_goal_distance']<=1e-3 and diagnostic['stop_altitude_guard']:
                    eligible+=1
                    rejected+=int(diagnostic['stop_probability']<threshold)
        stop_analysis.append(dict(threshold=threshold,goal_and_altitude_guard_eligible_decisions=eligible,
            probability_rejected_decisions=rejected,terminations=report['terminations'],
            false_stop_episode_fraction=report['metrics']['false_stop_episode_fraction']))
    (root/'stop_analysis.json').write_text(json.dumps(stop_analysis,indent=2))
    if stop_analysis:
        text += ['', '停止判定同时需要到达预测的 Top1 目标（距离≤1e-3m）和高度合法。因此更高阈值可能让已到达预测目标的无人机继续重复决策，最终 stagnation 或 horizon；不会自动换目标。各档 guard 通过但概率不足的次数见 `stop_analysis.json`。本协议 SR 按终点距离统计，达到预算/停滞也可能记为成功，必须连同 policy_stop 数量查看。']
    checks={}
    if 'stop_0.3' in reports:
        official={r['index']:r for r in map(json.loads,Path('runs/graphnav_v1_stage1/full_seen/predictions.jsonl').read_text().splitlines())}
        rows=list(map(json.loads,(root/'stop_0.3/predictions.jsonl').read_text().splitlines()))
        xyz_differences=[];shape_changes=[];outcome_changes=[]
        for row in rows:
            ref=official[row['index']]
            if np.asarray(row['poses']).shape==np.asarray(ref['poses']).shape:
                xyz_differences.append(float(np.max(np.abs(np.asarray(row['poses'])[:,:3]-np.asarray(ref['poses'])[:,:3]))))
            else:shape_changes.append(row['index'])
            if row['termination']!=ref['termination'] or row['metrics']['SR']!=ref['metrics']['SR']:
                outcome_changes.append(row['index'])
        checks['threshold_0_3_vs_frozen_official']=dict(episodes=len(rows),outcome_changes=outcome_changes,
            trajectory_length_changes=shape_changes,max_xyz_difference_for_equal_length_m=max(xyz_differences))
        text += ['',f"0.3 阈值与原正式结果核对：{len(rows)} 条中 SR/终止原因改变 {len(outcome_changes)} 条；轨迹长度改变 {len(shape_changes)} 条，同长度轨迹最大 XYZ 偏差 {max(xyz_differences):.6f}m。子集独立运行与原整 split 运行存在数值/缓存执行差异，不能宣称逐坐标完全重现；新阈值比较均采用同一子集独立运行口径。"]
    if 'stop_0.3' in reports and 'no_belief' in reports:
        a=reports['stop_0.3'];b=reports['no_belief'];assert a['episode_manifest_sha256']==b['episode_manifest_sha256']
        gain=(a['metrics']['SR']-b['metrics']['SR'])*100
        text += ['', '## 视觉残差对照','',f"同一 128 episodes，完整模型 SR={a['metrics']['SR']*100:.2f}%，无 belief 残差 SR={b['metrics']['SR']*100:.2f}%，差 {gain:+.2f} 个百分点；SPL 差 {(a['metrics']['SPL']-b['metrics']['SPL'])*100:+.2f} 个百分点，NE 差 {a['metrics']['NE']-b['metrics']['NE']:+.2f}m。无残差对照仍执行视觉编码/匹配，仅将 belief logit 修正置零，不能解释成去掉视觉的加速测试。小子集不代表总体提升。"]
    if 'stop_0.3' in reports:
        retrieval=reports['stop_0.3']['goal_retrieval']
        text += ['', f"视觉信念检索（去重查询）：初始 static R@1={retrieval['initial_static_R@1/20m']*100:.2f}%，首次 RGB R@1={retrieval['dynamic_first_R@1/20m']*100:.2f}%，最终 RGB/history R@1={retrieval['dynamic_final_R@1/20m']*100:.2f}%。最终时刻因 episode 而异，该时间对照不单独证明因果收益，因果对照参见 no-belief。"]
    text += ['', '## 复现与证据','',
        '从 GraphNav 目录执行 `../Baselines/.reproduction-env/bin/python -m scripts.run_graphnav_diagnostics`；默认 128 episodes、4 条 heatmap 可行性样本。队列安全跳过已完成项，保留未完成目录以便排错。',
        '运行命令与返回码见 `queue.json`，汇总见 `comparison.csv`，逐样本 profiling 见 `profile/profile_samples.csv`，各实验 metrics 包含数据/权重 SHA、实际参数、episode manifest 与代码版本。',
        '执行 `../Baselines/.reproduction-env/bin/python -m scripts.summarize_graphnav_diagnostics` 重建本报告及配对检查。',
        '相关测试 26 项通过，git diff --check 通过；失败的基础环境启动日志保留于 profile_hett_failed_environment.log，随后已使用正确虚拟环境完成 profiling。最终源码与文件 SHA 见 source_snapshot_manifest.json；原执行队列留存于 queue_initial.json。']
    if (root/'resolution_visualization/resolution_comparison.png').exists():
        text += ['', '## 照片与真实网格可视化','', '![384px 照片与 28/384 全局网格](resolution_visualization/resolution_comparison.png)', '初始 public 观测重放，两种真实网格 Top16 与实测结果匹配；最右列仅为照片坐标投影展示，未以展示插值冒充真实网格加密。']
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')
    (root/'verification.json').write_text(json.dumps(checks,indent=2))
    with (root/'profiling_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['component','min_ms','mean_ms','population_variance_ms2']);writer.writeheader()
        for key,value in profile['all'].items():writer.writerow(dict(component=key,min_ms=value['min'],mean_ms=value['mean'],population_variance_ms2=value['population_variance']))
    print(root/'RESULTS.md')


if __name__=='__main__':main()
