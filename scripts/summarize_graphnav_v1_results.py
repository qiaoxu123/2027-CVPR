"""Verify and publish GraphNav-V1 only, including the frozen full Test-Unseen run."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.closedloop import aggregate
from scripts.summarize_closedloop import read_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/graphnav_v1_stage1')
    args = parser.parse_args()
    root = Path(args.run)
    freeze = json.loads((root/'frozen_final_candidate.json').read_text())
    assert file_sha256(root/'best.pt') == freeze['checkpoint_sha256']
    records, audits = [], {}
    common_protocol = None
    for label, directory, count in [('Val-Seen', 'full_seen', 2470),
                                    ('Val-Unseen', 'full_unseen', 2697),
                                    ('Test-Unseen', 'full_test_unseen', 5281)]:
        report, predictions, _ = read_run(root/directory)
        assert report['metrics']['episodes'] == len(predictions) == count
        assert report['checkpoint_sha256'] == freeze['checkpoint_sha256']
        assert report['args']['stage1_visual_control'] == freeze['visual_control']
        assert report['args']['stage1_history_control'] == freeze['history_control']
        common_protocol = common_protocol or report['protocol']
        assert report['protocol'] == common_protocol
        computed = aggregate([row['metrics'] for row in predictions.values()])
        for key, value in computed.items():
            if isinstance(value, (float, int)):
                np.testing.assert_allclose(value, report['metrics'][key], atol=1e-10, rtol=0)
        moves = np.concatenate([np.diff(np.asarray(row['poses'])[:, :3], axis=0)
                                for row in predictions.values()])
        audit = dict(episodes=count, horizontal_over_50m=int((np.linalg.norm(moves[:, :2], axis=-1)>50.001).sum()),
            vertical_over_10m=int((np.abs(moves[:, 2])>10.001).sum()),
            horizon_over_20=sum(row['metrics']['steps']>20 for row in predictions.values()),
            active_observation_actions=sum(bool(d.get('active_observation'))
                for row in predictions.values() for d in row['step_diagnostics']),
            residual_max_abs=max(d['residual_max_abs'] for row in predictions.values() for d in row['step_diagnostics']))
        assert not any(audit[key] for key in ['horizontal_over_50m','vertical_over_10m','horizon_over_20','active_observation_actions'])
        assert audit['residual_max_abs'] <= .500001
        audits[label] = audit
        records.append(dict(split=label, **report['metrics'], seconds=report['seconds'],
            gpu_peak_allocated_gib=report['gpu_peak_allocated_gib'], checkpoint_sha256=report['checkpoint_sha256'],
            goal_retrieval=report['goal_retrieval']))
    atomic_json(dict(model='GraphNav-V1', frozen_selection=freeze, splits=records,
                     action_audits=audits), root/'GRAPHNAV_V1_METRICS.json')
    with (root/'GRAPHNAV_V1_METRICS.csv').open('w') as stream:
        columns = [key for key in records[0] if key != 'goal_retrieval']
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    lines = ['# GraphNav-V1 正式结果（含 Test-Unseen）', '',
        '主实验只列 GraphNav-V1。沿用已冻结的 C 第1轮；Test未用于选模、调参或训练。历史阶段与对照结果保留在归档中。', '',
        '|分割|Episodes|SR%|OSR%|SPL%|NE m|nDTW%|sDTW%|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"|{row['split']}|{row['episodes']}|{row['SR']*100:.4f}|{row['OSR']*100:.4f}|{row['SPL']*100:.4f}|{row['NE']:.4f}|{row['nDTW']*100:.4f}|{row['sDTW']*100:.4f}|")
    lines += ['', '|分割|False Stop/Stop%|错误Stop Episode%|平均步数|路径m|单步ms|峰值GiB|总时间s|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"|{row['split']}|{row['False_Stop_Rate']*100:.4f}|{row['false_stop_episode_fraction']*100:.4f}|{row['steps']:.3f}|{row['path_length']:.3f}|{row['inference_ms']:.3f}|{row['gpu_peak_allocated_gib']:.3f}|{row['seconds']:.2f}|")
    lines += ['', '|分割/观测时刻|唯一查询数|R@1%|R@4%|R@8%|R@16%|NE m|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        retrieval = row['goal_retrieval']
        for moment in ['initial_static','dynamic_first','dynamic_final']:
            values = '|'.join(f"{retrieval[moment+f'_R@{k}/20m']*100:.4f}" for k in [1,4,8,16])
            lines.append(f"|{row['split']}/{moment}|{retrieval['unique_queries']}|{values}|{retrieval[moment+'_NE_m']:.4f}|")
    lines += ['', 'R指标为20m半径、唯一map+instruction查询；最终观测时刻因Episode而异，不能混同初始定位。',
        '动作、预算、有界残差已核验；各Episode指标重新聚合与正式metrics.json一致。单步计时包括视觉编码和模型，排除观测渲染。',
        '当前执行选择信念Top-1目标，由预训练且联合微调的轨迹/Stop网络预测下一航点与停止；Guarded-Blend执行。没有主动观察、新GAT或关系辅助损失。',
        '正射RGB与高度代理Depth协议不代表真实透视相机或扫掠体碰撞验证。视觉/History因果结论仍以Seen128消融为准，Test只评测冻结完整配置。', '',
        f"Checkpoint SHA256: `{freeze['checkpoint_sha256']}`", '',
        '逐Episode轨迹在 runs/graphnav_v1_stage1/full_seen、full_unseen、full_test_unseen；输入SHA、源Commit和完整命令在各目录metrics.json。']
    (root/'GRAPHNAV_V1_RESULTS.md').write_text('\n'.join(lines)+'\n')
    status = json.loads((root/'test_unseen_status.json').read_text())
    status.update(status='complete', episodes=5281,
                  verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  metrics_sha256=file_sha256(root/'full_test_unseen/metrics.json'))
    atomic_json(status, root/'test_unseen_status.json')
    manifest = json.loads((root/'checkpoint_manifest.json').read_text())
    manifest['output_sha256']['full_test_unseen/metrics.json'] = status['metrics_sha256']
    manifest['test_unseen'] = dict(episodes=5281, checkpoint_sha256=freeze['checkpoint_sha256'],
        evaluated_after_freeze=True, test_used_for_selection=False,
        action_audit=audits['Test-Unseen'])
    atomic_json(manifest, root/'checkpoint_manifest.json')
    archive = Path('reports/graphnav_v1_stage1'); archive.mkdir(parents=True, exist_ok=True)
    for name in ['GRAPHNAV_V1_METRICS.json','GRAPHNAV_V1_METRICS.csv','GRAPHNAV_V1_RESULTS.md','test_unseen_status.json','checkpoint_manifest.json']:
        shutil.copy2(root/name, archive/name)
    shutil.copy2(root/'full_test_unseen/metrics.json', archive/'test_unseen_metrics.json')
    print(json.dumps(dict(complete=True, splits={row['split']:row['SR'] for row in records})))


if __name__ == '__main__':
    main()
