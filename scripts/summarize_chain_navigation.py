"""Summarize completed paired TRAIN navigation without model inference."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--full-context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directories = {name: args.run/name for name in ('original', 'real', 'gray-rgb', 'rgb-shuffle')}
    directories['full-context'] = args.full_context/'real'
    rows = {}
    reports = {}
    for method, directory in directories.items():
        report = json.loads((directory/'metrics.json').read_text())
        records = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        indexed = {row['index']: row for row in records}
        if len(indexed) != len(records) or len(records) != report['episodes']:
            raise ValueError('incomplete or duplicate trajectories')
        rows[method] = indexed
        reports[method] = report
    base = rows['original']
    result = {}
    for method, indexed in rows.items():
        if set(indexed) != set(base):
            raise ValueError('unpaired episode sets')
        corrections, harms = [], []
        first_inputs_equal = True
        for index, row in indexed.items():
            baseline = base[index]
            first_inputs_equal &= row['poses'][0] == baseline['poses'][0]
            first_inputs_equal &= row['step_diagnostics'][0]['observed_rgb_sha256'] == baseline['step_diagnostics'][0]['observed_rgb_sha256']
            success = row['metrics']['SR']
            original = baseline['metrics']['SR']
            if success and not original:
                corrections.append(index)
            if original and not success:
                harms.append(index)
        if not first_inputs_equal:
            raise ValueError('initial pose/observed RGB mismatch')
        result[method] = dict(report=reports[method], corrections=corrections, harms=harms,
                              net=len(corrections)-len(harms), initial_inputs_equal=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'paired_results.json').write_text(json.dumps(result, indent=2))
    lines = ['# V2关系描述：TRAIN开发集真实闭环对照', '',
             '25个相同物理目标，固定yaw0、冻结权重、相同控制器与Stop。属于开发集诊断，不能作为独立泛化或正式Val成绩。', '',
             '|方法|SR|OSR|SPL|NE m|相对Original纠正/破坏|', '|---|---:|---:|---:|---:|---:|']
    for method, entry in result.items():
        metric = entry['report']['metrics']
        lines.append(f"|{method}|{metric['SR']:.2%}|{metric['OSR']:.2%}|{metric['SPL']:.4f}|{metric['NE']:.4f}|{len(entry['corrections'])}/{len(entry['harms'])}|")
    lines += ['', 'SR沿用原工程最终XY误差≤20m定义，OSR为轨迹曾进入20m；因此需同时检查policy_stop及SPL，不能将SR当作Stop精度。',
              'gray-rgb为编码前恒灰；rgb-shuffle为编码前16px图块打乱，几何/质量来自干预前真实观测。GT只作离线评分。',
              '完整记录包含全部失败，不筛选改善样本。逐任务配对详见paired_results.json。']
    (args.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({name: entry['report']['metrics']['SR'] for name, entry in result.items()}))


if __name__ == '__main__':
    main()
