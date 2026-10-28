"""One unchanged-condition internal assessment; never a navigation claim."""
import json
from pathlib import Path
import shutil
import tempfile

import torch

from experiments.a2_waldo_landmark_belief.paths import sha
from .conditions import assess_condition
from .fitting import fit_and_freeze, ARMS
from .linear_fit import predict
from .packets import load_packets


REPORT_FILES = {'result.json', 'report_ZH.md', 'fit_curves.png', 'fit_curves.svg'}


def _cached(directory, signature):
    manifest = json.loads((directory/'manifest.json').read_text())
    files = manifest.get('files_sha256', {})
    if (manifest.get('status') != 'complete_internal_cpu_diagnostic'
            or manifest.get('signature') != signature or set(files) != REPORT_FILES
            or any(sha(directory/name) != digest for name,digest in files.items())):
        raise ValueError('Published internal report bytes or input commitments changed')
    result = json.loads((directory/'result.json').read_text())
    if (result.get('signature') != signature or result.get('objective_achieved') is not False
            or result.get('no_runtime_enablement') is not True):
        raise ValueError('Internal diagnostic report improperly claims deployment/success')
    return result


def _markdown(result):
    lines = ['# A2修改判断：固定CPU内部检查', '',
             '仅为现有自主行走训练与内部DEV的离线诊断，不是完整导航评估。', '',
             '两种方法均固定训练20轮；条件只在TRAIN预留目标上选择，随后保持不变。', '']
    if not result['internal_dev_evaluated']:
        lines += ['两种方法都未在训练预留目标上找到合格条件，因此没有在DEV上评估这两种修改规则。',
                  '前序导出已读取DEV并生成离线缓存；这些DEV结果没有参与学习或选择规则。',
                  '不把未检查的结果记为零提升或验证通过。']
    else:
        lines += ['| 方法 | 支持目标 | 纠正观测 | 改错观测 | 目标平均定位正确变化（百分点） | 平均距离变化（米） | 内部条件达标 |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for arm, title in (('baseline','原有信息'),('extended','增加候选信息')):
            r = result['internal_results'][arm]; s = r['summary']
            lines.append(f"|{title}|{s['supported_groups']}|{s['corrected']}|{s['broken']}|{100*s['mean_r1_difference']:+.3f}|{s['mean_ne_difference_m']:+.3f}|{'是' if r['passes_internal'] else '否'}|")
        lines += ['', '以上定位正确以距离目标20米内计算；观测数和目标数不可混用。', '',
                  '| 地图 | 原有信息定位变化（百分点） | 增加候选信息定位变化（百分点） |', '|---|---:|---:|']
        old, new = (result['internal_results'][arm]['by_map'] for arm in ARMS)
        for name in sorted(new):
            lines.append(f"|{name}|{100*old[name]['mean_r1_difference']:+.3f}|{100*new[name]['mean_r1_difference']:+.3f}|")
    lines += ['', '是否值得继续正式实验：' + ('是。' if result['promising_for_formal_iteration'] else '目前没有足够依据。'), '',
              '不启用任何实际修改，不改变原A0/A1/V4结果。尚未证明A2超过A0。', '',
              '完整逐目标统计见result.json；曲线随报告保存，TensorBoard与全部20轮权重保留在../fit。', '']
    return '\n'.join(lines)


def assess_and_report(directory, exported):
    directory = Path(directory)
    frozen = fit_and_freeze(directory, exported)  # Validate immutable choices first.
    choices = frozen['decision']['choices']
    evaluate = any(value is not None for value in choices.values())
    data = None
    if evaluate:
        data = load_packets(directory/'packets', role='dev',
                            expected_entries=exported['entries']['dev'], binding=exported['binding'])
        train_targets = {(r['map_name'],r['object_id']) for r in exported['entries']['train']}
        if train_targets & {(r['map_name'],r['object_id']) for r in data['identities']}:
            raise ValueError('TRAIN/internal DEV target overlap at assessment')
    signature = dict(binding=exported['binding'], fit_manifest_sha256=frozen['manifest_sha256'],
                     fixed_choice_sha256=sha(directory/'fit/fixed_choice.json'),
                     dev_manifest_sha256=data['manifest_sha256'] if data else None)
    destination = directory/'report'
    if destination.exists():
        return _cached(destination, signature)
    results, promising = None, False
    if data is not None:
        arrays = data['arrays']; x = torch.from_numpy(arrays['features']); results = {}
        for arm in ARMS:
            scores = predict(frozen['models'][arm], x).numpy()
            results[arm] = assess_condition(choices[arm], identities=data['identities'], scores=scores,
                available=arrays['available'], before=arrays['before'], after=arrays['after'])
        promising = bool(results['extended']['passes_internal'] and
            results['extended']['summary']['mean_r1_difference'] > results['baseline']['summary']['mean_r1_difference'])
    result = dict(signature=signature,
        status='fixed_internal_dev_assessed' if evaluate else 'no_safe_train_selection_conditions',
        internal_dev_evaluated=evaluate, internal_results=results,
        promising_for_formal_iteration=promising, no_runtime_enablement=True,
        objective_achieved=False, no_gpu_used=True, independent_or_benchmark_opened=False,
        scope='CPU feasibility only; not a trained navigation policy or A2>A0 conclusion')
    pending = Path(tempfile.mkdtemp(prefix='.report-pending-', dir=directory))
    (pending/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (pending/'report_ZH.md').write_text(_markdown(result))
    for extension in ('png','svg'):
        shutil.copyfile(directory/f'fit/fit_curves.{extension}', pending/f'fit_curves.{extension}')
    if fit_and_freeze(directory, exported)['manifest_sha256'] != frozen['manifest_sha256']:
        raise ValueError('Frozen choices or fitting artifacts changed during assessment')
    if data is not None:
        checked = load_packets(directory/'packets',role='dev',
            expected_entries=exported['entries']['dev'],binding=exported['binding'])
        if checked['manifest_sha256'] != data['manifest_sha256']:
            raise ValueError('DEV packet changed during fixed assessment')
    manifest = dict(status='complete_internal_cpu_diagnostic', signature=signature,
                    files_sha256={name:sha(pending/name) for name in sorted(REPORT_FILES)})
    (pending/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    pending.rename(destination)
    return _cached(destination, signature)
