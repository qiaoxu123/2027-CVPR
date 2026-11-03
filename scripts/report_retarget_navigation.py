"""Summarize completed TRAIN development rollouts; no model inputs or tuning."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve earlier reports')
    rows = {}
    cases = {}
    for label, folder, method in [
        ('Original', 'navigation_dev25_retarget_coarse', 'original'),
        ('FrozenV3', 'navigation_dev25_envelope', 'real'),
        ('Coarse', 'navigation_dev25_retarget_coarse', 'real'),
        ('SoftInstance', 'navigation_dev25_retarget_soft_instance', 'real'),
        ('SoftGray', 'navigation_dev25_retarget_soft_instance', 'gray-rgb'),
        ('SoftShuffle', 'navigation_dev25_retarget_soft_instance', 'rgb-shuffle'),
    ]:
        d = a.root/folder/method
        report = json.loads((d/'metrics.json').read_text())
        predictions = [json.loads(x) for x in (d/'predictions.jsonl').read_text().splitlines()]
        assert report['episodes'] == len(predictions) == 25
        by_id = {x['index']: x for x in predictions}
        assert len(by_id) == 25
        if cases:
            assert set(by_id) == set(next(iter(cases.values())))
        cases[label] = by_id
        rows[label] = report
    pairs = {}
    for baseline in ('Original', 'FrozenV3', 'Coarse', 'SoftShuffle'):
        good = []; bad = []
        for i, x in cases['SoftInstance'].items():
            before = bool(cases[baseline][i]['metrics']['SR'])
            after = bool(x['metrics']['SR'])
            if after and not before: good.append(i)
            if before and not after: bad.append(i)
        pairs[baseline] = dict(corrected=good, harmed=bad, net=len(good)-len(bad))
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(dict(results=rows, paired=pairs), indent=2))
    lines = ['# 重配对与软实例监督：真实TRAIN闭环诊断', '',
             '同25条已使用TRAIN开发任务；不是独立确认、正式Val或泛化成绩。固定3轮final、camera yaw0、原控制器与Stop；原模型和默认入口未修改。', '',
             '|方法|SR|OSR|SPL|NE(m)|nDTW|正确Goal丢失次数|', '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in rows.items():
        m = r['metrics']
        lines.append(f"|{name}|{m['SR']:.2%}|{m['OSR']:.2%}|{m['SPL']:.4f}|{m['NE']:.4f}|{m['nDTW']:.4f}|{r['lost_correct_goals']}|")
    lines += ['', '## 配对变化', '']
    for name, pair in pairs.items():
        lines.append(f"- SoftInstance相对{name}：纠正{len(pair['corrected'])}、破坏{len(pair['harmed'])}、净收益{pair['net']:+d}；具体任务见summary.json。")
    lines += ['', '## 监督和证据边界', '',
              '1165条拟合样本（原900＋新增265）；重配对查询重新计算全部语言条件特征，原纯图像特征共享。两组相同477次更新。软实例偏好不是像素金标准，未知样本跳过；1m位置核心仍可能配准错误。', '',
              '92开发状态：冻结V3 R1=79/92、NE17.2714m；Coarse=77/92、17.8391m；SoftInstance=79/92、17.4964m；后者打乱RGB=76/92、18.1206m。离线未证明全面改善。', '',
              '必须结合配对闭环及RGB消融判断是否保留；不得据25条开发结果宣布泛化，不读取旧Val128选模型，不启动全量。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(pairs, ensure_ascii=False))


if __name__ == '__main__':
    main()
