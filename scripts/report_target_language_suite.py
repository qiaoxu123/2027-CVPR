"""Report fixed-budget TRAIN target-language experiment without model selection."""
import argparse
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous report')
    status=json.loads((a.suite/'status.json').read_text())
    if status['status']!='complete':raise ValueError('require completed jobs')
    data={name:json.loads((a.suite/folder/'development.json').read_text()) for name,folder in [('role','role_s0'),('full','full_instruction_s0')]}
    methods={'FrozenV3':data['role']['frozen_baseline'],
             'FrozenV3+FullTarget':data['full']['input_only_reference'],
             'TrainedRole':data['role']['results']['real'],
             'TrainedFullTarget':data['full']['results']['real'],
             'FullTargetZeroRGB':data['full']['results']['zero'],
             'FullTargetShuffleRGB':data['full']['results']['shuffle'],
             'FullTargetNoGeometry':data['full']['results']['no_geometry']}
    paired={}
    new={r['frame_key']:r for r in methods['TrainedFullTarget']['rows']}
    for name,scored in methods.items():
        old={r['frame_key']:r for r in scored['rows']}
        assert set(new)==set(old) and len(new)==92
        paired[name]=dict(corrected=[k for k in new if new[k]['success'] and not old[k]['success']],
                          harmed=[k for k in new if not new[k]['success'] and old[k]['success']])
    a.output.mkdir(parents=True)
    (a.output/'summary.json').write_text(json.dumps(dict(methods=methods,paired=paired,
        checkpoint_sha256={k:v['checkpoint_sha256'] for k,v in data.items()}),indent=2))
    for name,folder in [('role','role_s0'),('full','full_instruction_s0')]:
        for filename in ('config.json','training.json'):
            shutil.copy2(a.suite/folder/filename,a.output/f'{name}_{filename}')
    shutil.copy2(a.suite/'status.json',a.output/'suite_status.json')
    lines=['# 目标通道补回原始指令：同预算训练对照','','相同1165条拟合样本、固定3×159次更新、seed0、同起点。参考通道、A0、冻结SigLIP2、RGB/Yaw0、架构与损失均相同，仅局部目标通道是否使用原始完整指令不同。独立TRAIN确认尚未进行。','',
           '|方法|Goal R1|NE(m)|纠正/误改A0|','|---|---:|---:|---:|']
    for name,d in methods.items():
        m=d['metrics'];lines.append(f"|{name}|{m['goal_r1']:.2%}|{m['goal_ne_m']:.4f}|{m['corrections']}/{m['harms']}|")
    lines+=['','## 配对比较','']
    for name in ['FrozenV3','TrainedRole','FullTargetShuffleRGB']:
        pair=paired[name];lines.append(f"- TrainedFullTarget相对{name}：纠正{len(pair['corrected'])}、误改{len(pair['harmed'])}。")
    lines+=['','## 边界','','这里的92状态来自25条已使用TRAIN开发任务，非独立泛化结果，不能当作闭环SR。旧冻结模型闭环22/25，原软实例模型19/25；本轮未新运行导航。','',
            'SigLIP2文本上限64 token，完整字符串仍可能截断：原400查询11条超限，新增236查询6条超限。5对已知碰撞在64-token输入中均可区分，不代表全部空间关系被保留或已被视觉理解。','',
            '完整原文目标通道保留了名称与参考内容，模型仍可能依赖语言/地图偏置；必须通过Shuffle和独立确认判断。不能把输入不同、梯度非零或训练loss下降当性能改善。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    names=list(methods);fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key,label in [(axes[0],'goal_r1','Goal R1'),(axes[1],'goal_ne_m','Goal NE (m)')]:
        ax.barh(names,[methods[n]['metrics'][key] for n in names]);ax.set_xlabel(label)
    fig.suptitle('TRAIN development only: 92 states / 25 tasks');fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
