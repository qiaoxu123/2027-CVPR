"""Publish actual negative training controls and frozen binding results."""
import argparse,json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 v=json.load(open(a.run/'frozen_binding_val/metrics.json'));assert v['coverage']=={'completed':2470,'expected':2470}
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for mode in ['legacy','ce8','hit20']:
  h=json.load(open(a.run/'adaptation'/mode/'history.json'));axs[0].plot([r['epoch'] for r in h],[r['mean_loss'] for r in h],marker='o',label=mode);axs[1].plot([r['epoch'] for r in h],[100*r['development']['R20'] for r in h],marker='o',label=mode)
 axs[0].set(xlabel='Epoch',ylabel='Training loss',title='Loss scales differ across recipes');axs[1].axhline(67.4984,color='gray',ls='--',label='Frozen baseline');axs[1].set(xlabel='Epoch',ylabel='TRAIN development Top1 <=20m (%)');axs[0].legend();axs[1].legend();fig.tight_layout();fig.savefig(out/'training_loss_and_development.png',dpi=150);plt.close(fig)
 arms=v['arms'];fig,axs=plt.subplots(1,2,figsize=(9,4))
 for ax,key in zip(axs,['R20','NE']):
  values=[100*m['raw']['1'] if key=='R20' else m['NE'] for m in arms.values()];ax.bar(list(arms),values,color=['gray','teal']);ax.set_title('Val-Seen Top1 (%)' if key=='R20' else 'Val-Seen NE (m)');ax.tick_params(axis='x',labelsize=8)
  if key=='R20':ax.axhline(50,color='red',ls='--',label='Goal >50%');ax.legend()
 fig.tight_layout();fig.savefig(out/'frozen_full_val.png',dpi=150);plt.close(fig)
 before=json.load(open(a.run/'frozen_binding_val/baseline_rows.json'));after=json.load(open(a.run/'frozen_binding_val/scoped_public_names_rows.json'));paired=[]
 for x,y in zip(before,after):
  assert x['episode']==y['episode'];paired.append(dict(index=x['index'],episode=json.dumps(x['episode']),map_name=x['map_name'],baseline_hit=x['raw']['1'],scoped_hit=y['raw']['1'],baseline_NE=x['NE'],scoped_NE=y['NE']))
 with (out/'paired_full_val.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 report=f'''# Top-1超过50%：阶段进展\n\n**当前未完成目标。** 同一冻结训练后V2 Geometry、FP32单条推理，完整HETT Refined Val-Seen2470条：公开名称提取修复使Top1从{100*arms['baseline']['raw']['1']:.2f}%升至{100*arms['scoped_public_names']['raw']['1']:.2f}%，纠正{v['corrected']}、破坏{v['damaged']}、净+{v['net']}。NE从{arms['baseline']['NE']:.4f}m降至{arms['scoped_public_names']['NE']:.4f}m；空间独立Top5从{100*arms['baseline']['nms']['5']:.2f}%升至{100*arms['scoped_public_names']['nms']['5']:.2f}%。\n\n此前47.09%来自BF16批量推理；本轮配对基线46.96%来自FP32单条推理，不混合两种精度口径。两组候选网格、20m标准、Checkpoint及全文语言相同，仅公开名称绑定/Alias改动。GT只在forward后评分，未通过GT筛地标、候选或视点。没有运行新导航或Test，也不宣称实例识别准确率。\n\n[第一轮训练负结果](phase1/REPORT_zh.md)：三个Loss方案各5轮，开发最优模型独立TRAIN适配确认净−8，因此没有扩大或拿它跑Val。[第二轮名称解析](phase2/REPORT_zh.md)：只在Cambridge TRAIN开发组修正语义作用域，保守版在Val评分前冻结；完整指标与逐任务对照已保存。没有据Val再改解析阈值。\n\n[第三轮保真训练](phase3/PLAN_zh.md)正在运行：名称解析器保持冻结，比较仅评分Head/Geometry核心，训练时对源模型正确预测保真、对错误预测纠正。GT只决定TRAIN Loss，不进入推理。只用TRAIN开发选择，已消费的Birmingham确认组不重用为独立确认。\n\n分支experiment/v2-top1-public-binding-20261107。原V2/V3/控制器权重未覆盖，不改main、Stop、Horizon或命中标准。服务与此前配准结果未动。真实权重仍在服务器，Git只保存代码与轻量结果。\n'''
 (out/'REPORT_zh.md').write_text(report);(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V2 Top1 progress</title><h1>目标尚未达到：完整Val Top1 48.26%</h1><a href="REPORT_zh.md">中文报告</a><p>训练Loss下降不代表泛化改善。所有图均来自实际日志。</p><img src="training_loss_and_development.png" width="900"><img src="frozen_full_val.png" width="800">')
if __name__=='__main__':main()
