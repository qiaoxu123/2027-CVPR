"""Report actual frozen-model TRAIN candidate coverage, not instance accuracy or SR."""
import json,argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 records=[json.loads(x) for x in open(src/'full.jsonl') if x.strip()];origins=np.array([r['world_bounds'][:2] for r in records]);summary=json.load(open(src/'summary.json'));audit=json.load(open(src/'parsing.json'));ks=[1,5,10,20,50];labels={'full':'完整原文','landmark_relation':'地标＋关系词','target_type_relation':'地标＋关系词＋目标名称/类型'}
 import shutil
 for name in ['summary.json','per_task.csv']:shutil.copyfile(src/name,out/name)
 (out/'extraction_all.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
 (out/'extraction_examples.json').write_text(json.dumps(audit[:30],ensure_ascii=False,indent=2))
 fig,ax=plt.subplots(figsize=(8,4))
 for arm,s in summary['metrics'].items():ax.plot(ks,[s[f'R{k}_20m']*100 for k in ks],marker='o',label=arm)
 ax.set_xlabel('Top-K grid candidates');ax.set_ylabel('Position hit within 20m (%)');ax.legend();fig.tight_layout();fig.savefig(out/'topk.png',dpi=180);plt.close(fig)
 arrays={k:np.load(src/f'{k}_predictions.npz') for k in labels}
 # Save actual WORLD XY, restoring map origin to local meter coordinates.
 ar=arrays['target_type_relation'];top=np.argsort(-ar['logits'],axis=-1)[:,:50];world=ar['grid']+origins[:,None,:];scores=np.exp(ar['logits']-ar['logits'].max(-1,keepdims=True));scores/=scores.sum(-1,keepdims=True)
 with open(out/'target_top50_world.jsonl','w') as f:
  for i in range(len(audit)):
   f.write(json.dumps(dict(index=i,episode=audit[i]['episode'],map=audit[i]['map'],world_xy=world[i,top[i]].round(4).tolist(),probability=scores[i,top[i]].tolist(),grid_ids=top[i].tolist()))+'\n')
 new=arrays['target_type_relation'];base=arrays['full'];ids=[]
 for cond in [(~base['r1'])&new['r1'],base['r1']&(~new['r1']),~new['r1'],new['r1']]:ids+=np.flatnonzero(cond)[:3].tolist()
 ids=list(dict.fromkeys(ids))[:12];(out/'cases').mkdir(exist_ok=True);example=[]
 for i in ids:
  fig,axs=plt.subplots(1,3,figsize=(12,4))
  for ax,arm in zip(axs,labels):
   ar=arrays[arm];g=ar['grid'][i]+origins[i];l=ar['logits'][i];prob=np.exp(l-l.max());prob/=prob.sum();sc=ax.scatter(g[:,0],g[:,1],c=prob,s=9,cmap='magma');top=np.argsort(-l)[:20];ax.scatter(g[top,0],g[top,1],s=15,facecolors='none',edgecolors='cyan');gt=ar['gt'][i]+origins[i];ax.scatter(*gt,marker='*',s=100,c='lime');ax.set_title(arm);ax.set_aspect('equal')
  fig.suptitle(f"TRAIN {i} | GT green, top20 cyan",fontsize=10);fig.tight_layout();name=f'case_{i}.png';fig.savefig(out/'cases'/name,dpi=150);plt.close(fig);example.append(dict(**audit[i],figure=f'cases/{name}'))
 (out/'cases.json').write_text(json.dumps(example,ensure_ascii=False,indent=2));
 lines=['# 结构化目标文本：冻结UnifiedPGVB-V2候选Top-K诊断','', '**范围：全部Birmingham TRAIN样本；这些样本已用于该Checkpoint训练，属于拟合诊断，不是独立泛化测试。** 未重新训练、未运行导航、未使用Val/Test。', '', '## 输入与公平对照','', '三组使用同一冻结全文Checkpoint、同一公开Polygon绑定、相同784个28×28世界网格。仅修改语言文本，并重新计算BERT及由文本派生的关系图/规则。没有复用原文的语言条件特征。目标名称/类别由已有spaCy+规则从原文提取，不读目标ID或标注类别；专有名称缺失时使用原文目标中心名词，不强行生成专名。颜色、尺寸等外观词未加入新增精简组。', '', '冻结模型本来以完整原文训练，因此新精简文本属于输入分布干预；本轮不能说明专门重训精简输入后的最终能力。公开多Polygon保留原绑定，不用GT选最近Anchor。GT坐标只在forward结束后评分。', '', f"Checkpoint：`{summary['checkpoint']}`；SHA256：`{summary['checkpoint_sha']}`。", '', '## 实测结果','', '|输入|N|Top1|Top5|Top10|Top20|Top50|Top1 NE(m)|目标20m概率质量|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for arm,s in summary['metrics'].items():lines.append('|'+labels[arm]+f"|{s['N']}|"+'|'.join(f"{s[f'R{k}_20m']*100:.2f}%" for k in ks)+f"|{s['NE_mean']:.3f}|{s['mass20']:.4f}|")
 lines+=['', 'Top-K命中指前K个概率最高的网格点中，至少一个距目标中心≤20m；不是BBox、实例匹配率或导航SR。相邻高概率点可能落在同一空间峰，不能将K个网格解释为K个不同目标。GT不参与排序。', '', f"全部784点的20m覆盖上界：{summary['metrics']['full']['grid_oracle20']*100:.2f}%。这仅检查网格支持，不是模型推理成绩。", '', '## 解析覆盖与结果变化','', json.dumps(summary['parse_coverage'],ensure_ascii=False), '', '相对完整原文的Top1纠正/破坏：'+json.dumps(summary['paired_vs_full'],ensure_ascii=False), '', '解析是自动代理，未独立人工核验；关系词采用现有名称前邻接词组抽取，间接链、远距离作用域可能被遗漏。未知目标类型和无具名地标任务保留分母，不按GT筛选。新增字段不等于保留完整视觉和实例约束。', '', '## 可视化与复现','', '[Top-K曲线](topk.png)、[逐任务结果](per_task.csv)、[提取样例](extraction_examples.json)、[成功失败热图](index.html)。绿色星号为事后GT，青圈为Top20网格；GT未进入预测。', '', '服务器实验目录：`'+str(src)+'`。保存三组完整logits、网格、GT评分和解析JSON；代码scripts/diagnose_structured_target_topk.py可复现。seed42；先2条GPU Smoke，再完整诊断。', '', '本轮优先回答候选覆盖与排序；若精简输入不及全文，不应直接加大训练，先检查语言信息删除和解析误差；若TopK明显高于Top1，才说明后续消歧存在空间，不能直接推断导航收益。']
 (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
 html='<meta charset="utf-8"><h1>TRAIN候选Top-K诊断</h1><p>GT仅事后评分。绿色GT、青色Top20。自动提取，非人工解析真值。</p><img width="800" src="topk.png">'
 import html as h
 for r in example:html+=f"<h2>{r['index']} {h.escape(r['map'])}</h2><p>{h.escape(r['original'])}</p><p>精简输入：{h.escape(r['minimal'])}</p><img width='1200' src='{r['figure']}'>"
 (out/'index.html').write_text(html)
 print((out/'REPORT_zh.md').read_text())
if __name__=='__main__':main()
