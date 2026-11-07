"""Summarize completed paired runs; never infer navigation SR from candidate recall."""
import argparse,csv,gzip,json,sys,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
from graphnav.closedloop import aggregate
Image.MAX_IMAGE_PIXELS=None


def main():
 p=argparse.ArgumentParser()
 for k in ['run','data','sbf-source','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'cases').mkdir(exist_ok=True);groups={};allrows={};manifest=json.load(open(a.run/'manifest.json'))
 for arm in ['original_v2','trained_v2']:
  m=json.load(open(a.run/arm/'metrics.json'));assert m['complete'] and m['coverage']=={'completed':2470,'expected':2470};groups[arm]=m;rows=[json.load(open(x)) for x in sorted((a.run/arm/'rows').glob('*.json'))];assert len(rows)==2470 and [r['index'] for r in rows]==list(range(2470));allrows[arm]=rows
  with gzip.open(a.output/(arm+'_trajectories.jsonl.gz'),'wt') as f:
   for r in rows:f.write(json.dumps(r)+'\n')
 paired=[]
 for x,y in zip(allrows['original_v2'],allrows['trained_v2']):
  assert x['episode_id']==y['episode_id'];paired.append(dict(index=x['index'],episode_id=json.dumps(x['episode_id']),map_name=x['map_name'],original_SR=x['metrics']['SR'],trained_SR=y['metrics']['SR'],original_OSR=x['metrics']['OSR'],trained_OSR=y['metrics']['OSR'],original_NE=x['metrics']['NE'],trained_NE=y['metrics']['NE'],original_termination=x['termination'],trained_termination=y['termination']))
 with (a.output/'paired.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 correction=sum(r['trained_SR']>r['original_SR'] for r in paired);damage=sum(r['trained_SR']<r['original_SR'] for r in paired)
 result=dict(groups=groups,paired=dict(corrected=correction,damaged=damage,net=correction-damage),manifest=manifest,by_map={arm:{m:aggregate([r['metrics'] for r in rr if r['map_name']==m]) for m in sorted({r['map_name'] for r in rr})} for arm,rr in allrows.items()});(a.output/'metrics.json').write_text(json.dumps(result,indent=2))
 sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
 from sbfnav.mapdata import GROUND_LEVEL
 from graphnav.renderer import CityNavObservedViewRenderer
 ds=CityNavDataset(a.data,'val_seen',version='processed_citynav');renderer=CityNavObservedViewRenderer(a.data/'rgbd',a.run/'case_observations',image_size=384);selected=[]
 for test in [lambda r:r['trained_SR']>r['original_SR'],lambda r:r['trained_SR']<r['original_SR'],lambda r:not r['trained_SR'] and r['trained_OSR'],lambda r:not r['trained_SR'] and not r['trained_OSR'],lambda r:r['trained_SR'] and r['original_SR']]:selected.extend([r['index'] for r in paired if test(r)][:2])
 cards=[]
 for i in dict.fromkeys(selected):
  ep=ds[i];fig,axes=plt.subplots(1,3,figsize=(14,5));m=ep.map_name
  with rasterio.open(a.data/'rgbd'/(m+'.tif')) as f:b=f.bounds
  im=Image.open(a.data/'rgbd'/(m+'.png'));im.thumbnail((1000,1000));ax=axes[0];ax.imshow(im,extent=[b.left,b.right,b.bottom,b.top]);ref=np.asarray(ep.trajectory);ax.plot(ref[:,0],ref[:,1],color='gray',alpha=.5,label='Official reference (scoring only)')
  for arm,color in [('original_v2','orange'),('trained_v2','cyan')]:
   r=allrows[arm][i];pose=np.array(r['poses']);ax.plot(pose[:,0],pose[:,1],'-o',ms=3,color=color,label=arm);ax.scatter(pose[-1,0],pose[-1,1],marker='s',color=color)
  ax.scatter(*ep.target_position[:2],color='red',marker='*',s=100,label='GT after rollout');ax.legend(fontsize=7);ax.set_title(f'Task {i} | {m}')
  for j,arm in enumerate(['original_v2','trained_v2'],1):
   r=allrows[arm][i];frame=renderer.current_rgbd(m,np.array(r['poses'][-1]),GROUND_LEVEL[m]);axes[j].imshow(frame['bgr'][...,::-1]);axes[j].axis('off');axes[j].set_title(f"{arm} final current RGB\nSR={r['metrics']['SR']:.0f} NE={r['metrics']['NE']:.1f}m {r['termination']}")
  fig.suptitle(ep.instruction[:180],fontsize=9);fig.tight_layout();fn='cases/task_'+str(i)+'.png';fig.savefig(a.output/fn,dpi=120);plt.close(fig);cards.append(f'<article><h3>{i}: {html.escape(ep.instruction)}</h3><a href="{fn}"><img src="{fn}" width="100%"></a></article>')
 fig,axs=plt.subplots(1,3,figsize=(10,3));arms=list(groups)
 for ax,key in zip(axs,['SR','OSR','NE']):ax.bar(arms,[groups[k]['metrics'][key] for k in arms],color=['orange','teal']);ax.set_title(key);ax.tick_params(axis='x',labelsize=8)
 fig.tight_layout();fig.savefig(a.output/'navigation_metrics.png',dpi=140);plt.close(fig)
 table='\n'.join('|'+arm+'|'+str(m['coverage']['completed'])+'|'+ '|'.join(f"{m['metrics'][k]*100:.2f}%" if k in ['SR','OSR'] else f"{m['metrics'][k]:.4f}" for k in ['SR','OSR','SPL','NE','nDTW','steps','inference_ms'])+'|' for arm,m in groups.items())
 report=f'''# V2公开文本绑定：完整Val-Seen导航实测\n\n完整2470/2470任务，两组seed=42；HETT官方Refined Val-Seen已逐字节核实。模型及控制器冻结，不训练、不接入未验证OSM、不运行Test。\n\n|方法|任务|SR|OSR|SPL|NE(m)|nDTW|平均步数|决策ms/步|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n{table}\n\n训练后Geometry相对原始Geometry：纠正{correction}条、破坏{damage}条、净{correction-damage:+d}条。全部失败计入分母，逐任务见paired.csv；完整实际Pose/Goal/Stop诊断见两份压缩轨迹文件。无候选不删除；若推理异常运行中断，不能生成完整metrics。\n\n## 协议与限制\n\n与上一轮候选召回使用相同公开文本地标绑定、28×28网格、冻结权重和公开Polygon。原有Top-K候选/Context同步使用模型输出，不将空间NMS候选替换为导航Goal；不使用GT挑选候选。原Guarded-Blend、20步Horizon、50m水平/10m垂直动作、20m成功半径、Stop阈值0.3及1mm到达守卫均保留。每步从实际预测Pose生成384px当前观测，Geometry控制不使用RGB分支。真实目标和参考轨迹仅在rollout结束后计分。\n\nREADME历史47.77%/50.24%使用此前标注引用元数据绑定，本次盲文本绑定不同，不可混作同一次对照。此处没有测试OSM增强或视觉更新的新增收益。候选Top-5覆盖率不能视作SR：控制器没有GT指出应选择哪一个候选，且终点/Stop/轨迹都影响成功。单seed，不宣称统计显著。\n\n初始Goal与上一轮离线Top-1的一致性（≤1m）及Horizon/Stop数量见metrics.json。两者不完全相同：离线以16条批处理，本轮逐条推理，存在近分数候选排序敏感性；原始组122条差异中，离线对应候选排名均在前4，logit差中位0.0153、最大0.1302。尚未单独隔离BERT/Geometry数值来源，不宣称已定位为特定算子bug；保留全部真实输出，未依据Val调整精度或排名策略。两组闭环均逐条推理，配对口径一致。推理时延为缓存初始Geometry后的逐步Controller决策，不包含本轮预计算初始Geometry、数据渲染或报告时间，不能直接与包含这些开销的端到端时延比较。所有任务rollout_wall_s已保留。\n\n## 产物\n\n[真实轨迹及终点RGB案例](index.html) · [完整指标](metrics.json) · [逐任务配对](paired.csv)。权重与数据SHA、脚本指纹和协议见metrics.json内manifest。\n'''
 (a.output/'REPORT_zh.md').write_text(report);(a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V2导航实测</title><h1>完整2470×2真实闭环</h1><a href="REPORT_zh.md">中文报告</a><img src="navigation_metrics.png">'+''.join(cards));print(json.dumps({k:v['metrics'] for k,v in groups.items()},indent=2))
if __name__=='__main__':main()
