"""Aggregate completed privileged Oracle diagnostics, with all failures retained."""
import argparse,csv,json,html
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import file_sha256


def pair(a,b,ids):
 return dict(fixes=sum(not a[i]['metrics']['SR'] and bool(b[i]['metrics']['SR']) for i in ids),damage=sum(bool(a[i]['metrics']['SR']) and not b[i]['metrics']['SR'] for i in ids))


def main():
 p=argparse.ArgumentParser()
 for n in ['run','baseline','output']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();r=a.run;o=a.output;o.mkdir(parents=True,exist_ok=True)
 offline=json.loads((r/'offline_summary.json').read_text());diag=json.loads((r/'offline_rows.json').read_text());fields=torch.load(r/'offline_fields.pt',map_location='cpu',weights_only=False)
 folders={'A':a.baseline,**{x:r/f'nav_{x}' for x in ['B','C','D1']}};rows={};metrics={};groups=defaultdict(list)
 for name,folder in folders.items():
  d=json.loads((folder/'metrics.json').read_text());assert d['coverage']['completed']==d['coverage']['expected']==2470
  rows[name]={v['index']:v for v in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())};assert len(rows[name])==2470;metrics[name]=d['metrics']
 for d in diag:
  i=d['index'];groups[d['group']].append(i)
  for name,folder in folders.items():
   b=rows['A'][i];q=rows[name][i];assert b['episode_id']==q['episode_id'] and b['poses'][0]==q['poses'][0]
   old=(a.baseline/f'belief_steps/{i:05d}/public_record.json').read_text();new=(folder/f'belief_steps/{i:05d}/public_record.json').read_text();assert json.loads(old)==json.loads(new)
   if name!='A' and q['step_diagnostics']:
    assert all(v['PRIVILEGED_ORACLE_DIAGNOSTIC'] and v['GT_range_or_position_at_runtime'] is False for v in q['step_diagnostics'])
 summary=dict(ORACLE_ONLY=True,deployable=False,offline=offline,closedloop=metrics,groups={},paired={},physical_target_groups=len(set(tuple(x['episode_id'][:2]) for x in rows['A'].values())))
 for group,ids in {'all':list(rows['A']),**groups}.items():
  summary['groups'][group]={}
  for name in folders:
   q=[rows[name][i]['metrics'] for i in ids];summary['groups'][group][name]={k:float(np.mean([m[k] for m in q])) for k in ['SR','OSR','SPL','NE','nDTW','steps','inference_ms']}
   summary['groups'][group][name].update(N=len(ids),**pair(rows['A'],rows[name],ids))
 for name in ['B','C','D1']:
  ids=list(rows['A']);pa=pair(rows['A'],rows[name],ids);pa['net']=pa['fixes']-pa['damage'];summary['paired'][name]=pa
  outcomes=Counter();records=[]
  for d in diag:
   i=d['index'];q=rows[name][i];step=q['step_diagnostics'];traj=np.asarray(q['poses']);gt=np.asarray(d['GT']);entered=np.linalg.norm(traj[:,:2]-gt,axis=1)<=20
   offlinegood=d[name]['R20'];sr=bool(q['metrics']['SR']);selected=np.array([x['selected_goal_xy'] for x in step]) if step else np.empty((0,2));goalhit=(np.linalg.norm(selected-gt,axis=1)<=20) if len(selected) else np.array([])
   label='success' if sr else 'entered20_then_failed_stop_or_left' if entered.any() else 'offline_correct_but_no_correct_selected_goal' if offlinegood and not goalhit.any() else 'correct_goal_but_not_reached' if goalhit.any() else 'offline_goal_incorrect' if not offlinegood else 'unknown'
   outcomes[label]+=1;records.append(dict(index=i,group=d['group'],category=label,offline_R20=offlinegood,ever_goal20=bool(goalhit.any()),ever_entered20=bool(entered.any()),SR=sr,termination=q['termination'],first_entered20=int(np.flatnonzero(entered)[0]) if entered.any() else None,oracle_top1_changes=max((v.get('oracle_changed_top1_steps',0) for v in step),default=0),selected_goal_differs_from_A=bool(step and rows['A'][i]['step_diagnostics'] and step[0]['selected_goal_xy']!=rows['A'][i]['step_diagnostics'][0]['selected_goal_xy'])))
  summary[name+'_failure_proxy_categories']=dict(outcomes);(o/f'failure_{name}.json').write_text(json.dumps(records,indent=2));summary[name+'_decision_changed_tasks']=sum(x['selected_goal_differs_from_A'] for x in records)
  physical=defaultdict(list)
  for i,q in rows['A'].items():physical[tuple(q['episode_id'][:2])].append(i)
  batches=list(physical.values());rng=np.random.default_rng(42);delta=np.array([rows[name][i]['metrics']['SR']-rows['A'][i]['metrics']['SR'] for i in ids]);boot=[]
  for _ in range(2000):
   ix=[i for j in rng.integers(0,len(batches),len(batches)) for i in batches[j]];boot.append(float(delta[ix].mean()))
  pa['SR_delta95CI']=np.quantile(boot,[.025,.975]).tolist()
 extra=r/'quantized8_diagnostic.json'
 if extra.exists():summary['quantized8_angles_sensitivity']=json.loads(extra.read_text())
 summary['grid_diagnosis']=dict(no_grid_within20=sum(not d['grid_support20'] for d in diag),A_top20_no20hit=sum(not d['A_top20_support'] for d in diag),nearest_grid_NE_mean=float(np.mean([d['grid_min_NE'] for d in diag])),nearest_grid_NE_max=float(max(d['grid_min_NE'] for d in diag)))
 summary['runtime_offline_top1_agreement']={name:dict(N=sum(bool(q['step_diagnostics']) for q in rows[name].values()),matched=sum(bool(q['step_diagnostics']) and q['step_diagnostics'][0]['final_top1']==diag[i][name]['top1'] for i,q in rows[name].items())) for name in ['B','C','D1']}
 summary['anchor_count_by_group']={g:dict(Counter(d['N_anchors'] for d in diag if d['group']==g)) for g in groups}
 summary['nearest_grid_rank']={name:float(np.mean([d[name]['rank_nearest'] for d in diag])) for name in ['A','B','C','D0.5','D1','D2','D4']}
 (o/'summary.json').write_text(json.dumps(summary,indent=2));(o/'offline_rows.json').write_text(json.dumps(diag,indent=2))
 paired=[]
 for d in diag:
  i=d['index'];line=dict(index=i,episode_id=json.dumps(d['episode_id']),group=d['group'],grid_min_NE=d['grid_min_NE'])
  for name in folders:
   for k in ['SR','OSR','SPL','NE','nDTW']:line[name+'_'+k]=rows[name][i]['metrics'][k]
   line[name+'_offline_rank_nearest']=d[name]['rank_nearest'];line[name+'_offline_NE']=d[name]['NE']
  paired.append(line)
 with (o/'paired2470.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 # Mechanism examples, successes AND damage, single anchors AND multi-ray privilege.
 chosen=[]
 for group in ['single_unique_polygon','same_name_multi_polygon','multiple_distinct_names','no_anchor']:
  for condition in ['fix','damage']:
   candidates=[d['index'] for d in diag if d['group']==group and ((not d['A']['R20'] and d['D1']['R20']) if condition=='fix' else (d['A']['R20'] and not d['D1']['R20']))];chosen+=candidates[:2]
 for name in ['B','C','D1']:
  for condition in [True,False]:chosen += [i for i in rows['A'] if bool(rows[name][i]['metrics']['SR'])==condition and bool(rows['A'][i]['metrics']['SR'])!=condition][:2]
 chosen=list(dict.fromkeys(chosen));images=[]
 for i in chosen:
  f=fields[str(i)];d=diag[i];fig,axs=plt.subplots(1,4,figsize=(17,4.5));bounds=f['bounds'];xy=f['grid'];gt=f['target']
  for ax,name in zip(axs,folders):
   z=f['scores'][name];prob=torch.tensor(z).softmax(-1).numpy();ax.imshow(prob.reshape(28,28),origin='lower',extent=[bounds[0],bounds[2],bounds[1],bounds[3]],cmap='magma')
   for poly in f['polygons']:
    poly=np.asarray(poly);ax.plot(*np.vstack([poly,poly[0]]).T,color='cyan',lw=.7)
   for anchor,theta in zip(f['anchors'],f['angles']):
    end=anchor+np.array([np.cos(theta),np.sin(theta)])*np.linalg.norm(bounds[2:]-bounds[:2]);ax.plot([anchor[0],end[0]],[anchor[1],end[1]],'g--',lw=.6);ax.plot(*anchor,'go',ms=2)
   top=np.argsort(-z,kind='stable')[:10];ax.scatter(*xy[top].T,marker='x',s=12,c='yellow');ax.plot(*gt,'r*',ms=8)
   for method,color in [(name,'lime')]:
    path=np.asarray(rows[method][i]['poses']);ax.plot(path[:,0],path[:,1],color=color,lw=1);ax.plot(path[0,0],path[0,1],'wo',ms=3)
   ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),title=f'{name} ORACLE' if name!='A' else 'A baseline');ax.set_aspect('equal');ax.text(.01,.01,f"GoalNE {d[name]['NE']:.1f}m; SR {rows[name][i]['metrics']['SR']:.0f}",transform=ax.transAxes,color='white',fontsize=7)
  fig.suptitle(f"Val {i} [{d['group']}] "+d['instruction'][:140],fontsize=8);fig.tight_layout();fn=f'case_{i:05d}.png';fig.savefig(o/fn,dpi=140);plt.close(fig);images.append((i,fn,d['instruction']))
 fig,axs=plt.subplots(1,2,figsize=(11,4));names=list(folders);axs[0].bar(names,[metrics[x]['SR']*100 for x in names]);axs[0].set(ylabel='SR (%)',title='Privileged ORACLE closed-loop, N=2470 each');axs[1].bar(names,[metrics[x]['NE'] for x in names]);axs[1].set(ylabel='NE (m)');fig.tight_layout();fig.savefig(o/'navigation.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(12,4));names=list(offline['all']);axs[0].bar(names,[offline['single_unique_polygon'][x]['R20']*100 for x in names]);axs[0].set(ylabel='Goal R20 (%)',title=f"Single unique Polygon N={offline['single_unique_polygon']['A']['N']}");axs[1].bar(names,[offline['all'][x]['R20']*100 for x in names]);axs[1].set(ylabel='Goal R20 (%)',title='All N=2470; multi-ray triangulation is privileged');fig.tight_layout();fig.savefig(o/'offline.png',dpi=150);plt.close(fig)
 report=['# Oracle方向角推理诊断','', '**所有B/C/D均使用目标GT方向，是特权诊断，不是可部署方法或泛化成绩。**','', '## 协议与隔离','', '冻结47.37%基线9ac51ee…，同一公开文本Polygon绑定、784网格、Controller、Stop=0.3、1mm到达守卫、Horizon20。未训练，未访问Test，未修改正常推理入口。GT仅预先生成每个既定Anchor的方位角；运行时接收角度而无GT坐标、距离或正确候选ID。无地标、无定义方向均回退原Belief。', '', 'Polygon Anchor沿用PR26的完整轮廓顶点均值，并非面积质心/目标最近边界。角度在工程World XY中计算，无需假定地理北。全部已绑定多Polygon等权使用，不用GT挑选。**多个GT方向可能三角定位，故多Polygon高分不能证明实例识别，更不能作为单角度能力。**', '', 'B=平均余弦，C=平均正向射线垂距/网格对角线并排除任何反向候选，无共同支持则回退；D=原logit+λ×4×平均余弦。λ=.5/1/2/4全部离线报告；闭环B/C/D1在看分数前固定，不依据Val选最佳。B/C固定≤1e-4原始排序微扰处理数值近似并列。', '', '## 离线定位','', '|分组/方法|N|R@20m|R@5|R@10|NE(m)|纠正/破坏|','|---|---:|---:|---:|---:|---:|---:|']
 for group in ['all','single_unique_polygon','same_name_multi_polygon','multiple_distinct_names','no_anchor','undefined_angle_abstain']:
  if group not in offline:continue
  for name,m in offline[group].items():report.append(f"|{group}/{name}|{m['N']}|{m['R20']*100:.2f}%|{m['R5']*100:.2f}%|{m['R10']*100:.2f}%|{m['NE']:.3f}|{m['fixes']}/{m['damage']}|")
 report += ['', 'R@5/R@10表示前K个网格中有点位于目标20m内，非实例匹配准确率。逐任务最近GT网格排名和首个20m网格排名见offline_rows.json。', '', '网格支持：'+json.dumps(summary['grid_diagnosis'],ensure_ascii=False), '', '## 完整闭环','', '|方法|N|SR|OSR|SPL|NE(m)|nDTW|单步ms|纠正/破坏|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,m in metrics.items():pa=summary['paired'].get(name,dict(fixes=0,damage=0));report.append(f"|{name}|2470|{m['SR']*100:.2f}%|{m['OSR']*100:.2f}%|{m['SPL']:.4f}|{m['NE']:.3f}|{m['nDTW']:.4f}|{m['inference_ms']:.2f}|{pa['fixes']}/{pa['damage']}|")
 report += ['', '闭环分组、配对物理目标bootstrap95%区间见summary.json。仅单种子，不能声称稳定泛化。Oracle评估最多两进程并行，时延受共享GPU负载影响，不能与历史单进程时延作严格效率差异推断。所有失败和无地标任务计入2470分母。', '', '## 机制结论与局限','', '1. 正确角度不能决定径向距离：单Anchor纯角度可能把同一射线上远处网格提到最前；射线垂距=r×sin(角偏差)，相近角偏差时可能偏好离Anchor更近的网格，同样不能知道目标半径。联合原Belief保留有价值的距离先验。', '2. 多Anchor Oracle方向提供额外三角定位信息。即使绑定的是语义不合适的地标，从它到GT的方向仍携带GT位置，因此高Oracle成绩不能证明Polygon绑定正确。', '3. 784网格离散上限与排序错误分开统计。世界坐标20m成功不是正确实例识别；道路/大地标的顶点均值会受形状采样影响，本轮未用GT改Anchor。', '4. 离线正确但闭环失败，检查failure_B/C/D1.json中的实际选Goal、轨迹是否曾入20m及终止。分类是观测事件代理，不能仅凭NE证明Stop或Controller内部根因。', '5. Oracle会同时改变候选XY、上下文、候选logit幅度，冻结Controller未必适配纯几何分数分布。全网格重排已同步IDs/XY/dense_context；实际首步Goal变化数量见summary。', '6. 本轮不依据Val增减权重或重训。后续应优先在TRAIN验证可学习方位预测及距离/多峰保真，再用独立数据确认；不能把Oracle方向上线。', '', '## 工程验证','', '完整CPU415项通过（含新增3项几何/回退/Context对齐测试）。最初输出目录所有权冲突发生在模型运行前，已隔离保留该目录并修复wrapper；有效闭环结果未覆盖。训练Loss的1m最小半径门控不符合本轮角度唯一干预要求，已撤除并保留未完成预检运行，不计入本报告。独立代码审查工具已实际调用，但workspace routing discovery超时，未完成外部审查；日志见adversarial_review_unavailable.txt。', '', '服务器原始轨迹、角度缓存和完整世界网格：`'+str(r)+'`。Checkpoint未写入；执行脚本、参数与SHA见运行清单。所有图使用真实模型输出与实际闭环轨迹，不含示意或生成RGB。']
 q8=summary.get('quantized8_angles_sensitivity',{}).get('single_unique_polygon',{})
 if q8:report += ['', '## 8方向量化敏感性（额外离线诊断）','',f"同一D1权重，单一明确Polygon连续Oracle角度R20={q8['continuous']['R20']*100:.2f}%，量化为8方向R20={q8['quantized8']['R20']*100:.2f}%。不是重新选择方案，也未用于调整闭环；说明连续角度与PR26八方向监督并非完全相同的精度设置。"]
 (o/'REPORT_zh.md').write_text('\n'.join(report)+'\n')
 page='<!doctype html><meta charset="utf-8"><title>Oracle bearing diagnostic</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}img{width:100%}article{padding:20px;border-bottom:1px solid #aaa}</style><h1>Oracle方向角诊断：不可部署</h1><p>GT方向仅离线生成；多Polygon含三角定位特权。红星=GT，青线=Polygon，绿虚线=Oracle射线，黄叉=Top10，绿实线=实际轨迹。点击图片查看原尺寸。</p><a href="REPORT_zh.md">中文报告</a><img src="offline.png"><img src="navigation.png">'
 for i,fn,text in images:page+=f'<article><h2>任务{i}</h2><p>{html.escape(text)}</p><a href="{fn}"><img loading="lazy" src="{fn}"></a></article>'
 (o/'index.html').write_text(page)
 print(json.dumps({k:summary[k] for k in ['closedloop','paired','grid_diagnosis']},indent=2))
if __name__=='__main__':main()
