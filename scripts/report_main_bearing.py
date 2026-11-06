"""Report fixed PR26 arms: Predicate effect separate from Angular effect."""
import argparse,csv,html,json,shutil
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.unified_direct_bearing_loss import DirectBearingDataset,predicate_ids_for_record
from scripts.main_bearing_groups import strata
from scripts.analyze_instruction_landmark_association import PublicNames

ARMS=('predicate','angle01','angle03')


def bootstrap(rows_a,rows_b,groups):
 rng=np.random.default_rng(42);g=list(groups.values());samples=[]
 delta={i:rows_b[i]['metrics']['SR']-rows_a[i]['metrics']['SR'] for i in rows_a}
 for _ in range(2000):
  ix=[i for j in rng.integers(0,len(g),len(g)) for i in g[j]];samples.append(np.mean([delta[i] for i in ix]))
 return np.quantile(samples,[.025,.975]).tolist()


def pair(a,b):
 fixes=sum(not a[i]['metrics']['SR'] and bool(b[i]['metrics']['SR']) for i in a);damage=sum(bool(a[i]['metrics']['SR']) and not b[i]['metrics']['SR'] for i in a)
 return dict(fixes=int(fixes),damage=int(damage),net=int(fixes-damage))


def main():
 p=argparse.ArgumentParser()
 for k in ('run','baseline-evaluation','source','public-objects','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();r=a.run;out=a.output;out.mkdir(parents=True,exist_ok=True)
 frozen=json.loads((r/'FROZEN_BEFORE_VAL.json').read_text())
 for path,sha in frozen['code'].items():assert file_sha256(path)==sha,'Frozen inference source changed: '+path
 for v in frozen['arms'].values():assert file_sha256(v['checkpoint'])==v['SHA256']
 offline=json.loads((r/'offline/summary.json').read_text());fields=torch.load(r/'offline/fields.pt',weights_only=True,map_location='cpu');audit=json.loads((r/'audit/data_audit.json').read_text());rows={};metrics={};folders={'matched_full_text':a.baseline_evaluation,**{n:r/f'val_seen_{n}' for n in ARMS}}
 for name,folder in folders.items():
  report=json.loads((folder/'metrics.json').read_text());assert report['coverage']['completed']==report['coverage']['expected']==2470
  rows[name]={x['index']:x for x in map(json.loads,(folder/'predictions.jsonl').read_text().splitlines())};assert len(rows[name])==2470;metrics[name]=report['metrics']
 base=rows['matched_full_text'];groups=defaultdict(list);pairedrows=[];matcher=PublicNames(json.loads(a.public_objects.read_text()));valgroups=[]
 for i in sorted(base):
  b=base[i];groups[tuple(b['episode_id'][:2])].append(i);ref=json.loads((a.baseline_evaluation/f'belief_steps/{i:05d}/public_record.json').read_text())
  record=dict(index=i,episode_id=json.dumps(b['episode_id']))
  for name in folders:
   n=rows[name][i];assert n['episode_id']==b['episode_id'] and n['poses'][0]==b['poses'][0]
   current=json.loads((folders[name]/f'belief_steps/{i:05d}/public_record.json').read_text());assert current['instruction']==ref['instruction'] and current['landmarks']==ref['landmarks'] and current['world_bounds']==ref['world_bounds']
   for m in ('SR','OSR','SPL','NE','nDTW'):record[name+'_'+m]=n['metrics'][m]
   record[name+'_termination']=n['termination']
   if name in ARMS:assert all(d['GT_world_bearing_at_inference'] is False for d in n['step_diagnostics'])
  public_map=matcher.maps.get(ref['map_name'],{})
  # Use original complete catalog for exactly the same public qualification.
  if i==sorted(base)[0]:catalog=json.loads(a.public_objects.read_text())
  _,idx=predicate_ids_for_record(ref,catalog[ref['map_name']],36)
  valgroups.append(strata(ref,idx>=0,matcher));pairedrows.append(record)
 paired={};comparisons=[('matched_full_text','predicate'),('predicate','angle01'),('predicate','angle03'),('matched_full_text','angle01'),('matched_full_text','angle03')]
 for b,n in comparisons:paired[b+'_to_'+n]={**pair(rows[b],rows[n]),'SR_delta95CI':bootstrap(rows[b],rows[n],groups)}
 summary=dict(Val_Seen=metrics,paired=paired,physical_target_groups=len(groups),single_seed=True,TRAIN_offline=offline,TRAIN_audit=audit,frozen=frozen)
 grouped={}
 for key in valgroups[0]:
  ids=[i for i,g in zip(sorted(base),valgroups) if g[key]];grouped[key]={}
  for name in folders:grouped[key][name]=dict(N=len(ids),SR=float(np.mean([rows[name][i]['metrics']['SR'] for i in ids])) if ids else None,NE=float(np.mean([rows[name][i]['metrics']['NE'] for i in ids])) if ids else None)
 summary['Val_overlapping_public_text_strata']=grouped;summary['automatic_strata_not_human_gold']=True
 atomic_json(summary,out/'summary.json')
 with (out/'paired2470.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(pairedrows[0]));w.writeheader();w.writerows(pairedrows)
 atomic_json(offline,out/'TRAIN_location_direction.json')
 for src,name in [(r/'FROZEN_BEFORE_VAL.json','FROZEN_BEFORE_VAL.json'),(r/'audit/data_audit.json','data_audit.json'),(r/'audit/eligible_TRAIN_audit.jsonl','eligible_TRAIN_audit.jsonl'),(r/'audit/raster_frame.json','raster_frame.json'),(r/'CPU_tests.txt','CPU_tests.txt'),(r/'real_angular_gradient_smoke.json','real_angular_gradient_smoke.json')]:shutil.copyfile(src,out/name)
 histories={name:json.loads((r/name/'progress.json').read_text()) for name in ARMS}
 for name,h in histories.items():atomic_json(h,out/f'training_{name}.json')
 fig,axes=plt.subplots(2,2,figsize=(10,7))
 for name in ARMS:
  h=histories[name]['history'];ep=[v['epoch'] for v in h]
  for ax,key,label in zip(axes.flat,('train_loss','mean_goal_loss','mean_angular_loss','mean_KL'),('Combined TRAIN loss','Goal CE: all21878','Angular CE: eligible21','KL: all21878')):ax.plot(ep,[v[key] for v in h],'o-',label=name);ax.set(xlabel='Epoch',ylabel=label)
 for ax in axes.flat:ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,3))
 for ax,key in zip(axes,('SR','NE')):ax.bar(list(metrics),[v[key]*(100 if key=='SR' else 1) for v in metrics.values()]);ax.set_ylabel(key+(' (%)' if key=='SR' else ' (m)'));ax.tick_params(axis='x',labelsize=7)
 fig.tight_layout();fig.savefig(out/'navigation_comparison.png',dpi=150);plt.close(fig)
 c=torch.load(r/'predicate/last.pt',map_location='cpu',weights_only=True)['config']['geometry_args'];ds=DirectBearingDataset(a.source,public_objects=a.public_objects,deduplicate_queries=False,grid_size=28,max_landmarks=c['max_landmarks'],max_roads=c['max_roads'],max_relations=c['max_relations'],prior_sigma_m=c['prior_sigma_m'],rule_sigma_m=c['rule_sigma_m'])
 train_ids=fields['TRAIN_indices'];positions={idx:j for j,idx in enumerate(train_ids)};g=json.loads((r/'offline/strata.json').read_text());selected=[i for i in train_ids if ds._predicates[i][1]>=0]
 for tag in ('same_name_multi_polygon','heuristic_indirect_cardinal_chain','no_public_anchor'):selected += [idx for idx,group in zip(train_ids,g) if group[tag]][:4]
 selected=list(dict.fromkeys(selected));cases=[];states=fields['outputs'];order=('matched_full_text',*ARMS)
 for idx in selected:
  j=positions[idx];rec=ds.records[idx];bounds=np.asarray(rec['world_bounds']);target=bounds[:2]+np.asarray(rec['target_xy'])*(bounds[2:]-bounds[:2]);fig,axs=plt.subplots(1,4,figsize=(15,4))
  vmax=max(float(states[name]['logits'][j].softmax(-1).max()) for name in order)
  for ax,name in zip(axs,order):
   z=states[name]['logits'][j];ax.imshow(z.softmax(-1).reshape(28,28).numpy(),origin='lower',extent=[bounds[0],bounds[2],bounds[1],bounds[3]],vmin=0,vmax=vmax)
   for lm in rec['landmarks']:
    xy=np.asarray(lm['contour'])*(bounds[2:]-bounds[:2])+bounds[:2];ax.plot(*np.vstack((xy,xy[0])).T,'c-',lw=.5)
   ax.plot(*target,'r*');ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),title=f'{name}\nNE={float(states[name]["ne"][j]):.1f}m');ax.set_aspect('equal')
  fig.suptitle(f'TRAIN {idx}, gate={ds._predicates[idx][1]>=0}: '+rec['instruction'][:150],fontsize=8);fig.tight_layout();file=f'train_{idx:05d}.png';fig.savefig(out/file,dpi=130);plt.close(fig);cases.append(dict(TRAIN_index=idx,instruction=rec['instruction'],image=file,public_strata=g[j]))
 eligible_cases={name:{v['index']:v for v in offline['angular'][name]['cases']} for name in order};angleids=sorted(eligible_cases['predicate'])
 fig,ax=plt.subplots(figsize=(8,3));theta=np.arange(8);GTcounts=np.bincount([v['gt_bin'] for v in eligible_cases['predicate'].values()],minlength=8);ax.bar(theta,GTcounts);ax.set(xlabel='GT map-plane bearing bin (+X=0deg, CCW)',ylabel='TRAIN eligible queries',title='Not a geographic north calibration');fig.tight_layout();fig.savefig(out/'GT_direction_distribution.png',dpi=150);plt.close(fig)
 for idx in angleids[:8]:
  fig,ax=plt.subplots(figsize=(8,3))
  for j,name in enumerate(order):ax.bar(np.arange(8)+(j-1.5)*.18,eligible_cases[name][idx]['predicted_direction_mass'],width=.18,label=name)
  ax.axvline(eligible_cases['predicate'][idx]['gt_bin'],color='r',label='GT TRAIN bin');ax.set(xlabel='Map-plane bin',ylabel='Main Belief angular probability',title=f'TRAIN {idx} induced direction mass');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'angle_{idx:05d}.png',dpi=150);plt.close(fig)
 atomic_json(cases,out/'cases.json')
 valcaseids=[]
 for name in ('angle01','angle03'):
  valcaseids += [i for i in base if not rows['predicate'][i]['metrics']['SR'] and rows[name][i]['metrics']['SR']][:2]
  valcaseids += [i for i in base if rows['predicate'][i]['metrics']['SR'] and not rows[name][i]['metrics']['SR']][:2]
 for i in list(dict.fromkeys(valcaseids)):
  rec=json.loads((folders['predicate']/f'belief_steps/{i:05d}/public_record.json').read_text());b=np.asarray(rec['world_bounds']);fig,axs=plt.subplots(1,4,figsize=(15,4))
  for ax,name in zip(axs,order):
   n=rows[name][i];field=np.load(folders[name]/f'belief_steps/{i:05d}/00.npz')['full'];prob=torch.as_tensor(field).softmax(-1).reshape(28,28).numpy();ax.imshow(prob,origin='lower',extent=[b[0],b[2],b[1],b[3]])
   for lm in rec['landmarks']:
    xy=np.asarray(lm['contour'])*(b[2:]-b[:2])+b[:2];ax.plot(*np.vstack((xy,xy[0])).T,'c-',lw=.5)
   xy=np.asarray(n['poses']);ax.plot(xy[:,0],xy[:,1],'w.-',lw=.8);ax.plot(*xy[0,:2],'go');ax.plot(*xy[-1,:2],'rx');ax.set(xlim=(b[0],b[2]),ylim=(b[1],b[3]),title=f'{name}: SR={n["metrics"]["SR"]:.0f} NE={n["metrics"]["NE"]:.1f}');ax.set_aspect('equal')
  fig.suptitle(rec['instruction'][:150],fontsize=8);fig.tight_layout();file=f'val_{i:05d}.png';fig.savefig(out/file,dpi=130);plt.close(fig);cases.append(dict(Val_index=i,instruction=rec['instruction'],image=file))
 atomic_json(cases,out/'cases.json')
 table='\n'.join(f'|{name}|{v["SR"]*100:.3f}%|{v["OSR"]*100:.3f}%|{v["SPL"]:.4f}|{v["NE"]:.4f}|{v["nDTW"]:.4f}|' for name,v in metrics.items())
 train_table='\n'.join(f'|{name}|{v["R20"]*100:.3f}%|{v["NE"]:.4f}|{offline["angular"][name]["N"]}|{offline["angular"][name]["angular_8bin_accuracy"]}|{offline["angular"][name]["mean_GT_direction_probability"]:.4f}|' for name,v in offline['metrics'].items())
 text=f'''# PR26 主Belief Angular Consistency Loss：完整训练与闭环

## 实现和监督边界

A仅Predicate，B加Angular0.1，C加Angular0.3；全部21878条Revised TRAIN，seed42、5Epoch、batch8、Geometry LR1e-5/Relation1e-4、KL0.05、原Goal CE，FP32/TF32关闭。三组实际初始化hash相同，原BERT/PGVB/Stop/Controller不变，不使用PR25角度弱头。Angular只在forward之后监督主784网格Belief诱导的8方向分布，不输入GT角度/目标坐标/距离到新模型。所有任务保留Goal/KL。

TRAIN及runtime统一原文+完整公开目录盲绑定。修复句尾规则全TRAIN0标签、别名、否定/复合方向/朝向/明确间接对象；支持可靠前置方位和后续句子。自动语法筛选不是人工真值。同名多Polygon、多Anchor及模糊关系不参与Angular，但不从数据/导航分母删除。修复无有效候选方位的假常数监督、Angular断点统计丢失；Experimental Checkpoint必须匹配runtime标志。Predicate映射0默认/1east/2west/3north/4south，原8条Embedding不变，缺省与全零输出位级一致。

数据审计：{audit}。只有21条自动资格、N/S8/13、E/W词样本0；5轮重复不是105条独立监督。GT半平面/8bin一致率只在原工程轴约定下统计，不能用它选样本或翻转坐标。TIF CRS=None、无标签，无法声明+Y是地理北；道路全Polygon顶点均值也不等于局部道路参照。Angle Loss训练真实GT地图方位，并不硬编码north=+Y标签。

CPU412项通过；真实TRAIN纯Angular对Geometry和Relation非零梯度，Predicate Embedding第二步收到梯度，首步0来自原score零初始化。随机4batch Smoke恰好无资格样本，另用按公开文本选择的2例验证Angular反传，未按GT筛选。后续每组2条TRAIN真实闭环只验证接口、不报告SR。

## TRAIN Birmingham拟合诊断（非独立Dev）

Birmingham7619条包含在TRAIN，GT仅损失/事后评价。概率质量是主Belief归入真实8方向类别的诱导概率，不是像素或实例识别。

|方法|R20|NE(m)|角度样本N|角度8bin准确率（比例）|GT方向概率|
|---|---:|---:|---:|---:|---:|
{train_table}

A对匹配全文：{offline['predicate_vs_matched_full']}；B/C对A：{offline['paired']}。多Polygon、自动间接链、无Anchor等重叠分层见TRAIN_location_direction.json；未判定直接不等于间接，链标签是公开词法代理而非人工语义真值。

## 全2470/2470 Val-Seen真实闭环

三个固定第5轮模型分别真实闭环，原起点/完整指令/Controller/Stop0.3/1mm守卫/Horizon20/20m均不变；RGB更新关闭以隔离本轮语言/几何方法。无Anchor/无资格/错误任务全部入分母，无GT角度在线使用，不访问Test。

主对照是已完成、同公开绑定重训的全文47.37%（9ac51ee2），不是早期旧训练绑定权重45.75%。前者使A与无Predicate模型的区别更纯；旧45.75%只保留历史上下文。Base/A/B/C所有Episode、起点、原文和公开Polygon逐条核对一致。

|方法|SR|OSR|SPL|NE(m)|nDTW|
|---|---:|---:|---:|---:|---:|
{table}

配对与物理目标分组95% bootstrap（区间为比例，乘100为百分点）：{paired}。单训练种子，不能将Episode CI解释为多种子稳定性。A−全文用于Predicate效果；B/C−A用于Angular新增效果，不选择Val最优α再假装独立确认。

完整Checkpoint/原始逐步轨迹/大体积离线字段保留{r}；代码、配置、曲线、全部配对CSV和真实热图/成功失败轨迹提交GitHub。冻结SHA见FROZEN_BEFORE_VAL.json，所有未训练模块逐张量与Original V2一致。未修改main，不覆盖旧权重，不更改阈值，不训练目标外观或新控制器。
'''
 (out/'REPORT_zh.md').write_text(text)
 gallery=''.join(f'<article><p>{html.escape(x["instruction"])}</p><a href="{x["image"]}"><img width="100%" src="{x["image"]}"></a></article>' for x in cases)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PR26 main bearing loss</title><body style="max-width:1600px;margin:auto"><h1>主Belief角度一致性实验</h1><p>TRAIN图红星GT仅事后；地图轴不等于已核验地理北，自动关系标签非人工真值。</p><a href="REPORT_zh.md">中文报告</a><img width="100%" src="loss_curves.png"><img src="navigation_comparison.png">'+gallery+'</body>')
 print(json.dumps(dict(Val_Seen=metrics,paired=paired),indent=2))
if __name__=='__main__':main()
