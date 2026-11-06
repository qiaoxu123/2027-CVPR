"""Report real TRAIN map-heldout matched bearing experiment. No navigation."""
import argparse,csv,json,math,textwrap,shutil,re
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import file_sha256
from graphnav.polygon_neighborhood_bearing import PublicPolygonNeighborhood

NAMES={'A':'text_name_control','B':'public_polygon_neighborhood','B_shuffled':'B_shuffled'}


def stats(rows):
 if not rows:return dict(N=0)
 return dict(N=len(rows),Top1=float(np.mean([r['top1'] for r in rows])),Top2=float(np.mean([r['top2'] for r in rows])),MAE=float(np.mean([r['error_deg'] for r in rows])),median=float(np.median([r['error_deg'] for r in rows])),NLL=float(np.mean([r['nll'] for r in rows])))


def main():
 p=argparse.ArgumentParser()
 for k in ['run','public-objects','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();r=a.run;o=a.output;o.mkdir(parents=True,exist_ok=True)
 raw=json.loads((r/'REPORT.json').read_text());conf=raw['protocol'];data=json.loads((r/'data_manifest.json').read_text());extra=json.loads((r/'diagnostics.json').read_text());records=data['records']
 paths={'A':r/'text_name_control_heldout_rows.jsonl','B':r/'public_polygon_neighborhood_heldout_rows.jsonl','B_shuffled':r/'B_shuffled_rows.jsonl'};rows={name:list(map(json.loads,path.read_text().splitlines())) for name,path in paths.items()}
 maps={name:{(v['index'],v['reference_index']):v for v in values} for name,values in rows.items()};keys=list(maps['A']);assert set(keys)==set(maps['B'])==set(maps['B_shuffled'])
 for key in keys:
  assert len({maps[n][key]['true_bin'] for n in maps})==1
 def physical(i):return ':'.join(str(records[i]['episode_id']).split(':')[:2])
 pairdata=[];queries=defaultdict(list);physical_groups=defaultdict(list);map_groups=defaultdict(list)
 for key in keys:
  x=maps['A'][key];y=maps['B'][key];i,j=key;ref=records[i]['landmarks'][j]
  v=dict(index=i,physical_target=physical(i),map=x['map'],reference_index=j,anchor_id=ref['id'],anchor_name=ref['name'],n_refs=x['n_refs'],instruction=records[i]['instruction'],GT_bin=x['true_bin'])
  for name in maps:
   q=maps[name][key]
   for z in ['predicted_bin','top1','top2','error_deg','nll','probabilities']:v[name+'_'+z]=q[z]
  v['fix']=int(not x['top1'] and y['top1']);v['damage']=int(x['top1'] and not y['top1']);v['delta']=y['top1']-x['top1'];pairdata.append(v);queries[i].append(v);physical_groups[physical(i)].append(v['delta']);map_groups[x['map']].append(v['delta'])
 def interval(group):
  # Resample independent physical targets/maps, preserving all correlated queries/anchors.
  sums=np.asarray([sum(v) for v in group.values()],float);counts=np.asarray([len(v) for v in group.values()],float);rng=np.random.default_rng(42);result=[]
  for k in range(3000):ix=rng.integers(0,len(sums),len(sums));result.append(sums[ix].sum()/counts[ix].sum())
  return np.quantile(result,[.025,.975]).tolist()
 summary=dict(protocol=conf,heldout_all={n:stats(v) for n,v in rows.items()},heldout_single={n:stats([x for x in v if x['n_refs']==1]) for n,v in rows.items()},heldout_multi={n:stats([x for x in v if x['n_refs']>1]) for n,v in rows.items()},fit_diagnostics=extra,anchor_fixes=sum(v['fix'] for v in pairdata),anchor_damage=sum(v['damage'] for v in pairdata),Top1_delta=stats(rows['B'])['Top1']-stats(rows['A'])['Top1'],delta95CI_physical=interval(physical_groups),delta95CI_maps=interval(map_groups),physical_groups=len(physical_groups),heldout_maps=len(map_groups),query_fixes=sum(bool(np.mean([v['delta'] for v in q])>0) for q in queries.values()),query_damage=sum(bool(np.mean([v['delta'] for v in q])<0) for q in queries.values()),query_tied=sum(bool(np.mean([v['delta'] for v in q])==0) for q in queries.values()),query_macro_Top1={n:float(np.mean([np.mean([maps[n][(v['index'],v['reference_index'])]['top1'] for v in q]) for q in queries.values()])) for n in maps},by_map={m:{n:stats([v for v in rs if v['map']==m]) for n,rs in rows.items()} for m in sorted(map_groups)})
 train_hist=np.asarray(extra['fit_label_histogram']);majority=int(train_hist.argmax());summary['fit_prior_control']=dict(majority_bin=majority,heldout_majority_Top1=float(np.mean([v['true_bin']==majority for v in rows['A']])),uniform_Top1=1/8,uniform_Top2=2/8,uniform_NLL=math.log(8),fit_histogram=train_hist.tolist(),heldout_histogram=np.bincount([v['true_bin'] for v in rows['A']],minlength=8).tolist())
 proxies={'left_right':r'\b(left|right)\b','front_behind':r'\b(front|behind|back|ahead)\b','indirect_lexical_proxy':r'\b(between|row|rows|parking|lot|intersection)\b','cardinal_lexical_proxy':r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b'}
 summary['unverified_lexical_strata']={s:{n:stats([v for v in rs if re.search(pattern,records[v['index']]['instruction'],re.I)]) for n,rs in rows.items()} for s,pattern in proxies.items()}
 # GT only AFTER predictions, to audit information actually available in the public neighborhood.
 public=PublicPolygonNeighborhood(json.loads(a.public_objects.read_text()),radius_m=conf['radius_m'],max_neighbors=conf['neighbors']);nearby={};coverage=[]
 for v in pairdata:
  i,j=v['index'],v['reference_index'];rec=records[i];ref=rec['landmarks'][j];mk=(rec['map_name'],str(ref['id']))
  if mk not in nearby:
   indexed=public._map_index(mk[0]);anchor=indexed[mk[1]][0];options=sorted([(float(np.linalg.norm(item[0]-anchor)),oid) for oid,item in indexed.items() if oid!=mk[1] and np.linalg.norm(item[0]-anchor)<=conf['radius_m']]);nearby[mk]=[x[1] for x in options[:conf['neighbors']]]
  target_id=str(rec['episode_id']).split(':')[1];v['public_target_in_neighborhood']=target_id in nearby[mk];coverage.append(v['public_target_in_neighborhood'])
 summary['posthoc_target_neighborhood_coverage']=dict(anchor_pairs=len(coverage),target_id_in_public_neighbors=sum(coverage),fraction=float(np.mean(coverage)),note='GT identity used only to audit already selected public polygons after inference; never used for selecting or weighting neighbors')
 coverage_lookup={(v['index'],v['reference_index']):v['public_target_in_neighborhood'] for v in pairdata}
 summary['posthoc_target_coverage_groups']={g:{n:stats([v for v in rs if coverage_lookup[(v['index'],v['reference_index'])]==flag]) for n,rs in rows.items()} for g,flag in [('present',True),('absent',False)]}
 from scripts.analyze_instruction_landmark_association import PublicNames
 from scripts.prepare_unified_v2_landmark_relation_ablation import extract_pairs
 matcher=PublicNames(public.catalog);attached={}
 for i in queries:
  rec=records[i];attached[i]=extract_pairs(rec['instruction'],matcher.match(rec['map_name'],rec['instruction']))
 typed={}
 for tag,types,pattern in [('road_left_right',{'TrafficRoad','Footpath','Bridge'},r'left|right'),('building_front_behind',{'Building'},r'front|behind|back|ahead')]:
  keep=set()
  for v in pairdata:
   ref=records[v['index']]['landmarks'][v['reference_index']]
   if ref.get('object_type') in types and any(str(ref['id']) in map(str,p['public_ids']) and re.search(pattern,p['phrase']) for p in attached[v['index']]):keep.add((v['index'],v['reference_index']))
  typed[tag]={n:stats([v for v in rs if (v['index'],v['reference_index']) in keep]) for n,rs in rows.items()}
 summary['typed_attached_relation_proxies']=typed
 from shapely.geometry import Polygon,Point
 audit_rows=[]
 for v in pairdata:
  rec=records[v['index']];ref=rec['landmarks'][v['reference_index']];oid=str(rec['episode_id']).split(':')[1];anchor=np.asarray(public.catalog[rec['map_name']][str(ref['id'])]['contour']);bounds=np.asarray(rec['world_bounds']);gt=bounds[:2]+np.asarray(rec['target_xy'])*(bounds[2:]-bounds[:2]);delta=gt-anchor.mean(0)
  audit_rows.append(dict(target_is_anchor=oid==str(ref['id']),target_geometry_present=v['public_target_in_neighborhood'] or oid==str(ref['id']),GT_inside_anchor=Polygon(anchor).covers(Point(gt)),GT_within1m=np.linalg.norm(delta)<=1.))
 summary['posthoc_label_and_geometry_audit']={k:sum(bool(v[k]) for v in audit_rows) for k in audit_rows[0]}
 effective=[]
 for (m,aid),ids in nearby.items():
  table=public._map_index(m);origin=table[aid][0]
  if ids:effective.append(max(float(np.linalg.norm(table[oid][0]-origin)) for oid in ids))
 summary['posthoc_label_and_geometry_audit'].update(N=len(audit_rows),unique_anchor_neighborhoods=len(nearby),actual_farthest_neighbor_median_m=float(np.median(effective)),actual_farthest_neighbor_mean_m=float(np.mean(effective)))
 summary['checkpoint_SHA256']={n:file_sha256(r/(name+'_last.pt')) for n,name in NAMES.items() if n in ['A','B']}
 assert torch.load(r/'text_name_control_last.pt',weights_only=True,map_location='cpu')['initial_digest']==torch.load(r/'public_polygon_neighborhood_last.pt',weights_only=True,map_location='cpu')['initial_digest']
 (o/'summary.json').write_text(json.dumps(summary,indent=2))
 with (o/'paired_anchors.jsonl').open('w') as f:
  for row in pairdata:f.write(json.dumps(row)+'\n')
 taskrows=[]
 for i,q in queries.items():taskrows.append(dict(index=i,physical_target=physical(i),map=records[i]['map_name'],instruction=records[i]['instruction'],anchors=len(q),**{n+'_Top1':float(np.mean([v[n+'_top1'] for v in q])) for n in maps}))
 with (o/'paired_queries.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(taskrows[0]));w.writeheader();w.writerows(taskrows)
 fig,axs=plt.subplots(1,3,figsize=(15,4))
 for ax,(name,rs) in zip(axs,rows.items()):
  matrix=np.zeros((8,8),int)
  for v in rs:matrix[v['true_bin'],v['predicted_bin']]+=1
  probs=matrix/np.maximum(matrix.sum(1,keepdims=True),1);ax.imshow(probs,vmin=0,vmax=1,cmap='Blues')
  for i in range(8):
   for j in range(8):ax.text(j,i,f'{matrix[i,j]}',ha='center',va='center',fontsize=6,color='white' if probs[i,j]>.5 else 'black')
  ax.set(title=name+' held-out',xlabel='Predicted bin (0=map+X, CCW)',ylabel='GT bin');ax.set_xticks(range(8));ax.set_yticks(range(8))
 fig.tight_layout();fig.savefig(o/'confusion.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for name,arm in [('A','text_name_control'),('B','public_polygon_neighborhood')]:
  hist=json.loads((r/(arm+'_history.json')).read_text());axs[0].plot([v['epoch'] for v in hist],[v['mean_loss'] for v in hist],'-o',label=name)
  fit=extra[arm+'_fit']['all']['top1'];held=summary['heldout_all'][name]['Top1'];axs[1].bar([name+' fit',name+' held'],[fit*100,held*100])
 axs[0].set(xlabel='Epoch',ylabel='Equal-query training CE');axs[0].legend();axs[1].set(ylabel='Top1 (%)');fig.tight_layout();fig.savefig(o/'training.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4));mps=sorted(map_groups);positions=np.arange(len(mps))
 for j,n in enumerate(['A','B','B_shuffled']):ax.bar(positions+(j-1)*.25,[summary['by_map'][m][n]['Top1']*100 for m in mps],width=.25,label=n)
 ax.set_xticks(positions,mps,rotation=15);ax.set(ylabel='Top1 (%)',title='Held-out TRAIN maps, no navigation');ax.legend();fig.tight_layout();fig.savefig(o/'maps.png',dpi=160);plt.close(fig)
 chosen=[];used=set()
 for condition in ['fix','damage','left_right','front_behind','indirect_lexical_proxy']:
  count=0
  for v in pairdata:
   cond=v.get(condition,False) if condition in ['fix','damage'] else bool(re.search(proxies[condition],v['instruction'],re.I)) and not v['B_top1']
   if cond and v['physical_target'] not in used:
    chosen.append(v);used.add(v['physical_target']);count+=1
    if count==3:break
 images=[]
 catalog=public.catalog
 for v in chosen:
  i,j=v['index'],v['reference_index'];rec=records[i];ref=rec['landmarks'][j];poly=np.asarray(catalog[rec['map_name']][str(ref['id'])]['contour']);center=poly.mean(0);bounds=np.asarray(rec['world_bounds']);gt=bounds[:2]+np.asarray(rec['target_xy'])*(bounds[2:]-bounds[:2]);fig,axs=plt.subplots(1,2,figsize=(12,6))
  for oid in nearby[(rec['map_name'],str(ref['id']))]:
   xy=np.asarray(catalog[rec['map_name']][oid]['contour']);axs[0].plot(*np.vstack([xy,xy[0]]).T,color='gray',alpha=.6,lw=.6)
  axs[0].plot(*np.vstack([poly,poly[0]]).T,color='cyan',lw=2,label='Public anchor');axs[0].plot(*gt,'r*',ms=10,label='GT (offline)')
  theta=math.atan2(gt[1]-center[1],gt[0]-center[0]);axs[0].quiver(*center,math.cos(theta)*75,math.sin(theta)*75,angles='xy',scale_units='xy',scale=1,color='red',label='GT direction')
  for n,color in [('A','orange'),('B','blue')]:
   theta=v[n+'_predicted_bin']*math.pi/4;axs[0].quiver(*center,math.cos(theta)*65,math.sin(theta)*65,angles='xy',scale_units='xy',scale=1,color=color,label=n)
   axs[1].plot(np.arange(8)*45,v[n+'_probabilities'],'-o',color=color,label=n)
  axs[0].set(xlim=(center[0]-170,center[0]+170),ylim=(center[1]-170,center[1]+170),xlabel='World map X',ylabel='World map Y');axs[0].set_aspect('equal');axs[0].legend(fontsize=7)
  axs[1].axvline(v['GT_bin']*45,color='r',ls='--',label='GT bin');axs[1].set(xlabel='Map-plane angle (degrees)',ylabel='Probability',title=f"A error {v['A_error_deg']:.1f}; B error {v['B_error_deg']:.1f}");axs[1].legend()
  fig.suptitle('\n'.join(textwrap.wrap(f"TRAIN {i} / {ref['name']}: "+v['instruction'],110)),fontsize=9);fig.tight_layout(rect=(0,0,1,.87));fn=f'case_{i:05d}_{j:02d}.png';fig.savefig(o/fn,dpi=140);plt.close(fig);images.append(dict(file=fn,index=i,reference_index=j,instruction=v['instruction'],anchor=ref['name']))
 (o/'cases.json').write_text(json.dumps(images,indent=2))
 # Entirely data-driven tables. Do not turn proxy language categories into ground truth.
 lines=['# Polygon邻域几何增强8方向预测：真实GPU结果','', '**结论：本轮未证实显著提升。只使用TRAIN，不加载导航器，不使用RGB、Val或Test；未修改GeomBelief。**','', '## 数据、隔离与公平性','',f"完整TRAIN {conf['expected_count']}条；拟合{conf['n_train']}，地图留出{conf['n_holdout']}；可评分留出5530条、22005个指令—Polygon对；170条无公开地标，不能定义方位标签，保留在覆盖率分母中。五个留出地图：{', '.join(conf['split']['heldout_maps'])}。",'', '地图、规范化重复指令和相同公开目标轮廓均无交叉；目标身份仅用于隔离/事后审计，不参与公开Anchor或邻域选择。所有合法绑定Polygon分别预测，不按GT挑实例。多Polygon按每条指令等权CE训练，主表按anchor对微平均，另报query等权结果。', '', 'A/B同一NeighborhoodBearingHead、同参数量、同初始化哈希、seed42、3epochs、batch32、lr1e-4、AdamW wd0.01、clip1、FP32，仅geometry_enabled不同。固定第3轮，未用留出选权重或改超参数。BERT冻结，真实最大125Token完整编码，96Token旧缓存仅对未截断条目复用，实测新旧BERT差异见diagnostics。', '', '邻域仅按公开Anchor顶点均值，150m内最近24个合法Polygon选择；使用相对坐标、距离、形状和类别。可能自然包含目标对象，但不输入其GT身份。Polygon无视觉Mask真值承诺。角标签是工程XY中从+X逆时针8类，非已验证地理north；间接关系同样是目标位置监督，不是语言方向真值。','', '## 留出指标','', '|组别/方法|指令—Polygon对|Top1|Top2|MAE|中位角误差|NLL|','|---|---:|---:|---:|---:|---:|---:|']
 for group,key in [('all','heldout_all'),('single','heldout_single'),('multi','heldout_multi')]:
  for n,m in summary[key].items():lines.append(f"|{group}/{n}|{m['N']}|{m['Top1']*100:.2f}%|{m['Top2']*100:.2f}%|{m['MAE']:.2f}°|{m['median']:.2f}°|{m['NLL']:.4f}|")
 lines += ['',f"全量B−A={summary['Top1_delta']*100:.3f}个百分点；物理目标分组bootstrap95%区间={np.asarray(summary['delta95CI_physical'])*100}个百分点；地图分组区间={np.asarray(summary['delta95CI_maps'])*100}个百分点。仅seed42、仅5个留出地图，不能把单次微小上升说成显著泛化收益。",'',f"Anchor级纠正{summary['anchor_fixes']}、破坏{summary['anchor_damage']}。Query平均anchor命中率上升{summary['query_fixes']}条、下降{summary['query_damage']}条、持平{summary['query_tied']}条；无标签170条不参与准确率比较。Query等权Top1：{json.dumps(summary['query_macro_Top1'])}。",'', 'B_shuffled使用同一冻结B模型，只打乱留出指令与场景对应，不改名称embedding、Polygon、GT标签。5700条文本对应均改变；不能把它等同于语法正确的关系词反事实。','', '## 分地图结果','', '|地图|对数|A Top1|B Top1|变化(pp)|','|---|---:|---:|---:|---:|']
 for m,methods in summary['by_map'].items():lines.append(f"|{m}|{methods['A']['N']}|{methods['A']['Top1']*100:.2f}%|{methods['B']['Top1']*100:.2f}%|{(methods['B']['Top1']-methods['A']['Top1'])*100:+.2f}|")
 lines += ['', '## 失败机制与边界','', '1. 几何分支确有非零梯度且训练内改善，但未迁移到未见地图。训练集A/B命中率、留出差距见training.png和diagnostics.json。不能只凭Loss下降称方位预测有效。', '2. 留出B与打乱语言B表现接近，当前预测对正确语言—几何对应缺乏可靠利用；NLL还应与均匀分布ln8=2.0794对比。不能把14%左右的Top1包装成足够可用的方向能力。', '3. 邻域表示匿名化、最多24个、单层聚合，不含RGB颜色、邻居名称或目标实例匹配。文字中的具名中间地标、车辆外观/序数约束未必可与这些形状Token对应，front/behind/left/right也可能缺少公开参考朝向。词面分层统计只是代理，不能当作人工验证的直接/间接标签。', '4. 单Polygon明确退化，多Polygon权重较大使微平均小幅上升；必须同时看query等权结果、分组和各地图。公开目标邻域覆盖率只作事后可用信息审计，不参与选择对象。', '5. 不能从负结果证明所有原始语言本身信息不足，亦不能证明Polygon没有方向信息；目前证据是现有表示/训练没有学习到可泛化的语言—匿名几何关联。三轮/单种子只验证本实现，未穷尽模型或数据上限。', '', '## 旧方向头为何不能作为成功基线','', '旧PR25的41.13%是参与拟合的Birmingham TRAIN统计，非地图留出；其头仅将全文Token均值和地标名拼接，没有Polygon邻域，融合还受zero-init gain/tanh幅度限制。本轮用实际Token Attention、独立地图留出和独立方向CE隔离测试，不复用旧分数作为泛化对照。', '', '## 下一步判断','', '暂不接导航，也不增加训练轮次追分。先在TRAIN核验能否把指令中的命名中间物体/目标类别绑定到邻域对象，并明确哪些相对方向在公开信息下可识别；再用同样地图留出验证。不要加入未经验证的参考系规则，不将GT对象ID用于推理。', '', '## 工程与复现','', '修复测试语法、非法自交Polygon、旧1m标签过滤、96Token截断、优化器/RNG恢复及跨图重复隔离。专项包含GT变化不改几何、geometry-off不受形状扰动、梯度、mask、重复隔离和0/360角误差测试。CPU日志见CPU_tests.txt。实际Checkpoint、完整训练日志、BERT缓存均留服务器；Github仅提交脚本、配置、日志摘要、逐条预测和可视化。', '', '独立审查工具已实际调用；如日志为workspace routing失败则未获得外部审查结果，不能称外部审查通过。', '', '服务器运行目录：`'+str(r)+'`。不读取正式Val/Test、不计算导航指标。']
 lines += ['', '## 道路左右、建筑前后专项（自动附着代理）','', '下表仅要求关系词组词面附着到相应公开名称，不证明关系主语就是最终目标，也不推断参考系。','', '|组|对数|A Top1|B Top1|','|---|---:|---:|---:|']
 for tag,arms in summary['typed_attached_relation_proxies'].items():
  if arms['A']['N']:lines.append(f"|{tag}|{arms['A']['N']}|{arms['A']['Top1']*100:.2f}%|{arms['B']['Top1']*100:.2f}%|")
  else:lines.append(f"|{tag}|0|不可估计|不可估计|")
 lines += ['', '原GT对象自然进入已选24邻域的比例为25.20%（按anchor对）；该身份只在预测完成后核查，绝不用于筛选/训练特征。受150m和24个对象预算、同名多Polygon及间接关系影响，几何输入经常不包含真实目标；这属于当前输入表示的覆盖限制，不是语言本身错误的证明。25.20%仅指邻居列表，不包括目标恰为Anchor自身的情况；含Anchor自身的完整统计见summary的posthoc_label_and_geometry_audit。', '', '**最终判断：没有显著提升，也不足以可靠预测。** 微平均A14.39%、B14.61%（+0.21pp），但query等权15.37%→13.37%，single13.40%→5.88%；B的训练内36.41%不能泛化到新地图。B打乱语言后14.42%，未出现明显损失。更直接的瓶颈是当前文字—匿名Polygon表示未建立可迁移对应，不能据此声称公开几何毫无信息或原指令普遍不可定位。']
 envpath=o/'ENVIRONMENT.json'
 if envpath.exists():
  env=json.loads(envpath.read_text());lines += ['', '## 实际GPU训练开销','',f"GPU={env['GPU']}，PyTorch={env['Torch']}，CUDA={env['CUDA']}；A/B均{env['parameters']}参数、1518updates。A训练{env['A_train_seconds']:.2f}s、B训练{env['B_train_seconds']:.2f}s；含缓存/数据和两组留出评价的完整正式运行{env['full_run_seconds']:.2f}s。每进程CUDA峰值约0.477GiB（不含其他作业）；原GeomBelief差异检查为空。"]
 lines += ['', '## 进一步输入审计','', '名义150m邻域在最近24个对象预算下，实际最远邻居距离的中位数仅27.23m，可能被密集小对象占满，缺少指令中更远的具名中间场景。22005个对中仅5571个包含公开目标自身几何（包括25个目标恰为Anchor）；这些只作事后审计，绝未据此选择对象。GT落在Anchor Polygon内1303个，距顶点均值不足1m的18个；位置角标签不保证具有明确的语言方向含义。', '', '即使公开目标已自然出现在邻居列表的5546个对上，B Top1为12.44%，仍低于A的14.98%。所以不能只归因于邻域范围；当前语言与匿名几何Token的关联和跨地图泛化同样没有验证成功。']
 (o/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
 for src,dst in [(r/'protocol.json','protocol.json'),(r/'diagnostics.json','diagnostics.json')]:shutil.copyfile(src,o/dst)
 for arm in ['text_name_control','public_polygon_neighborhood']:
  for suffix in ['history','gradient']:shutil.copyfile(r/(arm+'_'+suffix+'.json'),o/(arm+'_'+suffix+'.json'))
 page='<!doctype html><meta charset="utf-8"><title>Polygon neighborhood bearing</title><style>body{max-width:1300px;margin:auto;font:16px sans-serif}img{width:100%}section{border-bottom:1px solid #ccc;padding:15px}</style><h1>TRAIN地图留出：方向预测</h1><p>无RGB、无导航；红箭头/星为离线GT，橙色A，蓝色B，青色公开Anchor，灰色公开邻域。点击查看原图。</p><a href="REPORT_zh.md">中文报告</a> · <a href="paired_queries.csv">逐任务结果</a><img src="training.png"><img src="confusion.png"><img src="maps.png">'
 import html
 for im in images:page+=f'<section><h3>{im["index"]}: {html.escape(im["anchor"])}</h3><p>{html.escape(im["instruction"])}</p><a href="{im["file"]}"><img loading="lazy" src="{im["file"]}"></a></section>'
 (o/'index.html').write_text(page);print(json.dumps({k:summary[k] for k in ['heldout_all','query_macro_Top1','anchor_fixes','anchor_damage','Top1_delta','delta95CI_physical','delta95CI_maps']},indent=2))
if __name__=='__main__':main()
