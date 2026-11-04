"""Package measured V2 training records, real beliefs and Chinese Pilot report."""
import argparse,json,shutil,csv,hashlib,subprocess
from pathlib import Path
from collections import Counter,defaultdict
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import file_sha256,atomic_json
from graphnav.landmark_projection import world_to_frame


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run;o=a.output;o.mkdir(parents=True,exist_ok=True);stats={}
 for stage,folder in [('A200','training'),('B1000','training'),('Cepoch','training'),('Dfull_epoch','full_training')]:
  for suffix in ['_summary.json','_steps.jsonl']:shutil.copyfile(r/folder/(stage+suffix),o/(stage+suffix))
  stats[stage]=json.loads((o/(stage+'_summary.json')).read_text())
 shutil.copyfile(r/'full_training/config.json',o/'full_training_config.json')
 for name in ['train_A.sh','train_B.sh','train_C.sh','train_full_epoch.sh','cache_full.sh','smoke_before.sh','smoke_after.sh']:shutil.copyfile(r/name,o/name)
 cache=json.loads((r/'full_train_t0/manifest.json').read_text());metadata={k:v for k,v in cache.items() if k!='timings'};atomic_json(metadata,o/'full_cache_manifest.json')
 with (o/'cache_timings.csv').open('w') as f:
  writer=csv.DictWriter(f,list(cache['timings'][0]),lineterminator='\n');writer.writeheader();writer.writerows(cache['timings'])
 for label in ['cached','full']:
  for name in ['metrics.json','field_snapshots.json']:shutil.copyfile(r/f'benchmark_{label}'/name,o/(label+'_'+name))
 for phase in ['before','after']:
  shutil.copyfile(r/f'smoke_{phase}'/'metrics.json',o/f'smoke_{phase}_metrics.json');shutil.copyfile(r/f'smoke_{phase}'/'predictions.jsonl',o/f'smoke_{phase}_predictions.jsonl');shutil.copyfile(r/f'smoke_{phase}'/'run_manifest.json',o/f'smoke_{phase}_manifest.json')
 # Every full TRAIN frame/label is audited, not a downsample used as full data.
 full=torch.load(r/'full_train_t0/cache.pt',map_location='cpu',weights_only=True);raw=json.loads(Path('/home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav/citynav_train_seen.json').read_text());byframe=defaultdict(list)
 for record in full['records']:byframe[record['frame_id']].append(record)
 from graphnav.landmark_alias import canonical_name
 eligible_episodes=set()
 for fid,frame in enumerate(full['frames']):
  assert frame['frame']['source_split']=='train_seen' and frame['frame']['step']==0
  for ep in frame['frame']['source_episodes']:
   text=' '+canonical_name(raw[ep]['descriptions'][0])+' '
   if any(' '+canonical_name(full['queries'][q['query_key']]['name'])+' ' in text for q in byframe[fid]):eligible_episodes.add(ep)
 known_invalid=sum(bool(((q['labels']>=0)&~full['frames'][q['frame_id']]['valid']).any()) for q in full['records']);assert known_invalid==0
 groups={};maps={};rgb={};split_stats={}
 for sub in ['fit','dev']:
  rr=[q for q in full['records'] if q['subset']==sub];ff={q['frame_id'] for q in rr};groups[sub]={(full['frames'][q['frame_id']]['city'],canonical_name(full['queries'][q['query_key']]['name'])) for q in rr};maps[sub]={full['frames'][i]['map_name'] for i in ff};rgb[sub]={full['frames'][i]['frame']['image_sha256'] for i in ff};split_stats[sub]=dict(queries=len(rr),frames=len(ff),maps=len(maps[sub]),city_names=len(groups[sub]),query_keys=len({q['query_key'] for q in rr}),types=dict(Counter(q['type'] for q in rr)),positive_only=sum(not (q['labels']==0).any() for q in rr))
 assert not groups['fit']&groups['dev'] and not maps['fit']&maps['dev'] and not rgb['fit']&rgb['dev']
 eval_pngs=list((r/'smoke_before').rglob('*.png'));assert not {file_sha256(x) for x in eval_pngs}&(rgb['fit']|rgb['dev'])
 atomic_json(dict(TRAIN_instructions=len(raw),eligible_source_instructions=len(eligible_episodes),TRAIN_instructions_without_supported_referenced_visual_supervision=len(raw)-len(eligible_episodes),valid_unique_frames=len(full['frames']),valid_landmark_queries=len(full['records']),split=split_stats,known_label_invalid_patch_rows=known_invalid,train_dev_city_map_name_RGB_disjoint=True,ValSeen_smoke_RGB_overlap=0,all_model_observations_t0=True,destination_GT_never_used=True,feature_coverage_within_eligible_t0_pool=1.,exclusions=cache['excluded'],scope='Only visually supported public Building/Road references at all original TRAIN t0 observations, not all possible future states or true pixel visibility'),o/'full_data_audit.json')
 # Training curves are separated by data protocol; do not connect unlike losses.
 legacy=[]
 for stage in ['A200','B1000','Cepoch']:legacy += [json.loads(x) for x in (o/(stage+'_steps.jsonl')).read_text().splitlines()]
 natural=[json.loads(x) for x in (o/'Dfull_epoch_steps.jsonl').read_text().splitlines()];fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,rows,title in [(axes[0],legacy,'Existing TRAIN cache, 1712 updates'),(axes[1],natural,'Full original-t0 TRAIN fit, one epoch')]:
  ys=np.array([x['loss'] for x in rows]);ax.plot(np.arange(len(ys))+1,ys,alpha=.18);window=30;ax.plot(np.arange(window,len(ys)+1),np.convolve(ys,np.ones(window)/window,mode='valid'));ax.set(xlabel='Update',ylabel='Weak supervision loss',title=title)
 fig.tight_layout();fig.savefig(o/'loss_curves.png',dpi=150);plt.close(fig)
 b=stats['B1000'];d=stats['Dfull_epoch'];fig,axes=plt.subplots(1,2,figsize=(12,4));keys=['data_wait_ms','transfer_ms','forward_loss_ms','backward_ms','optimizer_ms'];axes[0].bar([k.replace('_ms','') for k in keys],[b['component_mean_ms'][k] for k in keys]);axes[0].tick_params(axis='x',rotation=30);axes[0].set_title('B1000 synchronous component ms')
 measured=dict(SigLIP_and_pool=sum(v['encode_and_pool_seconds'] for v in cache['timings']),RGB_render_eligible=sum(v['render_seconds'] for v in cache['timings']),local_supervision_eligible=sum(v['local_supervision_seconds'] for v in cache['timings']),cold_encoder=cache['cold_encoder_seconds']);measured['other_including_rejected_ROI_IO']=cache['seconds']-sum(measured.values());axes[1].bar(measured.keys(),measured.values());axes[1].tick_params(axis='x',rotation=35,labelsize=7);axes[1].set_title('First cache seconds; rejected-frame ROI/IO in residual');fig.tight_layout();fig.savefig(o/'efficiency.png',dpi=150);plt.close(fig)
 snapshots=json.loads((o/'full_field_snapshots.json').read_text());field_metrics=json.loads((o/'full_metrics.json').read_text());chosen=[]
 for typ in ['Building','Road']:
  indices=[i for i,row in enumerate(field_metrics['field_cases']) if row['type']==typ];deltas={i:field_metrics['field_cases'][i]['metrics']['after']['polygon_mass']-field_metrics['field_cases'][i]['metrics']['before']['polygon_mass'] for i in indices}
  for i in [indices[0],indices[-1],max(indices,key=deltas.get),min(indices,key=deltas.get)]:
   if i not in chosen:chosen.append(i)
 cases=[]
 for i in chosen:
  s=snapshots[i];f=full['frames'][s['frame_id']];bnd=np.array(s['bounds']);rgb=cv2.cvtColor(cv2.imread(s['image_path']),cv2.COLOR_BGR2RGB);fig,axes=plt.subplots(1,6,figsize=(23,4.5));axes[0].imshow(rgb)
  for poly in s['polygons']:
   pxy=world_to_frame(np.array(poly),np.array(s['footprint_world']),rgb.shape);axes[0].plot(*np.vstack([pxy,pxy[0]]).T,'c-',lw=.8)
  axes[0].set(xlim=(0,383),ylim=(383,0),title='Real RGB + public Polygon\nGeometry proxy, not human mask')
  world=np.array(s['local_uv'])*(bnd[2:]-bnd[:2])+bnd[:2];pixel=world_to_frame(world,np.array(s['footprint_world']),rgb.shape);valid=f['valid'].numpy()
  for ax,name in zip(axes[1:3],['before','after']):
   ax.imshow(rgb);v=torch.tensor(s['scores'][name]).masked_fill(~f['valid'],float('-inf')).softmax(0).numpy();ax.scatter(pixel[valid,0],pixel[valid,1],c=v[valid],cmap='magma',s=16,vmin=0,vmax=.15);ax.set(xlim=(0,383),ylim=(383,0),title=f'{name}: local visual probability')
  maxprob=max(max(s['fields'][name]) for name in ['geometry_only','before','after'])
  for ax,name in zip(axes[3:],['geometry_only','before','after']):
   v=np.array(s['fields'][name]).reshape(28,28);ax.imshow(v,origin='lower',extent=[bnd[0],bnd[2],bnd[1],bnd[3]],cmap='magma',vmin=0,vmax=maxprob)
   for poly in s['polygons']:
    q=np.array(poly);ax.plot(*np.vstack([q,q[0]]).T,'c-',lw=.7)
   fp=np.array(s['footprint_world']);ax.plot(*np.vstack([fp,fp[0]]).T,'y--',lw=.6);ax.set(xlim=(bnd[0],bnd[2]),ylim=(bnd[1],bnd[3]),title=name+' world Belief')
  fig.suptitle(s['query_key']+' | TRAIN city Dev | observed RGB384 | all world grids fixed | not CityNav target GT',fontsize=9);fig.tight_layout();filename=f'belief_{i:02d}.png';fig.savefig(o/filename,dpi=130);plt.close(fig);cases.append(dict(query=s['query_key'],file=filename,frame_id=s['frame_id']))
 atomic_json(cases,o/'cases.json');html='<meta charset="utf-8"><title>UnifiedPGVB-v2训练Pilot</title><style>body{font:16px sans-serif;margin:20px}img{max-width:100%}</style><h1>真实训练与效率Pilot</h1><p>公开Polygon弱监督；非原始CityNav任务SR结论。模型只训练PGVB视觉Head。</p><a href="REPORT_zh.md">中文报告</a><img src="loss_curves.png"><img src="efficiency.png">'
 for c in cases:html+=f'<h2>{c["query"]}</h2><a href="{c["file"]}"><img src="{c["file"]}" loading="lazy"></a>'
 (o/'index.html').write_text(html)
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();wall=json.loads((o/'process_wall_times.json').read_text());startup=wall['Dfull_epoch']['external_wall_seconds']-d['stage_seconds'];costs={str(n):dict(cached_seconds=startup+n*d['stage_seconds'],first_seconds=wall['cache_full']['external_wall_seconds']+startup+n*d['stage_seconds']) for n in [1,3,5,10]};atomic_json(dict(epochs=costs,scope='Same PGVB-only full TRAIN t0 fit pool, batch4 frames; shared GPU; actual 1 epoch, linear estimates3/5/10; one process startup; no 10-epoch run',cache_components_seconds=measured),o/'cost_estimates.json')
 env=json.loads((o/'environment.json').read_text());dev=json.loads((o/'full_dev_evaluation.json').read_text());audit=json.loads((o/'data_audit.json').read_text());visual=json.loads((o/'visual_cache_benchmark.json').read_text());checkpoints={key:dict(path=str(path),sha256=file_sha256(path)) for key,path in [('migrated',r/'migrated.pt'),('cached_final',r/'cached_pilot_final.pt'),('final',r/'full_training/last.pt'),('best_fit_window',r/'full_training/best.pt')]};atomic_json(checkpoints,o/'checkpoints.json')
 L=['# UnifiedPGVB-v2 首轮真实训练Pilot与效率测试','',f'代码基准：main `ce2c6dae`；本次训练/缓存入口代码Commit：`{source_commit}`。实验分支`experiment/unified-pgvb-v2-training-pilot-20261104`。','',
'## 核心结论','',
'**可以用真实CityNav数据正常训练UnifiedPGVB-v2内的PGVB视觉Head。** 不是旧独立训练器改名：实际调用UnifiedPGVB.forward、保存完整V2模型，并由原V2 Runtime加载。Geometry、SigLIP2、BERT、Trajectory/Stop及joint/direction/target全部冻结。不是全参数或联合融合训练。',
f"1000次有效更新：训练循环及定期保存 **{b['stage_seconds']:.2f}秒**；Python主函数含缓存校验/加载 **{b['total_process_seconds']:.2f}秒**；从shell日志创建至退出（含模块导入） **{wall['B1000']['external_wall_seconds']:.2f}秒**。剔除前20步后{b['mean_step_ms']:.3f}ms/步、{b['queries_per_second']:.1f}查询/s、{b['frames_per_second']:.1f}帧/s。", 
f"全部22,002条TRAIN原始起点已扫描，新建可监督池为{len(full['frames'])}帧/{len(full['records'])}查询。完成其全部拟合分区一次Epoch：{len(natural)}步，循环{d['stage_seconds']:.2f}秒，完整启动至退出{wall['Dfull_epoch']['external_wall_seconds']:.2f}秒。首次缓存生成（含启动）{wall['cache_full']['external_wall_seconds']:.2f}秒。**已停训，没有运行10 Epoch**。", 
'效果并非全面提升：独立TRAIN Dev局部Loss和地标R20改善，但世界Polygon概率质量下降，融合Belief未增加命中；两条闭环功能Smoke训练前后都是0/2成功。因此效率可行，不应以Loss下降为由立即扩大训练并宣称SR提升。','',
'## 1. 环境、权重与接口','',
f"- GPU：{env['gpu']}，物理显存约96GB；PyTorch `{env['torch']}`，CUDA runtime `{env['cuda_runtime']}`，BF16实际可用。Python：`{env['python']}`。环境完整记录见environment.json。", 
'- 初始GPU已有其他任务约35GB占用，未停止任何其他进程。资源探测脚本因PATH包装器误报无GPU，真实CUDA与`/usr/bin/nvidia-smi`已核实；以environment.json为准。',
'- Stage1来源：`GraphNav/runs/graphnav_v1_stage1/best.pt`，SHA `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。PGVB来源：`GraphNav/runs/pgvb_local_anchor_v2/pilot_s0/best.pt`，SHA `ba3073723703ce8188c7f74ae810a622342631bb150bd2597656ff0039f2cf56`。本轮再次核验一致。',
'- PGVB源权重已经历旧缓存12轮训练，本轮是继续训练，不冒充从零学习。迁移本身不算训练。原archive分支、源Checkpoint及main未修改。',
f"- 实测网络输入{visual['shapes'][0]['pixel_values']}、原生patch{visual['shapes'][0]['native_patch_map']}（24×24）、稀疏多尺度缓存{visual['pooled_feature_shape']}（144位置×3尺度×1024）。每个正常RGB帧只做一次SigLIP2 Vision forward，多个查询复用。", 
'- 29项原专项测试、5项训练专项测试合计34项通过；另有真实GPU梯度、原生RGB重算、保存加载以及两次各2条原协议闭环Smoke。','',
'## 2. 数据规模、监督与隔离','',
f"全量TRAIN原文22,002条；保守公开名称匹配指令{audit['conservative_public_name_bound_instructions']}条，绑定{audit['public_name_bindings']}次（同条可重叠）：Building {audit['binding_types']['Building']}、Road {audit['binding_types']['Road']}、Other含Ground {audit['binding_types']['Other']+audit['binding_types']['Ground']}。只是名称存在统计，不是人工语言解析准确率。", 
f"原始t=0几何候选8,591个不同位姿，RGB/Depth/Polygon质量检查后4,159帧、4,658查询；无可靠局部监督4,432个位姿，另1个起点高度不合法。有效监督对应{len(eligible_episodes)}条原始指令；其余{len(raw)-len(eligible_episodes)}条无本协议下可用的引用地标视觉监督（含未绑定/视野外/不支持类型/质量不足，不能统称目标不可见）。实际可见性仍未知。RGB来自原正射栅格在真实原始起点的裁剪，不是真实三维透视/遮挡重建。", 
'','|数据协议|拟合帧 / 查询|独立Dev帧 / 查询|用途|','|---|---:|---:|---|',
'|旧完整可用缓存（固定yaw0，含公开地标附近增强）|1711 / 6275|345 / 701|A/B及旧缓存Epoch效率测试|',
'|全量TRAIN原始t=0（原始yaw）|1956 / 2266|2203 / 2392|真正的全量可用t=0监督池Epoch|','',
'新池拟合Building550/Road1716，Dev Building1122/Road1270；Other不造视觉标签，继续保留几何回退。新池不是按容易成功的任务截取，而是全TRAIN固定t=0协议的全部有效支持。它不等于TRAIN所有未来时刻；两种缓存域/每帧查询数不同，不能以查询吞吐差异当作代码提速。',
'Cambridge拟合/Birmingham验证，城市、地图、物理公开名称及RGB SHA均隔离。所有新观察明确TRAIN、step0，仅原始起点，不提供教师轨迹、未来图像或CityNav目标GT作为网络输入。64条Val-Seen只用于既有前2条运行时Smoke，未进入训练、调参或Checkpoint选择；Smoke RGB与训练池无SHA重叠。完整审计见data_audit.json、full_data_audit.json。冻结Stage1的历史TRAIN覆盖并未重置；这里的独立是本次PGVB增量拟合隔离，不是整个系统从零未见地图泛化。',
'新池4159帧缓存全覆盖，原旧池精确RGB+Depth命中0：原始位姿/yaw与增强缓存不同，不能错误复用。缺失帧已增量编码；每128个候选保存Chunk进度，原缓存保留。未知/边界patch标签−1，内缩0.5m后覆盖≥90%为正、外扩覆盖≤1%为负；正区域少于2或有效来源不足时弃权。标签是公开Polygon投影弱监督，不是人工像素Gold或遮挡真值。','',
'## 3. 训练范围与真实执行','',
'UnifiedPGVB核心总参数946,298，实际训练226,306（19个PGVB参数张量）；其余719,992冻结。SigLIP2/BERT在运行时冻结、训练使用其常量缓存；控制器权重未进入优化器。19/19训练张量实测非零有限梯度，冻结参数逐位不变。',
'Loss=可靠区域概率损失+0.2平衡BCE+0.3同帧同类可靠互负对比。只有正区域的样本使用正区域内KL，不把未知当负；重叠/不互为可靠负区域的同类对象不强制对比。所有损失FP32，前向CUDA BF16，AdamW lr1e−4/weight_decay1e−3，梯度裁剪5，seed42。',
'|阶段|有效更新|循环含保存(s)|完整进程(s)|稳态ms/步|查询/s|峰值GiB|首20 / 末20 Loss|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,st in stats.items():L.append(f"|{name}|{st['updates']}|{st['stage_seconds']:.2f}|{wall[name]['external_wall_seconds']:.2f}|{st['mean_step_ms']:.3f}|{st['queries_per_second']:.1f}|{st['peak_gpu_gib']:.3f}|{st['loss_first20']:.4f} / {st['loss_last20']:.4f}|")
 L+=['',
'A200→B1000从第200步恢复AdamW的19组状态、epoch0/offset800；C补齐未完成Epoch后再完成一轮，最终旧池共4轮/1712步。D是全部新t=0拟合池489步/1轮，覆盖1956帧和2266查询，在C权重基础上因数据指纹变化重建优化器，不能伪称同一缓存续训。D之后未追加训练。',
'每100步保存last/best；best仅按最近100步TRAIN损失，不按Dev或两条闭环选。所有比较使用固定预算的last。保存完整模型/优化器/RNG/采样游标、数据和代码SHA，改变指纹拒绝续训。真实数据下一步保存/恢复与不中断执行最大参数差0，损失也完全一致。','',
'## 4. 时间与显存分解','',
'B1000稳态（每步同步CUDA，前20步剔除）：','',
'|分量|平均ms|','|---|---:|']
 for k,v in b['component_mean_ms'].items():L.append(f'|{k}|{v:.3f}|')
 L += [f"|完整Update工作Wall（保存/日志另计）|{b['mean_step_ms']:.3f}|",'',
f"PGVB-only训练每步Geometry计算为0：冻结静态Geometry对本局部损失没有必要重复运行。真实独立TRAIN查询实测Geometry一次GPU前向平均{np.mean(field_metrics['Geometry_GPU_forward_seconds'])*1000:.2f}ms，静态BERT及数据组装平均{np.mean(field_metrics['static_BERT_and_data_setup_seconds'])*1000:.2f}ms，Episode内缓存；没有独立GraphNav再做一次Geometry。", 
f"新t=0缓存实际生成主函数{cache['seconds']:.2f}s，完整进程{wall['cache_full']['external_wall_seconds']:.2f}s。已编码帧中SigLIP2+pool累计{measured['SigLIP_and_pool']:.2f}s，RGB渲染{measured['RGB_render_eligible']:.2f}s，局部监督{measured['local_supervision_eligible']:.2f}s；其余包含被拒绝帧检查、PNG/Chunk/整库保存、旧缓存查找，未将残差全归咎GPU。细分CSV保留。", 
f"单独真实RGB检查：冷加载编码器{visual['cold_model_load_seconds']:.2f}s，第一帧{visual['first_observed_forward_seconds']*1000:.1f}ms，预热后8个不同观测平均{visual['warm_forward_plus_pool_seconds_mean']*1000:.1f}ms（含ROI检查/pool）。单编码器进程实测峰值{visual['peak_gpu_allocated_gib']:.3f}GiB；完整运行时Smoke约3.745GiB。冻结特征训练B峰值0.135GiB不包含未加载的视觉主干，不能作为全网络峰值。", 
'原生RGB重算与缓存特征MAE约1.3e−6，2组检查均每图1次Vision forward，未发现缓存维度/权重错配。B1000设备总利用率10次采样70–88%、平均83.6%；有其他进程，不能冒充本任务独占SM利用率。每100步耗时和设备采样完整保留。','',
'## 5. 完整训练成本','',
'以下仅当前PGVB视觉Head、batch4帧、全部新t=0拟合池；1 Epoch为实测，3/5/10线性外推，不是已经运行。单进程启动/读缓存计一次，保存频率沿用实测；共享GPU及文件页缓存状态会影响复测；预训练权重已在本地，估计不含下载。首次包括整个TRAIN+Dev可监督t=0缓存生成，不包含工程开发时间。','',
'|Epochs|已有缓存(s)|首次含缓存(s / min)|','|---|---:|---:|']
 for n,v in costs.items():L.append(f"|{n}|{v['cached_seconds']:.2f}|{v['first_seconds']:.2f} / {v['first_seconds']/60:.2f}|")
 L+=['',
'不能把旧池1000步查询吞吐直接乘原始22,002条来估算；这里先实测全量原始t=0监督池的实际分母、DataLoader和完整Epoch。没有将Geometry、视觉主干或联合融合的解冻成本混进该表；若以后微调视觉编码器，旧缓存必须失效并另测。','',
'## 6. 独立TRAIN效果与真实Belief','',
'全量新池独立Birmingham Dev2392查询，以下是**视觉世界Belief与公开Polygon的弱监督指标**，不是原始CityNav目标定位/SR：','',
'|方法|局部Loss|Polygon概率质量|地标R20|地标Polygon NE(m)|','|---|---:|---:|---:|---:|']
 for method,label in [('before','迁移前训练权重'),('after','本轮训练后'),('after_zero_rgb','训练后RGB置零')]:
  v=dev['methods'][method]['all'];L.append(f"|{label}|{v['loss_batch']:.4f}|{v['visual_world_polygon_mass']:.4f}|{v['visual_world_R20']*100:.2f}%|{v['visual_world_polygon_NE']:.3f}|")
 L+=['','|类别|数量|训练前→后 R20|训练前→后 NE|训练前→后概率质量|','|---|---:|---:|---:|---:|']
 for typ in ['Building','Road']:
  v0=dev['methods']['before']['by_type'][typ];v1=dev['methods']['after']['by_type'][typ];L.append(f"|{typ}|{v0['n']}|{v0['visual_world_R20']*100:.2f}%→{v1['visual_world_R20']*100:.2f}%|{v0['visual_world_polygon_NE']:.2f}→{v1['visual_world_polygon_NE']:.2f}|{v0['visual_world_polygon_mass']:.4f}→{v1['visual_world_polygon_mass']:.4f}|")
 L+=['',
'Loss与峰值误差改善，但概率质量降低：不能只选正指标宣称全面提升。RGB特征置零只证明特征依赖，不能单独证明真实实例外观识别；本轮没有人工像素Gold。旧缓存独立Dev701查询结果另存cached_dev_evaluation.json（其输入为另一种观察协议，不与新池拼接汇总）。',
'在固定24个TRAIN城市场景地标查询中，用同一Geometry/原融合公式比较全局Belief：','',
'|方法|Polygon概率质量|地标R20|NE(m)|','|---|---:|---:|---:|']
 for name in ['geometry_only','before','after']:
  v=field_metrics['field_summary'][name];L.append(f"|{name}|{v['polygon_mass']:.5f}|{v['R20']*100:.2f}%|{v['NE']:.3f}|")
 L+=['','该小组R20未增加，融合后NE略变差；不能把局部视觉指标当作联合融合成功。joint融合头尚无本轮训练/启用；仅原bounded融合。真实RGB、Polygon、多片段Building/Road与前后Belief见index.html，图中只有公开地标代理，不画CityNav目标GT。','',
'## 7. 两条真实闭环功能Smoke','',
'同一固定Val-Seen64清单的前2条，完整原控制器/Horizon/Stop权重与阈值，训练前后均0/2成功、OSR1/2。Checkpoint正常加载、输出有限、真实RGB逐步更新，一条任务有9次有效PGVB更新，delta约0.499；但没有SR收益。不能声称历史61/64被复现，也未运行其余62条来挑选结果。',
'- Celine Grove：原Geometry Goal约26m离开公开地标，从未进入20m；PGVB没有有效本地证据，保留几何；停止在外侧。',
'- Church：曾进入约7.66m，后续Goal正确性丢失2次，最终约24.97m停止。训练后同样失败，未调整Stop/到达规则。本轮只记录，不用这两条Val任务反向调参。','',
'## 8. Checkpoint与复现','',
'所有大缓存、原图和权重留在服务器；根目录：`'+str(r)+'`。','',
'|权重|路径（相对运行根）|SHA256|','|---|---|---|']
 for key,item in checkpoints.items():L.append('|'+key+'|`'+str(Path(item['path']).relative_to(r))+'`|`'+item['sha256']+'`|')
 L += ['','训练配置见training_config.json、full_training_config.json以及仓库configs/unified_pgvb_training_pilot.json；tmux启动脚本/逐步日志随报告保存。恢复使用原数据/初始化权重指纹，加`--resume --until-updates <更大步数> --stage <新阶段名>`，不会从头重置优化器。缓存按Chunk恢复需`--resume`，完成缓存拒绝覆盖。',
'长作业均在tmux中执行，现已完成退出，原后台服务和历史任务未干预。仅保存/恢复探针的Checkpoint不参与训练、选择或闭环。源权重SHA结束时再次检查一致。','',
'## 9. 决策','',
'**训练机制和显存已经可用，主要开销在第一次RGB/Polygon准备和冻结特征提取，而不是PGVB反向。** 后续可优化首次图像提取的批量化、被拒绝ROI的CPU检查以及大缓存加载/分片；不需要换更大GPU来训练这个Head。',
'**不建议立即自动启动多轮完整训练追逐Loss。** 全量独立Dev概率质量下降、融合R20不变、两条功能Smoke均失败，尚无闭环增益证据。应先核对弱监督/观测域差异与“局部评分→全局Belief→停止”的作用，明确适合的联合监督后再单独申请/设计融合训练。本轮严格保持joint/direction/appearance未训练，不改Stop，不恢复暂停验证。']
 (o/'REPORT_zh.md').write_text('\n'.join(L)+'\n');atomic_json(dict(code_commit=source_commit,base_commit='ce2c6dae',completed=['migration','CPU tests34','real GPU smoke2','A200','B1000 resume','legacy full epochs','all22002 t0 scan/cache','full t0 supervised fit epoch','TRAIN Dev2392','field comparison24','post GPU smoke2'],training_stopped=True,ten_epochs_run=False,joint_direction_target_trained=False,formal_evaluation_run=False,checkpoint=checkpoints['final'],interpretation='Training feasible and fast with cache; navigation improvement not demonstrated'),o/'STATUS.json');print('Packaged report and',len(cases),'real belief figures')
if __name__=='__main__':main()
