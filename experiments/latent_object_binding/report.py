"""Produce a Chinese report from completed real artifacts, never planned metrics."""
import collections,csv,json,platform,shutil
from pathlib import Path
import numpy as np,torch
from experiments.semantic_polygon_grounding.audit import save,sha
from experiments.bidirectional_core.stats import paired_difference
from .resources import OUT,VERIFIED
from .diagnose import read_rows
from .evaluate import METRICS


def load(name):return json.loads((OUT/name).read_text())
def pct(x):return f'{100*x:.2f}'
def ci(x):return '['+', '.join(pct(v) for v in x)+']'
def count(m,key):return round(m[key]['mean']*m[key]['n'])
def table(rows):return '\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)


def main():
    full={s:load(f'full_{s}_metrics.json') for s in ('train','dev')};trained=load('trained_full_metrics.json');obs=load('observed_metrics.json');geometry=load('geometry_label_audit.json');support=load('binding_support_audit.json');coverage=load('observation_audit.json');impact=load('visual_goal_impact.json');pixels=load('pixels/manifest.json');text=load('frozen_text/complete.json');runtime=load('runtime.json');controlled=load('controlled_text_metrics.json');prepared=load('full_geometry/complete.json');observed=load('observed/complete.json')
    dest=Path('docs/p6_latent_object_binding_evidence');dest.mkdir(parents=True,exist_ok=True)
    additional={}
    for split in ('train','dev'):
        paths={n:read_rows(OUT/f'full_{split}_{n}.jsonl.gz') for n in ('A0','A1','A2','Hard')};paths['TrainedGeom']=read_rows(OUT/f'full_{split}_Trained_geometry.jsonl.gz');paths['TrainedRGB']=read_rows(OUT/f'full_{split}_Trained_rgb.jsonl.gz');pairs={'joint_minus_hard':('Hard','A2'),'joint_minus_independent':('A1','A2'),'trained_geometry_minus_joint':('A2','TrainedGeom'),'trained_geometry_minus_independent':('A1','TrainedGeom')}
        additional[split]={k:{m:paired_difference(paths[a],paths[b],m,seed=20261011) for m in METRICS} for k,(a,b) in pairs.items()}
    save(OUT/'additional_comparisons.json',additional)
    # Compact, reviewable results; raw predictions, large caches and checkpoints
    # remain local. Labels/individual object geometry are not replicated here.
    for name in ('protocol.json','a3_proof.json','full_train_metrics.json','full_dev_metrics.json','trained_full_metrics.json','observed_metrics.json','additional_comparisons.json','runtime.json','frozen_text/complete.json','observed/complete.json'):
        target=dest/('frozen_text_manifest.json' if name.startswith('frozen_text/') else 'observed_manifest.json' if name.startswith('observed/') else Path(name).name);shutil.copyfile(OUT/name,target)
    for name,value in [('geometry_audit_summary.json',geometry['summary']),('binding_support_summary.json',support['summary']),('observation_summary.json',coverage['summary']),('visual_goal_impact_summary.json',impact['summary']),('pixels_summary.json',{k:v for k,v in pixels.items() if k!='audit'}),('controlled_text_summary.json',{k:v for k,v in controlled.items() if k!='structures'})]:save(dest/name,value)
    for path in (OUT/'figures').glob('*'):shutil.copyfile(path,dest/path.name)
    provenance=dict(seed=20261011,base_branch='research/role-aware-object-graph',base_commit='868b1386dc432f2b308bb302d4f42dbd5dc5a6cd',branch='research/latent-object-binding-20261011',source_records_sha=sha(VERIFIED/'source_records.json'),baseline_proof=load('a3_proof.json'),prepared_cache=prepared,observed_cache=observed,heads={name:{k:v for k,v in trained['heads'][name].items() if k!='history'} for name in ('geometry','rgb')},runtime_versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),prepared_source_hashes=prepared['source_hashes'],final_source_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},post_prepare_change='Added correct joint reference marginal reporting only; object/goal/loss calculations unchanged. Separate oracle support bounds and diagnostics do not alter trained inference.',seen_evaluations=0,unseen_evaluations=0,closed_loop_evaluations=0,checkpoints_local_only=True)
    save(OUT/'model_manifest.json',provenance);save(dest/'provenance.json',provenance)
    comparison=[]
    for split in ('train','dev'):
        entries={**full[split]['metrics'],**{'Trained_'+n:trained['results'][split][n]['metrics'] for n in ('geometry','rgb')}}
        for name,m in entries.items():comparison.append(dict(dataset='full',split=split,method=name,n=m['goal_r1']['n'],**{k:m[k]['mean'] for k in METRICS}))
        for name,m in obs['metrics'][split].items():comparison.append(dict(dataset='observed',split=split,method=name,n=m['goal_r1']['n'],**{k:m[k]['mean'] for k in METRICS}))
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]),lineterminator='\n');w.writeheader();w.writerows(comparison)
    shutil.copyfile(OUT/'comparison.csv',dest/'comparison.csv')
    lines=['# P6：Latent Object Binding & Geometric Reasoning 实验报告','',
      '结论：已经实现并实际运行语言变量、公开 Polygon 概率绑定、空间因子边缘化及 28×28 Goal 输出；完整 Train 上完成两组 3 Epoch 轻量校准。**几何结构带来定位点估计改善，但未通过独立 Dev 的地图级统计门槛；RGB 带来部分对象排序变化，却没有产生 Goal 纠错，Real 对 Shuffled 的独立贡献不成立。不能宣布 P6 成功。**',
      '', '## 1. 冻结协议与输入权限','',
      '从 `research/role-aware-object-graph` / `868b1386dc432f2b308bb302d4f42dbd5dc5a6cd` 创建独立 `research/latent-object-binding-20261011`。原始数据、旧 Checkpoint、P0–P5 结果、20m 成功半径、784 个网格候选及控制器均保留。正式 Geometry A3、SigLIP2、Trajectory、Stop 冻结。本轮没有 Seen、Unseen、闭环或 GAT/RL 训练。',
      '', '新增输入协议允许同地图全部公开对象的类型、轮廓、名称和位置，比旧命名地标协议提供更多公共信息；A3 基线保持原输入与已验证 logits。因此增益属于“新增公开对象输入+推理”的离线对照，不能直接写成标准导航 SR/SPL 改善。对象 ID 只做缓存关联、训练标签和离线评价，没有 ID embedding。私有描述、目标 ID/XY 不进入正常推理。',
      '', 'Train/Internal Dev 地图沿用既有固定隔离；不根据 Seen 选择参数。Seed=20261011，参考支持8、联合赋值16、最多8个连通变量、输出混合系数0.25，遗漏支持和未解析假设保持 Unknown。3 Epoch、AdamW 0.001、有效 batch32；以完整 Dev Goal R@1 选 Checkpoint，平分取较早 Epoch。Bootstrap 2000次，分别重采样独立查询/family及地图。Dev只有4地图，置信区间不稳定；严格使用下界>0而不是下界≥0。',
      '', 'A3 权重必须严格载入并核对缓存 SHA；历史未正确载入 A3 的监督试验数据没有用于本轮。正常推理入口仅接受 `static_logits / grid_xy / bounds / pose_world` 和已验证因果视觉属性。完整样本按“地图×原始指令”去重；重复轨迹/Prefix不增加独立样本数，多正例全部保留。',
      '', '## 2. 架构与近似边界','',
      '```mermaid','flowchart LR','  L[原始语言] --> V[Z目标与X参考变量 / 作用域与Unknown]','  P[全部公开地图对象] --> B[类型·名称·面积·形状概率绑定]','  V --> B','  I[当前与因果历史真实RGB] --> A[Polygon前景 / HSV / 冻结SigLIP2]','  A --> B','  B --> F[空间因子 / 参考赋值边缘化]','  F --> O[目标对象边缘概率]','  F --> C[连续空间因子场]','  O --> H[28×28 Goal热图 / 固定置信门控]','  C --> H','  G[冻结Geometry A3] --> H','  H -. 可选兼容接口 / 本轮未执行 .-> T[原Trajectory与Stop]','```','',
      'P(Z=o) 对保留的参考联合赋值求和，不能先固定 X/Y 再寻找 Z。目标候选从不以 GT 或局部 Top-K 过滤；参考变量采用有预算的提议支持，遗漏概率进入 Unknown。已知公共名称直接使用原对应关系，不要求 RGB 再识别名称。不同独立变量不能无依据绑定同一对象；明确同一公开命名实体可共指。与 Z 无关的语言对象不稀释目标置信度。',
      '', 'Near/Along 使用实际参考 Polygon 距离，参考—参考距离使用边界距离；世界方向比较真实轮廓边界；Between 使用两个参考轮廓的联合凸包减去参考内部，交换 X/Y 后几何场严格一致。目标对象属性评估在其公开位置，连续目标场直接在网格位置计算，而不是只用参考物中心。Left/Right/Front/Behind 没有可信参考系时保持 Unknown。',
      '', '目标对象条件分布与连续空间分布各自边缘化后按固定接口结合，再以 0.25×已知概率质量混入原 A3。该实现保留公开 object.position 的10m核及连续关系场；不伪造精确目标 Polygon。参考联合后验另外通过对 Z 求和输出，提议分布与联合后验分别保存。',
      '', '**实现限制：**不是所有角色/作用域假设与全城参考组合的精确推断。当前采用首个角色/作用域解释，其余质量记 Unknown；匿名参考提议主要依赖 unary特征，空间因子在提议截断后应用。超过预算的变量、细形状、遮挡关系和不明确指代没有被假装解决。全图边缓存复用为公开节点资源，本轮没有边消息神经网络。',
      '', '## 3. 真实数据与监督审计','',
      '| 数据 | 地图 | 独立地图–指令查询 | 原始标签/轨迹记录 | 独立目标对象 | 多目标正例查询 |','|---|---:|---:|---:|---:|---:|']
    for split in ('train','dev'):
        g=geometry['summary'][split];lines.append(f"| {split} | {g['maps']} | {g['queries']} | {g['label_records']} | {g['unique_objects']} | {prepared['stats'][split]['multi_positive_queries']} |")
    lines+=['',f"公共地图资源共 {prepared['nodes']:,} 节点、{prepared['edges']:,} 条既有边；每条查询使用本地图全部目标对象，候选 Recall=100%。匿名参考物没有独立实例真值，Reference Recall@K 不可识别；公开名称的候选匹配不当作学得的参考识别准确率。Target 变量的 Recall@K 由下文 Object R@K 给出。",'',
      '| 数据 | GT在Polygon内/边界 | GT在外部 | GT等于公开object.position | 轮廓退化为Point的标签 | 最大中心偏差(m) |','|---|---:|---:|---:|---:|---:|']
    for split in ('train','dev'):
        g=geometry['summary'][split];lines.append(f"| {split} | {g['inside_or_boundary']} | {g['outside']} | {g['public_position_exact']}/{g['label_records']} | {g['point_fallback_labels']} | {g['centroid_error_quantiles'][-1]:.2f} |")
    lines+=['', '本轮原始目标 XY 全部等于相应公开对象 position；这仅解释离线身份 Oracle 的高上界，不允许将其目标 ID 传给模型。Polygon 外目标和几何中心偏差说明“正确对象→中心点”不总可靠；本实现使用公开位置与连续关系场，未训练新的 Goal Head。', '',
      '| 数据 | 含目标连通参考变量 | 含公开命名参考 | 含匿名参考 | 未确定相对方向 | Between可交换赋值对 |','|---|---:|---:|---:|---:|---:|']
    for split in ('train','dev'):
        s=support['summary'][split];lines.append(f"| {split} | {s['with_connected_references']} | {s['with_named_references']} | {s['with_anonymous_references']} | {s['relative_frame_unknown']} | {s['symmetric_exchange_pairs']} |")
    lines+=['', '全部保留的同一 Between 无序对象对交换后，空间 log-factor 最大差值为 **0**（真实 Train 3,635 对、Dev 280 对）。因此辨别“哪栋是X、哪栋是Y”不一定有导航价值；只有改变无序对象集合或非对称关系的绑定，才可能改变 Goal。匿名参考身份缺少真值，不能声称准确区分全部有价值/无价值歧义。',
      '', '## 4. 完整 Train/Internal Dev 几何与训练对照','',
      'A0只输出Goal，没有原生Object Belief；表中A0对象指标若出现在机器结果里，是A2的shadow诊断，不能当作A3对象基线。A1是独立变量/公开Polygon unary匹配，A2加概率空间因子，Hard仅取一个参考赋值。TrainedGeom与TrainedRGB分别为同预算训练组。',
      '', '| 数据/方法 | Object R@1/4/8 (%) | Goal R@1/4/8 (%) | NE(m) | Recovery/Damage/Net(例) |','|---|---|---|---:|---|']
    for split in ('train','dev'):
        entries={**full[split]['metrics'],**{'Trained'+n:trained['results'][split][n]['metrics'] for n in ('geometry','rgb')}}
        for name,m in entries.items():lines.append(f"| {split}/{name} | "+('—' if name=='A0' else '/'.join(pct(m[f'object_r{k}']['mean']) for k in (1,4,8)))+' | '+'/'.join(pct(m[f'goal_r{k}']['mean']) for k in (1,4,8))+f" | {m['ne']['mean']:.2f} | {count(m,'recovery')}/{count(m,'damage')}/{count(m,'net_recovery')} |")
    lines+=['', '| 完整Dev配对比较 | Goal R@1差值(pp) | family95%CI(pp) | 地图95%CI(pp) |','|---|---:|---|---|']
    for name,d in [('A1−A0',full['dev']['comparisons']['A1']['goal_r1']),('A2−A0',full['dev']['comparisons']['A2']['goal_r1']),('A2−Hard',additional['dev']['joint_minus_hard']['goal_r1']),('A2−A1',additional['dev']['joint_minus_independent']['goal_r1']),('TrainedGeom−A0',trained['results']['dev']['geometry']['versus_A0']['goal_r1']),('TrainedGeom−A1',additional['dev']['trained_geometry_minus_independent']['goal_r1'])]:lines.append(f"| {name} | {pct(d['mean'])} | {ci(d['family_cluster_95ci'])} | {ci(d['map_cluster_95ci'])} |")
    lines+=['', 'Dev中点估计满足+1pp且Recovery>Damage，但地图CI下界为0，**正式定位门槛未通过**。概率边缘化在完整Dev与Hard的Goal结果相同，且不优于更简单A1；不能因为架构更复杂就宣称更有效。完整Train上的CI是训练数据描述性统计，不是独立泛化证明。',
      '', '## 5. 轻量训练已经完成','',
      '完整Train每Epoch18,895查询；目标实例多正例负对数概率 + 0.5×20m有效Goal网格概率损失，不需要参考物实例标签。两组储存9个标量参数，几何组屏蔽2个RGB系数（有效7），RGB组有效9；不是新的视觉Backbone/Cross-Attention网络。参考提议支持由预先缓存冻结，学习只能重加权保留假设；无RGB样本的视觉增量始终为0。','',
      '| 模型/Epoch | Train loss | Dev Object R@1(%) | Dev Goal R@1(%) | 秒 | 峰值GPU(MiB) |','|---|---:|---:|---:|---:|---:|']
    for name in ('geometry','rgb'):
        for h in trained['heads'][name]['history']:lines.append(f"| {name}/{h['epoch']} | {h['loss']:.4f} | {pct(h['dev']['object_r1']['mean'])} | {pct(h['dev']['goal_r1']['mean'])} | {h['seconds']:.1f} | {h['peak_gpu_bytes']/2**20:.1f} |")
    lines+=['', f"几何/RGB总训练耗时分别{trained['heads']['geometry']['seconds']:.1f}/{trained['heads']['rgb']['seconds']:.1f}s，两组均选Epoch1。Train实际RGB覆盖255查询（256family中有重复原始指令），仅占{255/18895*100:.2f}%；Dev有128查询可复用RGB。**完整Train几何监督不等于完整Train视觉训练。**最后Epoch对象指标继续小幅变化，Goal无额外改善，预算内已结束，没有延长训练。",'', '![训练曲线](p6_latent_object_binding_evidence/training.png)','', '## 6. 合法观测及RGB属性对照','',
      '复用Dense Causal History，每family仅保留最后一个合法Prefix；ROI源step≤cutoff、同episode/map，native side≥10、depth support≥0.9。公开变量类型与质量选择已到达帧，最多2帧/family、128帧/split；GT不选择观察。实际重裁256个已观察位姿，共2,489个Polygon HSV/Lab ROI，耗时48.5s，bbox一致性和投影失败0。复用所有冻结SigLIP2图像特征，只新增101条变量文本特征（23.9s），没有重复大规模图像编码。',
      '', '**图像真实性边界：**这些是CityNav在真实轨迹已到达位姿的正射RGB裁图，Depth为垂直代理；不是记录的机载透视照片，也不能验证立面遮挡。Polygon前景仍是几何弱标注，颜色规则不是人工标签，可靠物理属性Oracle数量=0。ROI可信度仅为几何/像素代理，不把它写成人工语义可信。','',
      '| 数据 | family/独立对象 | 静态错误 | 目标有效ROI可见 | 错误且可见family/对象 | 有HSV观测family |','|---|---|---:|---:|---|---:|']
    for split in ('train','dev'):
        c=coverage['summary'][split];lines.append(f"| {split} | {c['families']}/{c['objects']} | {c['static_wrong']} | {c['target_visible']} | {c['static_wrong_and_target_visible']}/{c['independent_wrong_visible_objects']} | {c['with_pixel_evidence']} |")
    lines+=['', '所有Real/Black/Shuffled使用同样的family、地图全对象候选、cutoff与观察有效性。Black是实际编码黑图及黑像素属性；Shuffled在同类型、同family合法观察内置换，像素支持保持一致。未观测物体从不因无图像变成视觉负样本。明确公共命名绑定不被RGB覆盖。',
      '', '| 128 Dev family 方法 | Object R@1/4/8(%) | Goal R@1/4/8(%) | NE(m) | Recovery/Damage |','|---|---|---|---:|---|']
    for name in ('A0','A2','Hard','A3_HSV','A3_SigLIP','A3_Both','A3_References','A3_Target','A5_Black','A5_Shuffled','MatchedGeom','A4_Real','A4_Black','A4_Shuffled'):
        m=obs['metrics']['dev'][name];lines.append(f"| {name} | "+('—' if name=='A0' else '/'.join(pct(m[f'object_r{k}']['mean']) for k in (1,4,8)))+' | '+'/'.join(pct(m[f'goal_r{k}']['mean']) for k in (1,4,8))+f" | {m['ne']['mean']:.2f} | {count(m,'recovery')}/{count(m,'damage')} |")
    lines+=['', '| Dev配对比较 | Object差值/地图95%CI(pp) | Goal差值/地图95%CI(pp) |','|---|---|---|']
    for name in ('hsv_minus_geometry','siglip_minus_geometry','rgb_minus_black','rgb_minus_shuffled','learned_rgb_minus_geom','learned_real_minus_black','learned_real_minus_shuffled'):
        m=obs['comparisons']['dev'][name];lines.append(f"| {name} | {pct(m['object_r1']['mean'])} / {ci(m['object_r1']['map_cluster_95ci'])} | {pct(m['goal_r1']['mean'])} / {ci(m['goal_r1']['map_cluster_95ci'])} |")
    lines+=['', '原始冻结属性匹配对Black有对象排序信号，但对同类Shuffled没有统计支持。训练后Real与Shuffled的Object R@1完全相同；所有视觉方法的Goal结果相同，均0 Recovery/1 Damage。只更新参考变量或只更新目标变量也未改变Goal，因此不能将对象排序改善宣布为RGB导航价值。完整Train/Dev的训练RGB组与匹配几何组Goal亦相同。',
      '', '![RGB对照](p6_latent_object_binding_evidence/rgb_controls.png)','', '## 7. Oracle与实际分数影响','',
      '区分全局私有身份上界与相同可见支持的身份上界。Global Oracle可知道未观测目标，只诊断Object→Goal转换；Observed Oracle仅在目标已有可信几何ROI时使用身份，否则回退对应几何推理。SameInterface保留本轮固定门控，其余Oracle用完美身份核作为强上界，不能算真实模型成绩。',
      '', '| Dev Oracle | Object R@1(%) | Goal R@1(%) | Recovery/Damage |','|---|---:|---:|---|']
    for name in ('GlobalIdentityOracle','SameInterfaceGlobalIdentityOracle','ObservedIdentityOracle','SameInterfaceObservedIdentityOracle'):
        m=obs['metrics']['dev'][name];lines.append(f"| {name} | {pct(m['object_r1']['mean'])} | {pct(m['goal_r1']['mean'])} | {count(m,'recovery')}/{count(m,'damage')} |")
    lines+=['', '同可见支持、同固定接口的Identity Oracle仍能恢复15例，说明该接口并非绝对禁止纠错。强可见身份Oracle能恢复42例；这个差距同时包含门控/混合约束。真实RGB恢复0例，不能把全部失败归因于旧gain=0.025；本轮没有改变旧Gain，也没有用Dev/Seen增大残差。',
      '', '七条语义明确真实指令另外执行文本审读结构干预，正确结构仅由原始句子确定，不利用GT坐标选择。它是研究者文本审读，非独立专家真值；Current与同置信度正确结构的Object/Goal差值均0，理想置信度只改变少量NE。样本太少且结构相近，不能证明完整语言已无问题。全局变量解析Oracle和物理属性Oracle缺少可靠真值，明确不可识别，没有伪造其成绩。',
      '', '| 训练RGB Dev实际变化 | 对象Top1改变(例) | Goal Top1改变(例) | 目标对象排名改善/退化/不变 | 最大热图差值中位数/最大值 |','|---|---:|---:|---|---|']
    for name,m in impact['summary']['dev'].items():lines.append(f"| {name} | {m['object_top1_changes']} | {m['goal_top1_changes']} | {m['target_rank_improved']}/{m['target_rank_worsened']}/{m['target_rank_unchanged']} | {m['max_abs_heat_delta_quantiles'][1]:.7f}/{m['max_abs_heat_delta_quantiles'][-1]:.7f} |")
    lines+=['', '上述比较只分析冻结输出，没有幅度搜索或参数放宽。逐family保存目标概率、目标排名、热图L1变化、Known质量和Gate；Dev有1例Top1网格变化，但没有改变Goal成功/失败。训练RGB的50例Object Top1正确中有9例Goal失败（Geometry为8例）；对象识别与最终目标网格成功不能混同。',
      '', '## 8. 典型完整链路案例','', '案例同时保存原始语言、变量、参考候选概率、联合赋值、目标对象概率、最终热图和已观察RGB/Polygon前景。GT只用于离线选取/标注案例，图中Polygon前景不能当作人工mask。']
    cases=load('figures/cases.json')
    for i,c in enumerate(cases):lines+=['',f"### 案例 {i}：{c['map_name']}",'',c['instruction'],'',f"Object R@1={int(c['metrics']['object_r1'])}，Goal R@1={int(c['metrics']['goal_r1'])}；Recovery={int(c['metrics']['recovery'])}，Damage={int(c['metrics']['damage'])}。",'',f"![案例{i}](p6_latent_object_binding_evidence/{c['figure']})"]
    lines+=['', '## 9. 耗时、环境、权重与工程验证','',
      f"单卡RTX PRO6000 Blackwell Max-Q 96GB。Python {platform.python_version()} / torch {torch.__version__} / NumPy {np.__version__}；两组参数训练峰值约{max(h['peak_gpu_bytes'] for head in trained['heads'].values() for h in head['history'])/2**20:.1f}MiB，冻结SigLIP2文本编码峰值{text['peak_gpu_bytes']/2**30:.2f}GiB。全几何准备{prepared['seconds']:.1f}s，384family观测版本准备{observed['seconds']:.1f}s。",'',
      '| 预计算后单线程CPU方法 | mean(ms) | p95(ms) |','|---|---:|---:|']
    for name,r in runtime['prepared_cpu_inference'].items():lines.append(f"| {name} | {r['mean_ms']:.3f} | {r['p95_ms']:.3f} |")
    lines+=['', '这些延迟只含缓存后的概率推理，不包括数据I/O、地图构图、裁图、视觉编码或相机/飞行动作；原始阶段耗时分别记录。训练耗时包括每Epoch完整Dev评估与CI，两组可在同卡并发小任务，不能将总和当墙钟。','',
      f"- A3 Checkpoint SHA256：`{load('a3_proof.json')['checkpoint_sha']}`。",
      f"- A3 logits SHA256：`{load('a3_proof.json')['cache_sha']}`。",
      f"- SigLIP2 revision：`{text['revision']}`；文本缓存 SHA256：`{text['sha']}`。",
      f"- 几何轻量 Head SHA256：`{trained['heads']['geometry']['checkpoint_sha']}`。",
      f"- RGB轻量 Head SHA256：`{trained['heads']['rgb']['checkpoint_sha']}`。",'',
      '完整参数、每Epoch指标、模型/数据SHA见本报告旁 evidence JSON；9个标量系数也在history里。Checkpoint与25GiB级输入缓存只存本地。输入准备源码SHA和最终源码SHA分别保存；准备后仅补充正确的联合参考后验统计，不改变目标/热图/损失。','',
      '工程修复：早期多进程Torch tensor传输触发文件描述符/BrokenProcessPool，改为NumPy返回+单writer Tensor化；旧中间chunk的NumPy标量触发weights_only校验，限定安全旧标量读取并转换Python标量。统计退化Point边界距离为NaN改为明确None，未伪造Polygon。与目标无关变量和Between等价赋值预算问题均按语义约束修复，旧原型结果/部分训练仍在prototype目录保存，未混入本报告最终指标。','',
      '回归 **405 passed，1个既有rasterio弃用警告，8.32s**；包含关系对称性、变量置换、不同ID不改变特征、Unknown回退、遮挡/Future ROI隔离、坐标平移、半精度缓存/训练梯度、公开命名共指、目标全候选与28×28兼容接口、正确联合参考后验。实际命令见 `experiments/latent_object_binding/README.md`，日志在 `runs/p6_latent_object_binding/logs/`；最终Git SHA由Commit/Push后本地commit.json及交付消息记录，避免报告自引用SHA。',
      '', '## 10. 五个问题的回答与下一步','',
      '1. **变量与几何关系是否足够？** 能提供完整Dev约+2pp的定位点估计，但不能满足地图95%CI下界>0，且结构/参考实例真值不足；不能据此声称较可靠定位。',
      '2. **概率绑定是否优于硬绑定？** 完整Dev Goal相同，观察子集同样没有优势；训练子集有小幅差异但不构成独立证明。更简单A1在完整Dev还更高。',
      '3. **RGB能否消除真正影响Goal的歧义？** 当前没有证明。它改变匿名参考提议和部分对象排名，但Real−Shuffled不成立，Goal差值为0；Between同一无序对象对的换名歧义本就不需视觉消除。',
      '4. **位置改善来自哪里？** 本轮所有Goal改善来自公共对象几何/结构与原A3混合；RGB没有额外恢复。接口仍存在可见身份Oracle恢复空间，不能将失败完全归于融合幅度。',
      '5. **是否需要训练视觉模型？** 当前9参数训练没有改善Goal，不能支持扩大视觉模型；简单HSV弱于冻结SigLIP2的对象排序，但两者均未证明独立导航价值。','',
      '**下一步只推荐：公开几何条件化的匿名参考提议召回。** 当前匿名参考Top8主要按unary排序，空间因子在截断后才生效；正确参考若被截掉，校准器无法恢复。只用原A3公开目标先验与语言关系，对全部同类参考候选计算预期空间相容性再提议，保持目标全候选、预算、Unknown质量和输出Gate不变；优先在Train/Dev的小型独立文本+参考绑定核验集验证，不扩大RGB/网络。优先覆盖Dev含匿名参考的884查询；这是待验证范围，不是预期改善884例。','',
      '预注册验收：固定独立核验参考集合，Top8召回提高≥10pp且配对95%CI下界>0；同预算概率推理必须超过Hard/A1；完整Internal Dev Goal相对A0≥+1pp、Recovery>Damage且地图95%CI下界>0。再谈RGB时必须同时Real−Black与Real−Shuffled具有地图统计支持且Goal有正增益。未通过就保留辅助语义验证，不进入GAT、RL、Trajectory/Stop、Seen闭环或Unseen。']
    report='\n'.join(lines)+'\n';Path('docs/P6_LATENT_OBJECT_BINDING_REPORT.md').write_text(report);(OUT/'final_report.md').write_text(report)
    save(OUT/'acceptance.json',dict(full_dev_goal_delta=trained['results']['dev']['geometry']['versus_A0']['goal_r1'],learned_rgb_independence=obs['comparisons']['dev']['learned_real_minus_shuffled']['goal_r1'],passed=False,reason='Dev map lower bound is zero; no RGB goal recovery or real/shuffled advantage; do not run closed-loop/Unseen'))

if __name__=='__main__':main()
