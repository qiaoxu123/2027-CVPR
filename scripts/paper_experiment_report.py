"""Narratives generated only after the offline evidence checks pass."""
from pathlib import Path
import csv
import json
import os

from .build_paper_experiments import (HERE, ROOT, OUT, MAIN, SPLITS, METHODS,
                                      load, sha, save, csvout, mdtable, git)


def write_reports(result):
    from scripts.count_polybelief_parameters import run as count_parameters
    parameters = count_parameters()
    diff = result['difficulty']
    main = [r for r in result['unified'] if r['method'] == 'PolyBelief']
    sbf = [r for r in diff if r['method'] == 'SBFNav']
    pb = [r for r in diff if r['method'] == 'PolyBelief']
    comparisons = []
    for r in pb:
        s = next(s for s in sbf if s['difficulty'] == r['difficulty'])
        comparisons.append(dict(difficulty=r['difficulty'], episodes=r['episodes'],
                                PolyBelief_SR=r['SR'], SBFNav_SR=s['SR'], delta_SR_pp=r['SR']-s['SR'],
                                PolyBelief_SPL=r['SPL'], SBFNav_SPL=s['SPL'], delta_SPL_pp=r['SPL']-s['SPL']))
    csvout('difficulty_polybelief_sbf_gap.csv', comparisons)
    (OUT / 'difficulty_analysis.md').write_text('''# PolyBelief Test-Unseen 官方难度分析

采用 CityNav 发布的 `citynav_test_unseen_easy/medium/hard.json` 的精确 Episode ID 成员关系。
修订版为官方原始成员集合与 HETT refined 全集的交集。原始 5311 条，修订版 5281 条。
ID = (map_name, object_id, ann_id)，不按行号或描述文本连接。三个集合已验证无重复、互斥、完整覆盖；源文件 SHA 和全部 ID 在 difficulty_episode_manifest.json。

SBFNav 正文只说明沿用难度分组，并引用 OpenFly；HTNav §4.1 明确说基于起点到目标直线距离。
已发布 CityNav loader 直接按文件后缀加载集合，没有公布生成阈值程序。因而本次准确复用发布的成员清单，不自行发明距离阈值。
源清单中记录的 dist_start_to_target 的实际范围仅是核验性描述，不能解释为新分组规则。

参考：[SBFNav Table 2及数据说明](https://arxiv.org/html/2609.05841v1)，[HTNav §4.1](https://arxiv.org/html/2604.08883)，[CityNav官方loader](https://github.com/water-cookie/citynav/blob/main/gsamllavanav/dataset/mturk_trajectory.py)。

## 同一修订版集合、共同 XY 指标的本地结果

''' + mdtable([r for r in diff if r['method'] in ('PolyBelief', 'SBFNav', 'HTNav')],
              ['method', 'variant', 'difficulty', 'episodes', 'NE', 'SR', 'OSR', 'SPL']) + '''
## PolyBelief 与本地 SBFNav 的差距

''' + mdtable(comparisons, ['difficulty', 'episodes', 'PolyBelief_SR', 'SBFNav_SR', 'delta_SR_pp',
                           'PolyBelief_SPL', 'SBFNav_SPL', 'delta_SPL_pp']) + '''
这里是已完成的 SBFNav 独立论文重实现 epoch12 对照，不是作者发布权重。
epoch12 的三个 split 均有完整结果，并与 epoch14 的验证结果分开列出；不能跨权重拼表。
历史 checkpoint 选择使用 Val-Unseen，不能视作满足 Seen-only 选模的严格基线。
本轮只离线重算，未依赖 Test 指标调整模型。PolyBelief 在不同难度的变化请以表中实值为准；难度不是语义复杂性或指令长度标签。
PolyBelief的SR从Easy的37.72%降至Medium的34.62%，Hard反而为39.16%；SPL分别35.30%、33.87%、38.63%。
因此性能没有随官方难度单调下降。Hard的SPL更接近SR也受到较长最优路径在SPL归一化中的影响，不能单独解释为语义推理增强。
相对本地SBFNav，三个难度SR均有约6.08–6.90pp的点估计优势；未经配对地图级检验且控制器不同，不宣称这些差距是统计显著的模块收益。

## 原论文结果与其他本地协议

table2_paper_reported.csv 独立保存 SBFNav Table 2 引用值：论文使用 original，不能与上述 revised 直接作受控收益归因。
其中 Seq2Seq/CMA 没有 +GSM，不得把它们替换成 Table 1 的对应方法。
table2_difficulty_comparison.csv 另包含原始数据 AerialVLN+GSM、GeoNav 的已有本地结果，以及历史 GSM 难度汇总，均有 result_type 标识。
没有逐 Episode 证据的历史汇总只保留其原值，不能宣称重新验证了每个 ID。
''')

    protocol_rows = [
        ['Random', '已有无模型策略；输入姿态，输出离散动作', '无checkpoint', 'CityNav-GSM/htnav/baselines/random_policy.py',
         '历史21步ballistic/stop策略不同于论文未明确策略；eval_random有fit选项，禁止调用fit或追论文数值'],
        ['Seq2Seq+GSM', 'RGB/depth/text/GSM→动作', '历史checkpoint链接失效', 'https://github.com/water-cookie/citynav',
         '历史original/refined、cache/gdino、imagery配置不混排；官方无GSM权重不能替代+GSM'],
        ['CMA+GSM', '交叉模态RGB/depth/text/GSM→动作', '历史checkpoint链接失效', 'https://github.com/water-cookie/citynav',
         'SBFNav年代/会议标签按原样存档；优先以CityNav适配资源确认方法身份'],
        ['MGP', '地图/语言→目标+progress→控制动作', '官方228385010字节权重已恢复并核验SHA；旧本地引用仍不替换', 'https://github.com/water-cookie/citynav',
         '已有三个native版本，离线缓存、目标后处理与progress停止差异不能算同一protocol'],
        ['AerialVLN+GSM', 'RGB/depth/text/GSM→双GRU离散动作', '本地best.pt SHA见索引；作者CityNav适配权重未找到', 'https://github.com/AirVLN/AirVLN',
         '本地original三split已完成；alignment修正训练未完成；不是作者适配的严格复现'],
        ['FlightGPT', '当前/地图图像+语言→VLM动作', 'https://huggingface.co/ADJHD/Flightgpt；revision9eb6253；本机四shard缺失', 'https://github.com/Uavln/FlightGPT',
         '约16.6GB下载；还缺安全split/ID/resume evaluator、双图加载和完整native协议核查'],
        ['GeoNav', '多尺度地图/场景图+语言→高层规划+定位', '作者GPT-4o；本地DeepSeek Flash替换', 'https://github.com/Xhtshr/geonav-official',
         'original Test5311已有完整逐条指标；Seen/Unseen与原GPT-4o缺失；不能把替换模型当原模型'],
        ['HTNav', 'RGB/depth/text/地图→宏waypoint+微动作/value', '官方README仅上传中、未公布权重；本地两配置', 'https://github.com/tyrofanfan/HTNav',
         'reproduction三split、alignment仅两validation；结构/IL-RL损失假设未闭合；不能择优跨配置拼表'],
        ['HETT', '语言/地图/因果视觉历史→粗目标及细动作', 'DATA/hett/release/checkpoint/best_val_unseen存在', 'https://github.com/crotonyl/HETT',
         '官方发布权重选择名best_val_unseen照实标注；HETT/main为含轨迹belief改动的研究checkout，不能充作官方方法。隔离补依赖后help通过，仍缺兼容上游源码与隔离导出'],
        ['SBFNav', '观测RGB-D/静态轮廓/语言→belief候选+语义几何选择→控制', '本地独立重实现epoch12/14；不是官方权重', 'https://arxiv.org/html/2609.05841v1',
         '历史用Val-Unseen选模；epoch12三split和epoch14验证分开保留；补充B.4/B.6/B.7发布缺失，假设必须保留'],
        ['PolyBelief', '命名多边形/语言/姿态+因果RGB→目标belief、轨迹与Stop', '冻结完整模型SHA见table1；本轮不改方法', str(MAIN),
         '28x28/stop0.3；revised2470/2697/5281；384及新校准残差分别为消融，不能相加'],
    ]
    p = (OUT / 'baseline_protocol_audit.md')
    p.write_text('''# PolyBelief 外部 Baseline 协议审计

本轮以 SBFNav Table 1/2 为框架；仅原论文数值进入 paper_reported 表。
table1_ours_unified.csv 的“统一”指相同 revised Episode 集与共同 benchmark_2d 指标重算，保留各方法原生控制器。
它不表示统一网络输入权限/控制器/训练或严格论文复现。comparable=False 及missing列中的限制必须随表保留；不能据此直接做方法优劣归因。
original数据、替代API模型、只有历史文档或聚合值的记录进入 table1_local_other_protocols.csv，绝不伪装为统一实测。

## 原始资源与输入输出

''' + mdtable([dict(zip(['方法', '输入与输出', '权重', '代码或论文', '可比性限制'], r)) for r in protocol_rows],
              ['方法', '输入与输出', '权重', '代码或论文', '可比性限制']) + '''
## Revised Dataset 的准确边界

SBFNav Table 1 的 * 明确表示 revised dataset；正文称修订8437条错误指令并移除331条无有效地标引用的样本。
本机 HETT 发布版实际从32637变为32326，共删除311条，而非331。HTNav论文所述修订也为311条，但不能只凭数量断定两个发布版本完全相同。
本机逐 split 的文件SHA、ID差集、文本/目标/轨迹差异已实际计算，见 dataset_version_audit.json。
原始与修订导航split的保留条目中，指令文本、目标和轨迹均未改变；差异在过滤及processed_descriptions地标元数据，后者35196条中18038条变化，详见landmark_annotation_audit.json。此统计口径不能替代论文8437条的声明。
当前2470/2697/5281直接来自已发布的refined JSON，非为PolyBelief新增筛选；本轮验证预测ID完整覆盖，失败Episode保留分母。
SBFNav论文未给修订文件SHA，故只能确认同属HETT修订数据来源，不能声称与作者使用文件逐字节相同。
“8437条”是论文报告的数据修订声明；本机split文本diff不是相同统计口径，不将二者强行等同。

''' + mdtable(result['dataaudit'], ['split', 'original', 'revised', 'removed', 'added', 'instruction_changed', 'trajectory_changed', 'target_changed', 'maps']) + '''
## 统一指标与冻结限制

NE=最终XY到GT的欧氏距离；SR=NE≤20m；OSR=执行轨迹任一点到GT≤20m；SPL=SR×起点直线距离/max(实际XY路长,起点距离)。
仅在离线评价层使用GT坐标。新统计代码不调用模型、不训练、不改Checkpoint；原推理输入权限审计沿用冻结协议与视觉残差审计。
HTNav原生SPL的最优长度取离散生成的最短教师轨迹，和共同evaluator的欧氏起终点距离略有不同。统一表保存共同定义重算值，其他协议表保留原生值，两者的逐split差值见verification.json；不强行修正已有文件。
现有SBFNav基线的历史Val-Unseen选模是明确限制，本轮不再用Unseen/Test选模补救；任何需要新训练的严格复现都保留Pending。

原始资源快照URL、SHA和下载状态见 sources/manifest.json；Git分支/工作树及dirty状态见 repository_inventory.json；权重位置、存在性、SHA见 selected_checkpoint_inventory.csv。
14个本地GraphNav研究分支的只读树扫描见branch_inventory.csv。HETT官方权重SHA与兼容阻塞见hett_asset_preflight.json；MGP新恢复权重见mgp_recovered_asset.json，绝不据文件名推定与缺失历史checkpoint相同。
原评估命令以源metrics/queue记录为准；experiment_registry.csv列出可重建命令及证据路径，历史路径仅保留溯源，不能直接照抄运行。
新离线汇总使用 `python -m scripts.build_paper_experiments`，命令与环境另存build_provenance.json。
''')

    # Real filesystem/process evidence overrides stale status.json labels.
    qpath = ROOT / 'Baselines/comparison/runs/alignment_20261013/status.json'
    q = load(qpath)
    pids = {str(x): Path('/proc') / str(x) for x in (q.get('pid'), q.get('child_pid')) if x}
    save('baseline_queue_audit.json', dict(source=str(qpath), source_reported_status=q.get('status'),
                                          processes={k: dict(exists=p.exists(), cmdline=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace') if p.exists() else '') for k,p in pids.items()},
                                          interpretation='Do not mark Running from stale status alone. HTNav alignment complete validation; Aerial aligned full run incomplete. No historical job restarted.'))
    profilepath = ROOT / 'Ablations/GraphNav/profiling-val-unseen-20261016/profiling_val_unseen/full/profile.json'
    profile = load(profilepath)
    profile_metrics = load(profilepath.parent / 'metrics.json')
    csvout('efficiency_results.csv', [dict(component=k, **v) for k, v in profile['all'].items()])
    save('efficiency_existing_evidence.json', dict(source=str(profilepath), sha256=sha(profilepath),
                                                evidence=profile, parameter_count=parameters,
                                                peak_gpu_allocated_gib=profile_metrics['gpu_peak_allocated_gib'],
                                                note='Original frozen28 model, not calibrated residual or384. GPU timing and wall time definitions remain separate.'))
    tasks = []
    def todo(priority, name, status, dataset, source, path, missing, acceptance, cost):
        tasks.append(dict(priority=priority, experiment=name, status=status, dataset=dataset, source=source,
                          result_path=path, missing=missing, acceptance=acceptance, estimated_cost=cost))
    todo('P0','冻结PolyBelief完整模型三split复核','Done','Val-Seen / Val-Unseen / Test-Unseen','冻结checkpoint；现有完整轨迹','reports/paper_experiments/verification.json','无；本轮逐条重算与正式CSV一致','2470/2697/5281完整ID、NE/SR/OSR/SPL吻合','CPU离线数分钟；已完成')
    todo('P0','Table1原论文Reference及全方法缺口','Done','全部','SBFNav Table1；官方资源','reports/paper_experiments/table1_paper_reported.csv','严格本地复现仍有缺口，见各方法条目','11个方法三split有值或明确缺失；Human只作上限','离线；已完成')
    todo('P0','Table1本地共同指标与协议分表','Done','全部','已有metrics及逐Episode轨迹','reports/paper_experiments/table1_ours_unified.csv','未完成方法留空；不表示全部baseline完成','本地值与论文值分离，所有协议偏差保留','离线；已完成')
    todo('P0','官方Easy/Medium/Hard及Test离线统计','Done','Test-Unseen','CityNav发布子集ID；既有预测','reports/paper_experiments/difficulty_analysis.md','论文未发布阈值生成程序，准确使用官方清单','互斥且覆盖5281；全部ID、源SHA及计数核验','CPU数分钟；已完成')
    todo('P0','Polygon vs Centroid vs Bounding Box闭环','Pending','Val-Seen / Val-Unseen','历史几何分支仅部分静态对照','reports/paper_experiments/asset_inventory.csv','没有确认同一冻结正式模型的三组闭环证据；需定义移除形状后的可比输入，不自行改网络','三组同清单、同控制器和选模规则；静态结果另表','仅推理约3×15–40分钟；若需独立训练另估，不在本轮启动')
    todo('P0','Geometry-only vs Geometry + Visual Belief','Done','Val-Seen子集','正式冻结模型Seen128','reports/paper_experiments/ablation_results.csv','只有Seen128正式权重干预；全量双validation尚未补齐','当前完成范围明确为128条；后续扩展须匹配权重与样本','已完成；扩展约2×15–40分钟')
    todo('P0','Calibrated Visual Residual四组完整泛化','Done','Val-Seen / Val-Unseen','相同已训练calibrated_residual/best.pt','VISUAL_RESIDUAL_UNSEEN_VALIDATION.md','Test未运行且无需用Test选候选','四组2470/2697；逐条配对、Zero一致','已有全部结果，禁止重跑')
    todo('P0','视觉残差地图级Bootstrap','Done','Val-Seen / Val-Unseen','既有20,000次paired bootstrap','visual_residual_unseen_validation/bootstrap_results.json','仅4张Unseen地图；主比较SR CI下界=0，不能称明确跨地图正收益','固定seed；episode加权及地图等权；CI如实保留','已完成')
    todo('P1','Direct-to-Goal / Learned Trajectory / Guarded-Blend正式权重对照','Pending','Val-Seen / Val-Unseen','历史controller Seen128结果可复用','reports/paper_experiments/ablation_results.csv','历史头不同；尚无正式冻结模型三组全量受控比较','同正式权重和ID、只改变执行方式；先核查已有配置接口','仅推理约3×15–40分钟；不训练')
    todo('P1','Stop Head / Arrival Guard受控对照','Done','Val-Seen','冻结完整模型；六种guard全Seen','reports/paper_experiments/ablation_results.csv','Unseen扩展未做；不据Test选guard','全Seen六组2470，stop0.3固定，保留终止原因','已有；不重跑')
    todo('P1','Oracle Goal闭环瓶颈诊断','Pending','Val-Seen / Val-Unseen','已有候选覆盖oracle不是目标控制oracle','reports/paper_experiments/asset_inventory.csv','未核验真正GT目标替换的闭环；须明确仅诊断、不能作为方法输入','单独标注oracle权限，固定预算；严禁正式网络输入GT','推理约2×15–40分钟；先协议审查')
    todo('P1','Current RGB vs Current + History','Done','Val-Seen子集','正式冻结模型Seen128','reports/paper_experiments/ablation_results.csv','全量验证缺失，不能将早期不同stage权重混入','同权重128条成对；明确完成范围','已有；全量扩展约2×15–40分钟')
    todo('P1','28x28 vs384x384目标网格','Done','全部','原冻结权重、完整三split','full_resolution_384/COMPARISON_28_VS_384.csv','尚无384与独立校准残差联合评测，严禁收益相加','三split完整；只改变目标网格；固定stop0.3','已有全部结果；不重跑')
    todo('P1','官方目标距离难度分层','Done','Test-Unseen','官方难度ID','reports/paper_experiments/table2_difficulty_comparison.csv','自定义距离分箱若需使用必须另表，不能冒充官方难度','使用发布难度清单和完整ID','已完成')
    todo('P1','语言关系类别细分','Pending','Val-Seen / Val-Unseen','现有语言解析与原指令','reports/paper_experiments/asset_inventory.csv','没有完整闭环类别统计与经核验标签；不可把静态关系诊断当闭环','先固定解析规则/人工抽查，多标签策略及分母明确','CPU分析约0.5–2小时，无训练')
    todo('P1','参数量/每步推理/峰值显存','Done','Val-Unseen','2697条完整HETT-style profiling','reports/paper_experiments/efficiency_existing_evidence.json','只适用于原28模型；baseline同硬件profiling尚缺','完整样本、warmup/计时边界、参数组件及峰值定义保留','已有；不重跑')
    for method in ('GeoNav', 'HTNav', 'HETT', 'FlightGPT', 'Seq2Seq+GSM', 'CMA+GSM', 'MGP', 'AerialVLN+GSM', 'Random', 'SBFNav'):
        status = 'Blocked'
        cost = {'GeoNav':'约5–50k模型调用；费用/时长依API，当前不可可靠估计',
                'HTNav':'已有validation不重跑；严格复现缺作者配置/权重；若重训参考旧10epoch约15小时以上',
                'HETT':'上游兼容源码恢复/导出约1–4小时；全量推理约1–6 GPU小时，需smoke测定；不能直接使用当前研究策略',
                'FlightGPT':'下载16.6GB，1–4小时适配/加载检查；全量VLM推理预计数小时至数天，须实测',
                'AerialVLN+GSM':'对齐版全量5epoch预计数十GPU小时，当前未完成，不本轮重训',
                'SBFNav':'严格Seen-only重新选模需新训练/作者权重，历史Unseen选模不能洗白',
                'Random':'本地固定策略CPU数分钟；论文随机策略不明确，禁止fit',
                'MGP':'228MB官方权重已恢复；源码/输入/历史身份链路核查约0.5–2小时；若需新共同协议推理数小时，需smoke',
                'Seq2Seq+GSM':'精确GSM训练权重缺失；恢复优先，不重新训练',
                'CMA+GSM':'精确GSM训练权重缺失；恢复优先，不重新训练'}[method]
        todo('P1' if method in ('GeoNav','HTNav','HETT','FlightGPT') else 'P2',method+'严格本地复现缺口',status,
             'Val-Seen / Val-Unseen / Test-Unseen','官方资源；具体见baseline_protocol_audit.md',
             'reports/paper_experiments/table1_local_other_protocols.csv', result['reasons'][method],
             '代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份',cost)
    todo('P2','统一baseline多种子与同硬件效率','Pending','Val-Seen / Val-Unseen','先完成单种子严格复现','reports/paper_experiments/missing_results.md','部分历史Random为5seed均值，不能与单seed不加标识混排','按预先预算多seed；同硬件测量；不据Unseen/Test选择seed','约单seed预算×种子数；本轮不启动')
    csvout('todo_status.csv', tasks)
    intro = '''# PolyBelief 论文实验补充 TODO

更新：2026-10-17。结果来自实际文件和逐 Episode 验证；Done 只表示本行明确范围完成，不代表全部严格baseline复现。
正文方法统一称PolyBelief；旧源路径和文件内容保持原样以便溯源。正式冻结权重与三split结果未改。
2027-CVPR 是多仓库容器，新增代码/报告位于当前 research/visual-residual-20261013 工作分支；不修改其他研究工作树。
本轮无新训练或推理，完成已有资产重算与审计。没有活跃实验可据此标为Running，旧队列状态不作完成依据。

字段：数据集的“全部”指Val-Seen / Val-Unseen / Test-Unseen；数值比例在CSV均为百分比，NE为米。

'''
    columns = ['experiment','status','dataset','source','result_path','missing','acceptance','estimated_cost']
    for priority in ('P0','P1','P2'):
        intro += '\n## '+priority+'\n\n'+mdtable([r for r in tasks if r['priority']==priority], columns)
    (HERE/'docs/PAPER_EXPERIMENT_TODO.md').write_text(intro)
    (OUT/'missing_results.md').write_text('# PolyBelief 未完成实验与阻塞\n\n'+mdtable([r for r in tasks if r['status']!='Done'],columns)+'''
估算均为工程量级，不是运行承诺。模型/API不可用时不伪造速度与成本。已有资产的缺失源日志/commit/Checkpoint不能由新汇总日志代替。
下一步按GeoNav→HTNav→HETT→FlightGPT资源可用性推进。HETT已有发布权重但当前checkout含研究方法改动，尚缺上游兼容源码与隔离导出；MGP官方权重已另目录恢复。其他可复用的轨迹在本轮已优先离线统计。
''')
    (OUT/'PAPER_EXPERIMENT_SUMMARY.md').write_text('''# PolyBelief 论文实验数据与补充总结

本轮完成已有资产Audit、离线Reuse与可立即得到的Complete数据整理；外部严格Reproduce仍按真实资源阻塞，不重训全部baseline。
所有结果均区分原论文报告、本地共同XY指标重算、本地其他协议、历史仅聚合/文档。
本轮额外恢复MGP官方228MB权重并保存SHA，未覆盖旧权重；HETT隔离补依赖后help可运行，但当前代码含研究结构改动，不能冒充官方HETT执行评测。

## 正式冻结完整模型

''' + mdtable(main,['method','split','episodes','NE','SR','OSR','SPL','checkpoint_sha256']) + '''
本轮逐Episode坐标重新计算四指标，与正式三split CSV及原metrics吻合，未覆盖正式结果或权重。
方法名称暂统一PolyBelief，数值身份仍为原冻结完整模型；独立训练校准残差和384评估均是消融，不能替换其中任一split凑主表。

## Table 1 与数据审计

table1_paper_reported.csv保留所有目标baseline与Human参考，并额外保留HETT*/SBFNav*两修订数据行。PolyBelief原论文不存在，留空注明原因。
table1_ours_unified.csv保留共同revised数据+XY evaluator的本地值；原生控制器/历史选模与本地适配身份保留，不宣称同输入或严格复现。
SBFNav的epoch12有三split完整结果，epoch14仅验证另列，不择优拼表。HTNav原生SPL与欧氏最优距离定义略有差异，两个版本分别保存，不覆盖原值。
table1_local_other_protocols.csv收录原始数据、模型替换和历史记录。不存在的实验不填数值。
特别发现：SBFNav称删除331条，本机发布数据实际删311条；没有作者文件SHA，不能断言修订文件完全相同。

## Table 2

''' + mdtable(pb,['method','difficulty','episodes','NE','SR','OSR','SPL']) + '''
''' + mdtable(comparisons,['difficulty','episodes','delta_SR_pp','delta_SPL_pp']) + '''
上述差值相对本地SBFNav独立重实现epoch12；受其历史Unseen选模与控制器差异限制，不等同严格方法归因。完整难度统计和original论文表独立保存。

## 核心机制结论

Calibrated Visual Residual 已有四组完整Seen2470/Unseen2697，不重复运行。
Unseen Real−No-belief SR +1.8168pp，整地图cluster95%CI [0,3.4410]pp；SPL +1.2329pp，CI [-0.6561,2.7960]pp。
Zero与No-belief逐条行为一致。相对Shuffle的地图级SR/SPL区间为正，支持利用RGB对应信息；但仅4张独立Unseen地图，主比较CI下界=0，不能报告跨地图明确正收益。
Seen的地图级主比较同样跨零，不能沿用仅episode bootstrap的强结论。

Stop/Arrival Guard六组完整Seen、Current RGB/History的Seen128、全三split28/384及2697条Unseen profiling均复用。
缺少同正式权重的Polygon/Centroid/Bounding Box、三控制器、Oracle Goal和完整关系类别统计，按TODO保留Pending，不把早期静态/其他权重结果写成完成。

## 复现和资产

离线命令：在本工作树用Baselines/.reproduction-env/bin/python运行 `-m scripts.build_paper_experiments`。
experiment_registry.csv逐行列出结果路径、数据SHA、Checkpoint SHA、源commit、原推理重建命令、预测和日志存在性。
build.log/build_provenance.json记录本次统计；sources/manifest.json记录外部原始网页及SHA；repository_inventory.json记录分支/工作树/dirty状态。
asset_inventory.csv/checkpoint_inventory.csv为扫描清单，selected_checkpoint_inventory.csv对使用到的权重验SHA。未恢复的旧权重明确missing，不凭文件名替代。
verification.json记录逐Episode、分母和聚合校验。difficulty_episode_manifest.json保留官方清单到本地ID映射。

common_episode_metrics.csv保存本次可重算来源的逐Episode共同指标，通过experiment_id连接registry；原预测轨迹位置保持原地。
新增结果是可重建的离线产物，原始大规模轨迹与权重保持原地，Git提交源代码、报告、数据索引和SHA，避免复制共享DATA或checkpoint。
完整待办见 [PAPER_EXPERIMENT_TODO.md](../../../docs/PAPER_EXPERIMENT_TODO.md)，机器可读todo_status.csv同时保存。
''')
    summary = OUT / 'PAPER_EXPERIMENT_SUMMARY.md'
    with summary.open('a') as f:
        f.write('\n## 已有完整Val-Unseen效率数据\n\n')
        f.write(f"2697个Episode；每决策计时样本{profile['samples']}个。策略参数{parameters['policy_parameters']:,}，含冻结BERT和SigLIP双塔的驻留参数{parameters['all_resident_parameters']:,}；包含未激活的旧分支。峰值allocated显存{profile_metrics['gpu_peak_allocated_gib']:.4f}GiB。\n\n")
        f.write(mdtable([dict(component=k, **v) for k, v in profile['all'].items()],
                        ['component', 'min', 'mean', 'population_variance']))
        f.write('\n上述均值单位ms，方差ms²；GPU事件跨度和端到端wall time分别保留。完整计时边界、并发进程及首次冷启动规则见efficiency_existing_evidence.json和原launch记录；不据此冒充不同baseline的同机速度对比。\n')
