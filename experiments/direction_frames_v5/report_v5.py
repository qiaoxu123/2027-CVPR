"""Chinese report generated exclusively from real ledgers and frozen checks."""
from v5core import *
def pc(x):return '—' if x is None else f'{100*x:.2f}%'
def ci(a):return '—' if not a or a[0] is None else f'[{100*a[0]:.2f}%, {100*a[1]:.2f}%]'
def tab(h,r):return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+'\n'.join('| '+' | '.join(map(str,z))+' |' for z in r)+'\n'
def run():
 S=readj(RESULT/'summary.json');M=readj(RESULT/'method_comparisons.json');phase=readj(RESULT/'phase1_manifest.json');cases=readj(RESULT/'case_index.json');audit=readj(RESULT/'v4_ambiguous22_classification.json');abbey=readj(RESULT/'Abbey_failure_analysis.json');new=readlines(RESULT/'new_candidates.jsonl.gz');Q=readlines(RESULT/'geometry_checks.jsonl.gz');rot=readj(RESULT/'random_rotation_control.json')
 def summ(c,t='all'):return next(s for s in S if s['cohort']==c and s['tier']==t)
 text='''# CityNav 局部参考系 v5：跨指令、道路、地图的回顾性验证

基于 `experiment/citynav-semantic-frames-20261106` / `6ae8b68d4f73eaabbbb7aeb077169d0d7565adb4`；独立分支 `experiment/citynav-frame-generalization-v5-20261106`。2026-11-06。

## 结论先行

**v4 道路语义规则在少量新增关系上继续吻合，但尚不足以证明稳定的跨地图统一生成机制。** 本轮仅使用 TRAIN；没有读取 Val/Test，没有训练语言图网络、GeomBelief 或导航模型，也没有导航SR实验。所有TRAIN都曾经过历史扫描，因此新增数据不是独立盲测。

v4原44条完整复现为21符合、1冲突、22 ambiguous。v5新增13个可靠道路侧绑定，其中11个能用原封不动的v4几何构造唯一道路场，8个几何可判定且全部符合。去掉同对象/道路/词/方向场重复之后，新增可判定6条，6/6符合（Wilson95% CI 60.97%–100%），不能把它当作高精度已得到稳定保证。

相对v4的44个唯一道路场开发子集，新道路只有1个可判定样本，未覆盖地图有2条新绑定但0条唯一场；所以**没有得到新的未覆盖地图成功证据**。这不否定统一局部规则可能存在，只说明本轮证据不够。

## 1. 冻结协议、输入和验证资格

[PROTOCOL.md](PROTOCOL.md) 在本轮GT评价前记录规则与对照。`v4_lock.json`记录v4源码及4701账本、公开输入、冻结方向场和原始结果的SHA；v4目录完全不改。A方案始终调用原v4 `directed_field`，其进入端判断、12m连接阈值、15%端部范围、拓扑/连通比例、分辨率与切线稳定阈值均未调整。

新语言适配器只接收原文、具名对象名称/类型与语句范围。地图阶段只用具名参照物和公开道路轮廓，剔除导航目标/关系主语几何；字段先保存、SHA冻结后才读取新关系目标GT。未来轨迹未参与；初始yaw只在GT评价阶段作为对照。

v5新增的是**独立语义角色解析与绑定适配**，不是声称v4完整原始解析器已经自动覆盖所有新句式。`frozen_binding_support.csv`核查冻结v4绑定函数是否确实能发出同一关系；`frozen_language_comparison.csv`另比较给定新绑定后v4参考方向模块的输出。不能把“提供了新A/B角色后几何模块可用”等同于“原解析器端到端已泛化”。

开发资格分四个独立维度：地图、道路A、A/B组合、关系主语是否出现在v4的44个唯一样本中。这里的“未覆盖”仅相对该开发子集，**不是该地图从未在v1–v4出现**。不将已有TRAIN切分包装成外部盲测。

## 2. 全TRAIN检索与语义审计

原始TRAIN仍为22,002条、24张地图。复用全部4701个旧具名地标方向候选，新增36个候选ID，另复查5个旧候选，共4737个候选；不重复计算旧ID。41条新增/重新审查记录见 `new_relation_audit.csv`。

仅研究以具名地标为参照的关系。以下情况保持unknown或排除，不变成目标—主路直接关系：

- `right from the first alley ... on Road A` 的参照物是匿名巷道，不算Road A关系。
- `right side lane`、`right side of the corner turn`不能自动等同于目标在整条Road A右侧。
- `the sixth from B`只是计数起点，不能自动等同于沿A从B进入。
- 进入A后又转进匿名alley/driveway/停车场，后续左右不能继续强制使用A方向。
- 后续出现facing/looking/standing或新的turn，必须承认参考系可能重置。
- 明确 `left of Named Hall` 时，Hall不能被替换成行进道路A。
- 多个冲突A/B组合、同名非道路对象、原始描述与ann_id不一致、主语作用域不明继续保留失败。

句式关键字窗口命中不等于真正语义模板；特别是`coming from ... to your left`不是`coming from B to A`。下表将粗检索和具名角色模板分开：

'''
 text+=tab(['结构','原始窗口匹配指令数（可重叠）','真正匹配具名A/B角色模板的指令数'],[[r['requested_structure'],r['raw_TRAIN_instructions'],r['named_template_instructions']] for r in readj(RESULT/'requested_structure_inventory.json')])
 text+='''
`enter A via B`和`approaching A from B`在当前检索中没有真实实例，不能宣称这些结构已获得数据验证。facing-from属于observer而不是道路进入，不能硬改成road场。

新增41条候选的三类资格：

'''
 text+=tab(['类别','候选','绑定许可','唯一场','可判定','符合'],[[n,(s:=summ('v5_'+n))['n'],s['bound'],s['unique'],s['scorable'],s['consistent']] for n in ['previously_uncovered_expression','old_rule_semantics_missed_binding','duplicate_or_highly_similar']])
 text+='''
“旧语义结构”不意味着每一条都被冻结v4绑定函数接受；严格支持标志逐条保存。近重复阈值为实体名/序数掩码后与44个历史唯一样本最大SequenceMatcher相似度≥0.90。它只是语言相似度分层，不是人工模板金标准；语言近重复与对象重复另行去重。

![语义新颖性](figures/novelty_pipeline.png)

## 3. 先构造道路场，再检验方向

道路A及来源B独立绑定后，使用冻结的v4图、进入端和正向。所有同名Road分段均保留；不按目标位置挑分段。完整方向场在GT之前保存，包含每个中心线点/边、源端与正向。GT仅查询这个已固定的场做侧别检验，不改变路径、符号、分支或语义绑定。

“唯一”是**声明的2D地图提取模型下可确定有向道路场**，不是恢复一个唯一作者yaw。宽观察区域、入口不明或多分支继续multiple/unknown。

所有候选和失败都进入下表。绑定覆盖不是解析正确率：没有独立逐条人工关系金标准，候选失败中也包含真正信息不足/作用域不许可，不能都归罪于NLP模型。

'''
 text+=tab(['人群','候选n','绑定','唯一场','唯一覆盖','可判定/唯一','条件一致率及Wilson95%CI','全候选确认率'],[[c,(s:=summ(c))['n'],s['bound'],s['unique'],pc(s['unique_coverage']),f"{s['scorable']}/{s['unique']}",pc(s['conditional_consistency'])+' '+ci(s['wilson95']),pc(s['confirmed_all'])] for c in ['all_named_candidates','core_all_named','bound_core','v4_original_unique44','v5_all_new_candidates','v5_new_bound']])
 text+='''
总计55个唯一场中30个可判定、29符合、1冲突；25个仍ambiguous。29/30不是独立新测试准确率，而是历史开发+新增回顾性样本的混合结果。13个新绑定中1个multiple、1个unknown没有被删掉；原先4701个候选及所有原状态仍在v4文件中可核对。

完整与高置信分层（规则分数high≥0.95，不是校准概率）：

'''
 text+=tab(['人群/档','n','唯一','符合/可判定','Wilson95%CI'],[[c+'/'+tier,(s:=summ(c,tier))['n'],s['unique'],f"{s['consistent']}/{s['scorable']}",ci(s['wilson95'])] for c in ['bound_core','v5_new_bound'] for tier in ['all','high']])
 text+='''
### 去重及真正新增支持

以对象、道路、方向词和冻结方向场共同去重；同一目标同一指令重复说左右不能增加独立证据。道路/地图相关性另用聚类CI处理，“去重关系”也不意味着严格统计独立。

'''
 text+=tab(['去重人群','唯一关系单元','符合/可判定','Wilson95%CI','对象数','道路数','地图数'],[[r['cohort'],r['unique'],f"{r['consistent']}/{r['scorable']}",ci(r['wilson95']),r['independent_subjects'],r['independent_roads'],r['independent_maps']] for r in readj(RESULT/'deduplicated.json')])
 text+='''
对象/道路/地图聚类CI均保存在JSON；例如6/6的bootstrap会退化为[1,1]，不能替代Wilson区间的宽下界。不能据此宣称新增样本已证明不低于v4精度。

![历史与新增分母](figures/historical_new_denominators.png)

## 4. 跨地图、道路、主语与句式

'''
 text+=tab(['相对v4唯一开发集未覆盖维度','新绑定数','唯一场','符合/可判定','Wilson95%CI'],[[c,(s:=summ('v5_'+c))['n'],s['unique'],f"{s['consistent']}/{s['scorable']}",ci(s['wilson95'])] for c in ['new_map_vs_v4_unique','new_road_vs_v4_unique','new_road_pair_vs_v4_unique','new_subject_vs_v4_unique']])
 text+='''
未覆盖地图2条绑定均未产生唯一场，不能填成0%准确率，也不能称跨地图成功。新道路/组合只有1/1可判定（Wilson CI 20.65%–100%）。原路新对象、模板变体提供正向回顾性证据，但不是可靠跨地图统一规则的证明。

全部按地图、道路、句式、对象类别、方向词的计数、条件率、Wilson及对象/道路/地图聚类CI在 `grouped_results.csv/json`。确定性冻结规则没有参数可重训；逐图展示是可迁移性分组，不冒充重新训练的盲交叉验证。固定世界字典另做留地图/留道路拟合，测试组不参与其轴选择。

![新增地图结果](figures/new_map_results.png)

## 5. v4唯一冲突：Abbey Road

原文：

> The third building on the right of the Abbey Road when coming from the Walnut Tree Avenue.

关系主语为对象17的building，参照物为TrafficRoad 369，来源Walnut Tree Avenue包含372/398/400/416全部分段。原始TRAIN与CityRefer对应ann_id原文一致；GT与对象轮廓同在道路一侧，不是显然的ID错配。

独立地图审计：A中心线源端约(1953,1657)，另一端约(1964,1602)。源B只在源端接触，远端距B约48.56m；接触沿程约0–5.41m，A总长约62.63m。只有398与A直接相交，其余具名分段全部保留并报告距离；没有按GT选398。0.5/0.75/1.25/2m分辨率的方向及侧别结果逐项保存在 `Abbey_failure_analysis.json`。

在该独立进入解释下，正向沿A向远端，目标在反侧，冲突保留。相同A/B的另一个building描述符合，不能为了这一条把整路方向翻转。

| 可能原因 | 目前证据与边界 |
| --- | --- |
| 名称/分段绑定错误 | 地标与原文核对、保留全分段，未发现目标驱动消歧 |
| 道路拓扑或栅格进入端不稳定 | 单端接触及多分辨率稳定，不支持将本例解释为已证实的这类错误 |
| 语言作用域错误 | 原句是直接building—Road A关系，未发现匿名中间参照物 |
| Nav目标/ann_id明显错配 | 原文与对象索引一致、GT在对象轮廓内，不支持明显错配 |
| 作者视角/词义约定或方向标注差异 | 缺少原相机姿态及作者意图金标准，仍无法区分 |

因此可确认“该字面道路进入规则仍有一个冲突”，不能确认“已经找到真实标注错误”。同对象其他英文描述只作为上下文，不拿它们反推一个使GT匹配的正向；ann_id也不当作全局作者身份。

## 6. 原22条ambiguous逐条分类

直接几何触发：19条Polygon跨中心线、2条仅碰1m容差带、1条曲率/端部导致完整轮廓侧别不唯一。这是可复核的几何事实，不是对真实道路/目标标注的因果归因。

'''
 counts=collections.Counter(r['attribution_status'] for r in audit);text+=tab(['审计归类','数量'],[[k,v] for k,v in counts.items()])
 text+='''
20/22的GT点侧别符合，另外2个GT点也ambiguous。但这是点诊断，**没有替换主完整轮廓判据**。19条跨中心线不能靠容差从1m降为0自动解决；2条纯容差例在0m可通过，在0.5/1/2m仍ambiguous，说明判据保守效应存在，但没有据此挑0m当主阈值。

其中 `cambridge_block_33:128:1:21` 的Nav GT点距CityRefer Polygon约3.23m，需保留数据几何不一致；这不证明来源道路方向错。另有1条结果对栅格几何敏感，仍无独立车道中心线标注证明哪条线正确。**0条被宣称为已有独立道路金标准确认的中心线错误。**

逐条原因、GT点/质心诊断、轮廓面积、距离、0/0.5/1/2m容差与0.5/0.75/1.25/2m分辨率结果均在 `v4_ambiguous22_classification.csv/json`；所有原44条在 `v4_all44_reaudit.csv/json`。

![失败分类](figures/ambiguous_failure_types.png)

### 弯道、路宽与多分支核查

0.5/0.75/1.25/2m几何剖面、来源交汇和分支/连通状态完整保存。另用道路Polygon边界净空直径作为宽度代理，排除端部5m后计算p90/p10变化比，并按局部5m朝向变化分组（`geometry_strata.csv`）。这是冻结后的描述性分层，不参与方向/阈值选择；它不是实测车道宽度或因果中心线误差标签。实际物理路网分支可能被轮廓或固定剪枝代理简化，不能把模型无分支当成独立路网金标准。

## 7. A–F：相同支持与相同完整轮廓谓词

A–E共同使用同一冻结中心线、最近投影、完整目标Polygon边界采样、1m容差及相同几何mask，只改变正向。不是用±45°扇区比较道路侧半平面。所有55条第一阶段唯一场支持都保留，方法自己的ambiguous/unknown均算未确认，不仅统计成功子集。

- A：冻结v4道路方向场。
- B：世界固定正向0/90/180/270°全部保存；共享字典仅在其他地图/道路上选择，不按单样本GT选轴。这是在相同道路几何上施加世界正向的受控对照，**不是v2以地标质心计算的绝对世界方位率**。
- C：起点→地标；D：初始yaw仅为代理，不当作者视角。
- E：道路PCA两个符号等质量，绝不取匹配GT的符号。其值是候选兼容度，不是硬方向准确率。
- F：单独探索性中心线坐标平滑，±4m窗口、端点保持、位移≤2m并位于道路buffer(1m)；参数在新GT评价前写入协议并保存完整场。零长度收缩边在几何投影时忽略，未按GT选边。F可改变几何mask，因此另报告逐条变化，不能把它的收益归因于方向语义。

'''
 rows=[]
 for name in ['A_frozen_v4_field','B_world_crossfit_leave_map_out','B_world_crossfit_leave_road_out','C_start_to_landmark','D_initial_UAV_yaw','E_unsigned_PCA_equal_signs','F_exploratory_smoothed_geometry']:
  r=next(r for r in M if r['cohort']=='bound_core' and r['method']==name);rows.append([name,r['same_support_n'],pc(r['confirmed_or_compatibility_all_support']),pc(r['conditional_on_common_base_geometry']),ci(r['object_cluster95']),f"{100*r['difference_from_A']:+.2f} pp",ci(r['paired_map95'])])
 text+=tab(['方法','相同支持n','全支持确认/兼容度','共同A可判定集确认/兼容度','对象聚类95%CI','相对A净差','净差地图CI'],rows)
 text+='''
在这套严格控制的道路侧谓词下，A相对留地图世界对照提高12.73pp；对象/道路/地图聚类配对区间分别约[3.45,24.07]、[1.69,26.15]、[3.95,27.59]pp。但该差来自历史+回顾性混合支持，不能当作未见城市盲测提升，不能与v4那套直线/质心对照的净差直接相减，也不证明作者使用该参考系。

F总确认数由29变30，原22个ambiguous一个也未解决；新增Parsonage例因探索性几何变化从ambiguous变consistent。没有独立车道线真值证明F更接近真实道路中心，因此只能报告“几何判定变化”，不能宣称精度已提高。完整 `F_geometry_transitions.csv` 保留所有未变与失败样本，未逐例在A/F中挑正确者。

![相同支持比较](figures/same_support_methods.png)

## 8. 负对照

180°翻转保持原线及mask，仅反转正向；来源替换和错误地标绑定在GT前以同图另一具名道路构造，未按目标选替代。后两者可能无法构造唯一场，仍保留原55条支持，不能丢弃abstention后宣传低准确率。具体替代道路ID与完整场在 `negative_fields.jsonl.gz`。

'''
 text+=tab(['负对照','原支持','负对照唯一场','可判定','全支持确认率','条件一致率'],[[r['control'],r['support_n'],r['unique_alternative_fields'],r['geometry_scorable'],pc(r['confirmed_all_support']),pc(r['conditional_consistency'])] for r in readj(RESULT/'negative_control_coverage.json')])
 text+='\n如果替代来源/地标主要变为unknown，该负对照反映拓扑拒绝，不能当作在充分错误方向支持上的分类失败证明。\n'
 text+=f"\n另做300次固定mask随机旋转：真实确认率{pc(rot['observed_confirmed_all'])}，随机均值{pc(rot['null_mean'])}，随机95%范围{ci(rot['null95'])}，探索性单侧MC p={rot['one_sided_MC_p']:.4f}。未多重校正；不把人工错误绑定或随机角度当自然指令金标准。\n"
 text+='''
## 9. 真实地图案例

均为真实TRAIN正射图、原始地标/目标Polygon与离线GT。绿色是冻结v4场，橙色虚线是F探索，图像右=+X、上=+Y；不声称真实地理北方或原始作者视角。案例在冻结后用于说明，不参与方向选择或阈值调节。

'''
 for c in cases:text+=f"\n### {c['kind']} — `{c['occurrence_id']}`\n\n> {c['instruction']}\n\n状态：{c['state']}；主判据：{c['result']}；{'历史开发案例' if c['historical_development_case'] else '新增回顾性关系，非外部盲测'}。\n\n![真实地图](figures/{c['figure']})\n"
 text+='''
## 10. 五个问题的明确回答

1. **v4规则在新增独立样本继续有效吗？** 有有限正向证据：13个新绑定，11个唯一场，8/8几何可判定符合；去除已开发对象/同方向场重复后6/6符合，置信区间很宽。新道路仅1/1、未覆盖地图无可判定样本，因此不能宣称稳定跨地图泛化。
2. **最大限制在哪里？** 全候选覆盖受语言信息不足、绑定/作用域与道路语义许可共同限制，不能无人工关系金标准就断言纯粹NLP错误。可靠绑定后大部分可构造；已唯一场中25/55仍无法用完整轮廓判定，当前近端瓶颈是目标几何/有限道路域与严格侧别谓词。新地图例还受到公开道路轮廓/拓扑的限制。
3. **能否不降低精度提高覆盖率？** 语义适配使唯一道路场44→55，新增可判定8/8，但含重复且去重后仅6/6，尚不足以证明精度非劣。F不提高第一阶段唯一覆盖，只有一个探索性几何判定变化，不作为稳定收益。
4. **是否存在可靠统一生成规则？** 证据继续支持“明确进入语义+独立单端道路几何”这一部分机制；Abbey仍有冲突，真正新地图证据缺失。既不能确认全CityNav统一机制，也不能由局部失败否定其可能性。
5. **足以接入GeomBelief吗？** 不足以作为全局硬约束或替代世界方向。可支持下一步可关闭、带unknown与证据来源的离线接口/消融设计；在训练或导航闭环前，应先做冻结协议下独立外部检验及独立道路/车道几何标注。本轮未实施任何模型接入、训练或SR实验。

## 复现、审查与限制

[README.md](README.md)给出逐阶段命令与全部文件索引；所有原4701候选保留，4737条最终输出及所有失败均可按occurrence_id追溯。解析置信度、三个新颖性层、四种开发资格、聚类区间、负对照及F探索分开保存。

这是TRAIN内回顾性验证，语言适配器依据原文审计扩展；程序实现修复与保守范围守卫并不等于预注册盲测。冻结v4主规则和几何阈值没有根据目标GT修改；没有选GT最优道路、方向符号或中心线版本。GT只检验已固定假设。没有作者视角标签，所以本轮结果是几何吻合与可构造性，不是真实参考系识别准确率。

本地测试与全量复核见 `verification.json`；实际外部审查状态见 [REVIEW.md](REVIEW.md)，不能把运行时失败或本地自检写成外部审查通过。
'''
 (HERE/'REPORT.md').write_text(text);print('Chinese v5 report generated')
if __name__=='__main__':run()
