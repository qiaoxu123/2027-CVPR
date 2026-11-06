"""Chinese evidence report, populated from frozen predictions and actual checks."""
from shared import *

def pct(x):return '—' if x is None else f'{x*100:.2f}%'
def interval(a):return '—' if not a or a[0] is None else f'[{a[0]*100:.2f}%, {a[1]*100:.2f}%]'
def diffci(a):return f'[{a[0]*100:+.2f}, {a[1]*100:+.2f}] pp'
def tab(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def run():
 D=readj(OUT/'dataset.json');S=readj(OUT/'summary.json');E=readj(OUT/'evidence_summary.json');N=readj(OUT/'negative_controls.json');M=readj(OUT/'matched_baseline_summary.json');cases=readj(OUT/'case_index.json');P={p['occurrence_id']:p for p in readlines(OUT/'frozen_frames.jsonl.gz')};interventions=readj(OUT/'fixed_geometry_language_interventions.json');pairs=readj(OUT/'opposite_relation_pairs.json');shared=readj(OUT/'shared_view_constraints.json')
 def summary(c,t='all'):return next(r for r in S if r['cohort']==c and r['tier']==t)
 text='''# CityNav：语言语义与场景几何能否共同确定空间参考系？v4
+
+日期：2026-11-06。基于 `experiment/citynav-local-frame-rules-20261106` / `7f1bfb8a9ccd066519c04eeb2f0e9f6d4ce5313d`；独立分支 `experiment/citynav-semantic-frames-20261106`。仅 TRAIN，无 GeomBelief 训练，无导航闭环，无导航网络修改。
+
+## 核心判断
+
+**语言能指定参考系的构造方式，地图能补充独立几何；但给定当前数据，并非每种结构都足以唯一确定一个数值朝向。** 本轮得到新的跨地图有向道路证据，同时明确保留观察区域、道路分支和建筑正面的歧义。结果支持一个按语义角色组合的“语义→局部参考系”机制作为候选解释，尚未证明它就是 CityNav 的统一标注生成机制。
+
+必须区分三层：①文本/对象绑定可信；②在明确地图提取假设下参考系是否唯一；③冻结该参考系后方向关系是否成立。“几何模型下唯一”不等于已恢复注释者视角，方向成立也不能反推作者一定采用该参考系。
+
+## 1. 任务、数据及新增覆盖
+
+输入为原文、具名地标与公开道路/建筑/水体轮廓，以及可选起点。输出逐关系的主语、参照物、词、world/observer/road/intrinsic/unknown 类型、原点/原点区域、正向表达、证据span、候选与unknown。不同语句不自动继承同一观察方向。
+
+复用 v3 全部4,659条原始候选、全部1,349条已绑定输入。完整 TRAIN 22,002条、24张地图；其中具名地标候选涉及23张地图。显式语义词扫描3,991条TRAIN指令，含具名对象及方向词3,336条；这是词汇检索覆盖，不是已绑定关系数。检索账本包括没有最终可用绑定的原文。
+
+新增两类文本依据：
+
+- 7条旧候选的保守作用域修复：直接初始目标NP紧接方向关系，或以目标car开头、后续句子主语连续为 `It is` 的代词链。匿名中间对象、嵌套停车场或介词链继续排除。
+- 42条新道路锚关系，来自40条指令：如 `turn onto Road A coming from Road B ... this is ... to your left`。A/B均从独立名称和类型绑定，且方向属于目标指示语句；`car in the alley to your left` 不被简化成car相对Road A。道路A是显式行进锚，不是按目标最近距离找的地标。转弯动作中的left/right本身不算目标静态侧别。
+
+新增关系均校验同一目标ID/ann_id的原始CityRefer英文一致；同名道路保留全部路段，不按GT挑段。最终4,701个候选、1,398个绑定，四词候选3,940、四词绑定1,107。新绑定不代表几何可判定。v3原425个四词不相交配对保留；新道路语法带来另外8个不相交四词配对，共433，不能把新道路侧评价与旧扇区分母混用。
+
+逐条见 `results/relations.jsonl.gz`、`binding_extensions.csv`、`train_semantic_scan.jsonl.gz`；旧排除原因及新绑定规则全部保留。解析分数是规则分值，不是标注准确率或校准概率。
+
+'''.replace('\n+','\n')
 text+=tab(['检索结构','全TRAIN指令数','具名对象+方向词指令数'],[[r['structure'],r['all_TRAIN_instruction_count'],r['with_named_mentions_and_direction_words']] for r in readj(OUT/'semantic_search_coverage.json')])
 text+='''
+## 2. 两阶段隔离与“唯一”的定义
+
+第一阶段 `audit.py → frames.py`：公开场景剔除关系主语与导航目标的坐标/轮廓，仅保留ID作账本对齐。推断不打开任何GT文件，不用未来轨迹、初始yaw或最优角度。先保存 `frozen_frames.jsonl.gz`、完整 `road_fields.jsonl.gz` 与SHA256。
+
+第二阶段 `make_gt.py → evaluate.py`：必须先验证冻结SHA，才读取目标GT及初始pose进行离线检验。初始yaw仅作对照；trajectory仅索引 `[0]`。本研究继承了已看过的v1/v2/v3探索背景，**不是盲测或预注册实验**；代码中的输入隔离不等于研究者从未见过GT。
+
+输出三类基本状态：
+
+- **unique / unique_field**：在所声明提取模型与容差下唯一。`unique`为单一数值朝向（观察区域支撑宽≤20°才允许，精度±10°）；`unique_field`为沿道路随位置变化的有向坐标场，正向函数整体冻结，不能误报成每条关系已有一个确定yaw。
+- **multiple**：存在语义支持的候选、可行朝向范围或无向轴，未强行挑一个。
+- **unknown**：绑定/作用域不足、观察原点缺失、道路连接未知、真实北方未校准等。
+
+本轮单一数值朝向 `unique` 为 **0条**；44条为 `unique_field`。这不表示44条作者视角已知，也不表示其余指令没有真实参考系。公开输入只有区域时，通常不能恢复精确观察点或瞄准点。
+
+## 3. 语义生成规则 A–F
+
+| 结构 | 独立构造及原点 | 本轮可确定性边界 |
+| --- | --- | --- |
+| A. stand at A facing B | 原点为A/其区域，正向为A→B的可行向量集合 | A/B是区域时使用完整轮廓，不能把最近点或质心当成作者真实观察点；窄范围才近似唯一 |
+| B. looking from A towards B；facing B from A | 同样构造来源区域→目的区域，保留来源/目的span | 只有facing B而无A不能用目标/最近地标反推观察位置 |
+| C. on Road A coming from Road B | B与A的独立交汇确定进入端；原点在进入端；沿A中心线由起点向外为正向 | 中部进入、分支、多连通分量或分辨率不稳定时不唯一 |
+| D. along Road A | 可构造轴或中心线 | 不具备独立进入/行进方向时保留±；初始yaw不能替代文本 |
+| E. front/behind Building A | 原点在建筑；需要独立正面/入口语义或可靠朝向标注 | 当前缺少这些证据；PCA四方向或“建筑朝路”只能是候选 |
+| F. 多具名地标共同观察 | 非目标具名对象间的left/right等关系可约束同一视角的可行角集合 | 共享视角作用域仍需语言支持；二元侧别通常只给区间，不给精确yaw |
+
+### 道路几何不是一条全局PCA直线
+
+只用道路轮廓栅格化、Zhang–Suen细化，建立保留连通结构的中心线图；使用道路宽度剪除短端点毛刺，不取目标最近路段来构造图。取最大连通分量比例须≥95%，只有两端点且无分支、来源间距≤12m、交汇沿程距端点≤max(12m,道路长度15%)才允许给出有向场。进入端到其他点的图距离递增定义正向；6–8m局部邻域平滑切线。
+
+额外在0.75/1.0/1.25m三种栅格分辨率检查拓扑及朝向；替代分辨率也须唯一，公共地图采样点切线差的90分位≤15°。不稳定则降为multiple，不根据方向验证分数选参数。37组独立A/B道路配置中23组通过，覆盖44条关系；源路位于中部、断裂或拓扑歧义的配置继续保留失败原因。
+
+这是可复现的道路几何代理，未证明每条实际通行路线或作者视角已恢复。短支路可能受轮廓与剪枝影响，真实路网入口仍需外部核验。评价时在GT位置查询**已经固定的完整场**只是检验谓词，GT不会改变图、方向或候选选择；查询规则也不按哪个方向匹配得好来选择分支。
+
+### 船屋例重新界定
+
+v3取最近河岸法向得到2.83°误差，确实是一个吻合的独立几何代理。但“面对河流”没有说必须朝最近岸点的垂直方向。v4让观察位置在A区域、目的点在整个水体区域中变化，保留允许方向范围；该例因此为 **observer / multiple**，不是唯一作者yaw。该修正不否定相对语义的解释能力，而是避免把某个合理代理误称为已唯一确定。
+
+![观察区域歧义](figures/observer_region_ambiguity.png)
+
+## 4. 参考系覆盖：先于GT冻结
+
+'''.replace('\n+','\n')
 text+=tab(['人群','n','唯一道路场','多候选','unknown','唯一场覆盖'],[[c,(s:=summary(c))['n'],s['unique'],s['multiple'],s['unknown'],pct(s['unique_fraction'])] for c in ['all_named_candidates','core_all_named','core_bound','v3_core425','expanded_core_disjoint','road_semantic','new_semantic_road_anchors']])
 text+='''
+![唯一性覆盖](figures/identifiability_coverage.png)
+
+全四词候选中44/3,940=1.12%有唯一几何道路场，非唯一98.88%；已绑定四词中44/1,107=3.97%。44条涉及9张地图、23组参照物。**不能把这组低覆盖率解释成不存在统一语言规则**：它混合了解析未知、场景信息缺失、内在多解与当前提取模型的局限。
+
+## 5. 冻结后的方向检验
+
+道路采用有向曲线侧别：完整目标轮廓按≤1m边界步长采样，并检查整个Polygon是否与中心线1m缓冲带相交；稳定且整条轮廓在词所指侧>1m为consistent，整条在反侧为conflict，跨越/接近中心线或近邻切线不唯一则ambiguous。普通对象才使用±45°方位扇区及完整轮廓投影；不混用侧别与扇区准确率。逐样本 `metric`、边界裕量、歧义原因均保留。
+
+'''.replace('\n+','\n')
 text+=tab(['人群/置信档','冻结唯一场','几何可判定','符合/可判定','条件一致率 Wilson95% CI','符合/全部唯一场（保守下界）'],[[c+'/'+t,(s:=summary(c,t))['unique'],s['geometry_scorable'],f"{s['consistent']}/{s['geometry_scorable']}",pct(s['conditional_consistency'])+' '+interval(s['wilson95']),pct(s['unique_success_lower_bound'])] for c,t in [('core_bound','all'),('core_bound','high'),('road_semantic','all'),('road_semantic','high'),('new_semantic_road_anchors','all'),('new_semantic_road_anchors','high')]])
 s=summary('core_bound');text+=f"\n全部唯一场44条中，22条可判定：21符合、1冲突，条件一致率95.45%，但22条仍几何ambiguous。按对象聚类CI为{interval(s['object_cluster95'])}，参照物聚类{interval(s['reference_cluster95'])}，地图聚类{interval(s['map_cluster95'])}。可判定样本来自6张地图、11组参照物、21个目标对象，不能把相邻建筑视为大量独立证据。总体确认比例21/44=47.73%；占全部四词候选21/3,940=0.53%。\n"
 text+='''
+42条新道路锚关系中26条有唯一场，12条几何可判定且全部符合；其余14条唯一场关系仍有轮廓歧义。这是新增可靠支持，而不是42条全部成功。12条中含同一对象同一指令的重复侧别表达；去重敏感性如下。
+
+'''.replace('\n+','\n')
 text+=tab(['去重人群','唯一关系单元','可判定','符合','条件一致率及Wilson CI'],[[r['cohort'],r['unique'],r['geometry_scorable'],r['consistent'],pct(r['conditional_consistency'])+' '+interval(r['wilson95'])] for r in readj(OUT/'deduplicated_sensitivity.json')])
 text+='''
+![完整分母](figures/road_evaluation_denominators.png)
+
+逐词（全四词具名地标候选分母；唯一均为道路场）：
+
+'''.replace('\n+','\n')
 text+=tab(['词','候选','唯一场','多候选','unknown','符合/可判定','Wilson95% CI'],[[r['word'],r['n'],r['unique'],r['multiple'],r['unknown'],f"{r['consistent']}/{r['geometry_scorable']}",interval(r['wilson95'])] for r in readj(OUT/'word_counts.json') if r['word'] in MAIN])
 text+='''
+front/behind 没有新增唯一固有朝向证据，不能从道路left/right的成功外推。完整地图/参照类别/置信分层见 `breakdown.csv`，全部词、未知原因见 `word_counts.csv` 与 `frame_exclusion_reasons.csv`。
+
+## 6. 地图留出与历史基线
+
+本轮没有训练参数，每张地图只用自身公开对象执行同一套冻结规则；逐地图留出结果因此等于该图的独立规则应用。它检验地图间可迁移表现，不冒充对开发地图从未见过的盲测；阈值固定但研究者已知前轮案例。不同地图不重新拟合规则或方向。
+
+'''.replace('\n+','\n')
 text+=tab(['地图','唯一场','符合/可判定','条件率及Wilson95% CI'],[[r['heldout_map'],r['unique'],f"{r['consistent']}/{r['geometry_scorable']}",pct(r['conditional_consistency'])+' '+interval(r['wilson95'])] for r in readj(OUT/'leave_map_out.json') if r['unique']])
 text+='''
+![地图分组](figures/heldout_maps.png)
+
+在相同44条第一阶段唯一支持上比较：世界正向仅在其他TRAIN地图校准共享四轴字典；yaw、起点→地标、反向连线保留其原有定义，不把最优代理认作真实作者视线。世界/直线代理采用固定直线侧域，本轮采用冻结曲线场，均是道路侧分类任务，但决策域并不相同。ambiguous统一算“未确认”，避免删难例抬升；CI是逐关系差值聚类区间。
+
+'''.replace('\n+','\n')
 text+=tab(['对照','n','对照确认率','语义确认率','净差','对象聚类95% CI','地图聚类95% CI'],[[r['baseline'],r['support_n'],pct(r['baseline_confirmed_fraction']),pct(r['semantic_confirmed_fraction']),f"{100*r['paired_difference']:+.2f} pp",diffci(r['object_cluster95']),diffci(r['map_cluster95'])] for r in M])
 text+='''
+**当前总体净差区间包含0**，因此不能宣称语义模型已经在同支持上稳定优于世界/yaw/起点代理。小范围条件一致性高与总体净增益显著是两回事。
+
+![匹配基线](figures/matched_baselines.png)
+
+历史结果不能直接相减：
+
+- [v1](../direction_frames/REPORT_ZH.md)：212个全方向有效配对，保守目标主语范围；其四组正反例的强结论已由v2限定，不能据此排除全部统一局部规则。
+- [v2](../direction_frames_v2/REPORT_ZH.md)：423/250全部方向有效，高置信/完整集四词为207/344，提供世界旋转镜像、yaw、起点连线与离线运动诊断；未来运动未在本轮复用为推断证据。
+- [v3](../direction_frames_v3/REPORT.md)：425个四词有效集严格数值代理6例、扇区3/6，已知船屋占1例；道路次要集8/9。BERT候选hybrid的+8.25pp是概率兼容度，不是参考系识别准确率。本轮不重新训练BERT，也不把银标签路由当作真实参考系标签。
+- v4保留v3的425个原有配对，在这一支持中5条具有唯一道路场，侧别4/5；船屋降为区域多解。6/425与5/425采用不同“唯一”表达及几何谓词，**不能当作性能下降/提升直接相减**。v3原9条道路严格支持在v4的逐条结果见 `v3_strict_support_audit.csv`；新道路语法12/12（去重11/11）是额外证据。
+
+历史匹配数字与源文件SHA见 `historical_context.json`；完整原始v1/v2/v3产物在同分支保留。
+
+## 7. 负对照与语义干预
+
+随机方向保持道路几何、关系支持与GT几何资格不变，只旋转正向向量；180°翻转是相反行进对照。错误地标绑定使用同图其他具名道路，确定性选择且不访问目标；语义角色交换使原A/B进入角色互换。后两者可能输出unknown，保留原44条分母，并分别报告其唯一覆盖，不只统计残留易例。
+
+'''.replace('\n+','\n')
 text+=tab(['对照','原支持','唯一覆盖（若改变）','对照确认率','随机95%区间 / MC p'],[[r['control'],r['support_n'],str(r.get('unique_fields_emitted','保持44')),pct(r.get('null_mean',r.get('confirmed_fraction_all_support'))),interval(r.get('null95'))+(' / '+str(round(r['one_sided_MC_p'],4)) if r.get('one_sided_MC_p') is not None else '')] for r in N])
 text+='''
+随机旋转为300次探索性对照，未做多重比较校正。错绑定与角色交换降低覆盖/确认率，说明模型响应语义角色及几何连接；不能把这种人工错误当作真实文本标注样本。
+
+![负对照](figures/negative_controls.png)
+
+保持地图不变的 `stand at A facing B` ↔ `stand at B facing A` 文本干预：使用真实具名对象的原始Polygon，仅构造语言控制。两区域的保守几何支撑不交且有信息的组合，其允许方向区间中心均旋转180°；相交区域允许全360°时，保持不确定才是正确输出。完整记录包含这些无信息情形，不将构造文本加入自然样本准确率或案例图。另加固定原点A、仅把facing B改为facing C的干预（限有足够唯一具名建筑的地图），地图与A完全不变；B/C按公开具名建筑及独立角分离选择，不用目标。原点不变性、前后允许方向范围和角变化见 `fixed_origin_view_interventions.json`，同样不计为自然样本成功率。该测试证明程序遵守角色变换，不证明自然描述者使用该规则。
+
+## 8. 同地标指令、正反关系和多地标共享视角
+
+同Road A、同来源B的关系复用同一冻结方向场；不同目标、词或GT不能改变它。此不变性是实现纪律，不是经验上的作者视角一致。不同来源可生成不同方向，因此不要求同一地标的所有指令共用方向。
+
+'''.replace('\n+','\n')
 text+=f"共找到{len(pairs)}对同主语/同参照物的left/right或front/behind反词关系，其中0对能独立确认同一个唯一道路参考系。逐条状态和是否同一独立道路场见 `opposite_relation_pairs.json`；不能仅凭ΔXY相同就宣称真实标注矛盾。道路词与行进正向同时反转的代数一致性见 `negative_control_details.jsonl.gz`，这是几何实现检验，不是新增作者证据。\n\n"
 text+=f"另有{len(shared)}条可构造非目标具名对象间共享视角约束；全部保留允许角集合及共享作用域假设，未识别出唯一作者yaw。目标对象已从公开场景剔除，不用目标方向句来给自己反求朝向。`shared_view_constraints.json` 提供区间，不将二元left/right视作精确90°标尺。\n"
 text+='''
+## 9. 真实案例与失败分析
+
+以下图均来自真实TRAIN正射地图，叠加原始目标/地标轮廓及GT（仅展示）。绿色箭头为已冻结道路场；局部橙/蓝侧域由整个场计算，不围绕目标旋转。XY来自TIF affine，图像右=+X、上=+Y；本轮另逐一复核24张TRAIN地图的affine与缺失CRS（`raster_geometry_manifest.json`），仍不称+Y为真实北方。图不是原始注释者相机视角。
+
+'''.replace('\n+','\n')
 notes=['船屋面朝河流的完整允许方向范围；最近岸边法向是一个候选而非唯一视角，且该例为已知开发样本。','新增Bragg Road进入语义，来源路及完整中心线均在看GT前确定；道路弯曲由局部场表达。','新地图上的St Peter’s Street进入案例，使用相同规则，未在该图重新拟合轴。','Aylestone Road新语法案例；不是从GT选择道路的哪一面。','Abbey Road反例保留。局部场与语言侧别仍冲突；可能是语义/路径/轮廓/标注差异，未独立确认原因，不自动翻转符号。','目标GT可画出，但完整目标Polygon触及道路中心线，不能单凭质心或GT点放行；输出ambiguous。','修复代词链使主语与具名参照物可绑定，但facing Anatomy Building未给观察原点，仍unknown。','建筑主轴可计算，不等于入口或有向正面；保留候选，不利用目标GT选PCA符号。']
 for c,note in zip(cases,notes):text+=f"\n### `{c['occurrence_id']}` — {c['state']} / {c['geometry_result']}\n\n> {c['instruction']}\n\n{note}\n\n![真实地图案例](figures/{c['figure']})\n"
 text+='''
+## 10. 六个最终问题
+
+1. **哪些语言结构能唯一确定参考坐标系？** 精确观察点A和精确目的点B可唯一构造A→B；当前只有区域Polygon时必须保留方向范围。A/B显式语义明确不等于数值yaw唯一。道路进入结构在独立简单连通中心线、可定位进入端且分辨率稳定时能确定有向场；本轮44条、没有单一数值观察yaw的独立唯一样本。
+2. **能否解释位置？** 在44条冻结唯一道路场关系中，22条完整几何可判定，21符合、1冲突；95% Wilson CI为78.20%–99.19%。其余22条保持几何ambiguous，全部唯一支持确认率47.73%。这是一种有限支持，不是全CityNav统一规律的证明。
+3. **哪些缺信息？** 只说facing地标而无观察原点、来源/瞄准区域过宽、along无行进符号、岔路/中部进入、无入口/正面的建筑PCA、没有明确共享作用域的多地标描述，以及绑定/主语未知。
+4. **统一语义生成机制？** “语义角色确定原点/锚/方向约束，场景几何实例化，缺信息时输出多解”是可统一表达A–F的组合机制，且道路子类有新证据；尚未证明所有描述来自它，也不能因当前低覆盖率或低世界轴一致率否定尚未识别的统一规则。
+5. **相比v3有何新增证据？** 完整TRAIN扫描找出42条新具名道路锚关系并修复7个旧绑定；新语法26条可确定道路场，12条完整几何可判定且均符合，去重11/11。跨全部可判定结果涉及6图/11参照组。v3的最近岸边代理被重新标为区域多解；变化不能用不同分母/谓词的率直接相减。总体对世界/yaw/起点代理的净差CI仍含0，不能宣布稳定超越。
+6. **语言关系图网络该输出什么？** 绑定分布及作用域、逐关系frame_type、原点点/区域分布、面向/来源/行进语义角色与span；方向应能表示标量朝向、可行角区间、多峰±轴或有向道路场/路径分布；另输出绑定unknown、原点unknown、符号unknown、拓扑unknown及证据来源。建筑需独立入口/正面证据，不能监督成PCA符号。用目标GT最优朝向作参考系标签会制造循环证据；应优先补充独立观察点/朝向、道路路径及立面标注，再讨论训练。
+
+## 可复核性与局限
+
+本轮不训练任何分类器、BERT、GeomBelief或导航模型。阶段SHA、代码与输入指纹、全部4701关系、完整字段证据、GT离线评价、所有unknown、置信分层、地图分组与对照均提交。运行命令和文件索引见 [README.md](README.md)，测试与复算结果见 `results/verification.json`，实际外部审查状态见 [REVIEW.md](REVIEW.md)。
+
+高条件率只针对能独立绑定、几何场可确定且轮廓不含边界歧义的小子集；重复模板/对象、地图少、短支路处理、对象轮廓粗糙与未校准的观察点均限制外推。所有参数仍是探索性选择，不是预注册协议。不存在作者视角标签，所以本轮测的是独立可构造性和几何吻合，不是识别真实描述者参考系的准确率。
+'''.replace('\n+','\n')
 (BASE4/'REPORT.md').write_text(text);print('Chinese report generated')
if __name__=='__main__':run()
