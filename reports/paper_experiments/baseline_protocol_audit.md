# PolyBelief 外部 Baseline 协议审计

本轮以 SBFNav Table 1/2 为框架；仅原论文数值进入 paper_reported 表。
table1_ours_unified.csv 的“统一”指相同 revised Episode 集与共同 benchmark_2d 指标重算，保留各方法原生控制器。
它不表示统一网络输入权限/控制器/训练或严格论文复现。comparable=False 及missing列中的限制必须随表保留；不能据此直接做方法优劣归因。
original数据、替代API模型、只有历史文档或聚合值的记录进入 table1_local_other_protocols.csv，绝不伪装为统一实测。

## 原始资源与输入输出

|方法|输入与输出|权重|代码或论文|可比性限制|
|---|---|---|---|---|
|Random|已有无模型策略；输入姿态，输出离散动作|无checkpoint|CityNav-GSM/htnav/baselines/random_policy.py|历史21步ballistic/stop策略不同于论文未明确策略；eval_random有fit选项，禁止调用fit或追论文数值|
|Seq2Seq+GSM|RGB/depth/text/GSM→动作|历史checkpoint链接失效|https://github.com/water-cookie/citynav|历史original/refined、cache/gdino、imagery配置不混排；官方无GSM权重不能替代+GSM|
|CMA+GSM|交叉模态RGB/depth/text/GSM→动作|历史checkpoint链接失效|https://github.com/water-cookie/citynav|SBFNav年代/会议标签按原样存档；优先以CityNav适配资源确认方法身份|
|MGP|地图/语言→目标+progress→控制动作|官方228385010字节权重已恢复并核验SHA；旧本地引用仍不替换|https://github.com/water-cookie/citynav|已有三个native版本，离线缓存、目标后处理与progress停止差异不能算同一protocol|
|AerialVLN+GSM|RGB/depth/text/GSM→双GRU离散动作|本地best.pt SHA见索引；作者CityNav适配权重未找到|https://github.com/AirVLN/AirVLN|本地original三split已完成；alignment修正训练未完成；不是作者适配的严格复现|
|FlightGPT|当前/地图图像+语言→VLM动作|https://huggingface.co/ADJHD/Flightgpt；revision9eb6253；本机四shard缺失|https://github.com/Uavln/FlightGPT|约16.6GB下载；还缺安全split/ID/resume evaluator、双图加载和完整native协议核查|
|GeoNav|多尺度地图/场景图+语言→高层规划+定位|作者GPT-4o；本地DeepSeek Flash替换|https://github.com/Xhtshr/geonav-official|original Test5311已有完整逐条指标；Seen/Unseen与原GPT-4o缺失；不能把替换模型当原模型|
|HTNav|RGB/depth/text/地图→宏waypoint+微动作/value|官方README仅上传中、未公布权重；本地两配置|https://github.com/tyrofanfan/HTNav|reproduction三split、alignment仅两validation；结构/IL-RL损失假设未闭合；不能择优跨配置拼表|
|HETT|语言/地图/因果视觉历史→粗目标及细动作|DATA/hett/release/checkpoint/best_val_unseen存在|https://github.com/crotonyl/HETT|官方发布权重选择名best_val_unseen照实标注；HETT/main为含轨迹belief改动的研究checkout，不能充作官方方法。隔离补依赖后help通过，仍缺兼容上游源码与隔离导出|
|SBFNav|观测RGB-D/静态轮廓/语言→belief候选+语义几何选择→控制|本地独立重实现epoch12/14；不是官方权重|https://arxiv.org/html/2609.05841v1|历史用Val-Unseen选模；epoch12三split和epoch14验证分开保留；补充B.4/B.6/B.7发布缺失，假设必须保留|
|PolyBelief|命名多边形/语言/姿态+因果RGB→目标belief、轨迹与Stop|冻结完整模型SHA见table1；本轮不改方法|/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav|28x28/stop0.3；revised2470/2697/5281；384及新校准残差分别为消融，不能相加|

## Revised Dataset 的准确边界

SBFNav Table 1 的 * 明确表示 revised dataset；正文称修订8437条错误指令并移除331条无有效地标引用的样本。
本机 HETT 发布版实际从32637变为32326，共删除311条，而非331。HTNav论文所述修订也为311条，但不能只凭数量断定两个发布版本完全相同。
本机逐 split 的文件SHA、ID差集、文本/目标/轨迹差异已实际计算，见 dataset_version_audit.json。
原始与修订导航split的保留条目中，指令文本、目标和轨迹均未改变；差异在过滤及processed_descriptions地标元数据，后者35196条中18038条变化，详见landmark_annotation_audit.json。此统计口径不能替代论文8437条的声明。
当前2470/2697/5281直接来自已发布的refined JSON，非为PolyBelief新增筛选；本轮验证预测ID完整覆盖，失败Episode保留分母。
SBFNav论文未给修订文件SHA，故只能确认同属HETT修订数据来源，不能声称与作者使用文件逐字节相同。
“8437条”是论文报告的数据修订声明；本机split文本diff不是相同统计口径，不将二者强行等同。

|split|original|revised|removed|added|instruction_changed|trajectory_changed|target_changed|maps|
|---|---|---|---|---|---|---|---|---|
|train_seen|22002|21878|124|0|0|0|0|24|
|val_seen|2498|2470|28|0|0|0|0|23|
|val_unseen|2826|2697|129|0|0|0|0|4|
|test_unseen|5311|5281|30|0|0|0|0|6|

## 统一指标与冻结限制

NE=最终XY到GT的欧氏距离；SR=NE≤20m；OSR=执行轨迹任一点到GT≤20m；SPL=SR×起点直线距离/max(实际XY路长,起点距离)。
仅在离线评价层使用GT坐标。新统计代码不调用模型、不训练、不改Checkpoint；原推理输入权限审计沿用冻结协议与视觉残差审计。
HTNav原生SPL的最优长度取离散生成的最短教师轨迹，和共同evaluator的欧氏起终点距离略有不同。统一表保存共同定义重算值，其他协议表保留原生值，两者的逐split差值见verification.json；不强行修正已有文件。
现有SBFNav基线的历史Val-Unseen选模是明确限制，本轮不再用Unseen/Test选模补救；任何需要新训练的严格复现都保留Pending。

原始资源快照URL、SHA和下载状态见 sources/manifest.json；Git分支/工作树及dirty状态见 repository_inventory.json；权重位置、存在性、SHA见 selected_checkpoint_inventory.csv。
14个本地GraphNav研究分支的只读树扫描见branch_inventory.csv。HETT官方权重SHA与兼容阻塞见hett_asset_preflight.json；MGP新恢复权重见mgp_recovered_asset.json，绝不据文件名推定与缺失历史checkpoint相同。
原评估命令以源metrics/queue记录为准；experiment_registry.csv列出可重建命令及证据路径，历史路径仅保留溯源，不能直接照抄运行。
新离线汇总使用 `python -m scripts.build_paper_experiments`，命令与环境另存build_provenance.json。
