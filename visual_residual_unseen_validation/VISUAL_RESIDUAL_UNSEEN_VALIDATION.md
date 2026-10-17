# Calibrated Visual Residual：Val-Unseen 泛化与地图级统计验证

本报告只分析既有 Calibrated Residual，不重新训练、不重新选模型或阈值，不包含384×384实验。

## 配置与复现

- 原 GraphNav-V1 C Epoch1 SHA256：`00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。
- Calibrated Residual SHA256：`5fc1a3e115b71d14fe1e8fbaf3ed29c1863d8b2bc080455d89f7a4e503d25f11`。
- 目标网格28×28；校正Stop分数阈值0.3；原1毫米预测目标到达门控与高度门控保持不变；原引用地标输入。
- 动作预算20次决策，水平每步≤50m、垂直≤10m；SR按最终XY距离≤20m，停止/超时/停滞均依相同最终位置计分。
- 原几何、视觉编码器、匹配器、历史模块、轨迹/Stop及新残差头全部冻结；Unseen启动时逐参数检查requires_grad=False。
- 残差结构：视觉/文本1024→32投影、静态上下文128→16投影、106→64→2 MLP；中心化修正×置信度，有界幅度2.0，保留原use_pose=True。原训练3轮，固定复用Train Dev选出的第1轮best.pt。
- 训练仅使用原Train缓存及Train Dev整组划分选出的权重。Seen四组完整2470条原样复用，Unseen每组完整2697条。
- 四组均运行同一残差适配器：Real真实当前/历史特征；No-belief将残差置零；Zero将当前与历史视觉特征置零；Shuffle按固定种子0在每个有效尺度内打乱视觉向量与候选位置的对应，位置/有效性/质量掩码不变。
- 旧evaluator仅允许Seen残差、且held-out只允许real；本次最小修复加入独立冻结manifest，约束固定权重、完整Unseen清单和四控制。原正式模型/Test检查未放宽。Seen算法和已有输出未改变。
- 完整命令、PID、运行时间与返回码：`visual_residual_unseen_validation/queue.json`；日志为同目录各control.log。
- 逐episode原始输出：`visual_evaluations/calibrated_residual_full_seen_{control}/`及`calibrated_residual_full_unseen_{control}/`。保留predictions.jsonl、poses、map_name、metrics、termination、episodes.json。
- 版本与输入SHA：`launch_record.json`、`source_sha256.json`、`source_snapshot/`、`analysis_version.json`及本报告的metrics.json。
- 复现：在本worktree使用Baselines/.reproduction-env/bin/python运行`-m scripts.validate_visual_residual_unseen`；完成项验证后跳过，未完成项使用--resume。独立统计重建：`-m scripts.analyze_visual_residual_unseen`。

## 四组完整结果（Episode加权）

|Split|地图簇数|控制|Episodes|SR%|SPL%|OSR%|NE(m)|False-Stop/Stop%|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|val_seen|23|real|2470|48.7045|46.4941|58.1781|28.9303|51.2955|
|val_seen|23|no-belief|2470|47.3684|45.9585|55.3846|29.2530|52.6316|
|val_seen|23|zero|2470|47.3684|45.9585|55.3846|29.2530|52.6316|
|val_seen|23|shuffle|2470|46.9231|43.4193|59.5547|29.0381|53.0579|
|val_unseen|4|real|2697|23.9525|22.7625|34.5569|50.6818|76.0119|
|val_unseen|4|no-belief|2697|22.1357|21.5296|31.2940|51.7575|77.8479|
|val_unseen|4|zero|2697|22.1357|21.5296|31.2940|51.7575|77.8479|
|val_unseen|4|shuffle|2697|22.6177|20.8056|34.7423|51.6329|77.3403|

False-Stop Rate = 失败且主动Stop的episode数 / 主动Stop的episode数；分母为0时NA。所有差值使用同一残差适配器中的No-belief对照，不替换成原正式完整模型的指标。

## 地图等权结果

|Split|控制|地图数|SR%|SPL%|OSR%|NE(m)|平均地图False-Stop/Stop%|
|---|---|---:|---:|---:|---:|---:|---:|
|val_seen|real|23|44.4738|42.3431|52.8035|31.4400|55.5262|
|val_seen|no-belief|23|43.1123|41.6085|50.4276|32.5148|56.8877|
|val_seen|zero|23|43.1123|41.6085|50.4276|32.5148|56.8877|
|val_seen|shuffle|23|43.2525|39.9193|53.7787|31.5031|56.7399|
|val_unseen|real|4|23.6931|22.5156|33.9521|50.1254|76.2786|
|val_unseen|no-belief|4|22.3545|21.7367|31.2096|51.1025|77.6273|
|val_unseen|zero|4|22.3545|21.7367|31.2096|51.1025|77.6273|
|val_unseen|shuffle|4|22.5694|20.9168|34.1198|50.9941|77.3906|

地图等权对各地图指标取算术平均，地图大小不影响权重；False-Stop率只平均具有主动Stop的地图，maps_with_policy_stop见metrics.json。

## 配对效应与置信区间

每个split内部按episode_id严格配对，校验map_name、索引、清单及权重SHA；Seen与Unseen之间不配对。每项bootstrap 20,000次，episode随机种子20261017，地图种子20261018。区间为点态percentile 95% CI，未作多重比较校正，预先声明的主要比较为Unseen Real−No-belief。

Episode bootstrap仅作为诊断，假设episode独立过强；主要泛化判断采用整地图重采样。每次抽取M张地图（有放回），地图内保留全部episode，重复抽中的地图整体重复。Episode加权使用抽样地图效应总和/episode总数；地图等权使用抽样地图均值的均值。

### ΔSR（Real−控制，单位pp）

|Split|控制|Episode加权Δ|Episode bootstrap CI|地图cluster CI（Episode加权）|地图等权Δ|地图cluster CI（地图等权）|
|---|---|---:|---|---|---:|---|
|val_seen|no-belief|+1.3360|[+0.202, +2.510]|[-0.185, +2.770]|+1.3616|[-0.067, +2.798]|
|val_seen|zero|+1.3360|[+0.202, +2.510]|[-0.185, +2.770]|+1.3616|[-0.067, +2.798]|
|val_seen|shuffle|+1.7814|[+0.972, +2.591]|[+0.899, +2.644]|+1.2214|[+0.217, +2.191]|
|val_unseen|no-belief|+1.8168|[+0.853, +2.781]|[+0.000, +3.441]|+1.3386|[+0.000, +3.118]|
|val_unseen|zero|+1.8168|[+0.853, +2.781]|[+0.000, +3.441]|+1.3386|[+0.000, +3.118]|
|val_unseen|shuffle|+1.3348|[+0.630, +2.039]|[+0.556, +2.234]|+1.1238|[+0.475, +2.078]|

### ΔSPL（Real−控制，单位pp）

|Split|控制|Episode加权Δ|Episode bootstrap CI|地图cluster CI（Episode加权）|地图等权Δ|地图cluster CI（地图等权）|
|---|---|---:|---|---|---:|---|
|val_seen|no-belief|+0.5356|[-0.530, +1.616]|[-0.841, +1.830]|+0.7346|[-0.672, +2.139]|
|val_seen|zero|+0.5356|[-0.530, +1.616]|[-0.841, +1.830]|+0.7346|[-0.672, +2.139]|
|val_seen|shuffle|+3.0748|[+2.309, +3.843]|[+1.969, +4.147]|+2.4238|[+1.163, +3.760]|
|val_unseen|no-belief|+1.2329|[+0.344, +2.126]|[-0.656, +2.796]|+0.7789|[-0.569, +2.528]|
|val_unseen|zero|+1.2329|[+0.344, +2.126]|[-0.656, +2.796]|+0.7789|[-0.569, +2.528]|
|val_unseen|shuffle|+1.9569|[+1.325, +2.594]|[+0.648, +3.289]|+1.5988|[+0.445, +3.030]|

### 改善/退化/不变的episode数量

|Split|控制|SR改善|SR退化|SR不变|SPL改善|SPL退化|SPL不变|
|---|---|---:|---:|---:|---:|---:|---:|
|val_seen|no-belief|120|87|2263|404|395|1671|
|val_seen|zero|120|87|2263|404|395|1671|
|val_seen|shuffle|75|31|2364|522|240|1708|
|val_unseen|no-belief|110|61|2526|228|205|2264|
|val_unseen|zero|110|61|2526|228|205|2264|
|val_unseen|shuffle|64|28|2605|263|130|2304|

## 逐地图统计

|Split|地图|Episodes|Real SR%|No-belief SR%|Zero SR%|Shuffle SR%|ΔSR vs No-belief pp|ΔSR vs Zero pp|ΔSR vs Shuffle pp|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|val_seen|birmingham_block_1|320|65.000|60.625|60.625|60.938|+4.375|+4.375|+4.063|
|val_seen|birmingham_block_10|37|32.432|32.432|32.432|32.432|+0.000|+0.000|+0.000|
|val_seen|birmingham_block_11|2|0.000|0.000|0.000|0.000|+0.000|+0.000|+0.000|
|val_seen|birmingham_block_12|81|49.383|43.210|43.210|45.679|+6.173|+6.173|+3.704|
|val_seen|birmingham_block_3|161|30.435|24.845|24.845|27.950|+5.590|+5.590|+2.484|
|val_seen|birmingham_block_4|100|40.000|38.000|38.000|37.000|+2.000|+2.000|+3.000|
|val_seen|birmingham_block_7|39|43.590|41.026|41.026|38.462|+2.564|+2.564|+5.128|
|val_seen|birmingham_block_9|105|47.619|42.857|42.857|40.952|+4.762|+4.762|+6.667|
|val_seen|cambridge_block_12|157|49.045|47.134|47.134|47.771|+1.911|+1.911|+1.274|
|val_seen|cambridge_block_13|190|51.579|55.263|55.263|51.579|-3.684|-3.684|+0.000|
|val_seen|cambridge_block_17|44|18.182|18.182|18.182|18.182|+0.000|+0.000|+0.000|
|val_seen|cambridge_block_18|60|76.667|78.333|78.333|75.000|-1.667|-1.667|+1.667|
|val_seen|cambridge_block_19|47|46.809|48.936|48.936|53.191|-2.128|-2.128|-6.383|
|val_seen|cambridge_block_20|207|42.512|41.063|41.063|41.063|+1.449|+1.449|+1.449|
|val_seen|cambridge_block_21|190|43.684|46.316|46.316|42.105|-2.632|-2.632|+1.579|
|val_seen|cambridge_block_23|32|53.125|50.000|50.000|53.125|+3.125|+3.125|+0.000|
|val_seen|cambridge_block_25|23|65.217|56.522|56.522|65.217|+8.696|+8.696|+0.000|
|val_seen|cambridge_block_28|170|54.706|53.529|53.529|53.529|+1.176|+1.176|+1.176|
|val_seen|cambridge_block_33|99|40.404|42.424|42.424|40.404|-2.020|-2.020|+0.000|
|val_seen|cambridge_block_4|31|32.258|25.806|25.806|32.258|+6.452|+6.452|+0.000|
|val_seen|cambridge_block_6|33|36.364|42.424|42.424|36.364|-6.061|-6.061|+0.000|
|val_seen|cambridge_block_7|162|50.000|48.765|48.765|49.383|+1.235|+1.235|+0.617|
|val_seen|cambridge_block_9|180|53.889|53.889|53.889|52.222|+0.000|+0.000|+1.667|
|val_unseen|birmingham_block_5|914|16.193|12.035|12.035|13.676|+4.158|+4.158|+2.516|
|val_unseen|cambridge_block_10|919|28.074|26.877|26.877|27.312|+1.197|+1.197|+0.762|
|val_unseen|cambridge_block_2|293|17.406|17.406|17.406|17.065|+0.000|+0.000|+0.341|
|val_unseen|cambridge_block_3|571|33.100|33.100|33.100|32.224|+0.000|+0.000|+0.876|

Val-Seen有23个地图簇；Val-Unseen只有4个地图簇（两个城市中的地图块）。这是重采样单元数，不是2697个独立泛化单元，也不保证同城地图彼此完全独立。Unseen地图数过少，cluster区间非常离散、对单张地图敏感，无法覆盖未观测地图分布的不确定性。4张地图每次抽4张，实际只有35种地图计数组合；增加bootstrap次数不会增加独立地图信息。

## Zero / No-belief 行为验收

- val_seen：2470条，行为差异0条，最大位姿坐标差0m，最大Stop分数差0。
- val_unseen：2697条，行为差异0条，最大位姿坐标差0m，最大Stop分数差0。

比较全部逐episode位姿数组、终止原因、SR/SPL/OSR/NE/步数/路径长度，以及逐决策目标坐标、信念候选和Stop分数；不比较受运行负载影响的计时或仅外观诊断值。若出现差异，原样列于metrics.json，不修改预测以强制一致。Zero无有效非零视觉向量，因此修正为零；No-belief直接将修正置零，二者预期回退到同一静态信念。

## 改善与损害案例

从Real相对No-belief的成功翻转样本中，按|ΔNE|降序，每种方向优先选不同地图的最多两条；这是可复现的代表性诊断，不代替总体统计。

|Split|方向|Episode ID|Real SR/SPL%/NE(m)|No-belief SR/SPL%/NE(m)|Real/控制终止|
|---|---|---|---|---|---|
|val_seen|improved|`['birmingham_block_12', 148, 0]`|1/100.00/7.70|0/0.00/96.16|policy_stop/policy_stop|
|val_seen|improved|`['cambridge_block_6', 22, 5]`|1/84.86/16.32|0/0.00/91.00|policy_stop/policy_stop|
|val_seen|degraded|`['cambridge_block_28', 271, 0]`|0/0.00/98.07|1/76.07/18.32|policy_stop/policy_stop|
|val_seen|degraded|`['birmingham_block_3', 36, 3]`|0/0.00/94.75|1/91.25/19.92|policy_stop/policy_stop|
|val_unseen|improved|`['birmingham_block_5', 128, 2]`|1/100.00/19.59|0/0.00/272.23|policy_stop/policy_stop|
|val_unseen|improved|`['cambridge_block_3', 98, 3]`|1/91.28/10.12|0/0.00/76.11|policy_stop/policy_stop|
|val_unseen|degraded|`['birmingham_block_5', 106, 2]`|0/0.00/57.02|1/94.18/15.51|policy_stop/policy_stop|
|val_unseen|degraded|`['cambridge_block_10', 107, 1]`|0/0.00/45.69|1/95.16/8.98|policy_stop/policy_stop|

案例起点、终点、最终预测目标与最大残差值见metrics.json的typical_episodes；完整决策和轨迹均在对应predictions.jsonl，按episode_id/index定位。

## 私有信息隔离

已核查PublicEpisode字段、_batch_inputs白名单、Stage1.forward和残差forward参数/字典键：目标位置、目标标签、人类未来轨迹未进入推理网络；源代码检查摘要见public_input_audit.json。运行时所有网络参数冻结，残差加载检查base_sha。真实目标仅在rollout完成后的指标计算及本报告事后案例分析中使用。当前/历史视觉由实际观测与严格递增的历史记忆构建，没有未来图像库。

## 结论

- val_seen：Real相对No-belief，Episode加权ΔSR +1.3360pp、ΔSPL +0.5356pp；地图cluster 95% CI分别为[-0.185, +2.770]pp、[-0.841, +1.830]pp。地图等权结果及CI见上表。
- val_unseen：Real相对No-belief，Episode加权ΔSR +1.8168pp、ΔSPL +1.2329pp；地图cluster 95% CI分别为[+0.000, +3.441]pp、[-0.656, +2.796]pp。地图等权结果及CI见上表。
- RGB与空间对应：Unseen Real相对Zero有2299条行为不同，相对Shuffle有2189条不同；Real−Shuffle ΔSR +1.3348pp，地图cluster CI [+0.556, +2.234]pp。这证明是否对视觉干预敏感；只有真实RGB优于打乱且区间支持正效应时，才支持其对应关系带来有用信息。不能仅凭分数变化断言完整语义理解。
- 行为层面的RGB利用：是，非零视觉向量及其地图对应会改变闭环行为，不能解释成完全忽略RGB的静态旁路；这不等价于Unseen任务收益，也不单独证明目标外观语义理解。
- 真实视觉对应优于打乱：Unseen ΔSR及其地图cluster区间支持正效应；仍需与关闭残差对照分别判断，打乱更差不等于优于静态基线。
- Unseen任务点估计：相对No-belief，SR提高、SPL提高；是否具有地图级统计支持按上述cluster区间判断。
- 未满足“Unseen Real−No-belief ΔSR>0且地图cluster 95% CI下界>0”的验收门槛，不能报告“跨地图视觉残差有明确正收益”。即使episode bootstrap显著或点估计为正，也不能替代地图级支持。
- 所有负结果、跨零区间、帮助与损害样本原样保留，没有据此重选权重、训练或调整Stop阈值。

## 输出文件

- metrics.json：完整配置、逐split/控制的两种加权指标、SHA/版本/输入检查、Zero一致性及案例。
- paired_results.csv：Real相对每个控制、逐episode配对指标与结局。
- map_results.csv：每张地图、每控制的指标、权重及Real差值。
- bootstrap_results.json：20000次episode与地图聚类bootstrap、两种地图加权结果及区间。
