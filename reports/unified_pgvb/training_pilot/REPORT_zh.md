# UnifiedPGVB-v2 首轮真实训练Pilot与效率测试

代码基准：main `ce2c6dae`；本次训练/缓存入口代码Commit：`da0e162f7766d028afa7b1272a9ac378de755a7c`。实验分支`experiment/unified-pgvb-v2-training-pilot-20261104`。

## 核心结论

**可以用真实CityNav数据正常训练UnifiedPGVB-v2内的PGVB视觉Head。** 不是旧独立训练器改名：实际调用UnifiedPGVB.forward、保存完整V2模型，并由原V2 Runtime加载。Geometry、SigLIP2、BERT、Trajectory/Stop及joint/direction/target全部冻结。不是全参数或联合融合训练。
1000次有效更新：训练循环及定期保存 **10.93秒**；Python主函数含缓存校验/加载 **17.44秒**；从shell日志创建至退出（含模块导入） **19.23秒**。剔除前20步后9.537ms/步、1541.3查询/s、419.2帧/s。
全部22,002条TRAIN原始起点已扫描，新建可监督池为4159帧/4658查询。完成其全部拟合分区一次Epoch：489步，循环5.54秒，完整启动至退出15.18秒。首次缓存生成（含启动）496.20秒。**已停训，没有运行10 Epoch**。
效果并非全面提升：独立TRAIN Dev局部Loss和地标R20改善，但世界Polygon概率质量下降，融合Belief未增加命中；两条闭环功能Smoke训练前后都是0/2成功。因此效率可行，不应以Loss下降为由立即扩大训练并宣称SR提升。

## 1. 环境、权重与接口

- GPU：NVIDIA RTX PRO 6000 Blackwell Workstation Edition，物理显存约96GB；PyTorch `2.8.0+cu128`，CUDA runtime `12.8`，BF16实际可用。Python：`/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python`。环境完整记录见environment.json。
- 初始GPU已有其他任务约35GB占用，未停止任何其他进程。资源探测脚本因PATH包装器误报无GPU，真实CUDA与`/usr/bin/nvidia-smi`已核实；以environment.json为准。
- Stage1来源：`GraphNav/runs/graphnav_v1_stage1/best.pt`，SHA `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。PGVB来源：`GraphNav/runs/pgvb_local_anchor_v2/pilot_s0/best.pt`，SHA `ba3073723703ce8188c7f74ae810a622342631bb150bd2597656ff0039f2cf56`。本轮再次核验一致。
- PGVB源权重已经历旧缓存12轮训练，本轮是继续训练，不冒充从零学习。迁移本身不算训练。原archive分支、源Checkpoint及main未修改。
- 实测网络输入[1, 3, 384, 384]、原生patch[1, 576, 1024]（24×24）、稀疏多尺度缓存[1, 144, 3, 1024]（144位置×3尺度×1024）。每个正常RGB帧只做一次SigLIP2 Vision forward，多个查询复用。
- 29项原专项测试、5项训练专项测试合计34项通过；另有真实GPU梯度、原生RGB重算、保存加载以及两次各2条原协议闭环Smoke。

## 2. 数据规模、监督与隔离

全量TRAIN原文22,002条；保守公开名称匹配指令20854条，绑定33349次（同条可重叠）：Building 13778、Road 18536、Other含Ground 1035。只是名称存在统计，不是人工语言解析准确率。
原始t=0几何候选8,591个不同位姿，RGB/Depth/Polygon质量检查后4,159帧、4,658查询；无可靠局部监督4,432个位姿，另1个起点高度不合法。有效监督对应4159条原始指令；其余17843条无本协议下可用的引用地标视觉监督（含未绑定/视野外/不支持类型/质量不足，不能统称目标不可见）。实际可见性仍未知。RGB来自原正射栅格在真实原始起点的裁剪，不是真实三维透视/遮挡重建。

|数据协议|拟合帧 / 查询|独立Dev帧 / 查询|用途|
|---|---:|---:|---|
|旧完整可用缓存（固定yaw0，含公开地标附近增强）|1711 / 6275|345 / 701|A/B及旧缓存Epoch效率测试|
|全量TRAIN原始t=0（原始yaw）|1956 / 2266|2203 / 2392|真正的全量可用t=0监督池Epoch|

新池拟合Building550/Road1716，Dev Building1122/Road1270；Other不造视觉标签，继续保留几何回退。新池不是按容易成功的任务截取，而是全TRAIN固定t=0协议的全部有效支持。它不等于TRAIN所有未来时刻；两种缓存域/每帧查询数不同，不能以查询吞吐差异当作代码提速。
Cambridge拟合/Birmingham验证，城市、地图、物理公开名称及RGB SHA均隔离。所有新观察明确TRAIN、step0，仅原始起点，不提供教师轨迹、未来图像或CityNav目标GT作为网络输入。64条Val-Seen只用于既有前2条运行时Smoke，未进入训练、调参或Checkpoint选择；Smoke RGB与训练池无SHA重叠。完整审计见data_audit.json、full_data_audit.json。冻结Stage1的历史TRAIN覆盖并未重置；这里的独立是本次PGVB增量拟合隔离，不是整个系统从零未见地图泛化。
新池4159帧缓存全覆盖，原旧池精确RGB+Depth命中0：原始位姿/yaw与增强缓存不同，不能错误复用。缺失帧已增量编码；每128个候选保存Chunk进度，原缓存保留。未知/边界patch标签−1，内缩0.5m后覆盖≥90%为正、外扩覆盖≤1%为负；正区域少于2或有效来源不足时弃权。标签是公开Polygon投影弱监督，不是人工像素Gold或遮挡真值。

## 3. 训练范围与真实执行

UnifiedPGVB核心总参数946,298，实际训练226,306（19个PGVB参数张量）；其余719,992冻结。SigLIP2/BERT在运行时冻结、训练使用其常量缓存；控制器权重未进入优化器。19/19训练张量实测非零有限梯度，冻结参数逐位不变。
Loss=可靠区域概率损失+0.2平衡BCE+0.3同帧同类可靠互负对比。只有正区域的样本使用正区域内KL，不把未知当负；重叠/不互为可靠负区域的同类对象不强制对比。所有损失FP32，前向CUDA BF16，AdamW lr1e−4/weight_decay1e−3，梯度裁剪5，seed42。
|阶段|有效更新|循环含保存(s)|完整进程(s)|稳态ms/步|查询/s|峰值GiB|首20 / 末20 Loss|
|---|---:|---:|---:|---:|---:|---:|---:|
|A200|200|2.70|10.92|9.577|1498.4|0.122|0.1487 / 0.1389|
|B1000|1000|10.93|19.23|9.537|1541.3|0.135|0.1299 / 0.0969|
|Cepoch|512|6.48|14.74|10.666|1379.6|0.127|0.1103 / 0.0994|
|Dfull_epoch|489|5.54|15.18|9.322|498.0|0.056|0.9200 / 0.7069|

A200→B1000从第200步恢复AdamW的19组状态、epoch0/offset800；C补齐未完成Epoch后再完成一轮，最终旧池共4轮/1712步。D是全部新t=0拟合池489步/1轮，覆盖1956帧和2266查询，在C权重基础上因数据指纹变化重建优化器，不能伪称同一缓存续训。D之后未追加训练。
每100步保存last/best；best仅按最近100步TRAIN损失，不按Dev或两条闭环选。所有比较使用固定预算的last。保存完整模型/优化器/RNG/采样游标、数据和代码SHA，改变指纹拒绝续训。真实数据下一步保存/恢复与不中断执行最大参数差0，损失也完全一致。

## 4. 时间与显存分解

B1000稳态（每步同步CUDA，前20步剔除）：

|分量|平均ms|
|---|---:|
|data_wait_ms|0.692|
|transfer_ms|1.713|
|forward_loss_ms|3.467|
|backward_ms|2.412|
|optimizer_ms|0.955|
|完整Update工作Wall（保存/日志另计）|9.537|

PGVB-only训练每步Geometry计算为0：冻结静态Geometry对本局部损失没有必要重复运行。真实独立TRAIN查询实测Geometry一次GPU前向平均5.00ms，静态BERT及数据组装平均34.57ms，Episode内缓存；没有独立GraphNav再做一次Geometry。
新t=0缓存实际生成主函数493.65s，完整进程496.20s。已编码帧中SigLIP2+pool累计192.85s，RGB渲染30.86s，局部监督47.59s；其余包含被拒绝帧检查、PNG/Chunk/整库保存、旧缓存查找，未将残差全归咎GPU。细分CSV保留。
单独真实RGB检查：冷加载编码器3.85s，第一帧278.2ms，预热后8个不同观测平均45.0ms（含ROI检查/pool）。单编码器进程实测峰值3.323GiB；完整运行时Smoke约3.745GiB。冻结特征训练B峰值0.135GiB不包含未加载的视觉主干，不能作为全网络峰值。
原生RGB重算与缓存特征MAE约1.3e−6，2组检查均每图1次Vision forward，未发现缓存维度/权重错配。B1000设备总利用率10次采样70–88%、平均83.6%；有其他进程，不能冒充本任务独占SM利用率。每100步耗时和设备采样完整保留。

## 5. 完整训练成本

以下仅当前PGVB视觉Head、batch4帧、全部新t=0拟合池；1 Epoch为实测，3/5/10线性外推，不是已经运行。单进程启动/读缓存计一次，保存频率沿用实测；共享GPU及文件页缓存状态会影响复测；预训练权重已在本地，估计不含下载。首次包括整个TRAIN+Dev可监督t=0缓存生成，不包含工程开发时间。

|Epochs|已有缓存(s)|首次含缓存(s / min)|
|---|---:|---:|
|1|15.18|511.38 / 8.52|
|3|26.26|522.46 / 8.71|
|5|37.34|533.54 / 8.89|
|10|65.04|561.24 / 9.35|

不能把旧池1000步查询吞吐直接乘原始22,002条来估算；这里先实测全量原始t=0监督池的实际分母、DataLoader和完整Epoch。没有将Geometry、视觉主干或联合融合的解冻成本混进该表；若以后微调视觉编码器，旧缓存必须失效并另测。

## 6. 独立TRAIN效果与真实Belief

全量新池独立Birmingham Dev2392查询，以下是**视觉世界Belief与公开Polygon的弱监督指标**，不是原始CityNav目标定位/SR：

|方法|局部Loss|Polygon概率质量|地标R20|地标Polygon NE(m)|
|---|---:|---:|---:|---:|
|迁移前训练权重|1.2046|0.5809|87.92%|8.165|
|本轮训练后|1.0044|0.5665|89.38%|7.285|
|训练后RGB置零|2.9454|0.1306|24.54%|80.253|

|类别|数量|训练前→后 R20|训练前→后 NE|训练前→后概率质量|
|---|---:|---:|---:|---:|
|Building|1122|91.00%→92.07%|5.89→5.05|0.6686→0.6471|
|Road|1270|85.20%→87.01%|10.17→9.26|0.5034→0.4952|

Loss与峰值误差改善，但概率质量降低：不能只选正指标宣称全面提升。RGB特征置零只证明特征依赖，不能单独证明真实实例外观识别；本轮没有人工像素Gold。旧缓存独立Dev701查询结果另存cached_dev_evaluation.json（其输入为另一种观察协议，不与新池拼接汇总）。
在固定24个TRAIN城市场景地标查询中，用同一Geometry/原融合公式比较全局Belief：

|方法|Polygon概率质量|地标R20|NE(m)|
|---|---:|---:|---:|
|geometry_only|0.26426|100.00%|6.042|
|before|0.28993|100.00%|4.299|
|after|0.29025|100.00%|4.746|

该小组R20未增加，融合后NE略变差；不能把局部视觉指标当作联合融合成功。joint融合头尚无本轮训练/启用；仅原bounded融合。真实RGB、Polygon、多片段Building/Road与前后Belief见index.html，图中只有公开地标代理，不画CityNav目标GT。

## 7. 两条真实闭环功能Smoke

同一固定Val-Seen64清单的前2条，完整原控制器/Horizon/Stop权重与阈值，训练前后均0/2成功、OSR1/2。Checkpoint正常加载、输出有限、真实RGB逐步更新，一条任务有9次有效PGVB更新，delta约0.499；但没有SR收益。不能声称历史61/64被复现，也未运行其余62条来挑选结果。
- Celine Grove：原Geometry Goal约26m离开公开地标，从未进入20m；PGVB没有有效本地证据，保留几何；停止在外侧。
- Church：曾进入约7.66m，后续Goal正确性丢失2次，最终约24.97m停止。训练后同样失败，未调整Stop/到达规则。本轮只记录，不用这两条Val任务反向调参。

## 8. Checkpoint与复现

所有大缓存、原图和权重留在服务器；根目录：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42`。

|权重|路径（相对运行根）|SHA256|
|---|---|---|
|migrated|`migrated.pt`|`7a6ec452fff73bef574451901c87c82deaccbfd937f953d839fb6075665c3552`|
|cached_final|`cached_pilot_final.pt`|`c577eef6ae316a0a38a265004caa2f2da4eaeadc73cf0f44473b34d0a6637d51`|
|final|`full_training/last.pt`|`379cfc419ef098dc40ad85daa29419340058f46ae8a359415054625594d6c812`|
|best_fit_window|`full_training/best.pt`|`48cd96c659df34d72ca0243518ea4baaa951a00ac6cb39e96c630881aca4f8b2`|

训练配置见training_config.json、full_training_config.json以及仓库configs/unified_pgvb_training_pilot.json；tmux启动脚本/逐步日志随报告保存。恢复使用原数据/初始化权重指纹，加`--resume --until-updates <更大步数> --stage <新阶段名>`，不会从头重置优化器。缓存按Chunk恢复需`--resume`，完成缓存拒绝覆盖。
长作业均在tmux中执行，现已完成退出，原后台服务和历史任务未干预。仅保存/恢复探针的Checkpoint不参与训练、选择或闭环。源权重SHA结束时再次检查一致。

## 9. 决策

**训练机制和显存已经可用，主要开销在第一次RGB/Polygon准备和冻结特征提取，而不是PGVB反向。** 后续可优化首次图像提取的批量化、被拒绝ROI的CPU检查以及大缓存加载/分片；不需要换更大GPU来训练这个Head。
**不建议立即自动启动多轮完整训练追逐Loss。** 全量独立Dev概率质量下降、融合R20不变、两条功能Smoke均失败，尚无闭环增益证据。应先核对弱监督/观测域差异与“局部评分→全局Belief→停止”的作用，明确适合的联合监督后再单独申请/设计融合训练。本轮严格保持joint/direction/appearance未训练，不改Stop，不恢复暂停验证。
