# GraphNav 命名与诊断结果（2026-10-13）

工程统一为 **GraphNav — Graph-First Aerial Vision-and-Language Navigation**；正式主模型仍为 GraphNav-V1，冻结 C Epoch1。
导航消融使用 Val-Seen；附加性能 profiling 使用冻结配置的 Val-Unseen。结果独立存于消融目录，Test 未用于调参。

## HETT 原口径 profiling（整 Episode）

Val-Unseen 128 Episodes，冻结原配置；模型项使用 CUDA Event，rollout/环境使用同步墙钟。单位 ms；总体方差 ms²。

|项目|最小值|均值|总体方差|
|---|---:|---:|---:|
|单样本整 Episode rollout|382.56|681.91|47437.80|
|模型 GPU 推理合计|151.98|268.94|7402.31|
|语言编码器|16.72|32.08|653.60|
|视觉编码器与原生 ROI 池化|69.57|123.54|1025.07|
|静态几何图定位|2.61|6.07|571.11|
|视觉匹配 / 历史 / 残差更新|20.87|59.70|456.91|
|轨迹 / Stop 网络|21.98|47.55|165.79|
|环境观测|29.54|92.25|15523.84|
|其他 rollout 开销|168.49|320.72|6864.22|
|导航策略 GPU 合计（静态 + 残差 + 轨迹）|48.61|113.32|1827.53|

首条整 Episode 2.031s，观测 0.541s；排除首条 rollout 均值 671.28ms，方差 33364.62ms²。

|最慢 Episode|rollout ms|
|---|---:|
|birmingham_block_5 / 1 / 1|2031.13|
|birmingham_block_5 / 142 / 0|1280.46|
|cambridge_block_10 / 1 / 4|1214.28|
|cambridge_block_3 / 33 / 5|1181.04|
|cambridge_block_10 / 136 / 0|1148.00|

已读取原 HETT profiler `multiagent/profile_inference.py` 及其原始 JSON：Val-Unseen 2697 Episodes；rollout 均值 200.82ms，GPU 合计 148.16ms。两者均统计整 Episode（非单步）。GraphNav 本轮是 128 条子集，GPU 存在 HTNav 并发训练，且额外同步的细分阶段会增加开销，不能将两者直接解释成模型速度倍率。
CUDA Event 间隔遵循原 HETT 模块 hook 口径，可能包含主机发射间隙；视觉项包括原生 ROI 池化，残差项排除嵌套轨迹头。Episode 其他项按总墙钟减观测墙钟与 GPU 合计计算。

## 逐决策 profiling（定位瓶颈）

128 Episodes，共 1044 条逐观测/决策样本；单位 ms，总体方差 ms²。

|项目|最小值|均值|总体方差|
|---|---:|---:|---:|
|单样本 rollout|40.19|85.72|6764.73|
|模型阶段合计（同步墙钟）|33.01|59.67|1364.36|
|语言编码器（BERT + SigLIP 文本）|0.00|6.17|434.10|
|视觉编码器与 ROI 池化|25.49|40.30|74.24|
|静态几何图定位|0.00|0.74|77.94|
|视觉匹配 / 历史 / 残差更新|2.50|7.26|59.96|
|轨迹 / Stop 网络|4.02|5.21|3.19|
|环境观测|4.10|12.93|2532.15|
|其他 rollout 开销|2.56|13.11|354.79|

首条 1.555s，环境观测 0.456s。排除首条 rollout 均值 84.31ms，方差 4699.80ms²。

模型加载不计入 rollout。语言与静态定位在 episode 初次决策计算/缓存，后续零耗时代表复用。模块用 CUDA synchronize + perf_counter 计时，包含模块内部 CPU/传输；模型阶段合计不是纯 GPU kernel 时间，不能直接标成纯 GPU 推理。轨迹头在模型输出与原控制器内各调用一次，均计入轨迹项。残差项排除嵌套的轨迹与静态模块；其他项含数据准备、记忆更新、诊断与控制器开销。
当前 HTNav 训练并发使用相同 GPU；128 Seen 子集与用户 HETT 历史样本不相同，也未在同一环境重跑 HETT，不能据此声称正式加速倍率。

|最慢样本（episode / 决策步）|rollout ms|
|---|---:|
|birmingham_block_1 / 11 / 1 / step 0|1554.87|
|cambridge_block_20 / 4 / 0 / step 0|646.82|
|cambridge_block_9 / 43 / 2 / step 0|644.24|
|cambridge_block_21 / 20 / 1 / step 0|636.78|
|cambridge_block_6 / 27 / 4 / step 0|631.41|

## Heatmap 分辨率

将真实全局候选网格从 28×28 改为 384×384（照片原生 384×384），未用插值替代网络推理。候选数 784→147456，为 188.08 倍；同一冻结权重用于分辨率迁移诊断，未重训。注意：网格覆盖整个地图，照片覆盖局部视野，数组尺寸相同不意味着地面米/像素一致；SigLIP patch token 仍为 24×24，网格加密不会增加照片特征的原生分辨率。

|网格|Episodes|SR%|SPL%|NE m|rollout均值 ms|峰值 GiB|
|---|---:|---:|---:|---:|---:|---:|
|28×28|4|50.00|48.95|28.11|136.70|3.76|
|384×384|4|75.00|70.19|23.02|3144.26|12.32|

仅 4 个均匀索引 episodes 的可行性测试；不据此选取正式分辨率或宣称泛化提升。

## Stop 阈值闭环对照

所有阈值使用同一 128 条均匀索引 Episodes，重新执行完整闭环，不从截断轨迹推测较高阈值行为。阈值作用于 positive_weight=20 校准后的 Stop 概率；原有目标距离与高度 guard 仍参与停止判定。

|阈值|SR%|SPL%|NE m|False Stop / Stop%|平均步数|终止原因|
|---|---:|---:|---:|---:|---:|---|
|0.1|50.00|49.14|28.00|50.00|7.16|{'policy_stop': 128}|
|0.3|50.00|49.14|28.00|50.00|7.16|{'policy_stop': 128}|
|0.5|49.22|48.36|28.11|50.39|7.64|{'policy_stop': 127, 'horizon': 1}|
|0.7|49.22|48.36|28.11|48.62|10.61|{'policy_stop': 109, 'horizon': 17, 'stagnation': 2}|
|0.9|49.22|48.36|28.20|60.00|18.98|{'horizon': 102, 'stagnation': 21, 'policy_stop': 5}|

核验：各档实际运行阈值与命令一致，episode manifest 和 checkpoint SHA 相同。子集结果不足以替换正式 0.3 阈值。

停止判定同时需要到达预测的 Top1 目标（距离≤1e-3m）和高度合法。因此更高阈值可能让已到达预测目标的无人机继续重复决策，最终 stagnation 或 horizon；不会自动换目标。各档 guard 通过但概率不足的次数见 `stop_analysis.json`。本协议 SR 按终点距离统计，达到预算/停滞也可能记为成功，必须连同 policy_stop 数量查看。

0.3 阈值与原正式结果核对：128 条中 SR/终止原因改变 0 条；轨迹长度改变 1 条，同长度轨迹最大 XYZ 偏差 0.109375m。子集独立运行与原整 split 运行存在数值/缓存执行差异，不能宣称逐坐标完全重现；新阈值比较均采用同一子集独立运行口径。

## 视觉残差对照

同一 128 episodes，完整模型 SR=50.00%，无 belief 残差 SR=50.78%，差 -0.78 个百分点；SPL 差 -0.85 个百分点，NE 差 +0.02m。无残差对照仍执行视觉编码/匹配，仅将 belief logit 修正置零，不能解释成去掉视觉的加速测试。小子集不代表总体提升。

视觉信念检索（去重查询）：初始 static R@1=50.78%，首次 RGB R@1=50.78%，最终 RGB/history R@1=50.00%。最终时刻因 episode 而异，该时间对照不单独证明因果收益，因果对照参见 no-belief。

## 复现与证据

从 GraphNav 目录执行 `../Baselines/.reproduction-env/bin/python -m scripts.run_graphnav_diagnostics`；默认 128 episodes、4 条 heatmap 可行性样本。队列安全跳过已完成项，保留未完成目录以便排错。
运行命令与返回码见 `queue.json`，汇总见 `comparison.csv`，逐样本 profiling 见 `profile/profile_samples.csv`，各实验 metrics 包含数据/权重 SHA、实际参数、episode manifest 与代码版本。
执行 `../Baselines/.reproduction-env/bin/python -m scripts.summarize_graphnav_diagnostics` 重建本报告及配对检查。
相关测试 26 项通过，git diff --check 通过；失败的基础环境启动日志保留于 profile_hett_failed_environment.log，随后已使用正确虚拟环境完成 profiling。最终源码与文件 SHA 见 source_snapshot_manifest.json；原执行队列留存于 queue_initial.json。

## 照片与真实网格可视化

![384px 照片与 28/384 全局网格](resolution_visualization/resolution_comparison.png)
初始 public 观测重放，两种真实网格 Top16 与实测结果匹配；最右列仅为照片坐标投影展示，未以展示插值冒充真实网格加密。
