# GraphNav-V1 第一阶段完整训练报告

完整执行 A3轮、B3轮、C5轮；最终完整模型选择阶段 C 第 1 轮，选择仅使用 Seen128 SR、SPL 及内部 Train Dev。所有模型和推理参数在 Unseen 前冻结。

## 初始化及强基线

A3 Seed1 Epoch14、原 Trajectory/Stop、原控制器参考 A3 三份 SHA 与任务要求一致。初始完整模型固定 Seen128 的轨迹、候选、模式、Stop 概率与 static-transfer 逐项核验；结果见 initial_seen128_parity.json。原基线和阶段权重独立保留。
历史报告 SPL47.4130% 为四位小数显示，本次独立 B0 Seen128 为47.4129893%；该微小数值差异不隐去。与历史E1 Seed1独立重评的128条轨迹完全一致，候选、模式、Stop和协议也一致（historical_baseline_audit.json）。完整模型以本次保存的原始轨迹为精确重放参考。

## 全量导航指标

|模型/分割|SR%|OSR%|SPL%|NE m|nDTW%|sDTW%|False Stop%|
|---|---:|---:|---:|---:|---:|---:|---:|
|SBF seen|43.5223|60.4049|36.1184|31.5172|28.2789|14.7397|NA|
|B0 seen|48.5020|56.4372|47.0162|29.0544|30.1022|17.1760|50.9897|
|GraphNav-V1 seen|47.6113|55.6680|46.0028|29.1192|30.3978|16.9183|52.3887|
|SBF unseen|21.5054|38.7468|15.8907|47.5653|23.5093|7.5936|NA|
|B0 unseen|21.2458|30.7749|20.6388|51.3781|25.2298|7.8893|78.2453|
|GraphNav-V1 unseen|22.5436|31.7019|21.8205|51.5327|25.6228|8.4540|77.4397|

False Stop 以 policy Stop 为分母；错误 Stop Episode 占比保留在 full_comparison.csv。SBF 没有学习 Stop，NA不能解释为零错误。Seen2470、Unseen2697；不将 Seen128 与全量基线直接比较。

## 定位、视觉和 History 证据

静态唯一 map+instruction 查询指标分别保存 static_B0_seen/unseen.json、static_selected_seen/unseen.json；实际闭环的初始静态、第一次观测信念、最终观测信念 R@1/4/8/16 单独记录在 goal_retrieval.csv。最终观测时间随 Episode 不同。
阶段 A 的 Real/Zero/Shuffle/No-Belief、阶段 B 的 Current/History、最终候选同类消融均使用固定 Seen128。visual_ablation.csv、history_ablation.csv、analysis.json 保留指标及配对 Episode/地图 Bootstrap。消融在模型固定后执行，没有根据结果再调参。Loss或梯度非零不构成视觉收益证据。
训练使用实际 Train Seen 专家前缀的87512状态，内部查询组隔离83140训练/4372开发；不是87512个独立Episode。当前帧严格位于cutoff，历史严格早于当前帧；目标和专家轨迹只用于损失。无on-policy补充，训练与闭环状态分布差异仍存在。
History按可见比例和有效支持质量加权，并通过小型质量门融合当前/历史；位置关联为28×28网格，重复视角通过相邻已接收位姿2m阈值去重。它不是完整长期回访去重模型。
视觉使用实际384px观测的冻结SigLIP2上下文Patch池化32/64/128m区域；Depth用于可信区域/支持掩码，没有另行训练深度Backbone。所有区域共享当前帧Transformer上下文，不是独立重编码高分辨率ROI。困难负样本通过最高匹配分数选择，并排除可能覆盖GT的模糊区域；没有额外的逐物体颜色/类别真值，不能据此宣称完成同色同类物体识别基准。
视觉残差绝对值上限0.5，未观测候选残差为0，无有效证据回退原静态场。没有主动观察、新GAT或关系辅助损失。

目标Selector采用更新后信念Top-K中的第一名；Guarded-Blend关闭多候选探索，覆盖控制器联合候选选择，因此训练的rank分支不参与执行目标选择。所选目标的轨迹模式和Stop由预训练网络预测，C阶段联合微调；不能将本次结果解释为学习多候选Selector的收益。


|静态模型/分割|唯一查询数|R@1%|R@4%|R@8%|R@16%|NE m|
|---|---:|---:|---:|---:|---:|---:|
|B0/seen|2426|48.4749|68.6315|78.0297|86.9744|29.2174|
|selected/seen|2426|47.0322|68.5903|78.0297|86.9332|29.3563|
|B0/unseen|2679|21.3139|38.9324|49.9440|63.5685|51.4634|
|selected/unseen|2679|22.0978|38.9698|50.0933|63.6432|51.9686|

|实际闭环观测时刻/分割|唯一查询数|R@1%|R@4%|R@8%|R@16%|NE m|
|---|---:|---:|---:|---:|---:|---:|
|initial_static/seen|2426|47.1970|68.7139|78.1121|86.9744|29.3704|
|dynamic_first/seen|2426|47.2795|68.5903|78.1533|86.9332|29.3598|
|dynamic_final/seen|2426|47.4444|68.8788|78.2770|87.0981|29.2293|
|initial_static/unseen|2679|22.0231|38.9698|50.0560|63.4565|51.9212|
|dynamic_first/unseen|2679|22.2098|39.0071|50.1680|63.4938|51.9503|
|dynamic_final/unseen|2679|22.4711|39.4923|50.5039|64.0164|51.6795|

|固定Seen128因果对照|SR变化百分点|Episode配对95%区间|
|---|---:|---|
|A_real_minus_zero|0.0000|[0.0, 0.0]|
|A_real_minus_shuffle|0.0000|[0.0, 0.0]|
|A_real_minus_no_belief|0.0000|[0.0, 0.0]|
|B_history_minus_current|-0.7812|[-3.125, 1.5625]|
|selected_real_minus_zero|-0.7812|[-3.125, 1.5625]|
|selected_real_minus_shuffle|2.3438|[0.0, 4.6875]|
|selected_real_minus_no_belief|-0.7812|[-3.125, 1.5625]|
|selected_history_minus_current|-0.7812|[-3.125, 1.5625]|

## 训练、参数和效率

完成轮次训练时间合计 8248.04s（只计成功保存完整轮次；中断部分另计）。各轮吞吐、峰值显存、模块Loss和排名提升诊断见training.csv/history.json。单步延迟按冻结评测器的policy预测计时，包含原生视觉编码及模型推理，不包含prediction之前的观测构建/渲染；全量总时间另列。

|模块|参数量|
|---|---:|
|a3|727696|
|visual|264325|
|history|7|
|trajectory_stop|61261|
|frozen_bert|109482240|
|frozen_siglip2|881854466|
|trainable_at_selected_training_stage|779111|
|inference_trainable|0|
|complete_system|992389996|

A/B冻结A3和控制器；C学习率A3=1e-5、Visual/History=1e-4、Trajectory/Stop=2e-5，5%Warmup、余弦衰减、梯度裁剪1。BERT/SigLIP2全程冻结；原A3几何模式不使用的视觉/GNN分支也保持冻结。参数组和损失权重见config.yaml。

|评估|时间s|单步ms|峰值分配GiB|
|---|---:|---:|---:|
|baseline_full_seen|192.99|6.234|0.454|
|full_seen|1072.14|47.966|3.777|
|baseline_full_unseen|201.96|5.921|0.454|
|full_unseen|1157.37|45.532|3.783|

Training time sums successfully checkpointed full epochs; interrupted partial work and feature preparation are separate costs. Supplementary controls overlapped training. Final rollouts ran serially. No isolated speedup claim.

## 结果解释和发布候选

完整系统是否优于强基线以全量同协议对照和配对不确定性为准，所有失败或下降结果保留。完整Seen SR未超过B0；不宣称视觉或History已带来稳定收益，B0作为独立发布候选保留。
best.pt是Seen选出的完整模型；last.pt是C最终轮，包含优化器、调度器及RNG，可继续显式恢复；stages/A/B/C各有独立Best/Last。Backbone通过明确模型名和不可变revision解析，本地离线缓存必须存在；不能将小型完整训练状态文件称为包含所有Backbone张量的单文件离线包。
基准使用正射RGB和表面高度导出的垂直Depth代理，不代表真实透视相机或三维遮挡；不宣称真实飞行安全或扫掠体碰撞验证。历史SBF权重曾由Unseen proxy选出，该历史差异保留。

## 可复现运行

```bash
../2027-CVPR-static-geometry/.venv/bin/python -m scripts.run_graphnav_stage1 --resume
../2027-CVPR-static-geometry/.venv/bin/python -m scripts.audit_graphnav_stage1
```

所有评估均为已有scripts.evaluate_closedloop.py，具体命令保存在workflow_status.json及audit_status.json。检查点和输入SHA在checkpoint_manifest.json；大文件不提交Git。
