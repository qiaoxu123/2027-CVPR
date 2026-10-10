# GraphNav 完整联合方案：第一轮验证

状态：代码与真实模型 Smoke 已通过；执行一轮全量训练与 Seen128 验证，然后等待用户审阅。不是已完成的20轮实验报告。

## 当前用户要求

- SBFNav、HETT 不重跑，引用已有本地结果并说明协议差异。
- 计划20轮，使用20轮中 **Val Seen SR 最好** 的 Checkpoint；同SR时比较SPL。Best、Last均保留。
- 最新执行范围：先完整训练第1轮并评估 Seen128，返回指标与耗时，再决定后续19轮。
- 第1轮使用计划20轮的学习率调度。当前流程不会在第1轮后启动 Unseen。
- 后续完成20轮时，冻结 Seen 选出的 Best，再全量 Seen / Val Unseen / Test Unseen；不能用 Unseen 选模型。

## 实际模型和训练路径

完整 BERT Token + 每个候选对每个地标轮廓的几何输入 → 可训练 A3 → 全局784候选。

当前384 RGB/Depth + 真实位姿 → 冻结 SigLIP2 Large/384 的576个 Patch Tokens → 32/64/128m观测区域原生注意力池化 → 可训练图文匹配模块 → 仅观测区域的视觉残差 → 历史累积 Goal Belief。

更新后的 Belief / A3 Context / 当前位姿 → 可训练三模态、8 Waypoint 轨迹及 Stop → 有界控制器执行首步 → 新真实观测 → 重新更新和规划。主动观察依据预测信念、视觉匹配不确定性和公共位姿，最多2个动作，计入原20步。

A3可训练层、视觉匹配、视觉门控、轨迹及Stop共同反向传播。BERT/SigLIP2骨干冻结，仅缓存冻结表示与确定性输入几何；**不缓存训练中的A3预测或Context**。保留原始A3代码、权重与结果。未证明收益的GAT消息传递不纳入本次主干。

区域池化复用同一帧的上下文Patch Token，不等价于独立重新编码的裁剪，也不等于新增图像细节。只有新位姿对应的新环境观测能改变物理采样。正射Depth是垂直距离代理，不是透视深度或真实遮挡渲染。

## 数据与计算预算

复用 `DATA/sbfnav-data` 的 Revised CityNav；原始文件不改动。全量Train Seen21878个Episode，每个Episode使用与SBF采样公式一致的4个固定内部时序位置，合计87512状态/轮。当前和先前选定的观测形成历史，未来观测不进入该状态输入。

训练数据中的目标位置及剩余人类Waypoints仅进入Loss；推理接口仅接收公共地图、语言、当前/历史RGB与Pose。训练查询中的target_xy仅是全零打包占位，不是真实标签。缓存记录原始文件、实际RGB/Height、模型revision及源代码SHA256，可校验恢复。

seed0、batch16、BF16、workers4、AdamW、LR1e-4、weight_decay0.01、warmup5%、cosine、clip1。批次按种子打乱分片与片内样本，确保每轮完整覆盖且不重复随机读取大量冻结特征文件；该顺序与SBF的逐行全局打乱不同。

初始化使用历史A3五轮Checkpoint，因此联合20轮的**总计算预算包含额外A3预训练**，不能称为与SBF20轮从头训练等预算。HETT学生闭环训练与本方案的固定前缀监督也不同，不能称训练算法完全一致。

## 公平评估边界

GraphNav沿用SBF统一评估器：原始起点、20步、每步水平≤50m、垂直≤10m、20m XY成功半径、相同运动边界和指标定义。静态/动态候选Recall与闭环SR分别报告；候选Recall按map+instruction去重，闭环SR保留全部Episode。

SBF历史Val Unseen SR/SPL为21.5054%/15.8907%。HETT可用本地变体历史Val Unseen SR/SPL为25.2132%/20.2024%；其固定50m高度、原生边界行为与当前协议有差异。HETT同SHA训练标注由其Loader读取21877条，SBF读取21878条，该一条差异尚未解释。旧HETT完整20轮Checkpoint的符号链接失效，保留记录中的数字不能当新实测。**不重跑这些基线，不宣称已经实现完全统一的HETT对比。**

每轮记录各项Loss、训练/验证时间、samples/s、峰值显存、采样GPU利用率、Seen128 SR/SPL/OSR/NE/nDTW/sDTW/False Stop及动态R@1/4/8。另记观测覆盖率和图像空间检索指标；后者不冒充屋顶颜色/建筑形状/车辆识别准确率，这些人工属性标签仍未完成。

## 已完成的必要验证

- 单元测试174项通过，包含联合梯度、无私有标签forward、原生池化、观测支持与确定性分片采样。
- 真实HF模型输出 `[1,576,1024]` Patch Tokens；全区域手工池化与原生池化BF16最大差0.0078125。
- 8个真实Train Episode / 32个时序状态的联合前向及反向通过，A3、匹配、门控、轨迹/Stop均有梯度。只是Smoke，不是全量第1轮。
- 4个Seen Episode真实闭环Smoke通过，原20步限制；其小样本指标不作为性能增益证据。
- 修复旧A3配置未保存max_text_length引发的Runtime初始化错误，使用原实现一致的96默认值，并补回归测试。失败运行保留在本地。

## 执行与产物

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
/home/rental/20260922_1/Workspace/2027-CVPR-static-geometry/.venv/bin/python scripts/run_joint_active_20e.py \
  --first-epoch-only \
  --output runs/joint_active_first_epoch_20261010 \
  --cache data/joint_active_train_full_20261010 \
  --sbf-source /home/rental/20260922_1/Workspace/SBFNav-CityNav-Reproduction/runs/sbfnav_main_e20_bs16_s0_20260929/source \
  --sbf-config /home/rental/20260922_1/Workspace/SBFNav-CityNav-Reproduction/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml \
  --data-root /home/rental/20260922_1/Workspace/DATA/sbfnav-data \
  --a3-checkpoint /home/rental/20260922_1/Workspace/2027-CVPR-full-token-rgb/runs/final_a0_a5_evaluation/geometry_best.pt
```

状态：`runs/joint_active_first_epoch_20261010/workflow_status.json`。
缓存准备日志：同目录`prepare_frozen_features.log`。训练日志：`joint_training_20e.log`。
训练结果：`training/history.json`、`training/training.csv`、`training/last.pt`、`training/best.pt`。
Seen128：`training/seen128_epoch01/metrics.json`及逐Episode轨迹。第1轮完成标记：`training/review_hold.json`。

大数据、冻结特征和权重只保留本地，不提交。待第1轮完整训练及验证完成后，用实测耗时更新预计完成时间；32样本冷启动外推不作为完成时间承诺。


第一轮评估输入预核验：从当前Revised Seen原始数据与当前配置重新生成固定128 Episode清单，与历史SBF/A3清单逐项一致（Episode ID、索引、起点），SHA256为`ebef159ca4c9a0c7e9a3fcfdf83a413a7e34809166ba548c30340b6cd7ecc695`。20步、水平50m、垂直10m、高度10–100m、20m成功半径及二维距离声明协议一致。证据：`reports/active_perception_20261010/epoch1_seen128_protocol_preflight.json`。未重跑基线；历史控制器版本与SBF曾按Unseen选择权重的差异仍需披露，不能仅由清单相同宣称所有实现完全一致。第一轮输出产生后还须核验实际输出清单及协议。
