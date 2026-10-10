# Epoch 1 之后的保守修复与配对验证（2026-10-10）

基线 Commit `7dcf668`：Train Seen 87512 状态、1 个 Epoch；Seen128 SR **28.9063%**、SPL **22.9716%**、OSR **67.9688%**、NE **38.3867m**、False-Stop Rate **68.4685%**；动态 R@1/4/8=34.3750/60.1563/68.7500。结果已归档于 `JOINT_ACTIVE_EPOCH1_RESULTS_20261010.md`。**原权重、旧 metrics 和结果不修改，也不宣称新代码带来性能改善**。

## 查明的代码级问题

1. `graphnav/joint_active_runtime.py` 原版 Stop 条件只看概率与公共合法高度，不看**当前位姿与被选中预测目标的距离**。高 OSR、低 SR 和大量 false Stop 都支持做门控实验，但不能单独证明 Stop 是全部问题。旧 A3 guarded 模式也检查预测目标距离。
2. JointActive 旧 Runtime 在未 Stop 时**总是**调用启发式 `plan_active_observation`；Epoch1 实际产生256次主动动作。这使性能变化混有不需要研究的探索动作。
3. 视觉匹配梯度非零≠真实 RGB 起作用；缺少在同一视角覆盖/候选位置下的直接 causal ablation。旧 RGB 真实/清零/同类交换 Seen128 几乎相同，需对新 Epoch1 **单独**测。
4. Epoch1 是**开启主动观察的 Seen128**；新 Epoch2 不能将新被动推理的 SR 直接与旧结果混合挑选 Best。

## 本次代码改动

- `graphnav/joint_inference_controls.py`：无 GT 的 Stop 判定；要求满足 Stop 概率、公共合法高度，及到**被选中预测候选**的距离不超过门控半径（默认20m）；`0` 表示不使用距离门控。不会使用评估目标位置。
- `graphnav/joint_active_runtime.py`：主动观测默认 **OFF**，仅 `--joint-active-observation` 才启用；推理结构/权重不变。记录预测目标距离、Stop 拒绝原因、选择的候选、观察数。
- `graphnav/joint_active.py`：添加 `no-belief` 模式，精确移除视觉对 Belief logits 的残差，但保留与原输入相同的候选覆盖和轨迹 `observed_fraction`。
- `scripts/evaluate_closedloop.py`：联合模式独立控制 `--joint-stop-guard-m`、`--joint-stop-threshold`、`--joint-visual-control`、`--joint-active-observation`。旧 SBF/A3 评测不变。输出 `joint_inference_controls` 和各 Stop 拒绝原因**决策次数**；不要把它们当 Episode 数。
- `scripts/train_joint_active.py`：检测旧 Epoch1 是 active 状态并拒绝无校准续训。要求用户先重新评估**同一个 --resume checkpoint**，再传 `--review-rebaseline-metrics`。将旧 Seen128 指标保存在 `seen128_original_active`，用新配对基线重置 Best；后续 Epoch 均采用默认 passive/guarded 验证。
- `tests/test_joint_inference_controls.py`：无标签 Stop 守卫、合法高度、20m/0m 控制、零视觉 Belief 残差验证。

**边界：** 该修复只影响闭环控制与实验可比性。没有修改旧 Epoch1 参数或训练权重，不等于修复了视觉语言对齐和候选选错。20m 是一个预定义假设阈值，可能降低原本有效的 Stop，必须配对评估。

## 优先做：同一 Epoch1 checkpoint、同一 Seen128 的四组不重训实验

在 **本地有 CityNav/SBF 数据与 checkpoint 的机器**执行，下面为环境变量占位，替换成实际路径：

```bash
export SBF_SOURCE=/path/to/local/sbfnav
export SBF_CONFIG=/path/to/sbf_config.yaml
export CITYNAV_DATA=/path/to/revised_citynav
export CKPT=/path/to/joint_active_first_epoch_20261010/training/last.pt
export EXP=runs/joint_epoch1_controls_20261010

# C0：最大限度还原原 Epoch1 控制（主动2步、无预测目标距离门控）
python scripts/evaluate_closedloop.py --model joint-active --sbf-source "$SBF_SOURCE" --config "$SBF_CONFIG" --checkpoint "$CKPT" --data-root "$CITYNAV_DATA" --split val_seen --max-episodes 128 --output "$EXP/c0_active_legacy" --joint-active-observation --joint-stop-guard-m 0

# C1：仅关闭主动动作；无 Stop 目标距离门控
python scripts/evaluate_closedloop.py --model joint-active --sbf-source "$SBF_SOURCE" --config "$SBF_CONFIG" --checkpoint "$CKPT" --data-root "$CITYNAV_DATA" --split val_seen --max-episodes 128 --output "$EXP/c1_passive_unguarded" --joint-stop-guard-m 0

# C2：关闭主动动作，启用20m预测目标距离门控：新的主配置
python scripts/evaluate_closedloop.py --model joint-active --sbf-source "$SBF_SOURCE" --config "$SBF_CONFIG" --checkpoint "$CKPT" --data-root "$CITYNAV_DATA" --split val_seen --max-episodes 128 --output "$EXP/c2_passive_guard20"

# C3：与C2相同图像/覆盖及控制，但RGB外观不改变 Belief
python scripts/evaluate_closedloop.py --model joint-active --sbf-source "$SBF_SOURCE" --config "$SBF_CONFIG" --checkpoint "$CKPT" --data-root "$CITYNAV_DATA" --split val_seen --max-episodes 128 --output "$EXP/c3_no_visual_belief" --joint-visual-control no-belief

# C4：保留覆盖、尺度质量和语言，将有效 ROI 的视觉向量在空间位置间置换
python scripts/evaluate_closedloop.py --model joint-active --sbf-source "$SBF_SOURCE" --config "$SBF_CONFIG" --checkpoint "$CKPT" --data-root "$CITYNAV_DATA" --split val_seen --max-episodes 128 --output "$EXP/c4_shuffled_visual" --joint-visual-control shuffle-vectors
```

先核对 C0 是否大致复现原 28.9063/22.9716 和256次主动动作（注意浮点及源版本），C1-C0 测主动动作干扰，C2-C1 测预测目标 Stop 门控，C2-C3/C4 测 RGB 因果贡献。每组必须使用完全相同 `episode_manifest_sha256`（历史预核验 `ebef159ca4c9a0c7e9a3fcfdf83a413a7e34809166ba548c30340b6cd7ecc695`）、相同 checkpoint SHA、相同20步/50m XY/10m Z、相同数据与高度协议，保留逐 Episode 配对结果。补充 R@1/4/8、OSR、SR/SPL、NE、sDTW、policy_stop、False-Stop、20m经过未Stop、停止后选错目标和每个拒绝原因。不要仅凭 OSR−SR 归咎于 Stop。

## 如需从 Epoch1 恢复续训：先修正 Best 选择口径

```bash
# 使用 C2 的 metrics.json 作为新协议的 Epoch1 Seen128 基线，
# 重评分必须使用 **与 --resume 完全相同** 的 last.pt 文件。
python scripts/train_joint_active.py \
  --train-cache /path/to/cached_train_seen \
  --a3-checkpoint /path/to/a3_best.pt \
  --output /path/to/original_training_output \
  --sbf-source "$SBF_SOURCE" --sbf-config "$SBF_CONFIG" \
  --data-root "$CITYNAV_DATA" --resume "$CKPT" \
  --review-rebaseline-metrics "$EXP/c2_passive_guard20/metrics.json" \
  --epochs 20 --batch-size 16 --workers 4 --lr 0.0001 --seed 0 \
  --stop-after-epoch 2
```

注意：原 Epoch1 结果原样保留在提交报告中。训练脚本对新评测指标重新设置 Best，但输入指纹/原权重/optimizer/scheduler 的正确恢复仍必须由本地重新验证。继续更长训练之前先跑完 Epoch2，确认闭环及训练状态正常。完整20轮应在控制修复和视觉因果验证之后决定。禁止用 Val Unseen 来选择阈值或架构。

## 后续三个研究优先级

1. **Stop 与候选选择解耦（立即）**：按 C0/C1/C2 判断控制差异；重点关注高 OSR/低 SR 的真实分解以及错误候选后的 Stop。避免用 GT 作 inference guard。
2. **目标条件外观监督（若 C2>C3/C4 才继续）**：目前视觉定位软标签使用目标坐标并不保证颜色/形状学习。训练独立已观测 ROI、同地图同类负样本、颜色词替换，先测语言-视觉匹配再联合训练。
3. **历史可信度与候选重排序（稳定后）**：训练和测试的历史分布不同，当前 Runtime 简单等权平均特征。记录候选级置信度/重复视角，做质量加权与 on-policy 错误候选纠错，不急于加深 GAT。

## 验收与局限

此文是下一步复现实验计划，不是新实验结果。推理默认关闭主动观测，但保留代码给公平消融；训练与评测中的 RGB/Depth 来源和公平性约束保持不变。Epoch1 checkpoint 没有随本次 GitHub 提交到仓库，仅给出了本地 SHA 和路径。不能将尚未运行的 C0–C4 写成已观察到性能改善。
