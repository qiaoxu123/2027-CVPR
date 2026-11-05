# UnifiedPGVB-v2 主线测试与训练准备（2026-11-04）

## 0. 版本与备份

- 原GraphNav-V1完整保留在 `archive/graphnav-v1-pre-unified-20261104`，基准Commit为 `9098e2546fb1a018e9a31f34d562660c3f63000f`，不修改或删除。
- 主线当前统一定位网络为 `graphnav/unified_pgvb.py::UnifiedPGVB`，内部拥有A3 Geometry + 单一PGVB视觉评分及融合。
- 独立控制器：原 `CandidateTrajectoryStop` 及Guarded-Blend；只保留已训练的控制权重。
- V2权重格式是 `unified-pgvb-v2`，旧v1权重不能直接用于新Runtime。
- 统一视觉联合融合头 `joint_ready=False`、目标外观头 `target_ready=False`。权重打包不等于联合训练。
- 此版本首先用于**具名地标导航**诊断；原始CityNav目标+空间关系闭环尚未完成接线，不得将地标导航SR冒充原任务SR。

## 1. CPU检查（必要）

```bash
git fetch origin
git switch main
git pull --ff-only origin main
python -m pip install -e '.[dev,data]' pyyaml
python -m pytest -q tests/test_unified_instruction.py tests/test_unified_pgvb.py tests/test_unified_pgvb_v2.py tests/test_pgvb_closedloop_audit.py
python -m compileall -q graphnav scripts
```

额外检查本轮CI：统一模型确实持有 `geometry` 参数，输出28×28完整GoalField、Top-K候选和128维Context，无可信RGB时静态Belief原样返回；冻结时Episode内几何缓存一次，每个新Episode重置；单帧RGB只调用一次SigLIP2 Vision；可训练时几何与联合头有梯度。默认路径不存在第二套Matcher/History视觉评分。

## 2. 环境及Checkpoint核验

路径由服务器当前实验日志、真实文件与SHA确认，不要使用报告中的占位示例或虚构路径。
原GraphNav Stage1检查点SHA256参考：
`00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。
已有PGVB local Anchor权重SHA参考：
`ba3073723703ce8188c7f74ae810a622342631bb150bd2597656ff0039f2cf56`。
若实际文件与权威实验记录不符先停止，审查权重来源。

```bash
python -m scripts.build_unified_pgvb_v2 \
  --stage1 "$STAGE1_CHECKPOINT" \
  --pgvb "$PGVB_LOCAL_BEST" \
  --output "$V2_CHECKPOINT"
```

方向权重可选；本轮验证具名地标时不启用方向或Target Head。使用新路径输出，不覆盖原始权重，核对包内 `source` SHA、`geometry_integrated=True`、`jointly_trained=False`、`target_ready=False`。

## 3. 真实GPU两条Smoke

使用原冻结64条公开具名地标任务清单（`tasks_public.json`与`manifest.json`）；没有清单则用 `scripts.prepare_named_landmark_navigation` 从已有冻结的直接关系分类/起始位姿/公开Catalog确定性重建，不能依据结果挑任务。**不能用原始目标GT构建导航目标。**

```bash
python -m scripts.run_unified_pgvb_landmark_closedloop \
  --selection "$SELECTION_DIR" \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --unified-checkpoint "$V2_CHECKPOINT" \
  --sbf-source "$SBF_SOURCE" \
  --config "$SBF_CONFIG" \
  --data-root "$CITYNAV_ROOT" \
  --output "$V2_SMOKE_DIR" \
  --max-new-tasks 2
```

检查真实384×384 RGB、实际Footprint、相机原yaw、成功生成Belief/GoalField、输出有限值、动作与Stop正确、旧视觉Matcher未被调用，记录显存、每帧SigLIP2前向次数和端到端耗时。Smoke失败先修复再评测。

## 4. 固定64条完整闭环

```bash
python -m scripts.run_unified_pgvb_landmark_closedloop \
  --selection "$SELECTION_DIR" \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --unified-checkpoint "$V2_CHECKPOINT" \
  --sbf-source "$SBF_SOURCE" \
  --config "$SBF_CONFIG" \
  --data-root "$CITYNAV_ROOT" \
  --output "$V2_SMOKE_DIR" \
  --resume
```

**固定分母必须是64，包含Building32/Road24/Other8**。Other及混合Polygon只允许公开几何回退，不可宣称PGVB视觉识别成功；不得因无候选或不支持视觉而删除任务。完整报告SR/OSR/Polygon NE、正确Policy Stop、误停、horizon、SPL（协议若支持）、Goal跳变和平均时延，同时保存失败样例RGB+初始Belief+动态Belief+轨迹。不要复用历史E4的62/64替代新V2实验结果。

同任务对照历史E2原模型61/64、E4原模型+PGVB62/64、Polygon Oracle64/64，需清楚区分原报告与本轮实际重跑。64条为既有诊断样本，不是独立泛化测试。

## 5. 完整TRAIN与原始CityNav SR（后续开发门槛）

当前**尚无V2完整TRAIN联合训练器，也没有原始2470条Val-Seen指令的统一关系/目标视觉闭环入口**，因此严禁声称这些结果已完成。

先完成GPU Smoke和64条完整地标闭环；随后：
1. 为V2建立全量可用TRAIN地标训练器（复用PGVB缓存、单次视觉特征提取、seed=42、可断点恢复，不能误用目标GT推理输入）。
2. 全量具名地标评测覆盖Building、Road、Other、同名多Polygon等类别，报告几何对照及RGB/错名称消融，目标SR尽可能接近100%，同时检验Stop及耗时。
3. 训练并独立验证direct spatial relation Belief；核查坐标参考系、方向偏移和真实目标20m邻域概率质量，失败不能用外观头掩盖。
4. 最后才开启目标RGB外观Head与联合融合，基于**全部2470条Val-Seen原始任务**报告SR/OSR/SPL/NE（无候选亦计分母），并分别报告支持比例与各关系类别。禁止以128条Pilot替代正式全量结果。

所有后续训练结果、参数量、时延、Checkpoint SHA、实际样本分母、报告与修改均提交并推送。无需更改历史V1评测或原始权重。
