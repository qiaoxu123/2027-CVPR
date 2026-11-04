# PGVB + 原始GraphNav-V1地标闭环：接入与服务器执行

## 状态

已添加 **PGVB视觉独立导航与几何融合推理桥接** 和独立64任务回归入口，**当前环境没有服务器GPU、CityNav原始数据和PGVB best.pt，因此尚未完成真实闭环运行，SR/OSR/NE均待实测**。禁止以此前离线91.56% R20或旧融合63/64代替这次闭环成绩。

输入仍为 `Navigate to <landmark name>.`，使用冻结的原始64条具名地标闭环任务（Val-Seen来源），评价以公开地标Polygon并集的20m距离定义，不是原CityNav原始任务的SR。

### 流程和改动

- E2_original：完整GraphNavStage1Policy原始几何+原RGB残差+原Guarded-Blend。
- E4_PGVB：使用同一原模型和控制器，在输出的GoalField上增加**当前真实RGB**产生的PGVB局部概率证据，候选上下文、轨迹/停止权重和阈值不变。
- E5_PGVB_only：**不运行A3几何定位**，只使用当前PGVB观测域条件Belief产生Goal；控制器候选上下文置中性零，沿用原轨迹与Stop权重。无合法局部证据时明确弃权并保持当前位置，不假装能全局搜索。此诊断的低SR可能源于目标起始不可见，不可等价于地标身份识别失败。
- PGVB输入：名称、类型、公开Polygon局部注册token、当前RGB的原生SigLIP2多尺度Patch、Footprint及位姿到世界网格映射。
- **不使用原目标GT、Oracle Goal、未来帧和人工轨迹参与推理**；公开Polygon只在模型合法输入和事后区域评分中使用。
- 无局部可用Anchor、无有效RGB、全常数PGVB评分时严格回退原GoalField；改变的是概率排序而非控制器。
- PGVB增量上限0.5 logit（固定诊断融合，不是训练过的融合增益）。已知地标本身为导航目标，尚未训练直接关系定位。
- 其他/混合类别名称无法可靠区分时保守回退原模型；必须统计回退比例。当前名称任务64条中包含8条Other，不应混淆PGVB有效覆盖与全部样本指标。
- PGVB视觉编码目前是额外一次原生SigLIP推理：其每步计算时间应单独报告，不能宣称优化了推理效率。

## 在原服务器执行（真实运行前核对）

确保工作目录是服务器上该项目根目录，且当前branch为 `research/pgvb-closedloop-integration-20261104`。

先在服务器根据已有成功运行的 `scripts/run_named_landmark_navigation.py` 记录确定以下6个真实绝对路径，**不要猜测或自动使用最新Checkpoint**：

1. `SELECTION`：固定64地标任务目录（含`tasks_public.json`、`manifest.json`）。
2. `STAGE1`：原GraphNav-V1 Stage1 Checkpoint，SHA须为 `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。
3. `PGVB`：局部Anchor版本训练 `best.pt`（非早期PGVB、原生匹配或v1/v2）；核对相应 `freeze.json` 中的SHA。
4. `SBF_SOURCE`：原导航Python包源目录。
5. `CONFIG`：原64地标运行使用的配置文件。
6. `DATA_ROOT`：原CityNav数据目录（应含`cityrefer`和`rgbd`）。

执行：

```bash
python -m pytest -q tests/test_pgvb_navigation_runtime.py

python -m scripts.run_pgvb_landmark_closedloop \
  --selection "$SELECTION" \
  --checkpoint "$STAGE1" \
  --pgvb-checkpoint "$PGVB" \
  --sbf-source "$SBF_SOURCE" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --output GraphNav/runs/pgvb_named_closedloop64_v1 \
  --max-new-tasks 2

python -m scripts.run_pgvb_landmark_closedloop \
  --selection "$SELECTION" \
  --checkpoint "$STAGE1" \
  --pgvb-checkpoint "$PGVB" \
  --sbf-source "$SBF_SOURCE" \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  --output GraphNav/runs/pgvb_named_closedloop64_v1 \
  --resume
```

先核对两条smoke的RGB编码、模型权重、被选Goal、Stop概率，以及E5的无证据弃权，再启动全部64x3真实闭环。脚本逐条写入 `predictions.jsonl`，断点续跑要求所有输入和源代码SHA一致；不允许覆盖结果目录。

### 必须提供的结果

- E2/E4/E5各64条：SR、OSR、Polygon NE、按Building/Road/Other分类；平均步数、Stop/超时/无证据弃权原因；
- 逐条配对分析：新增成功、原有成功被破坏、两者均成功/失败；PGVB更新发生次数和Top1改变次数；
- 可视化不少于4条：原始RGB、地标Polygon叠图、PGVB证据热图、原始/更新后的Goal Belief和两条闭环轨迹；
- 对比原始64条历史E2回归：若SR不再是61/64，先排查环境或控制器差异，不得据结果调参；
- 运行前后两份Checkpoint SHA和训练数据/配置指纹；
- 当前64条小样本只做固定方案评估，不据此调增益或做模型选择。

## 验收结论

仅当原始E2与历史61/64可复现，且E4/E5均有合规的真实视觉推理，并且完整64条闭环执行完毕后，才能报告新PGVB的地标导航SR。

此实验是 **PGVB接入导航器的地标任务验证**，不代表直接关系目标定位或原始CityNav导航SR。
