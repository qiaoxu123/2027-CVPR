# Geometry Top-K 顺序候选尝试：固定128条 Pilot

日期：2026-11-07
分支：`experiment/v2-top1-public-binding-20261107`

## 结论

将最多4个 Geometry 候选按分数排序，并以20m间距去重后，尝试在到达候选但冻结 Stop Head 未触发停止时切换到下一个候选。固定 Val-Seen Pilot 得到 **69/128，SR=53.91%**，与同一运行器、同一128条任务的 `explore_candidates=0` 基线完全相同。

两组128条的完整位姿轨迹均相同；4候选方案在128条中**一次也没有前进到第二个候选**。全部任务都以 `policy_stop` 结束。因此，本次结果不能证明顺序候选访问能改善导航；它表明当前冻结 Stop Head 没有拒绝首个候选，不能直接充当候选身份验证器。完整 Val-Seen 不运行，遵守 Pilot 必须达到70%才进入全量的筛选门槛。

## 设置

- 同一固定 Val-Seen 前128条、相同原始起点、公开指令与扩展公开地标绑定。
- 冻结 V2 Geometry、Stage1 Trajectory/Stop、控制器、Stop 阈值0.3、原始动作协议及20m成功半径；未使用视觉融合、OSM或GT推理。
- 基线使用 Geometry Top-1，`explore_candidates=0`。
- 实验使用最多4个模型排序候选；以公开地图米制 XY 做贪心空间去重，候选中心至少间隔20m。到达候选后，只有冻结 Stop Head 低于原阈值时才尝试下一个；高于原阈值则按原规则停止。
- 两组使用同一组 V2 Geometry SHA256 `9ac51ee253eed46acbf66e6afb47888857b70258f5fcd195721723d5112dc387`、Stage1 SHA256 `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`、查询 SHA256 `9c959421dd7f94eb292385b240e80d44c36a6973a6eca16a472fffd34b20c382`。
- GPU：NVIDIA RTX PRO 6000 Blackwell；基线运行13.24秒、最终代码实验运行13.68秒，峰值显存约0.45 GiB。

## 指标

| 指标 | Geometry Top-1（无顺序尝试） | 最多4候选顺序尝试 | 差值 |
|---|---:|---:|---:|
| SR | 53.91% (69/128) | 53.91% (69/128) | 0 |
| OSR | 57.81% | 57.81% | 0 |
| SPL | 52.75% | 52.75% | 0 |
| NE | 24.982 m | 24.982 m | 0 m |
| nDTW | 0.2934 | 0.2934 | 0 |
| 平均步数 | 7.391 | 7.391 | 0 |
| 推理时间 | 5.36 ms/步 | 5.64 ms/步 | +0.28 ms/步 |
| 完全相同位姿轨迹 | — | 128/128 | — |
| 失败→成功 / 成功→失败 | — | 0 / 0 | 净0 |
| 曾前进到第二候选 | — | 0/128 | — |

每组均有59条失败、69条成功；128条均以策略停止结束。配对任务表见 `paired_episode_outcomes.csv`。

## 诊断与决策

本策略的“尝试下一个候选”条件依赖 Stop Head 在首候选处给出低停止概率。本批任务没有一个满足该条件。轨迹完全相同说明当前版本只是增加了一个未触发的分支，没有实际改变决策。停止在错误位置的结果也说明：只靠现有 Stop 输出不能区分“到达真正目标”和“到达错误候选”。

因此不运行全量、不把此策略作为SR改进，也不调整 Stop 阈值。后续若仍要探索 Top-K，应先用独立 TRAIN 证据训练或验证真正的候选确认信号（如真实实例级 RGB 审核标签）；在没有可靠确认信号前，顺序访问候选可能只会增加路径消耗或错误停止。已有200条人工审核页目前仍未收到人工标签。

## 复现文件

- `metrics.json`：两组聚合指标与配对统计。
- `paired_episode_outcomes.csv`：128条任务的成败、NE及轨迹一致性。
- `manifest_explore0.json`、`manifest_explore4.json`：输入、Checkpoint、数据和协议摘要。
- GPU服务器逐步轨迹保存在：基线 `/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_top1_public_binding_s42_20261107/pilot128_sequential_nms0/expanded/rows/`；最终代码实验 `/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_top1_public_binding_s42_20261107/pilot128_sequential_nms4_final/expanded/rows/`。

未提交完整轨迹和私有RGB缓存；原始权重及此前实验未覆盖。
