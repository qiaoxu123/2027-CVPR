# UnifiedPGVB-v2：稳定导航Goal，不冻结视觉Belief

状态：**实验分支实现；真实GPU的64条闭环增益尚未验证。**

本修改直接回应原固定64具名地标任务中TrackedPGVB的9条Horizon结束（均最终在20m内）、60次超过20m的Goal跳变及正常Stop从61/64降到52/64的问题。此前三种方案的具名Polygon代理SR均61/64，**不是**CityNav真实目标SR。

## 代码行为

- **原网络不变**：Geometry在UnifiedPGVB内生成全局Belief；SigLIP2仅编码一次当前RGB；PGVB有效时仍逐帧产生视觉Belief。保持原LandmarkAcquisition确认门控。
- **新增轻量控制接口**：`graphnav/stable_navigation_goal.py`中`StableNavigationGoal`维护一个Episode内的真实世界栅格候选ID；交给原`A3TrajectoryPolicy.predict_field`的仅是稳定排列的Top-K候选，**不覆盖实际Belief logits**。
- **必要时重规划**：新Top-1与当前Goal不同，而且新格点确实有当前有效RGB投影支持、PGVB已在Tracking状态、视觉修正实际激活，新Goal logits较旧Goal高至少0.35，并在连续2帧提出同一个新Goal，才允许切换。常量仅为未按Val-Seen调参的保守工程初值。
- **可靠性**：重新排列候选时，同时按同一个ID重取`candidate_xy`、`candidate_context`；不凭空制造视觉分数、不复制过期视觉、也不使用公开Polygon距离或目标GT作在线决策。任务切换清空持有Goal和确认计数。
- **Stop规则未变**：Stop概率阈值0.3、当前位置距当前Goal 1mm守卫、冻结Trajectory/Stop及Horizon保持不变。本次仅降低动态Goal对控制器到达判定的干扰，不能宣称已解决1mm在物理运动误差下的可达性问题。

## 兼容性与对照

`graphnav/unified_pgvb_runtime.py`现支持：
- `mode="geometry"`：原纯Geometry
- `mode="legacy"`：原固定PGVB修正
- `mode="tracked"`：192条实验中原Tracking融合
- `mode="tracked-stable"`：**本次新方法，默认**

`scripts/run_unified_pgvb_landmark_closedloop.py`新增`--methods StableTrackedPGVB`，并保留另外三种旧方法，以原固定64条任务组成完全配对对照。建议明确使用：

```bash
python -m scripts.run_unified_pgvb_landmark_closedloop \
  --selection "$SELECTION_DIR" \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --unified-checkpoint "$V2_CHECKPOINT" \
  --sbf-source "$SBF_SOURCE" \
  --config "$SBF_CONFIG" \
  --data-root "$CITYNAV_ROOT" \
  --output "$NEW_OUTPUT_DIR" \
  --methods Geometry TrackedPGVB StableTrackedPGVB
```

注意：新的输出路径，不覆盖旧192条轨迹；若使用`--resume`，必须保证原Fingerprint完全一致。

## 验收

先运行：

```bash
python -m pytest -q tests/test_stable_navigation_goal.py \
  tests/test_landmark_acquisition.py tests/test_unified_pgvb.py \
  tests/test_unified_pgvb_v2.py
```

服务器端再跑2条真实GPU Smoke，然后固定64条真实闭环，检查：
1. Geometry与原TrackedPGVB仍能复现之前61/64代理SR，不受新类影响；
2. 新方法的`stable_navigation_goal`日志中持有Goal、连续证据、真正重规划均可审计；
3. 动态Goal导致的Stop守卫阻塞、Horizon数量、正确Stop、SPL_polygon、Goal跳变是否改善；
4. 观察潜在风险：错误Geometry Goal被锁定、强视觉纠错被过度抑制、同名多Polygon及8条Other回退；
5. 复核原3条失败任务，不能把“已在20m但未正确Stop”与“根本没有到达正确区域”混淆。

**重要：** 这些是尚待真实GPU验证的验收项目，不是实际已取得的改进成绩。保持原权重、任务、yaw、时间步、成功20m定义以及真实GT隔离；仍需完整64分母。测试成功才考虑合并PR。
