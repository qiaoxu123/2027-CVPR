# GraphNav-V1 热力图诊断

来自最终 C Epoch1 权重，Seen/Unseen 各展示源顺序中的首个失败案例及另一地图的首个成功案例。未按视觉增益挑选。

四列依次是：初始静态信念、首次真实观测信念、最后一次真实观测信念、最后一次视觉/历史 logit 残差。前三列使用同一对数概率色标；第四列固定范围 [-0.5, 0.5]。红星和20m红圈是真值诊断，青色叉为热力图最大值，绿三角为起点，蓝线为已经执行的轨迹。GT仅在全部重放前向完成后用于绘图。

重放使用原评估保存的位姿前缀逐步取得实际RGB，按时间顺序更新记忆。每步Top16候选、执行目标和Stop概率与原结果核验通过。NPZ包含28×28完整原始logit、残差、世界网格和轨迹。此图没有修改模型或重新选择Checkpoint。

metrics.json / metrics.csv保存全量正式导航指标，manifest.json记录Checkpoint SHA和案例。生成入口：`python -m scripts.plot_graphnav_stage1_heatmaps`。输出目录已存在时拒绝覆盖。
