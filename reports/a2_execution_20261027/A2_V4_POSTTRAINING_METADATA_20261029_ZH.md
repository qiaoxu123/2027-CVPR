# V4 后续独立验证准备进度

远端科学代码保持 ac03d24，完整训练已启动；实际训练进程2168279，监督单元主进程2166345。训练日志已记录第200个批次、1,600条训练观测，当前没有训练异常。完整运行检查证据见同目录 A2_V4_REMOTE_RUNTIME_PREFLIGHT_20261029.json：A0、关闭修正、开启修正各32个训练场景；关闭修正与A0完全一致，开启修正实际影响控制器行为。该检查不证明导航提升。

新增独立验证元数据模块 reference_v4_confirmation_protocol.py，保留原定163个新目标（84用于确定设置、79用于独立检查）及130个旧封存目标，独立检查共209个目标。新记录关联新版特征文件及原数据、初始权重、代码校验值；旧目标划分和旧记录不改写。只核对封存文件字节，不解码其目标答案。

测试先观察到8个缺少实现的失败，再实现并通过。最终验证：599 passed, 1 warning in 12.45s，命令 pytest -q tests tests_confirmation_v3 tests_confirmation_v4。警告来自原有 rasterio 矩阵接口；测试日志在服务器外本地 runs/a2_waldo_landmark_20261027/revision_v4/confirmation_metadata_cpu_tests_20261029.log。10个新增测试含完整293目标的元数据写入、重复写入不变、拒绝旧目标重选、监督集合重叠和缺失特征校验值，以及禁止解码保留数据。

补充独立代码检查未发现具体缺陷，其提出的实际保存流程测试缺口已补上。实际 adversarial-review 工具已调用，日志为 runs/a2_waldo_landmark_20261027/revision_v4/adversarial_confirmation_metadata_20261029.log；本记录时工具正在网络重连，尚无有效审查结论，不声称工具审查通过。

此步骤没有采集独立验证观测，没有读取其模型表现，没有启用修正或启动最终导航评估。后续仍需完成实际新版观测、单次设置选择及独立检查、参数冻结和两个完整验证集；具体计划见 docs/superpowers/plans/2026-10-29-a2-v4-independent-evaluation.md。目前没有证据证明A2超过A0。
