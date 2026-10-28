# V4 远端启动与路径修复记录

科学代码固定为 ac03d24dde96c7ebe0bfe22e2a1e8196c0d1ea96，原 A0/A1/V3 结果保持不变。

首次启动在实际 CUDA 计算前因数据来源路径检查失败退出。原因是新代码工作目录改变了旧数据的规范路径。执行脚本现将新代码映射到原有进程内目录，同时明确映射原数据目录；没有放宽来源检查或修改数据归属。实际 TRAIN 33,440 条、DEV 5,647 条数据加载均通过，证据见同目录 PATH_BINDING_VERIFIED JSON。

修复后重新运行 CPU 检查：562 passed, 1 warning in 11.91s。初始权重不变。RTX 5090 实际完成排序训练反向计算 3 步和局部精修反向计算 3 步，数值有限、冻结参数未改变；完整证据见 REMOTE_CUDA_SMOKE JSON。小样本运行仅证明程序可训练，不作为提升结论。

运行单元 graphnav-a2-reference-v4-pipeline-20261029.service，实际主进程 2166345。本次观察在 2026-10-29 北京时间凌晨，处于 runtime_off 阶段，实际子进程 2167298 存活。三个运行检查依次执行，通过后自动进入完整 20+5 轮训练与产物核验；未启动第二个 GPU 任务。

执行脚本 SHA256：b606d697dc2af5c8e3b58cce779af69826a0a0d3ff074b6860e215bbac8da036。运行状态和逐阶段日志保留服务器 revision_v4/pipeline_status.json、pipeline_logs/。首次失败证据保留 v4_upgrade/incidents/path_namespace_20261029/。

独立校准和两个完整导航验证集尚未完成。当前没有证据证明新版 A2 超过 A0。
