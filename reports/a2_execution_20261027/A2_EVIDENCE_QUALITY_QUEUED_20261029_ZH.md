# 候选可靠程度内部检查：已部署CPU等待程序

当前尚未证明A2超过A0。原V4训练已完成12/25轮，显卡上只有原训练进程2168279；原独立确认和导航流程保持不变。

新工具只在原训练、逐轮核验、独立确认及原导航流程全部终止后进行CPU内部诊断。它比较原15项信息与增加候选/地标/投影质量后的22项信息，固定训练20轮、只在TRAIN预留目标上选规则，随后用同一规则检查内部DEV。不修改实际导航策略，不自动开展新正式实验，也不读取正式Seen/Unseen指标来调整方案。

## 实际部署证据

- 实现版本：07f3cd18ffbf9defad42d50262c7459b8c6d6f72。
- 只传输26个代码、测试和预先设计文件，压缩包30,183字节；未传输数据或权重。
- 部署前后核对原实验86个固定代码文件，内容全部不变。详见同目录 `A2_EVIDENCE_QUALITY_DEPLOYMENT_20261029.json`。
- 真实服务器CPU测试：80项通过，用时33.70秒；校验记录SHA为183399e83ea96f1b0e931adc11141670864ff6bf6b4bb9da08357c4c9a12ed78。详见 `A2_EVIDENCE_QUALITY_REMOTE_CPU_20261029.json`。
- 实际单次入口检查返回等待，随后启动持久程序 `graphnav-a2-evidence-quality-cpu-20261029.service`，每300秒检查原实验状态。
- 原生进程2226124，Python进程2226146，实际父子链2226146→2226145→2226124已核对。运行标识1297673f45934cae9b06899860ae1d8a。
- 实际进程环境中CUDA不可见；没有真实数据导出目录、拟合目录或新报告目录。详见 `A2_EVIDENCE_QUALITY_REMOTE_QUEUE_20261029.json`。

## 检查结论的边界

按要求实际调用了adversarial-review工具，返回码1，连接端报“workspace routing discovery timed out”，没有有效的结构化结论，不能记为检查通过。

补充只读代码检查未发现具体严重或重要缺陷，发现一处报告措辞问题：前序已导出DEV缓存，不能把未执行新的规则评估写成未打开数据。该措辞已在07f3cd1修正并得到再次确认；它不是指定工具的有效结论，也不是实际效果验证。详见 `A2_EVIDENCE_QUALITY_REVIEW_STATUS_20261029.json`。

设计文件与代码从部署起按实际CPU记录固定；后续进展在独立报告中追加，不能改动固定输入去迎合结果。

## 服务器产物位置

代码位于 `.../A2WaldoLandmarkBeliefV4/diagnostics/evidence_quality/`。全部新日志与后续产物位于原实际运行根目录的 `runs/a2_waldo_landmark_20261027/revision_v4/diagnostics/evidence_quality/`；持久程序日志位于服务器 `v4_upgrade/evidence_quality_supervisor.log`。

目前只有测试和等待记录，没有真实新模型的效果表。即使未来内部诊断有收益，也必须经过正式版本固定、可靠的独立检查和两个完整验证集闭环评估，才能判断是否超过A0。
