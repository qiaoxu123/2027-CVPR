# P2.5：冻结 RGB 语义纠错诊断

基于 `06ebf53`，使用独立分支 `research/rgb-feasibility-20261011`。不训练任何模型，不改变原 Head、Backbone、A3、正式 gain 或阈值；不运行闭环/Unseen。

- `protocol.json`：Dev开始前声明模型、语言/模态/RGB控制、family安全检查及离线理论幅度网格。主模型继承P2内部Dev选择，不按Seen重新选。
- `core.py`：并列分数期望排名、只使用已观察命名参考物的公开关系、AUC/ECE/risk，以及必须显式 `offline_only=True` 的理论残差函数。未知/未支持网格的残差为0。
- `run.py text`：复用原语言向量；仅补属性/交换文本tokens，不运行图像塔。`diagnose`复用P2真实因果ROI/context/patch及实际Black缓存。RGB-only关闭已有相对坐标bias，RGB+Geometry为预声明等权排名诊断。
- `run.py diagnose --split dev`：同时读取P2缓存的原始三分类logits，检查内部Dev置信度；完成后保存源码、协议、Checkpoint和文本SHA锁。
- `run.py diagnose --split seen`：拒绝源码/协议变化。只进行预声明一次排名、反事实和离线融合诊断，没有Seen置信度调参。
- `analyze.py`：从已完成结果生成中文报告、静态科研图和gzip预测审计副本。未支持属性/关系的人口分开报告；目标身份/坐标只进入标签及Oracle。

运行环境复用现有RemoteCLIP venv的Python；设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1`。

```bash
$TASK_PY -m experiments.rgb_feasibility.run text --p2 runs/instance_grounding --output runs/rgb_feasibility/text_cache
$TASK_PY -m experiments.rgb_feasibility.run diagnose --p2 runs/instance_grounding --text runs/rgb_feasibility/text_cache --split dev --output runs/rgb_feasibility/dev_complete
$TASK_PY -m experiments.rgb_feasibility.run diagnose --p2 runs/instance_grounding --text runs/rgb_feasibility/text_cache --split seen --lock runs/rgb_feasibility/dev_complete/seen_lock.json --output runs/rgb_feasibility/seen
$TASK_PY -m pytest -q
$TASK_PY -m experiments.rgb_feasibility.analyze --runs runs/rgb_feasibility --assets docs/assets/rgb_feasibility --report docs/P2_5_RGB_FEASIBILITY_REPORT.md
```

以上命令拒绝覆盖诊断结果。首个Dev失败的序列化输出另保留；正式Seen诊断只运行一次。图表标注的理论增益不是正式Goal改进；完整运行条件和耗时随报告交付。
