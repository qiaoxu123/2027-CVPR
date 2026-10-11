# P5 Language-conditioned Object Graph Grounding

独立研究分支 `research/language-object-graph-20261011`。P5 新协议明确允许全部
26,274 个对象的公开几何/类型/名称；与只使用命名地标的冻结 A0 区分。

- `objects.py`：公开字段白名单，全部匿名节点，STRtree 稀疏边、局部子图与缓存。
- `language.py`：只解析原始指令，目标/参考实体、属性、关系及相对方向假设。
- `reasoning.py`：非 GNN 对象概率绑定、Between 联合因子、连续空间分支。
- `model.py`：单层 32 维、2 头语言条件 GATv2；同输入无消息传递训练对照。
- `pipeline.py`：Object Belief → 784 个原候选；重建候选 ID/坐标/上下文后接冻结控制器。
- `rgb.py`：合法观测 Polygon ROI、侵蚀前景、HSV/Lab；未知证据为零。
- `evaluate_rgb.py`：A4/A5 同预算因果观测，官方冻结编码器接口，内容/模型指纹缓存；
  没有 Dev 支持的接受阈值时冻结模型相似度全部弃权。
- `run.py`：Train/Dev 固定三档概率混合，Dev 通过后才训练 GAT 和同输入对照；
  冻结配置后一次 Seen。门槛未通过则停止后续训练/闭环。
- `diagnostics.py`：独立 Train RGB 工程检查与分离的离线 Identity/连续网格 Oracle。
- `analyze.py`：配对统计、原 A0 精确复现核对、已冻结热图分数导出。
- `visualize.py`：实际对象图、热图、RGB、纠错/误伤案例与科学图表。

真实实验：Dev A2–A0 +1.5625 pp，地图 CI 包含负值；Seen 固定验证 +2.34375 pp，
CI 仍包含负值。GAT、RGB 训练与闭环/Unseen 未启动，不能解释为效果为零。
全部实现和失败结果保留；HSV/侵蚀前景不是人工语义标签。

## 复现

从工作树根目录执行，使用现有科学 Python 环境（不安装或更改主环境）：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_PYTHON=/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/.venv/bin/python
"$TASK_PYTHON" -m experiments.language_object_graph.run
"$TASK_PYTHON" -m experiments.language_object_graph.diagnostics
"$TASK_PYTHON" -m experiments.language_object_graph.analyze
"$TASK_PYTHON" -m experiments.language_object_graph.visualize
"$TASK_PYTHON" -m pytest tests -q
```

`run` 拒绝覆盖已完成的 `runs/language_object_graph/metrics.json`。所有大型缓存、
原始逐样本预测/分数与图缓存保存在 ignored `runs/language_object_graph/`，不提交数据或权重。
RGB runner 当前应停止：

```bash
"$TASK_PYTHON" -m experiments.language_object_graph.evaluate_rgb --split dev
# STOP: pre-registered geometry Dev gate failed. Do not run A4/A5.
```

语言解析是保守弱解析，不是经过人工确认的关系真值；复杂嵌套关系/指代和观察者方向
尚未可靠解决。Road Connectivity 仅为同高度相接轮廓的拓扑代理，并非已验证的道路路由。
A0 没有原生 Object Belief：原始 evaluator 中 A0 `object_*` 是 A1 单对象影子诊断，
不能当作原 A3 的实例识别；主对比 CSV/报告已省略该项。

正式结论、协议、门槛、统计与局限见 `docs/P5_LANGUAGE_OBJECT_GRAPH_REPORT.md`。
