# P5.1 Role-aware Language–Object Graph

冻结 P5 全对象图、Geometry、RGB 与导航控制器；只修复原始语言的实体角色、关系作用域和概率绑定。结果与限制见 `docs/P5_1_ROLE_AWARE_OBJECT_GRAPH_REPORT.md`。

输入白名单：原始 instruction、公开 ObjectGraph、`static_logits / grid_xy / bounds / pose_world`。`prepare.py` 的弱标签输出、`text_audit.json` 和评估标签只能用于离线监督/诊断；正常 `forward` 不接收它们。Target/Reference 名称匹配依赖名词实体 span，不能把全句出现的名称复制到 Target。

- A0：原冻结 Geometry；无原生 Object Belief，输出中的对象指标为标明来源的 shadow probe。
- A1：P5 原 A2，固定 weight=0.5，逐样本验证原结果一致。
- A2：角色修复，关系仍以 Target 为主体。
- A3：明确关系主体与端点，参考物间关系更新参考分布；Between 联合保留不同实例组合。
- A4：最多 8 个目标/作用域假设；裁剪概率质量退回 Geometry，不重新归一化成确定判断。
- 控制：同 alpha 的 A3、同 alpha/弃权质量的单假设，以及仅移除错作用域名称的 P5 干预。它们只做归因，不选 Seen 参数。

这是有限候选近似推理，不是通用、精确的全图概率推断。代词、属性协调、省略、序数、跨道路关系仍可能无法解析；没有参考实例人工真值，因此文本作用域准确率不等同于真实实例绑定准确率。

## 复现

从仓库根目录执行。`PY` 使用已有 scientific Python（torch/numpy/shapely/matplotlib），NLP 依赖仅安装在独立 `runs/role_aware_object_graph/nlp_env` 中，不能更改 GraphNav 环境。实际使用 Python 3.10.22、torch 2.8.0+cu128、numpy 1.26.4；隔离 NLP 使用 spaCy 3.8.7 / 官方 en_core_web_sm 3.8.0。

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/.venv/bin/python
$PY -m experiments.role_aware_object_graph.prepare
$PY -m venv runs/role_aware_object_graph/nlp_env
runs/role_aware_object_graph/nlp_env/bin/pip install spacy==3.8.7 click==8.1.8
curl -L --fail https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl -o runs/role_aware_object_graph/en_core_web_sm-3.8.0-py3-none-any.whl
runs/role_aware_object_graph/nlp_env/bin/pip install --no-deps runs/role_aware_object_graph/en_core_web_sm-3.8.0-py3-none-any.whl
runs/role_aware_object_graph/nlp_env/bin/python -m experiments.role_aware_object_graph.dependency_export --input runs/role_aware_object_graph/train_public_texts.json --output runs/role_aware_object_graph/train_dependencies.json
runs/role_aware_object_graph/nlp_env/bin/python -m experiments.role_aware_object_graph.dependency_export --input runs/role_aware_object_graph/dev_public_texts.json --output runs/role_aware_object_graph/dev_dependencies.json
$PY -m experiments.role_aware_object_graph.evaluate --phase dev
```

`dev_complete.json` 和 `frozen_config.json` 生成后才导出 Seen 原始文本并运行一次 Seen。已有完成标记时脚本拒绝覆盖；不要删除标记重选 Seen 参数。

```sh
$PY - <<'PYCODE'
import json
from experiments.language_object_graph.run import load
states, _, _ = load('seen')
with open('runs/role_aware_object_graph/seen_public_texts.json', 'w') as f:
    json.dump([s['metadata'] for s in states], f)
PYCODE
runs/role_aware_object_graph/nlp_env/bin/python -m experiments.role_aware_object_graph.dependency_export --input runs/role_aware_object_graph/seen_public_texts.json --output runs/role_aware_object_graph/seen_dependencies.json
$PY -m experiments.role_aware_object_graph.evaluate --phase seen
$PY -m experiments.role_aware_object_graph.controls --split train
$PY -m experiments.role_aware_object_graph.controls --split dev
$PY -m experiments.role_aware_object_graph.controls --split seen
$PY -m experiments.role_aware_object_graph.name_intervention
$PY -m experiments.role_aware_object_graph.analyze
$PY -m experiments.role_aware_object_graph.visualize
$PY -m pytest -q
```

训练参数数目为 0，不调用 P5 训练函数。逐样本完整对象概率、784 Cell 排名/Logits、参考分布、解析假设保存在 `runs/role_aware_object_graph/*.jsonl.gz`；大型缓存和 NLP 权重不提交。小型协议、环境、全部 Dev 消融统计、固定 Seen 统计和图表保存在报告 assets 中。默认的 20m 成功定义、Seed 20261011、地图/family 2,000 次配对 Bootstrap 不变。
