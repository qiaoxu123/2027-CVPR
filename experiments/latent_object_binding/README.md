# P6：Latent Object Binding & Geometric Reasoning

独立研究实现。冻结已验证 A3、SigLIP2、Trajectory/Stop，使用全部公开地图对象。
`language.parse → inference.prepare → inference.posterior` 是无 GT 推理入口。
可选接口 `graphnav.latent_object_binding.LatentObjectGrounder` 返回对象概率与原 28×28 Heatmap；未自动接入正式导航。

推理保留目标连通分量中的变量与 Near/Along/Between/世界方向因子；Between 对参考变量交换对称。目标候选是同地图全部对象，参考支持最多 8、联合赋值最多 16，遗漏概率保留为 Unknown。未确定的相对参考系、未枚举的角色/作用域假设不强行解释。该实现是有预算的近似概率基线，不是全城所有组合的精确枚举。

已知公开名称直接绑定；匿名对象使用粗类型、面积、形状和原 A3 先验。视觉输入仅来自合法观察，支持 Polygon HSV/Lab、冻结 SigLIP2 与真正编码的 Black/同类 Shuffled。无图像的候选视觉增量严格为零。分割是几何弱前景，不是人工语义真值。实际图像来源是 CityNav 在已到达位姿裁出的正射模拟观测，Depth 是垂直代理，不声称透视遮挡真实性。

训练仅学习 9 个标量匹配/关系系数（几何组 7 个有效系数），通过目标边缘概率和 Goal 热图损失监督；不使用参考物实例真值。完整 Train 用于几何监督，RGB 仅有已有缓存的 255 个独立查询可用，不声称完整 Train 有 RGB 覆盖。

## 复现

当前机器从 Workspace 的 `Ablations/GraphNav/latent-object-binding` 执行。共享 DATA 只读；`runs/` 中旧资源用链接访问。所需缓存与权重 SHA 在输出 manifest 和报告中，模型/缓存不进 Git。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/.venv/bin/python
$PY -m experiments.latent_object_binding.pixels
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -m experiments.latent_object_binding.encode_text
$PY -m experiments.latent_object_binding.prepare
$PY -m experiments.latent_object_binding.prepare_observed
$PY -m experiments.latent_object_binding.evaluate --stage geometry
$PY -m experiments.latent_object_binding.train --name geometry
$PY -m experiments.latent_object_binding.train --name rgb
# 中断续训：在相应命令后增加 --resume，不覆盖已有 last.pt。
$PY -m experiments.latent_object_binding.finalize
$PY -m experiments.latent_object_binding.controlled
$PY -m experiments.latent_object_binding.diagnose
$PY -m experiments.latent_object_binding.runtime
$PY -m experiments.latent_object_binding.impact
$PY -m experiments.latent_object_binding.visualize
$PY -m experiments.latent_object_binding.report
$PY -m pytest -q
```

新鲜输出目录才能准备数据；已有完成标记将拒绝覆盖。训练相同 Seed/数据/3 Epoch，Dev 最佳 Goal R@1 选择，平分选更早 Epoch。未评估 Seen/Unseen/闭环。`resources.py` 给出已严格载入 A3 权重的缓存位置；若迁移机器，修改资源根路径并重新核对 SHA，而不复用未验证 logits。

## 结果文件

`runs/p6_latent_object_binding/`：独立 public/labels 缓存、full/observed 逐样本压缩 JSONL、绑定候选与概率、训练每 Epoch 完整指标/可恢复 Checkpoint、2000 次地图/family 配对 Bootstrap、受控文本与不同可见支持的 Oracle、质量审计、耗时、案例图及配置。`docs/P6_LATENT_OBJECT_BINDING_REPORT.md` 与相邻小型 evidence 目录保存可审阅结果。

没有匿名参考物实例人工真值，因此不能报告其真实 Recall@K；只有 Target 变量有原始实例监督。属性 Oracle 缺少可信像素属性真值，样本数为 0。七条真实显式语言句的文本审读干预单独列出，不冒充独立专家标注或全局解析 Oracle。
