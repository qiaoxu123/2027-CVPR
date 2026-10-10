# RGB–Language Grounding 实验

实际结果和停止结论见 [中文报告](../../docs/RGB_LANGUAGE_GROUNDING_REPORT_20261010.md)。本目录是独立研究代码，不修改正式GraphNav/A3控制器。所有CLI拒绝覆盖已有产物；需要复现时使用新的输出目录。

## 数据与阶段

- `source_audit.py`、`download.py`：官方来源/许可元数据、直连优先再代理、流式预算/SHA记录。HF使用本机标准login token，不接受把token写入配置或命令行。
- `prepare_dev.py`：原Train内部Dev整family/map隔离；复用 `experiments.bidirectional_core.cache` 的FP16视觉＋FP32 pooling及数据保护。
- `city_features.py`：同真实384当前源ROI的whole/tight/masked/context/multiscale/旧patch与原生文本/token；`public-visual`模式不读取目标可见性决定推理适用范围。
- `prepare_aerial.py`、`aerial_features.py`：审计官方Train/Val图像重叠、同源视频隔离、显式颜色/类别相反负例与多正例。
- `refseg_features.py`、`external_leakage.py`、`dior_audit.py`：真实二值mask/归档/重复审计。RefSegRS、DIOR地理范围不明，不参与本轮训练。
- `evaluate_features.py`、`compare_foundations.py`：冻结视觉表示，官方OpenCLIP同Checkpoint图文两塔，Geo QuickGELU / Sky GELU，真实ROI精度检查。
- `head.py`、`train_head.py`：轻量Projection/Token–Patch VEN，多正例/质量/Hard Negative/confidence；冻结Backbone。
- `pixel_controls.py`、`counterfactual_language.py`：保持目的几何的真实同类RGB交换、零RGB与有弱标签支持的语言交换。
- `goal_evidence.py`、`run_goal_suite.py`、`foundation_goal.py`：Dev校准、固定Seen、当前/因果历史、未观察区域Unknown；`calibrate_foundation_scores.py`只对保存的Dev分数做强度审查。
- `benchmark_runtime.py`、`audit_corpus.py`、`cases_and_plots.py`、`summarize.py`：实际延迟/监督审计/逐样本结果/科学图表/小型证据归档。

## 本机实际命令

下面使用与实际运行相同的模块、关键配置及数据路径。实际产物已存在，不应原样覆盖。初始化失败、错误激活和重试目录仍保留，正式结果路径见报告及JSON manifest；本文件不声称从零训练完成了未运行阶段。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_PY=/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/.venv/bin/python
TASK_RUN=runs/rgb_language_grounding
TASK_OLD=/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-main/runs/bidirectional_core_test
TASK_ASSETS=/home/rental/20260922_1/Workspace/Ablations/GraphNav/rgb-grounding-assets

$TASK_PY -m experiments.rgb_grounding.prepare_dev --source "$TASK_OLD/data" --output "$TASK_RUN/data_internal_dev"
$TASK_PY -m experiments.bidirectional_core.cache --metadata "$TASK_RUN/data_internal_dev" --output "$TASK_RUN/cache_internal_dev" --split train_seen
$TASK_PY -m experiments.rgb_grounding.city_features --cache "$TASK_OLD/cache_fp16/train_seen" --output "$TASK_RUN/features_city_train"
$TASK_PY -m experiments.rgb_grounding.city_features --cache "$TASK_RUN/cache_internal_dev" --output "$TASK_RUN/features_city_dev"
$TASK_PY -m experiments.rgb_grounding.city_features --cache "$TASK_RUN/cache_internal_dev" --selection public-visual --output "$TASK_RUN/features_city_dev_public"
$TASK_PY -m experiments.rgb_grounding.city_features --cache "$TASK_OLD/cache_fp16/val_seen" --selection public-visual --output "$TASK_RUN/features_city_seen_public"

$TASK_PY -m experiments.rgb_grounding.train_head --train "$TASK_RUN/features_city_train" --dev "$TASK_RUN/features_city_dev" --output "$TASK_RUN/heads/city_only_projection"
$TASK_PY -m experiments.rgb_grounding.train_head --train "$TASK_RUN/features_aerial_train" --dev "$TASK_RUN/features_aerial_dev" --output "$TASK_RUN/heads/aerial_projection"
$TASK_PY -m experiments.rgb_grounding.train_head --train "$TASK_RUN/features_city_train" --dev "$TASK_RUN/features_city_dev" --initialize "$TASK_RUN/heads/aerial_projection/best_dev.pt" --output "$TASK_RUN/heads/aerial_city_adapt"
$TASK_PY -m experiments.rgb_grounding.train_head --train "$TASK_RUN/features_city_train" --dev "$TASK_RUN/features_city_dev" --initialize "$TASK_RUN/heads/aerial_projection/best_dev.pt" --token-attention --output "$TASK_RUN/heads/aerial_city_token_adapt"

$TASK_PY -m experiments.rgb_grounding.pixel_controls --features "$TASK_RUN/features_city_dev_public" --output "$TASK_RUN/controls_city_dev"
$TASK_PY -m experiments.rgb_grounding.goal_evidence --mode calibrate --cache "$TASK_RUN/cache_internal_dev" --features "$TASK_RUN/features_city_dev_public" --controls "$TASK_RUN/controls_city_dev" --heads "$TASK_RUN/head_candidates.json" --output "$TASK_RUN/calibration_dev_confidence_complete"
$TASK_PY -m experiments.rgb_grounding.pixel_controls --features "$TASK_RUN/features_city_seen_public" --output "$TASK_RUN/controls_city_seen"
$TASK_PY -m experiments.rgb_grounding.evaluate_heads --features "$TASK_RUN/features_city_seen_public" --controls "$TASK_RUN/controls_city_seen" --heads "$TASK_RUN/head_candidates.json" --episodes "$TASK_OLD/data/val_seen/pilot_episodes.jsonl" --output "$TASK_RUN/semantic_seen_frozen"
$TASK_PY -m experiments.rgb_grounding.run_goal_suite --suite "$TASK_RUN/goal_diagnostic_suite_frozen.json" --cache "$TASK_OLD/cache_fp16/val_seen" --features "$TASK_RUN/features_city_seen_public" --controls "$TASK_RUN/controls_city_seen" --output "$TASK_RUN/goal_seen_frozen"

$TASK_PY -m experiments.rgb_grounding.counterfactual_language --features "$TASK_RUN/features_city_seen_judged" --checkpoint "$TASK_RUN/heads/aerial_city_adapt/best_dev.pt" --output "$TASK_RUN/counterfactual_language_seen"
$TASK_PY -m experiments.rgb_grounding.benchmark_runtime --features "$TASK_RUN/features_city_seen_judged" --assets "$TASK_ASSETS" --heads "$TASK_RUN/head_candidates.json" --output "$TASK_RUN/runtime_verified"
$TASK_PY -m pytest -q
```

下载任务运行时不设置 `HF_HUB_OFFLINE=1`：

```bash
$TASK_PY -m experiments.rgb_grounding.download --spec "$TASK_RUN/download_spec.json" --root "$TASK_ASSETS" --workers 3
$TASK_PY -m experiments.rgb_grounding.prepare_aerial --assets "$TASK_ASSETS" --output "$TASK_RUN/aerial_pilot" --revision 3e158594d1788279264c8126ef3b7f25259eab69
$TASK_PY -m experiments.rgb_grounding.download --spec "$TASK_RUN/aerial_pilot/image_download_spec.json" --root "$TASK_ASSETS" --workers 6
$TASK_PY -m experiments.rgb_grounding.aerial_features --annotations "$TASK_RUN/aerial_pilot/train.jsonl" --images "$TASK_ASSETS/aerialvg/images" --output "$TASK_RUN/features_aerial_train"
$TASK_PY -m experiments.rgb_grounding.aerial_features --annotations "$TASK_RUN/aerial_pilot/internal_dev.jsonl" --images "$TASK_ASSETS/aerialvg/images" --output "$TASK_RUN/features_aerial_dev"
```

`head_candidates.json`将 `native`映射到字面值`native`，其余四名称映射到各自`best_dev.pt`。校准产物保存全部候选与checkpoint SHA；非零离线诊断从内部Dev固定grid选择，生产选择为gain0。基础模型表示选择文件、非零suite配置、下载spec、运行logs与源SHA都保留在上述实际runs目录。重建时需要先生成/绑定自己的配置，不得把本机路径或报告中的结果当作自动可移植数据。

`--selection judged`只用于训练/弱监督指标；Goal推理必须使用`public-visual`源和全部cached states。不得依据GT可见性过滤Goal评估。mask tile、属性语言、原始目标身份、静态XY20m命中和导航SR是不同指标。

本轮无方案通过Goal门槛，故没有新闭环/Unseen/LoRA/GAT训练命令，也没有伪造其结果。
