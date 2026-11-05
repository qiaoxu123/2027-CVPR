# UnifiedPGVB V2 主模型语言消融：只新训练两组

## 不再做的事

- **不重新训练完整原始指令版。** 使用已经完成的 UnifiedPGVB V2
  Full Relation Checkpoint（先前 main 主实验结果保留）。
- **不训练仅地标名称版。** 旧 PR #22 的独立小网络也不使用。
- 不训练 RGB，不重新处理 Stop/Controller，不修改主模型网络结构。

## 两组新增实验与现有基线

| 组 | 送入原始指令 BERT Token 通道的文本 | 是否重新训练 |
|---|---|---|
| full_text | 原始完整指令 | **否**；只加载已有完整指令 Checkpoint 来评测 |
| landmark_relation | 紧贴具名地标的空间词组＋对应地标名称 | **是** |
| relation_only | 只保留提取出的空间关系词组（不含地标名） | **是** |

所有组都使用 UnifiedPGVB-v2 原 Architecture、原始 V2 初始化
（新增两组）、28×28 Goal Belief、GAT、Relation Head 和
candidate-to-PUBLIC-Polygon 10D 几何。各组固定 seed42、训练5轮
且沿用原脚本超参数。

**重要：relation_only 只指「指令文本 Token 只包含关系词」。**
地图流仍保留相同具名地标节点、公共名字嵌入和对应 Polygon，
这是让关系词能够关联某个地图对象的必要条件。
该组不是删除地图地标信息，而是删除导航指令文本中的地标名。
多地标时无直接指令身份绑定，解码有可能退化，必须单独分析。

示例：

| 原始指令 | landmark_relation | relation_only |
|---|---|---|
| blue car north of One Stop | north of One Stop | north of |
| car in parking lot behind One Stop | behind One Stop | behind |
| car next to Blackwell Street | next to Blackwell Street | next to |
| no explicit public landmark | [UNK] | [UNK] |

此处 behind One Stop 只是词法邻接，不证明 car 与地标直接关系，
仍须检查错误作用域。不能把 behind/left/right 自动转换成世界方向。

## 输入链路检查

原 Unified V2 不止把语言送进 BERT：指令还决定候选地标关系图、
规则字段、Relation Head 的输入等。

准备器必须直接替换 paired JSONL 的 instruction 字段，
而不是仅替换 BERT embedding。所有文字相关的动态输入
（tokens、rules、relation_fields、GAT 关系）
必须从同一精简指令再计算。地标/Polygon 一律来自公开
CityRefer catalog 名称匹配，不根据 GT/目标ID选择。
三组输入除 instruction 外，参考 Polygon、查询次序及训练目标一致。

**已有 full_text Checkpoint 的来源差异：**
旧完整指令模型是在原始 preprocessed reference 条件下训练的；
这里为了可比性，把它冻结后直接放在与两组新模型相同的
公开 catalog Polygon 输入上评测。这构成输入分布变化，
**不能称为完全匹配的三组重新训练对照**。保留旧主实验的
原生评测指标作为历史参考，并明确区分这两种基线。
若现有模型在新 Polygon 绑定下性能下降，不应直接归因于删词。

## 本地运行（先核对真实路径）

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
MAIN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
STAGE1=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
ORIGINAL=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json
OUT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_landmark_words_only_s42
EXISTING_FULL="$MAIN/core/last.pt"

"$PY" -m pytest -q tests/test_unified_v2_landmark_relation_ablation.py

"$PY" -m scripts.prepare_unified_v2_landmark_relation_ablation \
  --train-jsonl "$MAIN/train_revised.jsonl" \
  --public-objects "$OBJECTS" \
  --output "$OUT/paired_inputs"

# 全文组只用于下方已有checkpoint的evaluation，绝对不重新训练。
# 新两组各先做3 batch smoke，再完整训练5轮。
for ARM in landmark_relation relation_only; do
  "$PY" -u -m scripts.train_unified_full_relation \
    --train-jsonl "$OUT/paired_inputs/$ARM.jsonl" \
    --stage1 "$STAGE1" \
    --original-v2 "$ORIGINAL" \
    --output "$OUT/smoke_$ARM" \
    --epochs 1 --batch-size 8 --seed 42 --max-batches 3
done

for ARM in landmark_relation relation_only; do
  "$PY" -u -m scripts.train_unified_full_relation \
    --train-jsonl "$OUT/paired_inputs/$ARM.jsonl" \
    --stage1 "$STAGE1" \
    --original-v2 "$ORIGINAL" \
    --output "$OUT/train_$ARM" \
    --epochs 5 --batch-size 8 --seed 42
done

"$PY" -u -m scripts.evaluate_unified_v2_language_ablation \
  --prepared-dir "$OUT/paired_inputs" \
  --existing-full-checkpoint "$EXISTING_FULL" \
  --checkpoint-landmark-relation "$OUT/train_landmark_relation/last.pt" \
  --checkpoint-relation-only "$OUT/train_relation_only/last.pt" \
  --output "$OUT/dev_report" --batch-size 8
~~~

路径采用之前实验的候选路径；本地应从现存复现脚本核对真实文件，
特别是 STAGE1/ORIGINAL。禁止覆盖已有模型。

## 验收指标

1. 所有 21,878 TRAIN 记录均保留；对比只在完整
   Birmingham TRAIN-Dev 7,619 条上做 R@20m、Top-5、Top-15、NE。
2. 同一任务、同一公共 Polygon、同一模型架构下比较
   现存完整指令模型与两个新模型，以及两新模型之间的纠正/破坏数。
3. 结果分组报告：有明确几何词、仅地标无词、无公开地标、
   多个地标，同名多个 Polygon。不要只报告可解析子集。
4. 对同一地标 Polygon 做 north/south、east/west 反事实；
   分开关闭手工 rule 贡献与保留 rule 的结果，判别网络是否真正
   使用了关系词。behind/left/right 参考系未知，不用任意世界方位判定。
5. 不使用 GT 选择 Polygon、输入 token 或候选区域；
   GT 只能作为最终 Goal 监督/评测。
6. 不跑 Val/Test/闭环，不改 Stop/main；完成 TRAIN-Dev 后先出报告。

无需重新训练 full_text，也无需仅地标模型。
