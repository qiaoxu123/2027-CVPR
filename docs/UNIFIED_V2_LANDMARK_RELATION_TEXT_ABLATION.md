# UnifiedPGVB-v2 主实验：只替换语言为「具名地标 + 邻接几何词组」

## 目标与 PR #22 的关键区别

**禁止再训练上一轮 PR #22 的独立 LandmarkGeometryWordsOnly 小模型。**
本实验使用原论文 UnifiedPGVB-v2 Geometry + 完整 BERT Token
Attention + candidate-to-Polygon 10D geometry + 原
CandidateAnchorRelation Head + 原 PGVB（本轮无 RGB）。
模型架构、28×28 网格、Optimizer、训练轮数、初始 Unified V2
Checkpoint、Stop/Controller 均沿用原 scripts/train_unified_full_relation.py。
仅更换三个输入变体的语言，
且每个变体分别进行完整的主干+Relation训练。

- full_text：完整原始指令。
- landmark_relation：只含「地标名称＋直接紧贴该地标的几何词组」。
- landmark_only：只含具名地标名称，不含几何词。

| 原句（仅数据准备器可读） | landmark_relation | landmark_only |
|---|---|---|
| The blue car north of One Stop | north of One Stop | One Stop |
| The blue car in the parking lot behind One Stop | behind One Stop | One Stop |
| The car next to Blackwell Street | next to Blackwell Street | Blackwell Street |
| The car without public named anchor | [UNK] | [UNK] |

注意：behind One Stop 的词法邻接不保证 car 直接 behind One Stop，
可能是停车场 behind One Stop。
不将 behind / left / right 硬编码为世界 X/Y 方向。
静态规则解析器只支持自己原有的部分空间词（north/south/near/along 等）。
需要将手工规则产生的变化与学到的几何关系能力严格区分。

## 为什么要先从公开对象目录重新绑定 Polygon？

原 Revised TRAIN 的 landmarks 来自 processed target-reference
记录，与目标标注有关。若简单照搬，则所谓“只输入地标”仍可能
隐含地获得答案相关的地标 ID。

本实验使用完整公开 CityRefer objects.json，
按照原始文本中实际出现的公开名称匹配地标，不用 GT 坐标选
Polygon；同一名称的多个物理 Polygon 全部保留，
超过 16 个容量则三组统一空地标并记录原因。
每条任务 original GT 坐标只作为网络 forward 之后的训练标签。

数据准备器同时替换 instruction 和 landmarks：
这意味着 BERT Token、静态规则/关系字段、GAT 关系节点、Node Name
以及可用 Polygon 均从当前同一变体的输入构造，
不会残留原文目标颜色、类别或位置词。
这是从 Unified V2 同一模型、同一初始化权重进行的
真实语言输入消融，而非单纯换 BERT 文本引入泄漏。

## 本地运行

训练前确认下列路径存在，尤其是官方 stage1 和 original-v2；
路径采用项目先前复现脚本中的值，如服务器已迁移请按实际位置更新。

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
MAIN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
STAGE1=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
ORIGINAL=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json
OUT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_landmark_words_only_s42

"$PY" -m pytest -q tests/test_unified_v2_landmark_relation_ablation.py

"$PY" -m scripts.prepare_unified_v2_landmark_relation_ablation \
  --train-jsonl "$MAIN/train_revised.jsonl" \
  --public-objects "$OBJECTS" \
  --output "$OUT/paired_inputs"

# 每组只跑3个batch先做CUDA梯度Smoke。不要用smoke作正式结果。
for ARM in full_text landmark_relation landmark_only; do
  "$PY" -u -m scripts.train_unified_full_relation \
    --train-jsonl "$OUT/paired_inputs/$ARM.jsonl" \
    --stage1 "$STAGE1" \
    --original-v2 "$ORIGINAL" \
    --output "$OUT/smoke_$ARM" \
    --epochs 1 --batch-size 8 --seed 42 --max-batches 3
done

# 同一原始Unified V2模型初始化，分别完整训练3组。
for ARM in full_text landmark_relation landmark_only; do
  "$PY" -u -m scripts.train_unified_full_relation \
    --train-jsonl "$OUT/paired_inputs/$ARM.jsonl" \
    --stage1 "$STAGE1" \
    --original-v2 "$ORIGINAL" \
    --output "$OUT/train_$ARM" \
    --epochs 5 --batch-size 8 --seed 42
done

# 三组都完成5轮后，统一在相同Birmingham TRAIN-Dev查询上测。
"$PY" -u -m scripts.evaluate_unified_v2_language_ablation \
  --prepared-dir "$OUT/paired_inputs" \
  --checkpoint-full-text "$OUT/train_full_text/last.pt" \
  --checkpoint-landmark-relation "$OUT/train_landmark_relation/last.pt" \
  --checkpoint-landmark-only "$OUT/train_landmark_only/last.pt" \
  --output "$OUT/dev_report" --batch-size 8
~~~

原 scripts.train_unified_full_relation 是 TRAIN 入口，它同时更新
Geometry 和 Relation，但没有当前因果 RGB 帧可监督，因此
不会重新训练 PGVB 视觉分支。本实验先检验静态
Goal Belief，而不是声称完整导航 SR 提升。

若训练中止，只能按原脚本 --resume 断点继续；
新目录必须保持和原 manifest 一致。切勿覆盖官方主实验权重。

## 验收

1. 所有组都是 21,878 条 Revised TRAIN 源，数据顺序一致；
   Cambridge/Birmingham 使用相同地图分割；无地标样本不能被丢掉。
2. 三组同源 original-v2、stage1、seed=42、5轮、
   相同学习率/损失，且 Polygon 输入逐样本完全一致。
3. 对抽样前向输入审计 BERT、node_names、
   static_rules、relation_fields、GAT 图，确认为
   每组重新由该组文本计算，不含被删的原始描述词。
4. Birmingham TRAIN-Dev 全量 R@20m、Top-5、Top-15、NE、
   20m 概率质量，以及相同任务的纠正/破坏；
   分“仅名称”“有关系词”“无公开地标”统计。
5. 重要机制核验：相同 Polygon 输入 north/south
   或 east/west，热图是否朝正确相对方向移动；
   若变化完全由现有显式规则 rules 驱动，
   必须如实区分模型学习与手工规则，不能称为网络理解几何。
   复杂 left/right/behind 参考坐标系仍保持未知。
6. 先 TRAIN-Dev；若精简版没有明确增益，不再运行2470条
   Val-Seen闭环。若有增益，再以同一 parser 接入
   UnifiedPGVBFullRelationPolicy 推理路径，并确认预测前
   不使用原始完整指令的隐式额外词汇、原 Controller/Stop 不变。

不要使用上一轮PR22的9.765%作为 Unified V2 主模型基准。
该数字来自另一个不共享参数的17.8万参数小网络，
只能作为独立弱基线。
