# PR28：先修复公开地标绑定，再做全量 Val-Seen 配对分析

## 实验目的与受影响任务

PR27 已核验：原全文基线全部2470条 Val-Seen 任务，SR=1130/2470，
NE均值37.9596m，最大385.8116m。126条没有公开Polygon，
其中122条导航失败；27条为>16容量而整行清空，8条为公开轮廓无效，
91条没有公开名称匹配。35条明确几何/容量问题全部失败。
这35条即使全部救回也只是本轮限定干预的理论SR上限+1.42pp，
**不是预期效果**；91条不允许基于目标GT猜地标。

## 输入修复边界：不重新训练、不用GT

使用历史 original full-text 旧 checkpoint、原完整指令、相同起点、
Trajectory/Stop、seed、20m成功定义、RGB off。只在**原本无Polygon**
的126条任务上做输入修复，其余2344条模型输入逐ID逐轮廓完全保留。

A. geometry：使用 shapely make_valid 尝试恢复 invalid contour，
  只接受恢复后唯一简单Polygon且面积变化<=3%、包围盒基本不变；
  出现MultiPolygon、空、退化或大幅改变则拒绝，不合成假Polygon。
B. capacity：保持模型现有16个地标上限，不改网络。遇到>16不再
  一律清空，而是按**指令中地标出现顺序**尽可能保留完整同名
  Polygon组；相互重叠的别名/ID组先合并，整组装不下就拒绝，
  **不得只留下同名组的部分Polygon，更不得根据目标GT选择**。
C. both：A+B。
没有被公开目录命中的91条，本阶段只做错误分析，不激活模糊猜测。
即使新方法对某条任务仍无可用Polygon，也保留原始输入/轨迹。

## 最小验收

1. 先运行 tests/test_public_anchor_repair.py 与
   tests/test_public_anchor_repair_rollouts.py；检查现有全部CPU测试。
2. 在不读取目标坐标/目标ID/教师轨迹的情况下核验修复前后ID、
   contour与公共catalog一致；新增字段全部明示GT_used=False。
3. 所有受干预模型都加载**同一个**旧完整指令checkpoint，
   绝不加载PR26/27新训练权重；单独测试no-flag原推理不变。
4. 三个修复组都必须完成全2470条真实Val-Seen，不筛选126条
   任务作分母。保存原 public_record.json 与每条
   binding_repair_audit.json。所有原已绑定任务在输入层完全不变。
5. 统计修复数量、有效新增PolyIDs、仍无法修复的容量组与几何、
   纠正/破坏任务、全体及受影响子集SR/OSR/SPL/NE/nDTW、最大NE、
   P50/P90/P95/P99、>=100m极端失败的减少量。
6. 如原本有Polygon的2344条任务出现输入变化，立即停止。
   如其输入相同而轨迹指标发生大变化，先排查确定性，不以此
   宣传修复收益。不得使用已看过的Val-Seen成绩选择新规则。
7. 只可声称诊断性**配对干预效果**；Val-Seen已被用于分析，
   不能称作完全未见过的独立泛化结果。

## 本地GPU服务器执行命令

核查实际路径与SHA存在后执行，三个新输出目录必须为空。
仓库分支：experiment/val-seen-public-anchor-repair-20261106。

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
STAGE="$BASE/graphnav_v1_stage1/best.pt"
V2="$BASE/unified_pgvb_v2_pilot_s42/full_training/last.pt"
CORE="$BASE/unified_full_relation_s42_20261105/core/last.pt"
OBJECTS="$DATA/cityrefer_original/objects.json"
OUT="$BASE/val_seen_public_anchor_repair_20261106"
ORIGINAL="$BASE/unified_v2_text_ablation_s42_20261105/eval_full_text"

"$PY" -m pytest -q tests/test_public_anchor_repair.py tests/test_public_anchor_repair_rollouts.py

for MODE in geometry capacity both; do
  "$PY" -u -m scripts.evaluate_closedloop \
    --model unified-relation --split val_seen \
    --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" \
    --data-root "$DATA" --checkpoint "$STAGE" --unified-v2 "$V2" \
    --relation-checkpoint "$CORE" \
    --instruction-mode full_text --public-objects "$OBJECTS" \
    --relation-control full --relation-visual-control off \
    --relation-geometry-source trained --public-binding-repair "$MODE" \
    --output "$OUT/$MODE"
done

"$PY" -u -m scripts.analyze_public_anchor_repair_rollouts \
  --baseline "$ORIGINAL" \
  --geometry "$OUT/geometry" --capacity "$OUT/capacity" \
  --both "$OUT/both" \
  --output "$OUT/paired_analysis"
~~~

若ORIGINAL并非同一Checkpoint的完整2470结果，先运行同版本、
同参数、无任何修复flag的最新原全文baseline，并用它作control。
若任一run缺少完整2470条、manifest变动、错误GT使用或源哈希
不一致，停止分析，不输出不可靠对比。

## 输出中文报告重点

- 27条容量清空、8条全无合法Polygon中实际恢复几条？
  91条未匹配名称未修的原因与具体例子，别名词典只做不自动应用的提案。
- 原126条无地标任务，修复后SR多少，>=100m的90条极端失败
  有多少变为100m以内？
- 原2344条已绑定任务轨迹是否保持完全一致？
- 全2470条净SR纠正/破坏、NE平均/分位数、最大误差及
  McNemar配对二元检验与物理目标聚类bootstrap不确定性。
- 分别比较geometry、capacity、both，避免混淆三个干预。

保存原始predictions.jsonl/episodes.json/repair audit/paired per_episode.jsonl，
生成失效案例可视化图册；Commit/Push到PR28，只做Val-Seen，
不跑Test、不改变main。
