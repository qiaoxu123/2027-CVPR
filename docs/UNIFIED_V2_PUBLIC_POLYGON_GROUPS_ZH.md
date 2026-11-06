# Unified V2: 公开地图绑定与同名多 Polygon 联合处理

## 根本问题

原TRAIN经 processed_descriptions[target_id][ann_id] 选参考地标名，再绑定公开objects Polygon；PR23/25在线则从完整指令盲匹配整张公开地图。虽然Polygon来自同一地图，但训练和推理选哪些锚点不同。公开绑定历史记录：1,221条无anchor、532次非法几何警告、234次超过16个容量警告；原全文Checkpoint换成盲绑定后，Birmingham训练内R20从70.09%降到64.20%。这些警告不能当互斥任务数，R20差不等于闭环SR差。PR25中同名多Polygon方向标签冲突说明不应强制学习单个真实yaw。

## 方案

1. 统一TRAIN和在线绑定：执行 scripts.audit_unified_public_binding 的 emit-blind-train，将全部21,878条 Revised TRAIN 的landmarks字段替换为原文+公开目录盲绑定结果（其余全文指令、目标GT标签、训练流程保持）。保留源ID对比和SHA/provenance；新训练脚本不接受旧的目标关联source。
2. 不合并不同Polygon为平均质心：同名/同类型多public object ID各自保持独立的Polygon和几何特征。仅对有保守直接的 north/south/east/west of NAME 关系，给该组全部成员同一个显式关系ID。若跨类型、不完整、容量溢出、参考系不明或间接关系，关系组不启用；任务仍参加Goal CE。
3. 新集合级方向Loss：给每个候选网格q和参照组G的每个Polygon计算方向余弦，定义S(q)为组内所有Polygon方向相容度的最大值（集合OR，不随碎片数膨胀）。L_group=-log(sum_q softmax(GoalBelief(q))*S(q))，只对方向约束有区分度的任务使用。不选择GT最近Polygon、不输入GT yaw、距离或目标坐标给网络。
4. 定位与执行模块继续使用原完整指令、原Geometry+Relation网络、28x28 Belief、Stop/Trajectory。训练GT目标仅进入原Goal Loss与最终评分。地图+X/+Y不是已经核验的真实地理东西/南北，报告应独立审查这一限制。

## 对照矩阵（训练全量21,878、5轮、seed42）

A：Blind Full baseline，predicate-mode=off，angular-loss-mode=none，angular-weight=0。
B：Blind Full + group Predicate，predicate-mode=group，angular-loss-mode=none，angular-weight=0。
C：Blind Full + group Predicate + Set OR Loss，predicate-mode=group，angular-loss-mode=group_word，angular-weight=0.1。

原始历史全文45.75% SR只做历史参照，不能代替相同新TRAIN绑定下的A。
三个新模型从相同官方原V2初始化；BERT、PGVB、Stop/Controller不变，
原Goal CE +0.05 KL不变，Geometry LR1e-5，Relation LR1e-4。
先TRAIN内部诊断再固定A/B/C，最后全部2,470条Val-Seen各跑一次。
无候选、无地标、同名多Polygon仍计入总分母。报告分层R20、NE、
SR/OSR/SPL/NE/nDTW、纠正/破坏、关系覆盖与Polygon集合大小。

## 本地命令（需核对真实路径）

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
SOURCE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/train_revised.jsonl
STAGE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
V2=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json
OUT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_public_polygon_group_20261106
BLIND="$OUT/train_public_blind.jsonl"

"$PY" -m pytest -q tests/test_unified_polygon_group.py tests/test_unified_direct_bearing_loss.py

"$PY" -u -m scripts.audit_unified_public_binding \
  --train-jsonl "$SOURCE" --public-objects "$OBJECTS" \
  --output "$OUT/binding_audit" --emit-blind-train "$BLIND"

"$PY" -u -m scripts.train_unified_full_angular \
  --train-jsonl "$BLIND" --public-objects "$OBJECTS" \
  --stage1 "$STAGE" --original-v2 "$V2" --output "$OUT/smoke" \
  --epochs 1 --batch-size 8 --max-batches 4 --seed 42 \
  --predicate-mode group --angular-loss-mode group_word --angular-weight 0.1

for ENTRY in "A  off   none       0" "B  group none       0" "C  group group_word 0.1"; do
  set -- $ENTRY
  NAME="$1"; PRED="$2"; MODE="$3"; WEIGHT="$4"
  "$PY" -u -m scripts.train_unified_full_angular \
    --train-jsonl "$BLIND" --public-objects "$OBJECTS" \
    --stage1 "$STAGE" --original-v2 "$V2" \
    --output "$OUT/$NAME" --epochs 5 --batch-size 8 --seed 42 \
    --predicate-mode "$PRED" --angular-loss-mode "$MODE" \
    --angular-weight "$WEIGHT"
done
~~~

新训练脚本拒绝原TARGET-ID关联 source；输出不覆盖原训练目录；
Smoke批次中可能无角度样本，不能把它当全量训练有效性。
省略的 --kl-weight 默认为0.05。

## 全量2470真实 Val-Seen

~~~bash
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
for NAME in A B C; do
  "$PY" -u -m scripts.evaluate_closedloop \
    --model unified-relation --split val_seen \
    --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" \
    --data-root "$DATA" --checkpoint "$STAGE" --unified-v2 "$V2" \
    --relation-checkpoint "$OUT/$NAME/last.pt" \
    --instruction-mode full_text --public-objects "$OBJECTS" \
    --relation-control full --relation-visual-control off \
    --relation-geometry-source trained --polygon-group-predicate \
    --output "$OUT/val_seen_$NAME"
done
~~~

必须报告：逐任务TRAIN原预处理vs盲绑定的ID集合差异、反向传播检查、
解析门控覆盖、同名多Polygon分层方向约束质量、
以及实际 Val-Seen 2470条同任务SR/OSR/SPL/NE/nDTW/修复破坏数。
如果数据目录、source哈希不一致或TRAIN/在线ID顺序不一致，
先修复后运行，不要以增加loss或更多epoch掩盖。
不跑Test、不修改main。


## 追加：2470条Val-Seen中无地标绑定与导航失败的配对统计

注意：TRAIN内1,221条无地标不是Val-Seen的无地标数量。
GitHub已经归档了完整2470条终点NE/SR（PR25 paired2470.csv），
但无锚点状态需从本地闭环日志的每条public_record.json查询，
才能和同一个 Episode ID 真实关联，不能估算外推。

新脚本：scripts/analyze_val_seen_binding_failures.py。
每个Episode重新执行同样的PublicNames + public_text_record盲绑定，
严格校验实际网络收到的Polygon ID顺序及全部Polygon轮廓。
缺失原始记录、绑定不一致、Episode ID错位、SR/NE阈值不一致均拒绝统计。
不需要重新导航，不看GT选择Polygon。

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
"$PY" -m pytest -q tests/test_val_seen_binding_failures.py
"$PY" -u -m scripts.analyze_val_seen_binding_failures \
  --paired reports/unified_pgvb/angle_only/paired2470.csv \
  --episodes reports/unified_pgvb/angle_only/episodes2470.json \
  --public-records-root "$BASE/unified_angle_only_s42_20261106/val_seen_predicted_angles/belief_steps" \
  --public-objects "$DATA/cityrefer_original/objects.json" \
  --output "$BASE/unified_public_polygon_group_20261106/val_seen_binding_failure_audit"
~~~

输出summary.json与per_episode.csv：全部2470条的公开地标绑定状态、
无公开名称匹配/容量超16/有名称但所有Polygon无效、
对应SR/失败数/NE分位数/最大误差/距离区间、
及全部失败中无地标的比例。

核查已完成的原全文基线：SR1130/2470，失败1340/2470，
平均NE 37.9596m，最大NE 385.8116m；
失败任务NE P50为38.3528m，P95为214.0615m，
152条失败任务终点大于或等于100m。
若本地日志与归档逐任务不一致，必须停止而非推测。

因果边界：“无地标且失败”只叫统计相关，不能直接声称
“因为绑定失败而失败”。下一阶段固定其余输入/模型/起点，
对本应可以匹配却未匹配成功的指令修复公开名称/几何绑定，
做配对真实闭环，才能估计可恢复的导航失败数量。
未具名的指令不能用目标GT偷偷添加地标。

## 本次已完成：2470条Val-Seen离线关联审计（2026-11-06）

详见 [完整中文报告](../reports/unified_pgvb/val_seen_binding_failure_report/REPORT_ZH.md)、
[逐任务CSV](../reports/unified_pgvb/val_seen_binding_failure_report/per_episode.csv) 与
[离线案例图册](../reports/unified_pgvb/val_seen_binding_failure_report/cases.html)（下载后用浏览器打开）。

- 2470/2470按Episode ID对齐，缺失/重复/错位均0；逐条原始指令、起点、原基线NE/SR与公开Polygon完整轮廓校验通过。
- 有绑定2344，无绑定126：无公开名称匹配91、全无有效Polygon 8、超过16容量27。
- 1340条失败中122条未绑定（9.10%）；无绑定失败率96.83%，有绑定51.96%。
- 失败率差44.86个百分点，按449个物理目标组bootstrap 95%区间[40.27,49.03]个百分点。
- ≥100米的152条失败中90条未绑定（59.21%）；全量最大NE 385.8116米。
- 只支持统计关联；多数失败仍有绑定。报告提供后续配对干预设计，本次未运行。

复现入口：`scripts/report_val_seen_binding_failures.py`，内部调用
`scripts/analyze_val_seen_binding_failures.py`。正式分析额外提供
`--baseline-predictions` 和 `--annotations`，防止仅按地图名校验遗漏同地图任务错位；
完整命令见报告目录 `reproduce.sh`。没有重新训练、导航、Test或main合并。
