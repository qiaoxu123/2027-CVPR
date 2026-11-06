# Unified V2 — 新增角度方向模态（不含横纵坐标差与距离）

本实验与 Graph Encoder PR #24 独立。保留既有完整指令 BERT、
Geometry、Relation、Polygon、StableGoal、Stop 和 Controller。
不再训练原始全文模型，只训练新增8方向预测及融合头。

## 方位角的严格定义

地图平面 +X 轴设为0°，逆时针增加：
theta = atan2(y_target-y_landmark, x_target-x_landmark)

GT方位角只在训练标签和TRAIN离线Oracle诊断中计算；正常推理只计算公开候选网格相对公开地标的方位角。新增模型输入不含
dx、dy、距离、距离桶、目标绝对位置或真实目标方位角。
已有主模型的10维 Polygon 几何不修改；**新增角度分支只提取
atan2 方位角**，不将已有几何的模长作为新增特征。

CityNav TIF 未验证 CRS，不能把地图 +Y 称为地理正北。
中心与目标重合（半径在1m以内）的方向标签无效。

### 两个绝不混淆的实验

Predicted Bearing：导航开始前，依赖完整指令 BERT 和公共地标
名字预测8方向概率 P(k)。所有候选网格位置相对各个地标的角度
从 PUBLIC Polygon 可计算，按角度相容度修正原 Belief。
只允许预测方向进入完整2470条 Val-Seen 闭环。

Oracle Bearing：由真实目标位置计算8方向标签，仅用于
TRAIN 离线 Goal R@20m 上界诊断。即使没有距离，仍属于
特权 GT 信息，**禁止用于真实闭环或论文公平导航指标**。

每个 candidate 对每个地标的方向核：
P_match = sum_k P_pred(k) * softmax_k(4*cos(theta_candidate - theta_k))
多地标不按 GT 选最近地标；取有效公共名称地标的平均。
残差最大 ±1 logit，gain=0 零初始化，未训练时与原全文模型相同；
无可用地标严格退回原 Belief。

## 训练与验证

训练全量21,878 Revised TRAIN，seed42，默认3轮，
冻结全文 Unified V2、PGVB、BERT、Stop/Controller。
目标函数=Goal CE + 0.5 Direction CE + 0.05 KL-to-baseline。
GT 仅在 forward 之后用于方向标签和 Goal 监督。
Birmingham 是完整 TRAIN 内的拟合诊断，不是独立泛化评测。
只在 Predicted 方向显示真实定位贡献时进入正式闭环。

## 本地命令

先核实这些路径是机器上实际存在的路径。

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
OUT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_angle_only_s42_20261106

"$PY" -m pytest -q tests/test_unified_angle_only_modality.py

"$PY" -u -m scripts.train_unified_angle_only \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --existing-full-checkpoint "$RUN/core/last.pt" \
  --public-objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json \
  --output "$OUT/smoke" --epochs 1 --batch-size 8 --seed 42 \
  --max-batches 12

"$PY" -u -m scripts.train_unified_angle_only \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --existing-full-checkpoint "$RUN/core/last.pt" \
  --public-objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json \
  --output "$OUT/full" --epochs 3 --batch-size 8 --seed 42
~~~

断点续训仅支持完成 whole epoch 后完全相同参数 --resume。
Source SHA不一致不能绕过；不会覆盖已有主实验。

## 可选真实2470条Val-Seen

先通过 GPU smoke、方向分类及离线 pred head 检查，
再执行只有**预测方位角**而不读 GT 的闭环命令：

~~~bash
STAGE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
V2=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data

"$PY" -u -m scripts.evaluate_closedloop \
  --model unified-relation --split val_seen \
  --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" \
  --data-root "$DATA" --checkpoint "$STAGE" \
  --unified-v2 "$V2" --relation-checkpoint "$RUN/core/last.pt" \
  --relation-control full --relation-visual-control off \
  --relation-geometry-source trained \
  --angle-head-checkpoint "$OUT/full/last.pt" \
  --instruction-mode full_text --public-objects "$DATA/cityrefer_original/objects.json" \
  --output "$OUT/val_seen_predicted_angles"
~~~

保留原20m成功判定，任务全2470条含无候选分母。
--angle-head-checkpoint省略时原Unified路径完全不变。
GT Oracle没有任何闭环开关。
预测方向残差修正后同步更新候选TopK、候选坐标及128维上下文，
Stop与控制器保持不变。

重点验收：实际 P(angle) 方向分类能力、关系词反事实敏感性、
地标数量覆盖、目标方向类别错误率、全部配对纠正破坏数；
最终与原全文SR45.75%比较，而非用离线 Oracle 上界冒充SR。


## 本次执行修复

原PR运行时从目标ID索引的processed_descriptions取得地标列表，不作为本轮盲推理输入。角度实验TRAIN及闭环现在都从原始文本匹配完整公开目录，复用PR23的public_text_record/full_text绑定，保留全部合法同名Polygon，无目标距离筛选；超16容量按原公开绑定策略回退空Anchor。原始21878条任务、指令、标签和原全文Checkpoint均保留，不重新训练主模型。无GT标签输入模型。

原45.75%对照是同一旧全文Checkpoint在公开名称绑定下的全量Val-Seen结果；逐条确认相同起点/公共Polygon后可复用。不能拿原生source-list绑定的新结果与之直接比较。

修复角度Policy在public-text模式下被普通文本Policy覆盖的问题；候选ID、XY、128维Context同步重排，回调同时保存真正角度修正后的field、angle_probs、angle_delta。默认普通导航不变，Oracle没有闭环命令参数。训练强制FP32/TF32关闭，复用经过模型名/revision/max_length核验的既有冻结BERT缓存，记录非零新头梯度/原主干无梯度及逐步耗时。
