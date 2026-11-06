# 研究记录：Unified V2 主网络加入地标—目标角度一致性监督

## 背景、假设和已有证据

- PR23 精简“地标+关系词”降低完整 Val-Seen SR：45.75%→34.09%；因此保留完整目标特征语言。
- PR25 新增独立8方向预测头，2470 Val-Seen SR：45.75%→45.71%；TRAIN方向准确率41.13%，仅24条改变初始Top1。独立头的弱残差不足以检验主网络的角度理解。
- 主实验 train_unified_full_relation.py 当前 Loss：
  L_main = L_goal(28×28 Gaussian软标签σ=0.9 cell) + 0.05 * KL(Geometry.detach() || Geometry+Relation)
  无单独世界方位角监督。Relation Head的显式predicate ID原来全为0。
- 新假设：不删除完整指令，也不增加独立弱头；直接约束主 Goal Belief 在相对公共地标的角度分布，使Geometry/Relation在目标定位监督之外显式学习空间结构。

## 新 Loss、监督和输入边界

L_new = L_goal + lambda_angle * L_angle + lambda_KL * L_KL。

L_angle将原有784网格Goal Belief按候选相对Polygon中心方位角聚合为8方向圆周软分布；
用仅在 TRAIN LOSS 计算的目标GT相对Polygon中心角theta_GT监督。
方位角按地图平面+X为0°、逆时针递增；不是UAV自身yaw；
地理北CRS和CityNav语义north并未独立核验。

新增角度学习不把目标GT、目标横纵差、距离、GT方向标签加入网络前向。
原模型已有10维候选—Polygon几何原样保留；角度Loss只对预测Belief和
已有公共几何施加梯度。

**可靠监督门控**：只有单个具名公开参照Polygon、全公开目录中同名只有一个物理ID、
原句存在直接的north/south/east/west of NAME，前文无in/inside/at/on等
明显间接中间对象语法，并且GT到参照中心距离大于1m，才计算Angular Loss。
同名多Polygon、多个地标、behind/left/right、间接关系以及其他模糊指令
全部不计算Angular Loss，但**仍参与原Goal+KL Loss与全部2470导航分母**。
门控只依赖公开文本和公开目录；不根据GT接近程度选正确Polygon。
这个保守语法门控并非经过人工核验的直接关系真值；报告必须统计覆盖和冲突。

显式 Predicate ID 沿用既有Relation Embedding(8条)，尺寸完全不变：
0=unknown、1=east、2=west、3=north、4=south；
非门控样本保持0。旧静态输入不含anchor_predicate_ids时，旧完整指令
模型仍使用全零ID，输出不变。

## 重训对照（全部从同一个官方Original V2初始化）

| 方法 | Predicate | Angular λ | KL λ | 训练 |
|---|---|---:|---:|---|
| 已有完整指令基线 | 原全零 | 0 | .05 | 复用，不重训 |
| A：Predicate control | 真实文本的四方向ID | 0 | .05 | 全量5轮 |
| B：Angular 0.1 | 同A | .1 | .05 | 全量5轮 |
| C：Angular 0.3 | 同A | .3 | .05 | 全量5轮 |
| 可选D：去KL | 同A | .1 | 0 | 全量5轮 |

训练源：全部21,878 Revised TRAIN、seed42、batch8、原Geometry LR=1e-5、
原Relation LR=1e-4、BERT冻结、PGVB和原Stop/Controller不动。
不改已有完整指令模型，不用PR25独立方向头权重。
先TRAIN拟合诊断，不在Val-Seen上调参/选epoch。

## GPU本地命令（先核对路径）

以下保留此前实际环境路径作为候选，请本地核验文件存在且SHA符合。

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
STAGE1=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
V2=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json
OUT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_angular_s42_20261106

"$PY" -m pytest -q tests/test_unified_direct_bearing_loss.py

"$PY" -u -m scripts.train_unified_full_angular \
  --train-jsonl "$RUN/train_revised.jsonl" --public-objects "$OBJECTS" \
  --stage1 "$STAGE1" --original-v2 "$V2" \
  --output "$OUT/smoke" --epochs 1 --max-batches 4 \
  --batch-size 8 --seed 42 --angular-weight .1 --kl-weight .05

for ENTRY in "predicate 0 .05" "angle01 .1 .05" "angle03 .3 .05"; do
  set -- $ENTRY
  NAME="$1"; A="$2"; KL="$3"
  "$PY" -u -m scripts.train_unified_full_angular \
    --train-jsonl "$RUN/train_revised.jsonl" --public-objects "$OBJECTS" \
    --stage1 "$STAGE1" --original-v2 "$V2" \
    --output "$OUT/$NAME" --epochs 5 --batch-size 8 \
    --seed 42 --angular-weight "$A" --kl-weight "$KL"
done
~~~

注意Smoke是少batch，可能恰好没有高置信角度样本；验证GPU正反传后
要检查完整TRAIN的覆盖率，不能由GT反向挑选样本。
如果完整训练有效角度样本数为0，必须停止排错而非声称成功。

## 实际2470条 Val-Seen（预测端没有GT角度）

冻结全部实验方案、检查高置信解析覆盖与角度梯度后，三组分别全量闭环。
--angular-main-predicate 在推理时按原文与PUBLIC目录恢复四方向predicate ID，
不会读取目标真实位置；原控制器、Stop和Success阈值不变。

~~~bash
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data

for NAME in predicate angle01 angle03; do
  "$PY" -u -m scripts.evaluate_closedloop \
    --model unified-relation --split val_seen \
    --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" \
    --data-root "$DATA" --checkpoint "$STAGE1" \
    --unified-v2 "$V2" --relation-checkpoint "$OUT/$NAME/last.pt" \
    --instruction-mode full_text --public-objects "$OBJECTS" \
    --relation-control full --relation-visual-control off \
    --relation-geometry-source trained --angular-main-predicate \
    --output "$OUT/val_seen_$NAME"
done
~~~

原完整指令45.75% SR来自既有完整Val-Seen，同一public目录绑定口径。
每组都2470条；无候选、无显式方向、同名多Polygon、间接关系全部计入。
输出 SR/OSR/SPL/NE/nDTW、纠正破坏、分方向/地标多义情况、角度Loss覆盖率、
GT-文本方向一致性、角度分布、训练loss组成及梯度量级。
不可在Val-Seen上调参数、不可使用GT角度在线推理、不可跑Test或合并main。

## 风险与科研解读

原Goal CE本身隐含目标相对地标的物理信息，因此Angular Loss可能冗余；
只有消融分开Predicate-only和Angular时，才能归因。
空间参考系未核验：north/south字面语义未必对应地图+Y/+(-Y)；
有冲突不能简单归咎于BERT。间接关系不等价于Target直接朝某个地标，
之后要使用多跳关系图监督，而不是强制单个世界yaw。


## 本轮执行修复与对照口径

- TRAIN在准备监督资格前先用原文和完整公开目录重新绑定，与runtime一致；不沿用目标ID索引的processed地标列表。
- 统一公开盲绑定后，末句式regex筛出0条。支持明确前置“South of X, this building…”、north/south side of及后续描述句；拒绝否定/复合方向/导航朝向/明显中间关系。保守自动资格为21/21878：north8/south13，east/west0。并非人工真值；GT一致性只事后统计，不作选择或轴翻转。
- 初始主模型实际张量hash三组完全一致；可使用--bert-cache复用经过模型协议核验的既有冻结BERT缓存，三组从同一缓存开始，不改上游缓存文件。
- 新marker权重必须匹配--angular-main-predicate，避免缺失Predicate输入；缺失与显式全零Predicate的旧模型输出位级一致，并验证ID进入原8条Embedding。
- Angular Loss无有效公共候选方位时不制造常数监督；断点续训保留Angular和Goal/KL累计统计。
- 主对照额外复用公开绑定下重训全文47.37%权重（9ac51ee2），以隔离A的Predicate新增收益；45.75%是早期旧绑定训练权重的公开绑定评测，单独说明分布变化。B/C与A比较才能归因Angular Loss。
- 不因监督稀少而排除其他TRAIN或Val任务，不在Val选epoch/阈值/α。无GT角度进入runtime。
