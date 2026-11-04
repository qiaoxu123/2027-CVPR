# 复现与恢复

训练代码Commit见REPORT_zh.md。先使用相同CUDA环境和本地冻结BERT/SigLIP2权重，在新目录输出，不覆盖本轮或历史权重。

```bash
TASK_PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
TASK_BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
TASK_DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
TASK_SBF=$TASK_BASE/../../Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
TASK_OUT=$TASK_BASE/unified_pgvb_v2_reproduction_s42
mkdir -p "$TASK_OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$TASK_PY" -m scripts.build_unified_pgvb_v2 --stage1 "$TASK_BASE/graphnav_v1_stage1/best.pt" --pgvb "$TASK_BASE/pgvb_local_anchor_v2/pilot_s0/best.pt" --output "$TASK_OUT/migrated.pt"
"$TASK_PY" -m scripts.train_unified_pgvb_v2 --data "$TASK_BASE/pgvb_local_anchor_v2/data" --checkpoint "$TASK_OUT/migrated.pt" --output "$TASK_OUT/training" --until-updates 200 --stage A200
"$TASK_PY" -m scripts.train_unified_pgvb_v2 --data "$TASK_BASE/pgvb_local_anchor_v2/data" --checkpoint "$TASK_OUT/migrated.pt" --output "$TASK_OUT/training" --until-updates 1200 --stage B1000 --resume
"$TASK_PY" -m scripts.train_unified_pgvb_v2 --data "$TASK_BASE/pgvb_local_anchor_v2/data" --checkpoint "$TASK_OUT/migrated.pt" --output "$TASK_OUT/training" --until-updates 1712 --stage Cepoch --resume
cp "$TASK_OUT/training/last.pt" "$TASK_OUT/cached_pilot_final.pt"

"$TASK_PY" -m scripts.cache_unified_pgvb_train --annotations "$TASK_DATA/original_citynav/citynav_train_seen.json" --catalog "$TASK_DATA/cityrefer_original/objects.json" --data-root "$TASK_DATA" --sbf-source "$TASK_SBF" --old-cache "$TASK_BASE/pgvb_local_anchor_v2/data" --output "$TASK_OUT/full_train_t0"
"$TASK_PY" -m scripts.train_unified_pgvb_v2 --data "$TASK_OUT/full_train_t0" --checkpoint "$TASK_OUT/cached_pilot_final.pt" --output "$TASK_OUT/full_training" --until-updates 489 --stage Dfull_epoch
"$TASK_PY" -m scripts.evaluate_unified_pgvb_pilot --data "$TASK_OUT/full_train_t0" --before "$TASK_OUT/migrated.pt" --after "$TASK_OUT/full_training/last.pt" --output "$TASK_OUT/full_dev_evaluation.json"
```

以上步数适用于本轮固定数据指纹和batch4帧。数据或batch改变后，按实际拟合帧数计算Epoch步数；不能继续使用489冒充完整Epoch。

长任务使用`tmux new-session -d -s <新会话> 'bash <脚本>'`，本轮实际启动脚本已随报告保存。中断后保持初始化Checkpoint、数据、学习率、batch、workers及代码指纹一致，增加`--resume --until-updates <目标累计步数> --stage <新阶段名>`；新阶段名用于保留旧日志。恢复训练不会重置优化器/采样游标。缓存生成中断时使用原命令加`--resume`；已完成缓存拒绝覆盖。

可调整`--batch-frames`、`--num-workers`、`--save-every`；比较效率时应控制数据/有效查询数，不把不同查询密度解释为提速。默认seed42、CUDA BF16、lr1e-4、clip5。

原协议功能Smoke使用同一冻结64清单，但显式限制两条，不运行完整64：参见smoke_before.sh和smoke_after.sh。不要将这些Val-Seen任务用于训练或Checkpoint选择。

Checkpoints中的随机未启用Head/序列化容器可能使重新打包字节SHA不同；本次恢复应使用checkpoints.json所列原文件。真实训练数值、参数冻结和下一步恢复一致性均已核验。
