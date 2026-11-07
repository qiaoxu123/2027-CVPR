# 可微分指令编码接口预检

当前既有Geometry训练使用FrozenBERTTokens及缓存，encode_tokens路径为推理张量，不能更新语言编码器。新增独立TrainableInstructionBERT，不替换原默认入口：完整指令实时BERT前向可反传，公开地标名称严格读取原冻结缓存；名称缺失直接报错，避免训练后名称嵌入悄然漂移。未使用的pooler冻结。

真实GPU两条文本前向/反向已执行：输出[2,13,768]，词嵌入梯度范数47.51，首次实测0.315s、峰值约0.984GiB。该测速包含短文本和初始化影响，不能外推完整TRAIN Epoch成本；尚需真实Geometry联合批次测速与Loss检查。没有训练Checkpoint或定位结果，不宣称端到端训练已完成。

下一步用TRAIN fit真实批次进行Geometry与指令BERT联合GPU Smoke，测量代表性长度、batch size及显存后确定预算。原V2、视觉权重及导航/Stop不变。现有确认组已被多轮历史诊断使用，后续必须说明划分暴露，不将重复使用的确认集冒充全新独立测试。

真实Geometry联合Smoke首轮发现并修复接口问题：_load_base已经删除视觉模块，冻结列表不应再次访问它们；第8步发现公开地标别名没有全部进入旧名称缓存。前7步Geometry/BERT梯度均有限非零，但Smoke未完成，不宣称通过。新增独立冻结原BERT副本按需补算缺失公开名称，指令仍由可训练BERT编码，避免使用更新后的指令模型生成名称。将从原权重重新执行Smoke，原中断参数不复用。

## 实测更新：联合训练Smoke通过

从原始权重重新开始的8步GPU Smoke完成，128条训练样本，FP32。
Geometry可训练453,517参数、指令BERT可训练108,891,648参数；两者梯度均非零，Loss均有限。
公开名称使用独立冻结原BERT补算缺失别名，不随指令编码器更新。
训练循环实测4.374秒（不含模型加载），峰值分配显存3.307GiB；warm-up后计算步约0.043–0.054秒，不能直接当作包含数据准备、评估和保存的全程吞吐。
不同批次Loss不可用来证明收敛，Smoke不代表定位或SR改善。

已启动tmux `instruction_geometry_joint`，拟合11133条TRAIN、开发3126条；3轮、最长4小时。
每轮在完整开发组统计R20、NE及纠正/破坏，保存联合Geometry+BERT最佳与最终Checkpoint、优化器和随机状态。
物理目标分组隔离；原基线预训练已使用官方TRAIN，因此此开发集仅用于适配诊断，不能宣称全程未见目标泛化。
本次不读取已消耗的7619条确认组、不访问Val/Test、不启动导航。
服务器目录：`GraphNav/runs/instruction_geometry_joint_s42_20261107/`。
训练尚未完成，收益待验证。后续仍遵守固定128条SR至少70%才扩大导航评测的计算门槛；该门槛不是全量SR的数学保证。

## 联合Checkpoint评估与编码器隔离测试

新增 `tests/test_trainable_instruction_encoder.py`：模拟优化后，指令Token必须改变；首次遇到的新别名与已缓存名称均保持冻结原编码结果，且冻结名称网络无梯度。
与 `tests/test_full_token_rgb.py` 合计实测9项通过（1.72秒）。
新增 `scripts/evaluate_joint_instruction_geometry.py`：联合恢复Geometry与指令BERT，公开名称仍用原始编码器；在完整3126条TRAIN开发组配对比较原始网格Top-1与固定local5解码，输出逐样本坐标、NE和纠正/误改。目标仅用于预测后的评分。
正在对训练中不可变硬链接快照执行真实GPU加载/定位接口核验，不依此调整超参数。正式训练按既定3轮继续，每轮开发集评估；不读取Val/Test。
截至实测第445步，未发现非有限Loss或零梯度。前100步与最近100步平均Loss分别0.9471、0.9565，尚不能声称收敛或有效提升。
