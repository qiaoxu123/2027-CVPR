# 可微分指令编码接口预检

当前既有Geometry训练使用FrozenBERTTokens及缓存，encode_tokens路径为推理张量，不能更新语言编码器。新增独立TrainableInstructionBERT，不替换原默认入口：完整指令实时BERT前向可反传，公开地标名称严格读取原冻结缓存；名称缺失直接报错，避免训练后名称嵌入悄然漂移。未使用的pooler冻结。

真实GPU两条文本前向/反向已执行：输出[2,13,768]，词嵌入梯度范数47.51，首次实测0.315s、峰值约0.984GiB。该测速包含短文本和初始化影响，不能外推完整TRAIN Epoch成本；尚需真实Geometry联合批次测速与Loss检查。没有训练Checkpoint或定位结果，不宣称端到端训练已完成。

下一步用TRAIN fit真实批次进行Geometry与指令BERT联合GPU Smoke，测量代表性长度、batch size及显存后确定预算。原V2、视觉权重及导航/Stop不变。现有确认组已被多轮历史诊断使用，后续必须说明划分暴露，不将重复使用的确认集冒充全新独立测试。
