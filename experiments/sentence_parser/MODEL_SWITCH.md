# 解析模型切换记录

2026-11-06，用户要求“模型换成 gpt-luna”。当前会话工具可选择的 Luna 正式型号为 `gpt-6-luna`，后续解析工作会话使用该型号，保持既有 `low` 推理强度。主控会话本身的模型身份不以此宣称发生改变。

切换时已终止三个 `gpt-6.1-sol / low` 解析工作会话；保存并校验了 1,780 条去重指令、4,854 条关系。未解析 29,975 条。这些旧结果保留原模型来源，不改标为 Luna。

后续直接撰写批次中的每条记录带 `author_model: gpt-6-luna` 和 `author_reasoning_effort: low`，展开后的 `provenance.model_requested` 及总体 `parsed_by_requested_model` 分开统计。模型名是会话请求配置，不冒充外部运行证明。

空间关系解析协议不变：不读取目标 GT，不以目标反推参照物/方向，不运行模型 CLI/API 或用规则生成替代标签，保留匿名实体、间接关系、作用域候选与 unknown。DeepSeek 方案已被用户取消，隔离结果不进入缓存。
