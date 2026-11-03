# 已核实事实

- 主权重00f2acc...的grid_size实际28，SigLIP2 Large/384，原视觉gain参数0.416814，tanh约0.394243。
- 代码delta=.5*tanh(gain)*tanh(centered/5)*observed，因此已训练权重单格修正绝对值上界约0.1971215，两格排序翻转最多跨越约0.394243 logits；这是数学上界，不是已测失败率。
- 原目标外观查询是关键词邻域拼接，随后与全文SigLIP文本向量相加；没有目标/中间区域/地标角色分离。
- 实际ROI为32/64/128m，来自同一384px图像24x24 contextual patch tokens，不是独立重编码小目标裁剪。
- 历史Seen128原模型真实RGB SR50%，zero与no-belief均50.78125%，shuffled47.65625%。仅为既有实验，不是新固定yaw结果；不能宣称原视觉收益稳定。
- 原data/graphnav_stage1_frames缺少manifest；恢复后的data/rebuilt_stage1_20261025/frames有完整87512状态manifest，不能直接把不完整目录传Stage1Dataset。
- 后续新训练必须重采固定yaw特征，历史变化yaw缓存只用于审计，不能冒充新方案训练数据。
