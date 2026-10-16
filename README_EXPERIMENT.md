# 原 V1 全量 384 网格与视觉残差改进

这是 GraphNav-V1 冻结模型的独立消融 worktree，不替换主结果。

`full_resolution_384/frozen_resolution.json` 在完整高分辨率评测前冻结原 C Epoch1 权重、384 网格、原控制器、stop=0.3；三个 split 均为完整样本。只读共享 DATA / 原权重，无大型复制。

ROI 优化保留真正的全局 147456 个候选：支持像素统计采用相同凸多边形栅格的整数计数，重复 patch mask 复用相同池化结果，仅处理有效尺度，半精度特征留在 GPU。旧 28/384 初始 Top16 重放匹配；支持 mask 与质量已对旧实现验证一致。高分辨率数值舍入差异、并发负载与续跑过程均保留记录；没有将显示插值冒充真实网格。

视觉残差候选仅训练新的小头，原几何模型与控制器冻结，利用实际当前/历史原生 SigLIP 向量、文本交互、几何上下文与质量置信度；Loss 用 20m 合法区域的概率质量，并限制不必要修正。分别比较中心化修正与允许实际可见负证据整体降权的非中心化修正。未观测、零向量证据均保持静态信念不变。

使用原 Train Dev 按整 map+instruction 组划分选 checkpoint；随后执行 Seen128 与完整 Seen 的 real / no-belief / zero / shuffle 对照。完整 Seen 的配对 bootstrap 区间与帮助/损害数量作为收益证据，不能用冒烟测试或极少样本声称提升。Test 不用于新残差模型选择。

执行工具：

- `scripts.run_full_resolution_parallel`：各 split 独立进程，完整 episode 续跑；固定实验配置，不做阈值选择。
- `scripts.train_calibrated_visual_residual`：全 Train 缓存训练及整组 Dev；`--uncentered` 是绝对负证据候选。
- `scripts.visual_residual_workflow`：串联第二候选和所有对照，自动更新实时状态 / 报告。依赖本地缓存的冻结 backbones。

最终完整指标在 [RESULTS.md](RESULTS.md)，原始证据在各输出目录。未完成项见 [TODO.md](TODO.md)，不把运行中的前缀 SR 当成全量 SR。
