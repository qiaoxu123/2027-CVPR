# 目标：同口径完整Val-Seen Top-1 >50%

起点：公开文本地标绑定，原V2 Top-1=44.66%，此前训练后Geometry=47.09%，HETT Refined Val-Seen全2470条、20m标准。历史元数据绑定结果不混用。

TRAIN审计：完整21878条，3998条名称绑定变化；无Anchor从旧91增至919。公开输入全部重建，target_xy保持占位零，真实目标仅在独立label文件用于训练Loss、分组、评分。

阶段1：同初始化、seed42、FP32/TF32关闭、冻结BERT、只训练已有Geometry。三组固定5轮：原CE；8m CE；8m CE+0.5×20m区域Mass Loss+0.1×候选Margin Loss。不加网络、不加OSM/RGB、不修改导航或命中标准。

划分：Cambridge拟合11133、开发3126；Birmingham确认7619。相同物理目标、重复指令、相同公开形状分组不跨拟合/开发。Birmingham与拟合/开发城市和地图隔离。初始化历史V2已经训练过完整官方TRAIN，因此这些组只证明本次适配留出表现，不能冒充预训练阶段未见样本。

开发组选择最佳方案/轮次，再封存一次性Birmingham确认。正收益后按冻结设置使用全TRAIN重训，冻结后完整Val-Seen评分一次。不得反复用Val筛超参数，未达50%不标记目标完成。

CPU Loss/梯度测试与三组真实GPU各3更新Smoke已通过；稳定每步约0.04–0.05s，峰值约2.75GiB。正式对照由tmux `v2_top1_train_20261107`运行，服务器目录`GraphNav/runs/v2_top1_public_binding_s42_20261107/`。总预算4小时，支持断点恢复；当前未进行新Val/Test推理。
