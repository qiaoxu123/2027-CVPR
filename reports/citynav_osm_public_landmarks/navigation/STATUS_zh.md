# 运行状态：已完成

原始V2、训练后V2 Geometry各2470/2470，HETT Refined Val-Seen，seed42。没有训练，没有OSM增强，没有修改Stop/Goal阈值。

[完整导航报告](REPORT_zh.md) · [1911任务观测支持报告](observability/REPORT_zh.md)。权重、协议、绑定口径和初始批量/逐条推理差异均已记录，不能混用历史元数据绑定结果。

服务器输出：`GraphNav/runs/v2_public_navigation_20261107/`。逐任务原子保存，输入指纹固定；tmux脚本再次运行会跳过已完成任务。
