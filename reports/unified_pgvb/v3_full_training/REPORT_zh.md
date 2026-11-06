# UnifiedPGVB-V3 完整TRAIN训练与评测

完整TRAIN输入21878条；本阶段拟合17473条、物理目标留出4405条，物理分组重叠0。初始化V2曾使用TRAIN，内部留出不是完全未见泛化。

当前已完成完整Epoch：1；训练结束：False。

Geometry和轻量视觉模块训练，BERT/SigLIP2冻结并读取缓存；seed42，原权重保留。四小时累计训练预算，至少三轮后连续两轮相对Loss改善<0.5%提前停止。未修改Stop=0.3、1mm守卫或Horizon。

训练视点由公开地标选择，属于离线地图引导观测；不能作为无人机真实已看到的图像。导航/初始Goal评测使用真实起点及因果历史，保留原yaw，存在需检查的训练—部署分布差异。Polygon/Depth均是弱监督/2.5D代理，不是人工像素真值。

|Epoch|拟合样本|TRAIN Loss|本轮Wall秒|GPU峰值GiB|
|---|---:|---:|---:|---:|
|1|17473|13.351958|756.05|0.144|

![训练Loss](training_loss.png)
![Loss分量](loss_components.png)

## 全量评测状态

尚无全部2470条Val-Seen定位结果；不报告子集作为正式效果。

尚无完整闭环结果，不宣称SR提升。

训练原始日志/权重留在服务器：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v3_full_s42_20261106`。本报告只汇总真实文件，不预测未完成结果。
