# 权重与恢复说明

原几何/完整GraphNav-V1：

`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt`

SHA256 `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。原文件未覆盖，不将本轮模型设为默认。

新学生权重目录共同前缀：

`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/`

|版本|文件|SHA256|状态|
|---|---|---|---|
|v1候选蒸馏|pilot_s0_v1/final.pt|990a8613d5561025ad2421a4095adb4a43c7fae3d3e69b0eb841724d79f72658|评测前冻结，完成64地标回归及置零控制，未达到全部门槛|
|v2世界蒸馏修正|world_pilot_s0_v2/final.pt|c48af4ed37e616e2d95a19039513233c1e963a43886e3941d31fa9956a9a0a47|仅TRAIN Dev代码回归，未重新使用确认/Val64选择版本|

每版`last.pt`保存optimizer、步数、Python/Torch/CUDA RNG及源SHA，可使用对应训练脚本`--resume`；不把v1 last传给v2。最终权重为固定12轮版本，不以Val挑选best。teacher_v1为原冻结A3新名称查询缓存，不是新训练主干。

原SigLIP2为`google/siglip2-large-patch16-384`，revision `1b426889ea62b5a72bf9839009a1b184bfc9c178`，后验文件SHA核验见source_audit.json。仅使用已有原生模型，不使用AerialVG；Python环境路径中aerialvg_env只是既有环境名称。

所有完整数据/权重保存在服务器，不提交Git。Git保存模型代码、配置、freeze清单、训练日志、逐样本预测和代表性PNG；第二阶段目标关系模型没有产生任何权重。
