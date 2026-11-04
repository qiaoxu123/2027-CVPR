# 权重与恢复

原权重未覆盖。

- 新增模型代码：`graphnav/polygon_guided_visual.py`，独立Visual Belief。
- 服务器根目录：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_structured_v1`。
- `pilot_s0/best.pt` SHA256 `ebcd6db3f0561a06486a64c35148e61287ca4a4e58d60bf889964729c2c72304`。
- `pilot_s0/last.pt` SHA256 `f15ae6f15360a29d72a8dd5d24100e2ac13178363fe97f930730a671b78e8973`，保存优化器、Python/Torch/CUDA RNG和轮末状态。
- 新权重不加入Git；原A3/视觉分支Checkpoint保持原位。
- 冻结SigLIP2版本和SHA见`data_manifest.json`；不使用LoRA，不宣称全参数训练。
- 复现命令与真实日志见`logs/`；原输出保护，重新实验需新建路径，恢复用`--resume`。
- 配置冻结后仅评分一次开发组，没有正式导航SR结果。
