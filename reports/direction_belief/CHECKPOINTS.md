# 新增方向模块权重

服务器目录：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/direction_belief_v1/pilot_s0`。原Checkpoint未覆盖。

- `nodirection_synthetic.pt`：SHA256 `f81b9dbdab0215ca6988346b11bb1fa326f2cc40a221a51cfadd63a1c88eb3ae`。
- `direction_synthetic.pt`：SHA256 `665d36fafa127c028078153206ee9cbf59dd89920a43acb629646675daacb668`。
- `nodirection_real.pt`：SHA256 `68c6cf0828f6f74f5d539f8e0dd13a023516a4e0b72e002703c1458e2acbdf3d`。
- `direction_real.pt`：SHA256 `eed5ed21af969c1a1039321ed32db19bcae284bf1f1be39d698afebb8de34478`。

所有新模型均32参数。`learned_parameters.json`保存完整参数及其米制解释；`config.json`保存数据/源码指纹。训练恢复使用同一train.sh命令加`--resume`，保存模型、Adam状态及阶段step；优化器恢复一致性已单元测试。单种子固定步数，没有Dev最优Checkpoint选择。
