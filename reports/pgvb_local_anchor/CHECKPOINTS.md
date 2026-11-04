# 权重及缓存（仅服务器保存）

运行根目录：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2`。

- 新best：`pilot_s0/best.pt`，SHA256 `ba3073723703ce8188c7f74ae810a622342631bb150bd2597656ff0039f2cf56`。
- 新last：`pilot_s0/last.pt`，SHA256 `61e3b281052d5aefb07011e597e4b672582afe93b828ebdc3fe1f802ca84ffee`。
- 新缓存：`data/cache.pt`，SHA256 `45efdb638745f8b859533616ed2f99bf6624f8a627bc1faec2a5cea83760473d`。
- 原PGVB：相邻`pgvb_structured_v1/pilot_s0/best.pt`，SHA256 `ebcd6db3f0561a06486a64c35148e61287ca4a4e58d60bf889964729c2c72304`，不覆盖。
- 原GraphNav：`runs/graphnav_v1_stage1/best.pt`，SHA256 `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`，不覆盖。

缓存/训练/评估完整命令见logs/*.sh；配置、源文件哈希及权重冻结记录见freeze.json和evaluation/metrics.json。断点恢复使用原训练命令的resume参数，先读取脚本帮助；本轮已完成，不自动追加训练。
