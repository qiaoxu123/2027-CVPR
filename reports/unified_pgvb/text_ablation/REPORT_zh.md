# PR23 Unified V2语言消融：执行中

基于f8f21b3，只新训练landmark_relation，取消relation_only，不重新训练full_text或仅名称模型。原架构及训练超参数不变：seed42、5轮、batch8，Geometry1e-5/Relation1e-4；PGVB/BERT/控制器冻结。

全21878条TRAIN均参与训练，Birmingham7619条是训练内诊断，不是独立Dev。已有full_text/checkpoint曾用原processed引用绑定，当前公共目录重绑定构成输入变化，将另评native绑定以分离影响。

公开目录重新匹配后12458条有邻接关系词、8199条仅名称、1221条空Anchor（含234条容量溢出）；532次非法Polygon形状明确记录。所有任务保留。公开名称别名同步到所有语言路径，解决省略The后Rule绑定失败；7项专项测试通过，3步GPU Smoke中Geometry/Relation梯度非零。5轮训练进行中。

已完成4次TRAIN真实导航接口Smoke：2任务×full/compact输入，均正常Policy Stop，未评分、不作为性能。共享原Stage1控制器与Stable Goal，不改Stop/Goal规则。full Val-Seen将在最终Checkpoint冻结后运行全部2470条，两组共享公开Polygon绑定；不运行Test。

Rule消融与方向反事实将区分直接加分、规则特征/合成节点以及网络学习。原Rule的north=+Y无地理配准佐证（本地栅格CRS为空），只能报告工程坐标响应率。

服务器：GraphNav/runs/unified_v2_text_ablation_s42_20261105，tmux unified_v2_text_training；每200更新可恢复Checkpoint。已尝试adversarial-review，服务workspace routing discovery timed out，不列为独立审查通过。
