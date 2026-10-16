# 384 全量评测与视觉残差改进

状态会自动更新；未完成的 split 不报告整体 SR。

|384 split|已完成 / 总数|整体 SR|状态|
|---|---:|---:|---|
|val_seen|2470 / 2470|48.8259%|complete|
|val_unseen|2697 / 2697|23.1739%|complete|
|test_unseen|5281 / 5281|38.2693%|complete|

冻结原 C Epoch1，实际网格 384×384，stop=0.3；Test 不参与视觉模型选取。目录内保留冻结 manifest、输入/权重 SHA、完整逐 episode 轨迹及续跑记录。

## 视觉残差

两种训练候选：几何条件/质量置信度残差，及允许真实可见负证据整体降权的非中心化残差（用原生图文匹配代替 ego 距离捷径）。二者只训练新的残差头，静态主干、轨迹与 Stop 参数冻结；仅用 Train 标签，按原 Train Dev 整组划分选权重。

|候选|训练完整|Dev 静态 R@1|Dev 修正 R@1|
|---|---|---:|---:|
|calibrated_residual|True|67.338%|66.651%|
|absolute_residual|True|67.338%|67.132%|

### calibrated_residual 闭环配对对照

|控制|Seen128 SR|全 Seen SR|
|---|---:|---:|
|real|50.0000%|48.7045%|
|no-belief|50.7812%|47.3684%|
|zero|50.7812%|47.3684%|
|shuffle|46.8750%|46.9231%|

完整 Seen：帮助 120 条，损害 87 条，SR 差 +1.336pp，配对 bootstrap 95%CI [+0.162, +2.510]pp。只有区间下界>0 才标为有明确正收益；同时查看 zero/shuffle 结果验证实际 RGB 贡献。

### absolute_residual 闭环配对对照

|控制|Seen128 SR|全 Seen SR|
|---|---:|---:|
|real|48.4375%|47.8543%|
|no-belief|50.7812%|47.3684%|
|zero|50.7812%|47.3684%|
|shuffle|47.6562%|47.3279%|

完整 Seen：帮助 88 条，损害 76 条，SR 差 +0.486pp，配对 bootstrap 95%CI [-0.526, +1.538]pp。只有区间下界>0 才标为有明确正收益；同时查看 zero/shuffle 结果验证实际 RGB 贡献。
