# PR22 地标名称＋相邻几何词组：正式TRAIN实验

基于PR22 `73ff64ec5383c33446382c5613e168b6fe3ce5ad`；工作分支`experiment/landmark-geometry-words-only-20261105`。原模型与main未改动，没有Val/Test或闭环导航。

## 输入与解析核验

模型从零初始化。文本只来自公开CityRefer名称和紧邻的白名单几何短语，分别经过冻结BERT得到768维CLS。无完整原始指令编码、目标颜色/类别/外观、原Geometry隐层、RGB、轨迹、目标ID或目标坐标输入。名称中自带的Church/Road等是专名的一部分，不是额外目标类别。`across the road from`按文档视为完整几何构式，不单独编码中间对象。

使用FULL cityrefer_original公共目录，只按原文名称匹配，同名全部Polygon保留；不使用旧目标专属processed引用表。目标GT只在forward后计算损失和评分。解析只是词法邻接，不证明Target语义上直接连接；例如“停车场behind地标里的车”仅提供behind构式，不能据此声称车本身具有该直接关系。

全Revised TRAIN21,878条纳入：Cambridge14,259拟合、Birmingham7,619最终确认。名称＋短语12,367条（56.53%）；有可用名称20,657条（94.42%）。仅名称8,290、无公开名987、超16 Polygon容量234；后两类均匀回退，保留分母，不用GT选片段。

检查并修复自交/零面积Polygon接受问题；复合方向`north-east/north east`不再被截成`east`，直接否定`not north of`保留否定。无相对方向到世界南北的硬编码。复杂关系作用域、并列继承及参考系仍不保证准确，覆盖率不能叫解析准确率。

BERT只缓存639种允许字符串，每段含CLS/SEP的长度范围3–21，中位5.0；96是上限，不是每条固定Token数。每个Anchor有名称和词组各一枚CLS摘要，最多16个Polygon槽；网络不是原完整句Token模型。

## 训练及对照

seed42、3轮、batch16；每轮完整14,259条fit，固定最后Checkpoint，无Dev选参。AdamW2e-4、裁剪1、FP32；训练参数177,601。冻结BERT，原Geometry/Relation/PGVB/导航器均不加载。

训练65.761s，含Dev评估总计102.681s；峰值1.328GiB。12步真实GPU Smoke非零词组梯度、有限Loss通过；385项CPU测试通过，详见CPU_tests.txt；恢复后192条Smoke任务三组预测逐位一致，原实验未覆盖。恢复支持完成epoch的优化器/模型/RNG，不宣称步内恢复。

三组为同一最终模型的输入消融，**不是分别重训的三个最优模型**。原batch内roll会依赖batch大小，已改为全Dev固定置换，按查询/名称绑住同名多Polygon词组，保留原word_mask；4197个短语单元中3252个实际改变，不用GT置换。未知任务不排除。

## 全部7,619条Birmingham TRAIN-Dev

|方法|N|R@20m|NE(m)|纠正vs名称|破坏vs名称|净收益|
|---|---:|---:|---:|---:|---:|---:|
|Name only|7619|9.148%|81.7809|0|0|+0|
|Name + geometry words|7619|9.765%|81.0803|101|54|+47|
|Scrambled geometry words|7619|9.896%|80.1582|109|52|+57|

|输入子组|N|名称R20|真实短语R20|打乱短语R20|真实短语NE(m)|
|---|---:|---:|---:|---:|---:|
|all|7619|9.148%|9.765%|9.896%|81.080|
|paired|3957|9.704%|10.892%|11.145%|64.603|
|name_only_input|3112|9.254%|9.254%|9.254%|65.337|
|no_public_name|550|4.545%|4.545%|4.545%|288.708|

参考Polygon组合分组bootstrap（192组）真实短语对名称R20差值95%区间[0.139,1.357]个百分点；真实对打乱为[-0.540,0.365]个百分点。按公共参考组合分组不等于严格物理目标独立，因此不作广泛显著性结论。单种子、仅TRAIN-Dev，不能推出导航SR或未知场景泛化。

正确几何词未稳定优于两种对照；当前最小网络未证实可靠的几何词定位增益，不能据此扩大闭环。

无公开名称时输出均匀分布，Top-1按固定argmax首格计分；不能把该组偶然命中称作有效定位。历史完整Geometry+Relation使用不同输入、权重及数据拟合范围，不能直接把其约70%与本实验作公平性能差值。

名称/关系相同的不同车辆等目标被刻意映射到同一模型输入，无法恢复被丢弃的实例身份；方向样本不均衡、front/behind参考系未知也会限制结果。是否响应关系词与是否定位正确是两个问题，不能仅凭热图移动判断成功。缺少人工关系真值，不能报告语言解析准确率。

## 交付与恢复

[交互图册](index.html)、[提取示例](extraction_examples.json)、[训练曲线](training_curve.png)、[逐任务指标](paired.json.gz)、[机器指标](metrics.json)。服务器`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_geometry_words_s42_20261105/full`保存完整提取审计、BERT允许文本、Checkpoint、预测与日志。

最终Checkpoint：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_geometry_words_s42_20261105/full/last.pt`；SHA256 `5a89379b9d1c422fcaa37ae78bc7a6d011720bf0de17dd460d8870cbeef28fce`。未覆盖旧权重。adversarial-review调用因workspace routing discovery timed out失败，不列为独立审查通过。
