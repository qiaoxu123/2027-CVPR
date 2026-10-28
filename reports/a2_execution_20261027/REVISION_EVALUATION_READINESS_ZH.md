# 修订版闭环评估准备与证据界限（2026-10-28）

修订版完整20+5轮训练已完成。全部25个epoch的样本数、优化器更新、Checkpoint别名及TensorBoard均经核验；选定第23轮，SHA256为`b5f7ae68659d1550c77e6d1cb4c32447433e9c2d459b2e4b2ac221ef45a2fc9d`。完整训练曲线见revision_v1_training_curves.png/pdf/csv；数值见REVISION_TRAINING_COMPLETE.json。大权重及原始日志保留服务器。

4618个内部验证teacher观测的定位R@1由68.644%升至71.048%，NE由19.909m降至18.890m，修正271、破坏160。此为内部定位指标，不能写成导航SR提升。

实际CUDA功能核验使用32个TRAIN目标、96个稀疏teacher观测：关闭分支时坐标、Stop、Belief Top16与原A0逐项一致；诊断开启改变52次目标选择、88次动作。它证明接入了原控制器，不证明闭环性能提升。

独立确认仍在采集：使用训练前预留的256个全新TRAIN来源目标，拟合127、留出129。只由A0执行动作，新增分支旁路计算；完整轨迹结束后才计算GT监督。采集结束后同一GPU锁下自动串行校准，留出否决后不得换配置或降门槛。尚未产生本修订版完整Seen/Unseen导航结论。A0/A1的原始完整Seen2470/Unseen2697评估已完成，并保持不变。

## 完整评估核验

- 入口不支持部分Episodes或覆盖控制器配置。pilot固定64个内部目标；正式Seen/Unseen固定全部2470/2697源索引。
- 每Episode保存原始轨迹、公开观测覆盖位图、真实检测投影、动作命令、Stop、原始Belief候选与同步计时。GT只在整段rollout结束后用于指标与覆盖率。
- 每条trace以SHA sidecar提交并绑定源Episode、代码、权重、校准、数据及冻结协议。恢复时逐条校验，全部结束才写完整标记。
- 用原BoundedWaypointController重放命令，核对每一步pose及policy_stop/stagnation/horizon终止；重新计算全部导航指标、Recall与候选覆盖。
- 冻结程序重验真实配对pilot、唯一拟合决策、校准原始行/packet及其SHA，并绑定原A0/WALDO权重、外部源码和完整评估数据。失败则不启动正式全集评估。
- 最终报告再次验证所有完整trace及汇总一致性，给出全部指标、逐地图结果、修正/引错比例、回退、覆盖及延迟。配对置信区间按目标实例聚类重采样，保留Episode加权均值；McNemar仅作描述性补充，不能替代聚类区间或跨种子验证。

独立CPU只读审查覆盖控制器重放、冻结谱系和配对统计。额外复算150条真实原控制器生成轨迹（3种终止各50）以及40组不等簇大小bootstrap，均符合预期。实际adversarial-review插件也已调用，服务器日志`adversarial_revision_controller.log`记录workspace routing discovery timed out；这不属于插件审查通过。没有因此跳过独立只读审查。

## TRAIN诊断控制

REVISION_TRAIN_CONTROLS.json使用固定哈希预选128个TRAIN目标各1个已有teacher观测、冻结模型、未校准scale1/threshold0，仅检查机制，不参与门控选择：

- 真实新增分支R@1=71.094%，A0=66.406%；检测失败及参考歧义均128/128精确回退。
- 去掉box-reference pair几何后R@1=72.656%，链接置换为71.094%。该小样本不支持声称几何/链接已产生稳定收益。
- no_geometry仍保留A0上下文和候选支持；no_text仅去掉新增分支文本，A0 prior仍含语言。image-relative从实际footprint构造图像坐标方向，其他几何量不变。
- 有合法对应目标的反事实语言控制尚未完成；没有通过更改方向却保留错误GT来伪造控制结果。

当前结果不足以确认目标“A2超过A0”。必须等待独立门控、真实配对pilot、冻结后两个完整验证集的导航对照；如失败则保存结果并在TRAIN上继续迭代，不强制启用。
