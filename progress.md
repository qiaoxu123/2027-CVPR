# 进度

2026-11-04 关系因子实测：d8bdff2后已完成tmux GPU Smoke、恢复及Hidden/Typed各477更新。TRAIN Dev92帧：冻结V3 85.87%/17.2714m，Hidden86.96%/16.9301，Typed86.96%/17.0578；均相对V3+1/-0，Typed不优于Hidden，参照RGB打乱92个Goal全部不变。Typed拟合净-1，缺乏关系学习收益；不进入确认/闭环，不加轮数。唯一开发纠正原文还存在between作用域错挂，不能当语义成功。拟合12条AI复核发现4条明显错误/歧义、1楼层不可直接观测、1参考系缺失，仅6可用部分关系，非人工金标准。下一步先审计可靠参照定位/作用域监督，不扩网络。日志/配置/权重/真实RGB全保留，原权重逐轮未改。

2026-11-04 关系因子预案：新增多参照公共关系提案，400任务中204可用、74多参照、16普通中间节点关联公开锚点；保留单分支作用域，不将between伪装成两条near边。新增WorldRelationFactor在冻结V3世界支持上更新，Null中性、无参照严格回退，Forward输入白名单无GT。原图像缓存保留，仅242参照文本编码4.41s。27项模型/基线相关测试及20项句法/元数据测试通过。已固定Hidden/Typed各3轮同预算TRAIN方案，真实GPU Smoke/恢复与训练结果尚待运行；不访问Val/Test。

2026-11-04 当前状态补记：固定Val-Seen128已完成（f3c4c08）：Original45/128，新方法52/128，+11纠正/-4误改，未通过至少84/128门槛，未启动全量/Test。当前新增关系输入审计：400条仅38条通过保守单参照提案；四组left/right/near/behind旧局部视觉文本输入相同，新增公共关系码可区分。14 tests passed。没有新训练/新SR，下一步先扩展TRAIN关系表示覆盖，不能只依赖38条训练。

2026-11-03：全量语言关联分析已推送2303ce1。创建独立Goal分支；实读Stage1、matcher、patch pooling与runtime。读取真实权重核对grid和gain，定位完整恢复缓存。尚未新训练、尚无新SR。

2026-11-03：新增read-only审计脚本及中文设计报告。使用恢复完整TRAIN缓存首个有支持帧，复制原matcher权重在RTX PRO6000上完成forward/backward（合成梯度检查目标，非训练），梯度有限。原始tests/test_stage1.py与tests/test_appearance_matching.py合计10 passed（1.29s）。原模型文件未写入，未启动导航。新分支目前仅审计/设计，没有声称新模块收益。

2026-11-03 第二阶段：独立FixedYawObservation实现且原入口未改。新增7项相关测试，连同原Stage1/appearance共17 passed。实际采集24物理实例/16地图/87帧，保存独立公开图像输入与private labels；原/小ROI两套冻结SigLIP2编码均完成。分离predict与score、先封存预测后读标签。原ROI22帧无候选，小ROI13帧；目标点深度支持但全帧无ROI从10降4。小ROI全文空间命中3→11/87，但shuffle也11，尚不能证明视觉有效；分尺度角色规则没有稳定优势，不据此扩大正式测试。生成24真实RGB案例图册；下一阶段需观测内细粒度位置头+独立TRAIN训练确认，不再调本24例固定权重。

2026-11-04 第三阶段：新150338参数LocalChainPositionHead已真实GPU训练3轮，非冻结规则。400物理目标/1454帧，排除前24；fit351目标1271帧(18图)，dev25目标92帧(3图)，confirm24目标91帧(3图)，地图和目标隔离。sourcecache、公共角色解析与原A3先验/当前SigLIP2 patches完成。最佳epoch1 dev81.52% vs A3 78.26%，先freeze后一次confirm真实59.34% vs A3 53.85%，NE24.761 vs34.074，9纠正4误改。Shuffle60.44%、Zero56.04%、去几何56.04%、换全文59.34%，不能将收益归因成正确视觉对应/关系链；没有新SR。best SHA a2a850160452ecc13f7b7b15402075b7e2f30f287181a7254a5186e9fa762ad5。实际Smoke恢复到epoch2通过；25 tests passed。下一阶段应针对几何/先验捷径设计视觉专属位置监督和新独立确认，别在旧confirm24上调参或将其称完整原Stage1对比。

2026-11-04 第四阶段：local_chain_runtime与独立TRAIN真实闭环入口完成，原控制器与Stop参数不变；2条接口Smoke后，冻结24 TRAIN确认目标跑5组120条真实轨迹（Original完整RGB历史/A3/Real/Zero/Shuffle），全部结束。SR分别13/24、13/24、15/24、14/24、15/24；Real和Shuffle逐任务SR完全一致。Real SPL.399<Original.517，全部20步无Stop；末次预测Goal距离中位2.932m vs.000137。9失败最后一次Goal全错，7从未正确，不能主归因Stop。Code/真实轨迹/归因/6案例18决策位姿重建RGB已保存navigation24；28 tests passed，完成后resume校验无重复导航且SHA不变。下一阶段先以原固定世界784格接收概率避免运动候选/熵支持变化，再做视觉专用世界位置监督与中间参照解析，新确认目标；仍未跑Val128，不是65%门槛结论。

2026-11-04 第五阶段：确定Direct相机patch点不含光心，合法高度下离agent至少.5908m>.001，Stop不可能；写入全局lessons后修复为概率投回原28×28世界格，默认grid，原控制器不改。另一组TRAIN dev25相同权重对照：Original21/25 SR.84 SPL.8059；Direct18/25 SR.72 SPL.4925 Stop2；Grid20/25 SR.80 SPL.7234 Stop25；GridZero19/25、GridShuffle18/25。Grid修复相对Direct净+2但相对Original净-1，尚无有效新方法结论。新观测SHA日志+重建2帧内容完全一致；32 tests passed。原确认24未重调，后验空间支持Real7neverGoal失败中6在Footprint/5深度/6近20m候选，均代理不是真实可见性/实例标签。下一步保持Grid，输入级灰图/像素块Shuffle核查位置编码捷径，再视觉专属弱定位/对比监督与中间参照NP，新确认物理目标；最后128ValSeen门槛未测。
