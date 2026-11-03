# 进度

2026-11-03：全量语言关联分析已推送2303ce1。创建独立Goal分支；实读Stage1、matcher、patch pooling与runtime。读取真实权重核对grid和gain，定位完整恢复缓存。尚未新训练、尚无新SR。

2026-11-03：新增read-only审计脚本及中文设计报告。使用恢复完整TRAIN缓存首个有支持帧，复制原matcher权重在RTX PRO6000上完成forward/backward（合成梯度检查目标，非训练），梯度有限。原始tests/test_stage1.py与tests/test_appearance_matching.py合计10 passed（1.29s）。原模型文件未写入，未启动导航。新分支目前仅审计/设计，没有声称新模块收益。

2026-11-03 第二阶段：独立FixedYawObservation实现且原入口未改。新增7项相关测试，连同原Stage1/appearance共17 passed。实际采集24物理实例/16地图/87帧，保存独立公开图像输入与private labels；原/小ROI两套冻结SigLIP2编码均完成。分离predict与score、先封存预测后读标签。原ROI22帧无候选，小ROI13帧；目标点深度支持但全帧无ROI从10降4。小ROI全文空间命中3→11/87，但shuffle也11，尚不能证明视觉有效；分尺度角色规则没有稳定优势，不据此扩大正式测试。生成24真实RGB案例图册；下一阶段需观测内细粒度位置头+独立TRAIN训练确认，不再调本24例固定权重。

2026-11-04 第三阶段：新150338参数LocalChainPositionHead已真实GPU训练3轮，非冻结规则。400物理目标/1454帧，排除前24；fit351目标1271帧(18图)，dev25目标92帧(3图)，confirm24目标91帧(3图)，地图和目标隔离。sourcecache、公共角色解析与原A3先验/当前SigLIP2 patches完成。最佳epoch1 dev81.52% vs A3 78.26%，先freeze后一次confirm真实59.34% vs A3 53.85%，NE24.761 vs34.074，9纠正4误改。Shuffle60.44%、Zero56.04%、去几何56.04%、换全文59.34%，不能将收益归因成正确视觉对应/关系链；没有新SR。best SHA a2a850160452ecc13f7b7b15402075b7e2f30f287181a7254a5186e9fa762ad5。实际Smoke恢复到epoch2通过；25 tests passed。下一阶段应针对几何/先验捷径设计视觉专属位置监督和新独立确认，别在旧confirm24上调参或将其称完整原Stage1对比。
