# PR25训练阶段完成

全21878条、seed42、3轮完成；每轮69709个有效Anchor方向标签，仅新增231817参数优化。最终gain=0.151301；训练记录累计514.43秒，首轮325.49秒，后续复用几何/BERT缓存。原主模型所有张量与原Checkpoint一致。

Birmingham7619条属于TRAIN内拟合诊断，不是独立Dev：

|方法|R@20m|NE(m)|
|---|---:|---:|
|冻结全文基线|64.1948%|33.8403|
|预测8方向|64.1554%|33.8358|
|Oracle8方向，仅TRAIN离线|64.9560%|33.3591|

有效方向标签17381个、预测分类准确率41.1254%。预测定位纠正6、破坏9、净−3，尚未形成定位增益。Oracle仅检验相同已训练有界融合下的GT角度贡献，不是理想位置Oracle上界。

2条TRAIN真实闭环仅做推理接口Smoke（不评分）：只公开objects的catalog，GT角度标签函数被替换为抛错哨兵，推理仍通过；正常路径不需要任何标签。冻结第3轮后已启动完整2470条Val-Seen，未改Stop/Goal/动作协议，Oracle没有运行时入口。

Checkpoint SHA256：c30eb96ad408f07eb96bb5f02dd88ebcf4fcdd283c632a1542dfef190502182f。

CPU392项通过。独立adversarial-review调用失败：workspace routing discovery timed out，未计为审查通过。后续等待完整导航成绩，不根据Val调参。
