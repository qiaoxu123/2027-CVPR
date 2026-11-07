# 两套冻结Geometry等权集成

仅TRAIN development3126条，FP32前向；固定比较当前V2、Stage1 Checkpoint内原Geometry及二者概率50/50平均，均使用相同公开绑定和5×5/T0.5解码。未拟合、不搜索权重、不读取确认或Val/Test。私有目标仅在三组预测生成后评分。

当前V2 R20=72.94%、NE20.942m；Stage1原Geometry69.71%、22.543m（相对V2纠正77/破坏178）；固定集成71.94%、21.295m（纠正45/破坏76，净-31）。不采用集成，不依据开发误差继续调权重，不进入导航。

已有模型错误存在部分互补，但本次固定平均不能可靠利用它；不能把GT挑选两模型正确结果当可部署收益。原始逐任务在服务器runs/frozen_geometry_ensemble_20261107/results.json。脚本evaluate_frozen_geometry_ensemble.py可复现，原权重未变。
