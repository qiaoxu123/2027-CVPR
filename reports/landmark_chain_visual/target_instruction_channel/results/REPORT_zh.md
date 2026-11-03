# 目标通道补回原始指令：同预算训练对照

相同1165条拟合样本、固定3×159次更新、seed0、同起点。参考通道、A0、冻结SigLIP2、RGB/Yaw0、架构与损失均相同，仅局部目标通道是否使用原始完整指令不同。独立TRAIN确认尚未进行。

|方法|Goal R1|NE(m)|纠正/误改A0|
|---|---:|---:|---:|
|FrozenV3|85.87%|17.2714|8/1|
|FrozenV3+FullTarget|85.87%|17.0930|8/1|
|TrainedRole|85.87%|17.4964|8/1|
|TrainedFullTarget|86.96%|17.0528|8/0|
|FullTargetZeroRGB|82.61%|18.7596|6/2|
|FullTargetShuffleRGB|82.61%|18.2125|5/1|
|FullTargetNoGeometry|85.87%|17.3873|7/0|

## 配对比较

- TrainedFullTarget相对FrozenV3：纠正1、误改0。
- TrainedFullTarget相对TrainedRole：纠正1、误改0。
- TrainedFullTarget相对FullTargetShuffleRGB：纠正4、误改0。

## 边界

这里的92状态来自25条已使用TRAIN开发任务，非独立泛化结果，不能当作闭环SR。旧冻结模型闭环22/25，原软实例模型19/25；本轮未新运行导航。

SigLIP2文本上限64 token，完整字符串仍可能截断：原400查询11条超限，新增236查询6条超限。5对已知碰撞在64-token输入中均可区分，不代表全部空间关系被保留或已被视觉理解。

完整原文目标通道保留了名称与参考内容，模型仍可能依赖语言/地图偏置；必须通过Shuffle和独立确认判断。不能把输入不同、梯度非零或训练loss下降当性能改善。
