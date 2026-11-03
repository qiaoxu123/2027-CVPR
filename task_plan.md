# 地标关系链视觉分支 Goal

- [完成] 原始SigLIP2/主权重/协议审计；固定camera yaw0，原入口不改，无AerialVG。
- [完成] 24例机制、400实例1454状态、150338参数GPU训练3轮、恢复与独立TRAIN确认。
- [完成] TRAIN确认24×5组120真实闭环。Real15/24 vs Original13/24，但Shuffle同15/24，且Direct无Stop/SPL恶化。
- [完成] 修复确定性接口错误：相机patch点作为Goal不能到达Guard；概率投回原固定世界784格，原Stop/动作不改。
- [完成] 同TRAIN开发25例Direct/Grid配对。Grid20/25、Direct18/25，Stop2→25，但Original21/25；未超过原系统。32 tests passed。
- [进行中] 检验真正图像内容贡献：Gray RGB/像素块Shuffle保留真实Mask/坐标，排除特征位置编码与相机中心捷径；视觉专属世界位置/必要对比损失，目标/中间参照NP角色，最终固定网格定位/保真。
- [待做] 新物理目标独立确认及真实闭环，旧24确认集不再调参；保留原模型/所有负结果。
- [待做] 冻结模型和128 Val-Seen清单，SR严格>65%(至少84/128)且相对Original有增益才全量。不得用TRAIN80%当Val成绩。
- [进行中] 分阶段commit/push代码/报告/真实轨迹/Checkpoint说明；大文件server，Test仅最终冻结后。

当前无新Val/Test；Goal仍以真实导航指标为终点，不能把定位R1、回放或开发SR当正式成绩。
