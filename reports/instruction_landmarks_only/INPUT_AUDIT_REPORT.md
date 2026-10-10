# GraphNav-V1：只保留指令地标的输入审计

用户澄清：只保留该条指令提到的目标相关地标，其他地标不放；不强制截取两个，不用真实目标坐标挑选。

当前正式模型输入已经是该条指令的CityRefer名称绑定，经保守名称/别名解析得到地图多边形。max_landmarks=16是多边形上限，不代表输入16个无关地标；同名道路等可包含多个碎片。max_roads=0，不加入额外道路中心线。其他地图对象只用于查找绑定，不整体送入模型。当前框架不声称仅凭指令能够挑出某个同名道路的目标附近片段。

|分割|核验Episodes|筛选前后输入SHA一致|改变输入|额外未绑定地标|
|---|---:|---:|---:|---:|
|Val-Seen|2470|2470|0|0|
|Val-Unseen|2697|2697|0|0|

全部5167个Episode的public_static_record逐项核验。过滤后的地标名称或别名必须可追溯到该Episode的reference_names；未改语言、位姿、几何边界、视觉或控制器。保留同名合法多边形，不以GT距离消歧。完整逐Episode原始/筛选输入SHA保存在runs/instruction_landmarks_only/val_seen_input_audit.json和val_unseen_input_audit.json。

这是输入不变性审计，不是一次新的导航运行。该筛选不会形成不同的模型输入，因此不重复训练/评估、不将已有SR/SPL标为新测消融成绩；主模型正式三split结果仍从GraphNav报告读取。正式权重保持冻结C第1轮。

曾检查字面名称匹配方案，但它会误删真正提到的地标，例如Police Custody Suite简称、Food Warehouse简称及拼写变体。因此该尝试未运行导航，源码与统计保存在runs/instruction_landmarks_only/rejected_literal_name_matching，不作为正式消融结果。不能把名称绑定失败冒充删除无关地标的收益。

204项测试通过。独立worktree来自主实验1678eb0；正式runtime和evaluator与主实验字节一致。审计入口：

```bash
/home/rental/20260922_1/Workspace/2027-CVPR-static-geometry/.venv/bin/python -m scripts.audit_instruction_landmarks --main-run /home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1 --output runs/instruction_landmarks_only
```
