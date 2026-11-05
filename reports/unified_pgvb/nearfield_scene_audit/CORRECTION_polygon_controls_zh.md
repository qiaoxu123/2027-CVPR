# 名称对照解释更正

上一轮只替换name_text、保留Polygon A及ROI A，测量的是名称敏感性。该实验不能证明PGVB无法区分不同地标实例，也不能证明未使用Polygon轮廓。

新一轮同步替换Polygon、Token、ROI及候选区域：34个同类困难查询中，完整PGVB为52.94% Top1地图代理，Polygon Token置零为38.24%。这证实Polygon输入会影响匹配，但不等于真实像素边界已对齐。原名称对照的数值保留，超出干预范围的解释撤回。

完整设置、共同子集及负结果见[Polygon—RGB实验报告](../polygon_rgb_matching/REPORT_zh.md)。所有指标仍为地图弱监督代理，非独立实例/像素真值准确率。
