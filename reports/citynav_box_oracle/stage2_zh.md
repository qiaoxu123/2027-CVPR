# 阶段二：可编辑数据集、隔离测试与Smoke

annotation_set/含全部1275条自动框提案、未核验模板、64个不同实例的当前PNG、可编辑review.html和静态预览。HTML JavaScript语法检查通过；支持拖动画框、确认/修正/拒绝/不可见、真人声明、图像SHA绑定、JSON导入导出与浏览器草稿。

冻结候选阶段按原流程计算：raw900、valid900、guided NMS、空间抑制Top15/30/60；原始15特征/世界坐标逐条对齐缓存。正常选择/打包函数没有GT参数。私有目标仅在正常分数算完后进入评价或显式Oracle路径。

32状态Smoke完成（17个高几何支持假设状态）。Oracle A从对应当前真实RGB框裁剪，用同一冻结AerialVG取得真实query特征，不造向量；注入候选的深度质量也从实际bbox/depth mask计算。B保留实际score/null/世界映射/原更新；C对比像素框中心与已知目标点、bounded与unbounded理想位置证据。

用户无人工数据：human-verified A/B/C严格为NA；投影假设敏感性结果另存，未核验状态不自动成为真人已批准。低支持/视野外/无效轮廓不注入。完整1275状态诊断使用固定参数，不训练或读取Dev。没有将Smoke结果当完整结论。
