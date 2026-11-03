# 局部视觉文本输入碰撞审计

实际新增训练样本265，涉及51张来源图；同图不同查询共949对。完全相同目标/参考向量5对，其中两边均参与软实例监督且正核心不相交5对，涉及10个查询。配对和查询不是独立导航样本。

这是输入信息丢失的实测证据：同图、相同视觉特征和相同目标/参考向量，局部visual_score必定相同，不可能让同一个确定Top1同时落到两个不相交目标核心。并不意味着全部损失必然无解（多个高分候选可满足软平均margin），也不意味着完整Goal必然相同：A0全文和空间分支仍可能区分。核心是弱地图标签，不能冒充真实像素实例标签。

## 原始指令与实际局部文本

### train_00733_0030 / retarget_00645 vs retarget_00590

- 目标输入：`The black car`
- 参考输入：`the 12th row of cars`
- 原文A：The black car that is 2nd from the right in the 12th row of cars in the parking lot behind the One Stop Building, when viewed aerially with the long side of the One Stop building to the right.
- 原文B：The black car that is 1st from the right in the 12th row of cars in the large parking lot behind the One Stop Building, when viewed aerially with the long side of the One Stop Building to the right.
- 地图教师中心距离5.02m，正核心不相交=True；完整文本向量相同=False。

### train_01899_0316 / retarget_01263 vs retarget_01943

- 目标输入：`The gray car`
- 参考输入：`the 6th row of cars`
- 原文A：The gray car that is 5th from the right in the 6th row of cars in the large parking lot behind the One Stop Building, when viewed aerially with the long side of the One Stop Building to the right.
- 原文B：The gray car that is 2nd from the right in the 6th row of cars in the large parking lot behind the One Stop Building, when viewed aerially with the long side of the One Stop Building to the right.
- 地图教师中心距离7.43m，正核心不相交=True；完整文本向量相同=False。

### train_07126_0029 / retarget_07162 vs retarget_07105

- 目标输入：`The black car`
- 参考输入：`the 2nd row of cars; the massive parking lot`
- 原文A：The black car that is 14th from the left in the 2nd row of cars in the massive parking lot for the West Midlands Police Custody Suite, when viewed aerially with Holford Drive to the right.
- 原文B：The black car that is 16th from the left in the 2nd row of cars in the massive parking lot for the West Midlands Police Custody Suite, when viewed aerially with Holford Drive to the right.
- 地图教师中心距离4.71m，正核心不相交=True；完整文本向量相同=False。

### train_07126_0029 / retarget_07550 vs retarget_07050

- 目标输入：`The black car`
- 参考输入：`building`
- 原文A：The black car that is 1st from the bottom in the column of cars to the right of the West Midlands Police Custody Suite.
- 原文B：The black car that is backed up to the side of the West Midlands Police Custody Suite. It is the third car in the row.
- 地图教师中心距离4.69m，正核心不相交=True；完整文本向量相同=False。

### train_18380_0020 / retarget_19074 vs retarget_18241

- 目标输入：`The gray car`
- 参考输入：`the 2nd row of cars`
- 原文A：The gray car that is 6th from the left in the 2nd row of cars in the parking lot behind the Grafton West building, when viewed aerially with the Grafton West building to the left.
- 原文B：The gray car that is 13th from the right in the 2nd row of cars in the parking lot behind the Grafton West building, when viewed aerially with the Grafton West building to the left.
- 地图教师中心距离16.96m，正核心不相交=True；完整文本向量相同=False。

## 下一步方案及边界

优先修复目标语义接口，保留颜色/类别，同时把目标自身的序数、排列和实例限定词送入局部视觉分支；参考地标/区域继续单独表示。未知参考系保留语言，不强制转为世界方向。A0保持原全文。先验证输入碰撞被消除、原目标属性不丢失，再用相同训练预算和独立TRAIN确认。

不能直接把全文塞给所有角色当成已验证方案：此前full_text消融并无稳定收益。此次只证明特定序数信息被删除，不证明修复后SigLIP2一定具有数数能力，更不能据5对解释全部导航失败。若现有特征无法表示车排/序数，需明确报告，而非继续强加不可辨别标签。

## 真实GPU前向确认

对旧冻结V3与新软实例模型分别执行这5对样本的实际GPU前向：每对的144个视觉分数最大差均为0，Top1索引完全相同。两模型均未命中这10个弱目标核心（0/10）；这不是像素级识别准确率，只是地图核心代理结果。GT只用于前向完成后的核心匹配，公开前向输入为原8字段。详见gpu_forward.json。

影响范围为265个新增样本中的10个（3.77%），949个同图查询对中的5对（0.53%）。不能据此推断这是大多数导航失败的主要根因，也不能通过删除这10条就宣称问题解决。它明确限定了下一步：局部语义接口必须保留实际用于区别实例的限定词，再评价视觉特征能否利用这些词。
