# 独立关系图核验约定

下载`review.html`后在浏览器直接打开，或在实验目录用静态HTTP服务打开。原文与目录来自TRAIN；不显示GT、误差、轨迹或目标图像。

1. 在未查看模型候选时，选中英文中的实体文字，用页面按钮添加实体。目标是指令要求到达的对象；地标是实际提及的专有名称；中间实体是串联目标和地标的普通对象。无关的参考对象标reference。不能因在目录出现就添加不存在的实体。
2. 目标描述保留颜色、形状等修饰，不把参考地标当目标；原文不唯一时备注，不强行指定。目标可能自身具名，应记录其目标角色并注明身份歧义。
3. 添加subject→relation→object。例如`car in lot next to Library`为car→inside→lot及lot→next to→Library。`Library behind the target building`保留Library→behind→building，不能简单把主客体反转。链的遍历方向另行记录。
4. 不依赖坐标解释front/behind/left/right。不在本轮专项标注指代、序数、否定；遇到影响关系判断的歧义请保留备注，不编造中间对象。
5. 填写审核人代号，完成后勾选并保存。查看模型候选后保存会标为辅助审核，不自动计入独立金标准。可直接编辑JSON修正，跨度采用原文字符索引。
6. JSON/CSV导出在本地下载。重新打开同一个HTML可从浏览器localStorage恢复；换电脑请导入导出的JSON。不同浏览器或清空网站数据不会共享保存内容。

推荐两名人工独立标注再裁决，先固定规范，记录争议。当前仓库没有宣称存在这种标准标签。

正式评分命令：
`python -m scripts.report_parser_benchmark --run <SERVER_RUN> --output <REPORT_DIR> --gold human_graph_reviews.json`

评分只接受reviewed=true、independent=true、source=human_editor且审核人非空的记录。目标/地标/中间实体采用严格字符跨度及role；关系三元组保留主客体，少量表面同义词归一化（in/inside、beside/next to等）。完整链准确率仅对人工图确实含目标—地标路径的任务计算，不将空链算作成功。所有模型输出、模型共识与本Agent案例意见均不自动成为GT。

A/B的实体类别在原解析器不输出类型时，使用相同的原文Span词典作type补全，标记type_source=shared_lexical_span_heuristic；它不是新训练的类别识别器。C同样补全unknown类型，D保留其原始类别。A/B原始置信路径过滤与通用图展示分别保留：图上存在路径不代表方向网络当时实际采用了该路径。
