# CityNav 离线句子解析协议 v1

本任务由当前会话模型直接阅读、解析指令并写入 JSONL，不运行模型 CLI、API 或规则解析器。请求模型为 gpt-6.1-sol，reasoning effort 为 low；该信息是请求配置，不冒充外部 API 回执。程序只做输入导出、结构校验、证据 Span 定位和缓存查询。

输入为原始文本及 SHA256 text_id。模型不读取 target_positions、trajectory、object_ids、对象 GT 描述或测试标签。全部 split 的文本可以解析，因此这不是未接触测试文本的盲测。难度子集只建立别名索引。同一文本共享语义解析，地图对象绑定留给独立后续阶段。

每条记录包含 entities、relations、frames、uncertainties。实体 ID 在句内有效，mention 必须是原文子串；kind 区分 named_landmark、object、group、region。每条关系记录 subject、predicate、reference（允许多参照物）、evidence、confidence（high/medium/low）及 frame ID。qualifiers 保留颜色、形状等属性及原文依据；不通过几何检验修改语义。只标记文本显式的具名地标，匿名建筑不能因为是目标就获得地图对象 ID。

frame 类型为 world/observer/road/intrinsic/unknown。status 为 unique/multiple/unknown/not_required。origin 与 forward_to 为符号实体 ID，不提供虚构角度。候选方向由 candidate_descriptions 文字记录，不能利用目标选择。方位词地理 north/south 使用 world，但不把它映射为地图 XY。道路侧别未给行进方向时是 unknown；建筑 front/behind 可表明 intrinsic 候选，却不证明建筑的有向正面已知。

关系主语不必是目标。间接关系必须分解：A behind B on Road C 不能自动变成 A behind C。对跨句共指只在语言支持时绑定；存在语法作用域歧义时记录多候选并降低置信度。关系 confidence 与参考系可确定性独立。

执行顺序：冻结本协议；分批直接阅读指令并写入解析；程序检查完整性、原文证据与实体引用；生成进度及未完成账本；通过 ID/hash 查询已有结果。未完成记录必须报 missing，不能伪造空解析或在线调用模型补齐。

本缓存是模型解析结果，不称作完美标注。语义置信度未经概率校准。当前阶段不读取地图 Polygon，不声称完成空间朝向求解，也不训练或修改导航模型。
