# CityNav 直接离线解析缓存（进行中）

用户要求：当前模型直接解析，指定 gpt-6.1-sol low；不用模型 CLI、外部 API、在线解析或其他模型。解析结果提前保存，使用时查询文件。

基于 v5 分支提交 `7193b0a9e435494bd36aab2297438e45def42a08` 开展独立工作。旧 v1–v5 实验与导航模型未修改。本任务获准读取所有 split 的指令文本，与旧实验 TRAIN-only 的统计结论分开。

## 当前实际进度

- 共 31,755 条不同文本，43,272 条 split/难度子集查询别名。
- 已直接解析 120 条、319 条空间/定位关系；剩余 31,635 条。
- 缓存尚未全量完成，不能用于完整数据集的导航效果评测。
- 首批顺序由文件名排序决定，首批为 40 条测试指令，后续 80 条具有 TRAIN 别名（其中 3 条也出现在测试文本中）。跨 split 的相同文本共享解析，不作为独立泛化证据。
- 实际进度以 [parsed/progress.json](parsed/progress.json) 为准；所有未完成 ID 保存在 `parsed/pending_ids.json`。没有模型后台任务在自动补齐剩余解析。

模型直接撰写的语义记录在 `direct_batch_*.json`，`i` 为冻结 `inputs/instructions.jsonl` 的零起始行号。`cache.py` 仅将这些记录展开为具名字段、定位原文 Span、校验引用并建立缓存，不生成关系标签。`N/U/I/R/W/F` 仅为序列化简写，解释见代码与协议。每条展开结果保留来源批次。

## 使用

Python 中无需调用任何模型：

```python
from experiments.sentence_parser.cache import lookup
record = lookup(split_file="citynav_test_unseen.json", episode_index=0)
record = lookup(text_id=record["text_id"])
```

未完成 ID 抛出 `KeyError: Parsing missing`；不会自动降级到规则解析或返回伪造空结果。`records.jsonl` 是便携解析文件；`cache.sqlite` 是可重建的本地查询加速副本。实体地图对象 ID 暂未绑定，frame origin/forward_to 为句内符号角色，不是伪造地图朝向。

```bash
/home/20261024_1/miniconda3/envs/graphnav/bin/python experiments/sentence_parser/cache.py build
/home/20261024_1/miniconda3/envs/graphnav/bin/python -m unittest experiments.sentence_parser.test_cache -v
```

## 首批保留的失败与歧义

- building / parking area 的关系从句附着不唯一：保留候选主语。
- `behind church` 可修饰 short street，不强行提升为目标建筑的直接关系。
- `take a right ... first building ... left`：right 为转弯动作，left 才是对象侧别；保存来源道路、进入道路、转弯方向，但未读取拓扑前不宣称有唯一朝向。
- `house facing Road`：房屋是 facing 主语，不是观察者；不能因此确定旁边目标的 left/right 视角。
- `when viewed aerially with Livingstone Road to the left`：保存明确视图条件，但没有地图几何就不生成数值 yaw。
- 具名道路 behind 匿名目标建筑：保留真实主客体，不因希望统计具名地标参照关系而倒置。

置信度是模型自评等级，未做人工概率校准。本产物不称为“完美解析”；目前仅完成句子语义层，不等于已经完成地图绑定或语言参考系的独立几何验证。
