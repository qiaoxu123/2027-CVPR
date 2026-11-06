# PR26 主Belief Angular Loss 执行审查

全部21878条Revised TRAIN，seed42，5轮，A Predicate-only/B λ0.1/C λ0.3；同一Original V2文件SHA379cfc41和相同实际初始化张量SHA a0b1cc535a4d1ffd15eb85c3d7c89f1ef29c7929fea0b8f96d9ae0c5961deeb6。LR Geometry1e-5/Relation1e-4、KL0.05、batch8、FP32/TF32关闭、BERT/PGVB/Controller冻结。三组并行训练，后续闭环串行，保留历史权重。

修复训练/推理绑定不一致：原训练源预处理地标替换为原文+完整公开目录盲绑定，不读取目标ID挑Anchor。原PR句尾匹配在真实数据中筛出0条，修复前置方位和有后续句子的明确直接关系，支持公开别名；排除否定、复合方向、导航朝向、明确中间对象。语法自动筛选不是人工语义真值，不通过GT方向吻合度筛样本或翻转坐标。

保守资格21条（0.096%），north8、south13、east/west0；21条GT距Anchor中心均大于1m。按现有工程轴假设：8bin词义匹配1/21，半平面匹配6/21，不能称已核验地理方向准确率。TIF CRS=None、无tags，保持原轴约定。Angle Loss本身监督实际GT地图方位，并不把north强制作为+Y标签。

全部样本仍参与Goal/KL，Angular只对上述21条；重复5轮不是105条独立监督。GT只在forward后Loss/事后评分。修复无候选方位支持时的假常数loss，以及断点恢复Angular计数丢失。新Checkpoint必须匹配--angular-main-predicate，避免省略Predicate默默评测错误输入。

CPU412项通过。真实TRAIN纯Angular梯度：Geometry L1=6780.64、Relation=18.65；第二更新Predicate Embedding梯度L1=0.0007266，证明主网络可反传。首步Predicate梯度0源于原Relation score零初始化，非接口断路。随机4batch Smoke无21条资格样本，因此不能用它冒充Angular反传验证；另行完成仅按公开文本选择的2例专项GPU验证。

三组训练已完成；离线定位和全量闭环尚在执行，最终成绩另行生成。主要对照复用同公开绑定重新训练的全文47.37%，以隔离Predicate收益；早期45.75%旧权重在绑定训练分布上不同，只作历史上下文。
