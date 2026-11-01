# AerialVG 官方实现理解、测试与适配

日期：2026-11-01。官方仓库：https://github.com/Ideal-ljl/AerialVG 。

## 可复现来源

- 仓库：`/home/20261024_1/Workspace/third_party/AerialVG`。
- commit：`db33e1dc733911853c5e132394d2e00b024334db`。
- 官方权重：`https://huggingface.co/Ideallll/AerialVG`，经 hf-mirror 传输；本地 `../aerialvg_assets/aerialvg.pth`，725,198,638 bytes。
- 权重 SHA256：`f563437fd0b6f3a2e8e1b037ce15ef6ad40c81aead385fb81dae01c7953cc812`。
- 独立环境：`../aerialvg_env`，继承已有 graphnav 包，新增依赖只安装到该 venv。
- README 当前仅发布推理代码及权重，训练代码 TODO；数据集有访问申请条件。本轮不声称复现整个测试集准确率。

## 官方代码实际做了什么

1. `model/AerialVG/aerialvg.py`：Swin 多尺度视觉特征、BERT 完整 token、图文融合 encoder 和检测 decoder。900个检测query，经文本token分数筛选Top-K（配置15）。
2. `model/AerialVG/relation.py`：将K个decoder query两两拼接，得到K²个关系token；先对这些token做512维self-attention并投影到256维，然后交替堆叠3层文本cross-attention和关系self-attention。
3. 同文件：使用 `ContrastiveEmbed` 输出每个query对与文本token的点积分数，reshape为 `[B,K,K,256]`，对两个候选维度分别取mean后平均，输出 `[B,K,256]`。**实际发布代码的输出是候选—文本token分数，不是直接返回显式关系分类标签。**
4. `aerialvg.py`：Top-K分类logits以0.4关系分数、0.6原分类分数融合。关系模块位于框预测之后，本身只重排logits，不修改预测框。
5. `fuse_modules.py::similarity_update`：在四层空间相似度图之间做top-down双线性上采样、bottom-up共享卷积更新。`alpha=beta=sim_param`，当前模型常数0.3；该函数的 `up_conv` 形参没有实际参与计算。四层空间位置要对齐，不能把32/64/128m的池化ROI当作同样结构。

## 原始模块测试

`test_official_modules.py` 直接导入官方模块：

- 关系模块前向/反向、文本padding屏蔽通过。
- 替换有效文本向量确实改变输出；这是随机输入结构检查，不代表任务准确率。
- 候选置换等变性通过。
- 层级更新系数为0时严格identity；非零时输出改变且梯度可回传至输入及官方下采样卷积。
- 结果：`official_module_results.json`；日志：`module_test.log`。

实现限制：关系模块把最终token长度256与hidden_dim=256耦合，不能直接改为64维使用；它对K²个关系token做self-attention，计算随K约按K⁴增长。直接用于全局784/147456网格不可行；需保留Top-K。

## 官方权重整图推理与干预

`run_official_inference.py` 严格加载986项state（无missing/unexpected keys），使用仓库 `images/demo.JPG`（原图4000×3000，按官方demo预处理为800×1066）。

环境未加载官方编译CUDA扩展。兼容脚本仅在进程中将算子dispatch指向仓库自带 `multi_scale_deformable_attn_pytorch`，在CUDA tensor上运行；没有改动上游Python源码。结果不用于官方CUDA速度比较。

|条件|Top-1得分|Top-1 query|说明|
|---|---:|---:|---|
|完整模型|0.707163|0|框落在右侧道路的蓝色汽车上|
|删除关系重排|0.601453|0|框与Top-1不变，分数降低|
|删除层级更新|0.720243|0|候选排序/框略变，Top-1仍是同一车辆|
|两者都删除|0.612857|0|同checkpoint干预|
|above改为below|0.703149|0|部分候选分数/排序改变，Top-1仍不变|

关系删除使用原分类头结果替代关系分数，保证融合严格退化为原logits，避免 `0 * -inf` 造成padding NaN。层级删除令更新系数为0。单图没有新指令的人工GT，因此不能用分数或Top-1变化判断关系推理正确率。

结果：`demo_results/results.json`；可视化：`demo_results/full.jpg`；完整张量及5种条件图片均保留。

## 与此前方案的关键差别

此前自建的候选消息/平均或单层注意力没有复现官方的K²关系token与交替文本cross-attention；不能把其无增益结果归为“AerialVG无效”。新适配方案将实际加载上述官方关系源码及44个关系权重tensor，保持256维内部接口和完整文本token，通过可训练的视觉/语言投影桥接导航候选特征。官方关系权重先冻结，残差门零初始化、未观测候选零更新。

它仍是导航适配：SigLIP ROI与GroundingDINO decoder query不同，投影需训练；不声称直接迁移就会有效。HCA的四层空间特征接口尚未接入；不会用ROI插值替代。此前三路角色解析/弱监督代码暂不用于本轮官方适配对照。

补充算子检查：官方PyTorch可变形注意力的CPU/CUDA输出最大绝对误差为2.3842e-7，且value、sampling locations、attention weights均可回传梯度。结果保存在 `deformable_fallback_results.json`；未测试编译CUDA扩展。

导航适配的14项相关测试通过（含实际官方模块、预训练参数加载、梯度、零初始化、无效观测屏蔽、共享初始化与旧接口）。这是软件集成证据，不能代替定位/导航收益验证。
