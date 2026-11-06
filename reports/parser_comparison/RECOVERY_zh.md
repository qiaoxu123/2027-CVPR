# 复现与恢复

服务器RUN：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/parser_comparison_s42_20261106`。

- 正式输入：`data/public_inputs.json`，V2 SHA `fc9821b1d679b3299873b6e74ee6fd03c46ff6052bce355acd23bb0ebfc3d376`。推理命令不得改用PRIVATE_sampling.json或任何含GT数据文件。
- A/B解释器：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106/parser_env/bin/python`；spaCy3.8.7，sm/trf3.8.0，NumPy1.26.4，CuPy13.6.0。
- C/D解释器：`$RUN/gliner_env/bin/python`，GLiNER2 1.2.4，Transformers4.57.6，Torch2.8.0+cu128，Accelerate1.15.0。未修改主conda环境。
- C权重：`/home/20261024_1/Workspace/models_local/gliner2-base-v1`，官方revision `f9634218e53580c56edf0de97ca1a7d3f1c2354e`。
- D权重：`/home/20261024_1/Workspace/models_local/Qwen3-14B`，官方revision `40c069824f4251a91eefaf281ebe4c544efd3e18`，BF16，非thinking、greedy，不调用外部LLM API。

首次推理：在项目根目录，设置`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4`，运行：

`<PYTHON> -m scripts.run_parser_benchmark --inputs "$RUN/data/public_inputs.json" --output "$RUN/<TAG>" --parser <TAG> [--model <LOCAL_MODEL>] [--batch 32]`

A/B/C实际batch1，D正式batch32。D的batch2与batch8输出仅为工程/时延试验，不能与最终400条混为一组。

每批结果追加到predictions.jsonl，已完成index自动跳过；config和源代码SHA不一致时拒绝混合恢复。若主脚本在运行后改动，可使用该运行目录保存的SOURCE.py原样恢复：先将项目根目录加入PYTHONPATH，再用同样参数执行`$RUN/D/SOURCE.py`。保持输出目录、模型、batch和输入完全一致。不删已有结果、不用失败过滤分母。

C有5条图的关系端点按原模型字符偏移修复（避免同词形不同位置误合并）；原始模型输出保持不变。补丁说明见C/postprocess_patch.json，修复前数据在服务器C_before_endpoint_fix。`repair_gliner_endpoint_offsets`可在原始副本上重建图，无需重跑神经网络或读取GT。

汇总：依次执行`report_parser_benchmark`、`build_parser_review`、`build_parser_comparison_cases`。加入独立人工JSON后才使用`--gold`生成真实解析指标。

V1清单及其A/B/C结果保留于`$RUN/protocol_v1`和Git b33c0ed。不要与V2结果混用。浏览器自动测试未生成任何真实人工标准图。

D原始生成后增加唯一大小写字面Span规范化（例如the building→原文The building），不是根据GT改写语义，也不在多个重复提及中猜选。正式推理结束后执行`normalize_qwen_parser_spans --run <RUN>`；该脚本拒绝在推理未完成时覆盖JSONL。原始生成、修复前图和标记全部保留。未能唯一对齐的实体单独统计，不能与“模型完全没输出目标”混为一谈。

C是span架构的GLiNER2-base-v1，max_width=8，固定20类空间关系schema、threshold=.5；不是GLiNER2.5 JointIE或GLiREL的性能结论。A/B提供完整公开名称表给PublicNames；C以同目录的字面出现名称作schema提示；D接收完整目录。因此这是三种实际系统的比较，不是只替换编码器的统一架构实验。

A/B的统一角色映射还区分：原生路径内部普通节点为intermediate，其余普通名词为reference，避免人为抬高中间实体输出率。旧SOURCE.py生成后可执行`normalize_spacy_graph_roles --run <RUN>`；只重映射已保存的原生解析，不重跑spaCy、不读GT，原图保留在A/B_before_intermediate_role_mapping。
