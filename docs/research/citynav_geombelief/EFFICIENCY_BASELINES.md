# 04 · 全量 Profiling、HETT 历史比较与公开基线边界

## GeomBelief 原正式完整模型 Profiling（Done）

- Revised Val-Unseen：**2,697 Episodes / 22,144 decisions**；默认 28×28 候选位置，Stop=0.3。
- GPU peak **allocated**：**3.7825 GiB**；policy 子网络参数 1,053,290；连冻结 BERT/SigLIP2 与未激活旧模块共驻留参数 992,389,996。
- 模块 GPU CUDA Event，环境同步 host wall time；有嵌套跨度与异步发射间隙，**子项不可机械相加**。CUDA/CPU 测量口径不同，原记录中存在后台 GPU 并发活动。
- 测试对象 **不是**另行训练的校准视觉残差或 384×384 候选网格。

| Stage | 最小值 ms | 均值 ms | 总体方差 ms² |
|---|---:|---:|---:|
| Rollout / decision | 31.16 | **48.17** | 595.40 |
| GPU model inference total | 17.47 | **24.33** | 110.04 |
| Language encoder | 0.00 | 3.07 | 72.94 |
| Visual encoder | 11.20 | **13.13** | 1.46 |
| Geometric localization | 0.00 | 0.35 | 4.06 |
| Visual residual/history | 2.16 | 3.44 | 0.24 |
| Trajectory/Stop | 3.94 | 4.34 | 0.19 |
| Environment observation | 3.92 | 6.72 | 110.30 |
| Other rollout overhead | 9.14 | 17.11 | 140.51 |

首个决策 `birmingham_block_5/1/1` 用时 1,493.78ms（GPU 754.42、环境 490.15、其他 249.22）；去掉它之后 Rollout 平均 48.10ms、方差 501.05ms²。Top 5 慢样本均为 Step 0，涉及环境准备及冷启动。

[已核查完整 JSON、参数与计时元数据](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/efficiency_existing_evidence.json)。

## 历史 HETT Val-Unseen Profiling（来自 Outline 归档；不同配置）

历史 HETT 加载 20 Epoch 训练中第 9 个 Epoch 的 best_val_unseen 权重，完整 2,697 样本；其中语言编码 5.32ms、视觉编码 91.38ms、导航网络 51.45ms、环境观测 27.69ms。

| 项目 | HETT 历史记录均值 ms | GeomBelief 归档均值 ms |
|---|---:|---:|
| Per-sample/decision Rollout | 200.82 | **48.17** |
| GPU model inference | 148.16 | **24.33** |
| Language encoding | 5.32 | 3.07 |
| Visual encoding | 91.38 | **13.13** |
| Environment observation | 27.69 | 6.72 |
| Other overhead | 24.97 | 17.11 |

表中平均比值是 Rollout 4.17×、Model 6.09×。**不是严格同 GPU 型号、并发负载、模块边界和环境渲染开销下的有效 Speedup 结论。** HETT 历史首样本 2,435.2ms（环境 1,728ms），去冷启动后平均 199.99ms；它与我们的“剔除首个决策”规范也需检查是否同义。

Outline 历史 HETT 报告：[HETT 性能 Profiling](http://127.0.0.1:12115/doc/hett-profiling-qcGKCl0MzO)。

## 与其他公开方法的对比原则

- **Published**：原论文使用自己的模型/评估器报告的数值；同为 Revised 并不一定文件/控制器完全一致。
- **Locally reproduced**：本地checkpoint 和完整逐 Episode 预测，并明确是否改动核心组件。
- **API substitution**：如 GeoNav 用 DeepSeek Flash 替代原 GPT-4o，属于新配置，不冒充作者原模型。
- **Original vs Revised** 数据版本严格分表。Original Test=5311，Revised Test=5281，GeoNav DeepSeek 原始 Test SR 26.66% / SPL 20.38% 不能直接拿来对 revised GeomBelief SR 37.15%。
- 内部历史复现只作研究背景，不将无严格可比条件的历史实验排入正式论文模型排行榜。

[外部基线协议审计](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/baseline_protocol_audit.md) · [论文实验总汇](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md)。
