# 诊断复现

- 基础提交：`fcfde0bcd9e7f6cc66d0a1f23302ecd4ec9289f6`。
- 主报告：[REPORT_ZH.md](REPORT_ZH.md)。机器可读汇总：`candidate_summary.json`、`navigation_summary.json`。
- `--experiment-root` 指向包含既有原生Query缓存、Train/内部Dev prepared缓存及三个128归档的本地实验目录。无需重新训练或编码图像。

```bash
python -m scripts.diagnose_aerialvg_candidate_bottlenecks --experiment-root /path/to/visual-residual-20261013 --output /tmp/aerialvg-candidates
python -m scripts.diagnose_aerialvg_navigation_failures --experiment-root /path/to/visual-residual-20261013 --output /tmp/aerialvg-navigation.json
python -m scripts.render_aerialvg_bottleneck_report --candidate /tmp/aerialvg-candidates/candidate_summary.json --navigation /tmp/aerialvg-navigation.json --output /tmp/REPORT_ZH.md
python -m pytest -q tests/test_aerialvg_bottleneck_diagnostics.py tests/test_aerialvg_diagnostic_artifacts.py
```

环境使用已有`aerialvg_env`（Python3.10、Torch2.8）；该工程依赖原有`2027-CVPR`工作区布局。候选/融合统计只读Train和内部Dev；导航按请求只读已存128条Val-Seen轨迹，不读其源标注、不跑新闭环。20m几何Oracle不是实例匹配准确率。30/60候选的融合是原15候选训练模型的离线预算外反事实，不能据此调正式评测参数。

阶段日志与逐状态大JSON保留在本地`candidate_v2/`（不提交）；输入manifest、Checkpoint和归档轨迹哈希在汇总中。首次运行的空候选汇总错误已修复并有回归测试。原正式验证进程按本轮诊断范围暂停，既有输出保留，未自动恢复。

## 融合Oracle补充

参见 [FUSION_ORACLES_ZH.md](FUSION_ORACLES_ZH.md)。运行：

```bash
python -m scripts.diagnose_aerialvg_fusion_oracles --experiment-root /path/to/visual-residual-20261013 --output /tmp/fusion-oracles.json
python -m pytest -q tests/test_aerialvg_fusion_oracles.py
```

Oracle为GT参与的离线数学干预，不用于推理/调参；同样固定σ、gain、bound和观测掩码。既非新训练模型，也非正式R@1/SR结果。
