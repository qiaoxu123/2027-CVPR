# 已完成且保留的 A0 / A1 完整对照

A0、A1均已完成完整Val-Seen2470及Val-Unseen2697，原始结果不修改。A0采用既有正式完整训练归档；A1完成A3/B3/C5全部训练后按预定Seen128及内部验证规则选择C1快照。单种子归档A0对新训练A1不能作为匹配重训的因果证明。

| 验证集 | 指标 | A0 | A1 | A1−A0 |
|---|---|---:|---:|---:|
| Seen2470 | SR (%) | 47.611 | 45.789 | −1.822 pp |
| Seen2470 | SPL (%) | 46.003 | 40.686 | −5.316 pp |
| Seen2470 | NE (m) | 29.119 | 29.812 | +0.693 m |
| Seen2470 | 最终Belief R@1/20m (%) | 47.444 | 45.672 | −1.772 pp |
| Unseen2697 | SR (%) | 22.544 | 24.805 | +2.262 pp |
| Unseen2697 | SPL (%) | 21.821 | 22.080 | +0.260 pp |
| Unseen2697 | NE (m) | 51.533 | 50.722 | −0.811 m |
| Unseen2697 | 最终Belief R@1/20m (%) | 22.471 | 24.673 | +2.202 pp |

A1改善Unseen，但损失Seen成功率及路径效率，不能称为全面超过A0。原SR逐Episode配对95%区间：Seen[−3.198,−0.486]pp，Unseen[+1.075,+3.411]pp；Unseen SPL区间包含零。原报告另列query-cluster区间及单种子限制。

完整OSR/nDTW/sDTW/False-Stop、全部Recall、逐地图和配对结果、训练曲线及10334条轨迹重算核验见原分支：

- [完整结论及曲线](https://github.com/qiaoxu123/2027-CVPR/blob/e070786d9dbd3f94446e950336814a310fce6bc4/reports/core_pair_20261027/ANALYSIS_ZH.md)
- [完整机器可读结果](https://github.com/qiaoxu123/2027-CVPR/blob/e070786d9dbd3f94446e950336814a310fce6bc4/reports/core_pair_20261027/results.json)

本摘要不包含尚未完成的修订版A0/A2完整评估。内部teacher定位R@1不能替代导航SR，完整A0/A1不能充当完整A0/A2。

源文件内容SHA256（本次只读核对）：

- ANALYSIS_ZH.md: `58916360830b1b0fb1f4383e7c112d84f687b617ed319c93d161f123d44704d0`
- metrics.csv: `b41c7c6b5efe52cba52c78183850796e3298bb0a0621d198b104f8614502e79c`
- completion_verification.json: `751dd25feee5ef70102d1e6748124eb18df078ceafc7484aa4dcf034648b372a`
- recall_verification.json: `d655562bf4cfe8fb6020673b491cf5a9538c478a7f343336344f79db5c83cc7d`
