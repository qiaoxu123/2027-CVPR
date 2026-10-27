# LocalSemanticRelationRefinement 0.1

主模型负责宏观二维定位；独立分支在其多峰局部邻域内，联合指令语义、box外观、深度与参照物几何关系，重排实例并给出有界位置偏移。保持原GraphNav 28×28权重、原控制器和飞行高度逻辑。旧BoxRelationCorrection不被覆盖。

## 网络和数据

- 冻结现有WALDO、SigLIP和BERT；使用完整1024维RGB向量、逐token语言特征和32×32双通道深度/mask。
- 新增3层depth CNN、256维4头Cross-Attention、两层候选—参照物关系更新，包含双参照物“之间”特征。实际参数数目以verification.json为准（首版约4.62M）。
- 辅助头预测期望方位/距离/弱表面高差；主头输出候选分数、局部XY偏移和matchability。
- private labels与public inputs严格分开。目标ID/坐标/轮廓用于离线关联和监督，禁止传入模型或线上门控。
- 公共地标链接与原基线一致；不声称已经解决开放地标检测/指代。深度是正射DSM代理，高差只是可见表面弱证据，不是物体顶部/完整3D标签。
- 新schema重建缓存；目标实例不跨train、calibration、dev。身份不明确候选不进入排序softmax分母。

完整设计：[spec](docs/superpowers/specs/2026-10-27-local-semantic-geometry-refinement-design.md)。实现计划见docs/superpowers/plans，执行记录见LEDGER.md。

## 运行

以下均在本目录运行，Python必须使用现有conda环境。所有输出拒绝覆盖已有训练/校准/评测；源码或缓存不匹配会停止，勿手改签名绕过。

```bash
/home/20261024_1/miniconda3/envs/graphnav/bin/python -m unittest discover -s tests -v
/home/20261024_1/miniconda3/envs/graphnav/bin/python scripts/prepare.py
/home/20261024_1/miniconda3/envs/graphnav/bin/python scripts/verify.py
```

独立训练默认20轮排序关系、5轮加入偏移，AdamW lr3e-4，8个独立观测梯度累积后更新；冻结特征离线缓存。每轮内部dev选模，不用论文Val。测试链路可用 `train.py --smoke --steps 4 --output runs/smoke_new`；smoke权重无论结果如何都不能启用导航修正。

```bash
/home/20261024_1/miniconda3/envs/graphnav/bin/python scripts/train.py
/home/20261024_1/miniconda3/envs/graphnav/bin/python scripts/calibrate.py --checkpoint runs/branch/epoch_001.pt --output runs/branch/calibration.json
```

上面的epoch_001只是命令格式示例，实际应使用training.json中selected_checkpoint；如果为空，表示没有满足内部dev门槛的权重，继续保持诊断模式。高置信实例关联仍须抽检，不应把初始IoU规则称为已验证精确实例真值。

```bash
/home/20261024_1/miniconda3/envs/graphnav/bin/python scripts/evaluate.py --split val_seen --arm baseline --limit 2 --output runs/baseline_new
/home/20261024_1/miniconda3/envs/graphnav/bin/python scripts/evaluate.py --split val_seen --arm refinement --checkpoint runs/branch/epoch_001.pt --calibration runs/branch/calibration.json --limit 2 --output runs/refinement_new
/home/20261024_1/miniconda3/envs/graphnav/bin/python scripts/visualize.py --limit 6
```

省略评测limit运行固定paired128；不要把两任务smoke或paired128与全量论文结果混比。可视化带checkpoint时可显式使用`--diagnostic-alpha .25`查看强制局部反馈，但图中会标DIAGNOSTIC，不代表校准策略或效果提升。

## 接入与限制

反馈仅重新分配候选支持区域内原概率质量，区域外概率保持不变；α≤0.5。校准不足、dev出现新增错误、无有效支持或smoke权重时，门控保持关闭并精确回退原策略。matchability不是“物理不存在”的概率。

候选分数/连续偏移可独立评测，但原控制器仍消费28×28位置；相同栅格中的两个实例可能重排成功却无法改变执行位置。原始连续偏移是有界提议，不能当成已验证可达或安全航点。首版只验证离线局部修正和原栅格反馈，不宣称完整三维导航/亚栅格控制。

`no_language`、`no_geometry`等控制针对固定缓存上的网络通路；候选集合仍由原指令的公开提议过程生成，不应将其解释成端到端无语言系统。深度/表面高差有各自掩码，必须另报有效深度子集结果。

正式比较前需要：实例关联质量抽检、困难样本覆盖、完整内部训练/校准和独立配对导航；实现测试通过不代表指标提升。当前实际运行状态和数据数量见reports/STATUS.md。

## 远程复现

分支包含代码、runtime源码快照、配置和报告；原主checkpoint、WALDO权重、全部860视图冻结特征与本次最终验证权重放在配套GitHub Release完整资产包中，避免大文件进入Git历史。原始CityNav/RGBD和通用BERT/SigLIP模型缓存属于外部依赖。逐文件SHA清单见SYNC_MANIFEST.json。runtime优先复用原工作区，缺失时自动使用本分支快照，不修改原仓库。

在该分支根目录恢复完整资产：

```bash
gh release download local-semantic-geometry-refinement-20261027-v0.1.1 --repo qiaoxu123/2027-CVPR --pattern local-semantic-geometry-refinement-20261027-v0.1.1.tar.zst
tar --zstd -xf local-semantic-geometry-refinement-20261027-v0.1.1.tar.zst
```

Release说明记录压缩包SHA256；SYNC_MANIFEST.json逐项记录解包后的文件哈希。资产包也含代码快照，可在独立目录完整恢复；旧版调试缓存和过期smoke权重只在本机归档，不作为当前可用版本发布。

使用v0.1.1或之后的包；初始v0.1漏打Ultralytics的data子包，已由恢复测试发现并修复，不能独立恢复。

仅用已同步特征训练可使用 `scripts/train.py --cached-only --output runs/new_training`；这仍核验缓存/配置、主模型、WALDO及源码哈希，但不打开未参与该次训练的外部原始annotation。重建观测/闭环评测需要设置 `CITYNAV_DATA_ROOT`、`CITYNAV_RGBD_ROOT`，必要时设置 `CITYNAV_PROJECT_ROOT`；加载冻结编码器仍需原Hugging Face缓存。缓存条目支持换机器后的本地路径重定位，不能绕过内容校验。
