# 全样本 384 网格与视觉残差改进

- [x] 建立独立 worktree；共享只读数据/权重，不覆盖 GraphNav-V1 正式结果。
- [x] 加速 ROI 支持统计与半精度传输；通过原 28/384 初始 Top16 重放一致性及 24 项相关测试。
- [x] 预先冻结原 C Epoch1 + 384 grid + stop=0.3，启动全部三 split，支持按完整 episode 续跑。
- [x] 完成 Val-Seen 2470 / Val-Unseen 2697 / Test-Unseen 5281，分别报告 SR/SPL/NE。
- [x] 从 Train 缓存诊断静态候选分数差、视觉残差量级、可观测覆盖与外观区分能力。
- [x] 训练带静态先验条件、局部置信度与当前/历史证据的增强残差；静态主干及控制器冻结。
- [x] 同一 Seen 子集 real/no-belief/zero/shuffle 对照，内部 Train Dev 与独立 Seen 评测，不用 Test 调参。
- [x] 汇总全量指标、收益/损害样本、运行限制与下一步依据。

当前状态：三个 split 全部完成，共 10448 episodes，队列退出码均为 0；完整 SR/SPL/NE 及相对 28×28 冻结模型的差值已生成。此前于 2026-10-14 22:25:52（北京时间）按用户要求通过 `--resume` 续跑。两组视觉残差训练及所有 28×28 Seen 对照已完成并跳过。原暂停记录保留在 `pause_record.json`，新进程与恢复进度见 `resume_record_20261014.json`，实时进度见 `live_status.json`。

`scripts.summarize_resolution_when_complete` 已完成，核验同一权重、stop、episode 清单和完整逐 episode 索引，将 SR/SPL/NE 与正式 28×28 冻结结果及差值写入 `full_resolution_384/COMPARISON_28_VS_384.csv` 和 `comparison_complete.json`。其日志为 `full_resolution_384/comparison_watcher.log`；失败时保留错误记录 `comparison_failure.json`，不改写主实验结果。
