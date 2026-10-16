# 暂停与恢复记录

用户于 2026-10-14 要求记录当前进度并停止；恢复必须等待用户再次提出。
当前暂停的是 384×384 闭环评测，两组视觉残差训练及 28×28 全 Seen 对照均已完成。
精确完成数量、停止时间及原进程命令保存在 `pause_record.json`，逐 episode 数据保留在原输出目录。

## 后续恢复

在本 worktree 目录执行以下命令，自动对未完成 split 添加 `--resume`，已完成 episode 跳过；中断中的 episode 重新执行。

```sh
cd /home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
/home/rental/20260922_1/Workspace/2027-CVPR/Baselines/.reproduction-env/bin/python -m scripts.run_full_resolution_parallel >> full_resolution_parallel.log 2>&1
```

另一个终端可运行状态汇总器（视觉对照已完成，等待大网格任务结束后汇总）：

```sh
/home/rental/20260922_1/Workspace/2027-CVPR/Baselines/.reproduction-env/bin/python -m scripts.visual_residual_workflow >> visual_residual_workflow_offline.log 2>&1
```

保持冻结权重、384 网格、stop=0.3、数据及 episodes 清单一致；恢复时脚本会校验配置和权重。不要删除或覆写现有输出目录。
恢复会有冷启动开销；最终汇总中的运行秒数仅代表最后一次运行，不代表累计执行时间，累计时间需结合暂停及 queue 记录。
