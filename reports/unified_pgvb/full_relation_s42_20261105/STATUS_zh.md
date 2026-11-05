# PR17任务最终状态

已完成：全Revised TRAIN 21878条×5轮Geometry+Relation，249帧真实RGB×3轮PGVB，CPU/GPU Smoke与恢复检查；完整Val-Seen四组各2470条，共9880条真实闭环，失败和回退全部计入。

结果：原Geometry47.77%，训练后Geometry50.24%，+Relation50.24%，+RGB50.00%。新分支没有带来优于训练后Geometry的SR；负结果完整保存。

详细结果、分层、配对、Belief/Stop失败归因见REPORT_zh.md；曲线、12个真实案例、完整轨迹压缩文件与Checkpoint SHA已保存。原权重不变，不使用Val调参，Test/Val-Unseen未运行，main未合并。
