# PGVB 64地标闭环执行计划

Goal: 在eed893a基础上冻结权重、任务与控制器，完成E2/E4/E5各64条真实闭环。
Architecture: 使用现有PGVB桥接和因果当前观测；仅修复接口错误、补充可追溯诊断。禁止训练、目标GT/未来帧/Oracle进入策略。
Tech stack: PyTorch CUDA, SigLIP2, Shapely, 原GraphNav控制器。

- [x] 拉取并确认eed893a，定位历史64条任务及原控制器配置。
- [x] 检查1024维三尺度特征、16x4局部Polygon、28x28世界网格、候选context与Stop接口；补充回归测试。
- [x] 测试全部相关单元测试，真实CUDA前向，两条配对闭环Smoke。
- [x] 冻结输入和源代码SHA，tmux运行E2；核对61/64历史结果，若不符先归因。
- [x] 相同64任务完成E4与E5，支持按任务恢复。
- [x] 统计配对SR/OSR/NE/步数/Stop/有效更新/Goal跳变/耗时，保存至少4条真实RGB和世界Belief/轨迹可视化。
- [x] 核验所有权重未改变，完成中文报告、测试、commit/push。

Review focus: 不可见弃权、Other类别、坐标网格次序、PGVB更新是否真正到达controller、E2完全相同与任务无变更。用户已授权执行，无需再次确认。adversarial-review技能未找到，不能声称已执行该技能。
