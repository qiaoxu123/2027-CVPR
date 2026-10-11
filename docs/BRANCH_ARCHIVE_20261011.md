# GraphNav 分支归档与清理结果（2026-10-11）

> **实际完成**：在 GitHub Actions [38111070896](https://github.com/qiaoxu123/2027-CVPR-Dev/actions/runs/38111070896) 中，通过先创建并核验远端轻量 Tag、再按固定 SHA 校验删除分支的流程，**12条旧/临时 GraphNav 分支成功归档并删除，12个 `refs/tags/archive/graphnav-20261011/*` 标签可恢复原始提交**。早期两次因标签核验与 PR 权限限制安全中止，不构成历史丢失；最终已修正完成。

> **最终保留5个分支**：
> - `main`：GraphNav-V1 C Epoch1 正式代码、全量 Seen/Unseen/Test-Unseen 报告，保留 A3 强基线和后续研究代码。
> - `2027-CVPR/graphnav-relational-visual-belief`：原 V1 完整提交链（含 `9e458d7`）。
> - `research/multiview-rgb-grounding-20261011`：多视角 RGB 研究头。
> - `research/active-view-selection-20261011`：推送后新更新，SHA 不再匹配原计划，因此自动跳过并保留。
> - `backup/20261011/detached/instruction-landmarks-only-60a4efe7`：detached HEAD 历史备份。

> **证据边界**：本报告只验证 `qiaoxu123/2027-CVPR-Dev` 的远端引用状态。用户提到的95/96本地分支、25个 detached HEAD 以及 HETT 丢失17对象的问题需以不可访问的 `.workspace/git_push_audit_20261011/PUSH_REPORT.md` 和各仓库原始资料单独核验；本次未接触其它仓库。权重文件 `.pt` 未入 Git，正式性能由已冻结 Checkpoint SHA 及原实验报告支持，不等于本次重新训练或 GPU 重评。

---
# GraphNav 分支归档计划（2026-10-11）

本次**只涉及** `qiaoxu123/2027-CVPR-Dev` 仓库，不涉及 HETT、SBFNav 或本地工作区中其它仓库的95/96分支，也不删除任何 detached HEAD 备份。

前提：
- 正式 GraphNav-V1 C Epoch1 的代码、报告、原始 `9e458d7` 提交均已并入 main，且远端 V1 分支是 main 的祖先。
- 只要 SHA 与固定清单不一致就**跳过删除**，不覆盖新提交。
- 对每条计划删除的远端分支，先创建 `archive/graphnav-20261011/<branch 名替换 / 为 -->` 标签，核对指向同一 SHA，再执行 GitHub Ref 删除。**不删除源提交对象**。
- 这是一次性 workflow，仅当合并提交标题以 `chore(branches): archive redundant GraphNav heads` 开头才实际执行。执行成功后移除 workflow。

保留分支：
1. `main`（V1 正式发布 + A3 强基线 + 历史实验代码）
2. `2027-CVPR/graphnav-relational-visual-belief`（原始 V1 训练提交快照）
3. `research/multiview-rgb-grounding-20261011`（最后一轮 RGB 研究）
4. `backup/20261011/detached/instruction-landmarks-only-60a4efe7`（独有 detached 提交备份）

准备归档后清理的12条冗余分支：

- `2027-CVPR/graphnav-closedloop-sbf-parity` → `64294e3ad9764f7ba1a9a18d8c1dad7cf276c8ea`
- `2027-CVPR/semantic-geometry-localization` → `4d6c81c2395505fb4473a08383d0476576acd776`
- `integration/graphnav-v1-main-20261011` → `b3018e7b9e9f0766243b16a9e86637408f8fcea7`
- `maintenance/main-release-audit-20261011` → `141c940d4f9e4fe770e5326507ee3c62963e18e3`
- `research/active-view-selection-20261011` → `043d3897a97af8f35ee60889fa3640f59393c4ad`
- `research/instance-rgb-grounding-20261011` → `06ebf5398d051ecfc105298821ce3b12eced0131`
- `research/rgb-feasibility-20261011` → `62044c124c2a188e15358cd8c412115c04209590`
- `research/rgb-language-grounding-20261010` → `350f99d3a55d305969f934e31fea6faf08d05fe8`
- `research/visual-goal-association-20261011` → `4a69bdf73fabaa931ee92354b52eb6672430ccd3`
- `validation/bidirectional-p0-p2-local-20261009` → `11e0409b322f08db4459888c6d38ba08547da730`
- `validation/full-token-rgb-local-20261009` → `8a75d9ed5396933f89492624360a3c2a9e8b530f`
- `validation/static-geometry-20261009` → `e6344eac1c6d03307af43b187a0cf927f01bdf7e`

对 GitHub 上新出现的其它分支，不做操作。GH Actions 上传 `branch_archive_results.csv`，失败不宣称完成；根据该文件验证残余分支。

用户所述 `.workspace/git_push_audit_20261011/PUSH_REPORT.md` 未在当前可访问项目附件或远端 Git 树中，本次核验**仅基于远端 GraphNav refs**。特别注意旧 HETT `codex/hett-sbf-candidates-siglip` 仍缺17个 Git 对象，未修复也未触碰。
