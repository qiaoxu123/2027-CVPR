# 实际外部 adversarial-review 状态

2026-11-06，按用户AGENTS.md要求实际调用插件命令：

```bash
node /home/20261024_1/.codex/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs adversarial-review '--wait --scope working-tree Challenge direction_frames_v4 scientific validity. Audit language-only landmark/subject binding, per-relation scope, region-valued viewpoint ambiguity, whether a directed road field is honestly distinguished from a unique scalar author yaw, GT-free centerline construction and spatial query, topology sensitivity, matched-support baselines, abstention denominators, semantic interventions, and historical comparability. Read-only, no edits.'
```

Job `review-mw4cdau5-i96xg6`，线程 `01a1aea3-153e-74e3-9461-83d44ed245d3`。运行时创建线程及turn，随后报告 `Codex error: Reconnecting...`，状态接口出现 `status=running`、`phase=failed`。确认运行时失败后仅取消该任务以清理挂起。

没有外部科学审查结论返回，不能称“审查通过”，也不能称实现被审查拒绝。没有用本地自检或文字描述替代实际调用。原始日志、取消前状态与取消返回保存在 `results/adversarial_review*`。

本地规则测试和全量公开输入推断/道路场复算是独立的实现检查，不证明作者采用这些参考系，不取代外部同行审查。后续保守区域支持的最终复核与代码SHA以 `freeze_manifest.json`、`verification.json` 为准；失败的审查未返回对任何修订的意见。
