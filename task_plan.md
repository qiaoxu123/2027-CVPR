# V3 observation and fusion diagnosis
Baseline 1f545e00. No training, no model/runtime changes, no Val/Test execution. New diagnostic scripts/reports only.
- [in_progress] Audit source, freeze inputs and match 4405 TRAIN heldout IDs to official starts.
- [pending] Run paired positions/yaw/frame-count and gate diagnostics on all heldout.
- [pending] TRAIN fixed fusion counterfactuals and weak Polygon/loss/gradient audit.
- [pending] Inspect stored Val failure categories without inference; plot actual examples.
- [pending] Verify checkpoint/source integrity, test diagnostic scripts, report and push independent branch.
