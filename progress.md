# Progress
Created independent diagnostics worktree from 1f545e00; clean parent. Read runtime, cache and six losses. No network weights or production code modified.
GPU smoke caught a diagnostic-only interface error: world_grid requires explicit dtype. Added dtype to both diagnostic calls; no production change. Failed smoke retained separately.
Smoke_v2: 3 actual TRAIN cases, all six views and frozen counterfactuals completed on GPU. Original 15 CPU tests pass. Full 4405 TRAIN paired run launched in tmux v3_obs_fusion_diagnosis; no Val/Test execution.
Saved prior Val artifacts regrouped without new Val input/model execution. Full TRAIN paired run and Polygon audit live in tmux. Runtime/training/model source remain unchanged.
Polygon audit completed all 4405; four real GPU per-component autograd probes complete. Data remains TRAIN only; original checkpoints untouched. Full six-condition paired run progressing.
CPU tests: original V3 plus offline diagnostic tests, 17/17 pass. Diagnostic tests verify no update outside observed mask, original scale formula, target-free gate signature and uniform-confidence abstention. Full run has passed 2925/4405; no tuning from partial results.
