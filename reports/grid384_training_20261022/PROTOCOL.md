# Real 384-grid training, preregistered 2026-10-22

User authorized new 384 training and complete evaluation. Isolated worktree: research/grid384-training-20261022. Original GraphNav C1 weights, formal results, data and paper remain unchanged.

## Model identity

- Stage A/B/C all use 384×384 = 147,456 real world-coordinate candidates, including loss normalization and proposal generation. RGB remains 384×384. Native map rasters remain at their original resolution.
- Initialize from the original verified A3 Epoch14 and trajectory/Stop checkpoints (28-grid pretraining), not the old integrated C1 weight. This is **384-grid stage training with 28-grid pretrained initialization**, not training from random initialization or 384 pretraining of every component.
- Original frozen BERT and SigLIP2 revisions, geometry architecture, loss weights, residual bound 0.5, guarded-blend and Stop settings retained.
- Dense geometry skips padded entities whose masked softmax weights are zero. Forward/gradient parity is checked against the original implementation with identical dropout RNG. Parameter shapes/architecture unchanged.

## Data and optimization

- Reuse original full Train-Seen sampling metadata: 21,878 episodes, 87,512 states, same seed 0 and map+instruction 5% internal Dev exclusion. All original source cache chunks are SHA-verified; only poses, text and labels are reused.
- Recompute observation support and frozen ROI vectors at true 384 candidate coordinates from real orthophoto pose-conditioned observations. No interpolation/relabeling of 28-grid ROI arrays. History accepts strictly earlier frames with original pose deduplication.
- Candidate geometry uses a bounded32-query CPU LRU to reuse public geometry across the four sampled states without unbounded RAM growth.
- New derived feature cache is bounded at 160 GiB and stops writing below 40 GiB disk free; missing entries can be recomputed. Raw datasets stay read-only.
- Goal supervision preserves normalized sigma 0.9/28: sigma_cells=0.9×384/28=12.342857 at384. Keeping0.9cells would silently narrow the physical label width13.7×. Old trainer defaults remain0.9; new training explicitly passes the scaled value.
- A=3, B=3, C=5 complete epochs. Microbatch 4, gradient accumulation 4: effective batch 16. Original AdamW group learning rates and cosine schedule measured in optimizer updates.
- Limitation for controlled comparison: eligible visual/ranking losses are averaged within each microbatch before accumulation. This is not algebraically identical to original batch16 eligible-only reduction; report this difference. No claim that only grid size changed.
- A/B selection follows Seen128 SR, SPL, internal Dev belief recall. Final model is the best **C** checkpoint by the same Seen-only key, ensuring trainable map geometry received 384-grid updates. Unlike the historical workflow, B is not eligible as the final model. This rule is fixed before training.
- Save model, optimizer, scheduler, RNG, sampler position and running metrics every64 microbatches at accumulation boundaries. Resume explicitly; no overwrite of prior experiments. Training freezes A3/controller during A/B; C trains the original permitted branches.

## Completion gate

1. Finite CUDA smoke A/B/C checks; no smoke weights enter full training.
2. All A3/B3/C5 epochs and Seen128 selection finish, with actual sample counts, gradients, memory, time and complete checkpoints.
3. Freeze chosen C checkpoint SHA before any held-out evaluation.
4. Full Revised Val-Seen2470, Val-Unseen2697, Test-Unseen5281, sequentially on the same GPU. Never select by held-out results.
5. Seen128 real/zero/shuffle/no-belief controls; isolated Seen16 profiling with phase breakdown. Check GPU contention before interpreting speed.
6. Save full real384 B0/Bt/residual/world coordinates/observed support/RGB for all steps of the first three episodes in each split. Raw RGB-containing outputs stay in ignored local runs, never auto-published.
7. Offline recompute NE/SR/OSR/SPL against GT and exact Episode IDs; official Easy/Medium/Hard membership; belief L1/KL/top1 shift; source SHA summaries. Only then mark workflow complete.

## Locations and operation

- Local outputs: `runs/grid384_20261022/`; workflow: `workflow_status.json`; logs: `train_A.log`, `train_B.log`, `train_C.log`; per-stage `progress.json` and `last.pt`.
- Launch: reproduction-env Python `-m scripts.preflight_grid384` (detached process logs to `runs/grid384_20261022/launcher.log`).
- Resume a failed/stopped workflow only when no child is running: `python -m scripts.run_grid384_training --output runs/grid384_20261022 --resume`.
- Final metrics: `summary/RESULTS.csv`, `summary/VERIFICATION.json`, `summary/belief_changes.csv`.
- This document is a plan, not completed experiment evidence. Background status and completion gates determine actual progress.
