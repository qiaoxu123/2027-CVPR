# A2 Target-Balanced Training Design

## Intent

Improve A2 candidate localization on new target instances using only the already collected TRAIN data. The current V4 proposal is correct within 20 m on 42–43% of real TRAIN observations but only 21–24% of real DEV observations. The A0 belief is correct on 65.3% of those DEV observations. V4 has 2,025 real rows across 253 target groups and 31,415 teacher rows across 1,908 groups; its fixed real-row weight of 8 still gives real observations about 34% of row-weighted training mass.

## Goal

Run a distinct V5 training experiment that gives each `(source family, map, target)` group equal total weight within its source family while preserving the existing 8:1 real-to-teacher group weight. Normalize weights so their mean matches V4's mean per-row weight. Keep V4 checkpoints, source code, and reports unchanged.

The immediate success signal is higher real-DEV proposal recall than V4's selected checkpoint, measured with the existing unchanged metric and target-disjoint DEV split. This is only a candidate-quality signal; it does not establish an A2 navigation gain. Final success still requires independent gating and the complete paired A0/A2 navigation benchmark.

## Constraints

- Use only existing TRAIN and internal DEV data; do not resume data collection.
- Do not read Val-Seen or Val-Unseen for fitting, checkpoint choice, or calibration.
- Keep A0, WALDO, BERT, SigLIP2, controller, 28×28 grid, model architecture, loss definitions, optimizer, seed, and 20+5 epoch schedule fixed.
- Keep the existing risk gate and admission rules unchanged; never force A2 on.
- Do not alter V4 artifacts or source files. V5 writes to a new revision directory.
- Keep all GPU work serial. Start V5 only after the currently running V4 verification and queued post-checks release the GPU.

## Approach

For every training row, assign a base source weight of 8 to real A0 closed-loop observations and 1 to teacher snapshots. Divide by the row count for that row's source-family/map/target group. The current inventory has 127 V1 real groups, 126 V2 real groups, and 1,908 teacher groups, so the resulting aggregate source masses are 2,024 real and 1,908 teacher before normalization. Apply one shared scale factor so the mean row weight equals V4's mean base weight (47,615 / 33,440). This preserves overall loss scale while making repeated frames from one target unable to dominate its source family.

V5 will wrap the existing immutable V4 feature dataset and change only TRAIN `labels.sample_weight`. DEV rows retain V4 weights and the existing selection metrics. A new training entry point, protocol, source signature, checkpoint directory, and verifier will bind the weighting code and preserve the existing 25-epoch workflow.

## Risks and anti-goals

- Small groups receive larger per-row weights and may contain noisy observations. Report per-target and per-map DEV metrics; do not hide regressions behind pooled results.
- Equal weighting may improve target transfer but reduce teacher-only performance. Existing teacher-preservation gate remains mandatory.
- This change may not fix poor candidate-to-landmark semantics. If the proposal does not improve on real DEV, stop this weighting line and inspect candidate visibility, projection, and language-reference pairing before another training run.
- Do not claim A2 beats A0 from localization-only metrics.

## Verification

1. Unit tests prove row weights sum to the intended normalized total, each target group receives the same aggregate weight within its source family, real-to-teacher group mass remains 8:1, and malformed/non-TRAIN inputs are rejected.
2. V5 preflight confirms the exact source populations and weighting manifest without opening calibration or benchmark data.
3. A serial CUDA smoke pass confirms finite gradients and frozen assets before the full 20+5 run.
4. Independently recompute every V5 DEV checkpoint and verify hashes before selecting a checkpoint.
5. Only an eligible checkpoint may enter independent calibration and the frozen paired navigation evaluation.
