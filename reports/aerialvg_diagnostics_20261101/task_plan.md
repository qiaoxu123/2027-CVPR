# AerialVG candidate/bottleneck diagnostics

Base: fcfde0bcd9e7f6cc66d0a1f23302ecd4ec9289f6. Diagnostic-only branch; no training, checkpoint writes or protocol changes.

1. [complete] Validate Train/internal-Dev raw900 caches and prepared static beliefs; pin input/checkpoint hashes and split groups.
2. [complete] Compare Native/A0-guided, shared validity/NMS and15/30/60 budgets; attribute losses and compute position-only oracle/correctability bounds.
3. [complete] Frozen ranker/belief diagnostics, corrected/damaged A0 counts and ranking/fusion decomposition.
4. [complete] Read only existing128 paired archived trajectories (baseline/current/history8); diagnose OSR/SR failures with explicit observable bounds and unknown categories, no new validation labels or rollouts.
5. [complete] Meaningful diagnostic unit/invariant tests and concise Chinese report. Publication is verified by the final remote Git ref rather than this pre-commit plan.

Definitions: A0 is the cached frozen static base_logits/grid_world (GraphNav A3 geometry), not an extra trained checkpoint.20m coverage is geospatial proximity, not target-instance matching. Candidate/fusion statistics are separate Train and internalDev. Archived128 navigation is descriptive only and does not tune Val-Seen.
