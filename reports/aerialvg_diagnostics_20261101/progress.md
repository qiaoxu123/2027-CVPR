# Diagnostic progress

- Created independent branch directly fromfcfde0b; existing experiment files and weights unchanged.
- Suspended previous formal validation coordinator751047 and child757617 withSIGSTOP to honor Train/internalDev diagnostic scope; outputs/process state preserved, not resumed.
- Frozen-cache scan:1275 Train +635 internalDev states,900 native queries each. Deployed proposal selection reproduced1910/1910. Checkpoint SHA before/after identical.
- First candidate summary run encountered missing optional statistics for an empty proposal set. Fixed safe aggregation and added regression; preserved first failed log locally. Corrected full run completed.
- Navigation analysis consumed only three existing128-episode archives and stored endpoint NE, never source Val-Seen/Test labels or new rollouts. Distance bounds and unknown attribution retained.
- Six new diagnostic tests pass: validity/score strategy, NMS versus budget, oracle empty-set semantics, public endpoint bounds, complete accounting invariants and archived128-only evidence limits.
- Report generated reproducibly. Top-level conclusions: deployed visible-target position coverage is high; ranking and transfer to global belief lose correction opportunities. History reduces jumps but harmed7 episodes, with accumulation mechanism unknown.

Read-only follow-up: matched frozen actual outputs reproduced exactly on1910 states. Dev37 correct-ranked failures: top1-only/same-mass repairs7; any positive-query single-point/same-mass13; same/query mass1 repairs15; GT-anchor/mass1 repairs16. Among21 not repaired by either full-mass Oracle,12 are impossible even under independent signed-grid residual relaxation,9 remain unexcluded and mechanistically unknown. No exhaustive-mixture bound or deployable-model claim. Existing validation remains SIGSTOP; no weights changed.

Further diagnostic-only follow-up: necessary posterior-mixture LP over frozen kernel family. Train42 residual cases:3 actual-update witnesses,38 numerical infeasible relaxations,1 unverified/unknown; Dev9:1 witness,8 numerical infeasible. Only witnesses verified by actual frozen update count as Oracle repairs; no claim of trained-model gain or formal infeasibility proof. Same checkpoint hash before/after, no new validation labels or rollouts; existing formal validation remains SIGSTOP.
