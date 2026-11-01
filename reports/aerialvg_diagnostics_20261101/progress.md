# Diagnostic progress

- Created independent branch directly fromfcfde0b; existing experiment files and weights unchanged.
- Suspended previous formal validation coordinator751047 and child757617 withSIGSTOP to honor Train/internalDev diagnostic scope; outputs/process state preserved, not resumed.
- Frozen-cache scan:1275 Train +635 internalDev states,900 native queries each. Deployed proposal selection reproduced1910/1910. Checkpoint SHA before/after identical.
- First candidate summary run encountered missing optional statistics for an empty proposal set. Fixed safe aggregation and added regression; preserved first failed log locally. Corrected full run completed.
- Navigation analysis consumed only three existing128-episode archives and stored endpoint NE, never source Val-Seen/Test labels or new rollouts. Distance bounds and unknown attribution retained.
- Six new diagnostic tests pass: validity/score strategy, NMS versus budget, oracle empty-set semantics, public endpoint bounds, complete accounting invariants and archived128-only evidence limits.
- Report generated reproducibly. Top-level conclusions: deployed visible-target position coverage is high; ranking and transfer to global belief lose correction opportunities. History reduces jumps but harmed7 episodes, with accumulation mechanism unknown.
