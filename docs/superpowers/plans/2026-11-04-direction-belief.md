# Minimal direction Belief transform (latest scope)

Supersedes full direct-relation head plan for this round. Freeze PGVB, SigLIP2, GraphNav and Stop. No appearance recognition or navigation.

- [x] Audit explicit world-cardinal TRAIN relations; keep unknown-frame relative words separate. Public anchor and private target labels in different files.
- [x] Minimal learned direction-conditioned spatial kernel applied to Anchor Belief on existing28 grid. Known-heading adapter for synthetic front/behind/left/right; unknown heading must abstain.
- [x] Unit tests for coordinate sign, convolution shift/normalization, finite gradients, unknown frame rejection, no GT input.
- [x] Genuine CUDA smoke, synthetic independent samples for all8 directions, same-budget no-direction control.
- [x] TRAIN Cambridge fit / Birmingham evaluation with physical city+name grouping; synth initialization followed by tiny real-supervised head fitting. No frozen-module updates.
- [x] Seal real evaluation predictions before scoring; report per-word data counts, direction consistency, GT20m mass, R20 and mean/median NE. Do not invent missing direction samples.
- [x] Actual Anchor/Target/GT heatmaps and Chinese report; checkpoint hashes, logs, tmux resume, commit/push.

Synthetic accuracy is a controlled mechanism test, not CityNav performance. Direction-only cannot determine distance or visual instance identity. PGVB optional current-observation anchors remain unknown when no supported current view; no target crop or future frames.

Outcome: synthetic direction control verified; true-target coverage remains7/34, reference-frame/anchor semantics unresolved. No further Dev tuning or navigation.
