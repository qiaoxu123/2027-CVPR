# Target + features + landmark + relations: execution plan

Goal: train the existing Unified V2 Geometry/Relation model on all 21,878 TRAIN queries while preserving target identity and the full original constraint chain.
Architecture: no network changes. Preserve original clause order and all content tokens; remove only a strictly recognized initial navigation imperative. This is a conservative content-preserving text control, not LLM semantic reconstruction or a new appearance head. Public Polygon binding uses the original text, exactly as the preceding PR23 control.

- [x] Test content preservation, scope, negation, ordinals, public names and no-GT inputs.
- [x] Prepare complete paired data from audited public-bound full_text; verify original provenance, labels, counts and hashes. Rebuild all text-conditioned model inputs.
- [x] CUDA Smoke, then seed42 fixed five epochs using original initialization, batch8, original learning rates/loss. Freeze PGVB/BERT/controller as in prior comparison; no fabricated RGB training.
- [x] Evaluate all Birmingham TRAIN (resubstitution, not held-out); reuse preceding full-text predictions only if exact public binding and source match.
- [x] User updated scope: freeze checkpoint and protocol, run only128 evenly spaced Val-Seen paired tasks against cached full-text/compact baselines. Do not launch2470 full evaluation or Test.
- [x] Save report/curves/heatmaps/paired outcomes; tests; commit/push independent branch without main merge.

Review focus: unknown parsing preserves text; same-name Polygon binding; no target GT in inference; language changes invalidate BERT features; old experiments and services remain untouched.

Ruling: all21878 TRAIN strings already omit navigation imperatives, so this is complete original-text training under the corrected public binding, not a four-slot semantic rewrite. Explicitly reported; no invented parsing claims.

Latest user steering: after the128 Pilot, run the complete2470 Val-Seen using exactly the same frozen checkpoint and protocol; no Test, no further fitting or selection. Preserve Pilot artifacts separately.
