# Actual closed-loop visual evidence coverage

This analysis reads completed public policy diagnostics, not subsampled human-trajectory P0 frames. Every record corresponds to a decision with an actual observation; an unobserved terminal waypoint is excluded. It does not inspect target coordinates/IDs or add supervision to inference.

| Seen model | Observed decisions | Decisions with trusted regions (%) | Decisions with referenced-node evidence (%) | Episodes with any trusted region (%) | Final observed grid (%) |
|---|---:|---:|---:|---:|---:|
| Full seed1, current | 22257 | 70.02 | 35.78 | 98.18 | 32.32 |
| Full seed2, current | 22332 | 69.22 | 35.33 | 98.14 | 32.04 |
| Legacy joint, current | 22848 | 67.59 | 34.00 | 98.26 | 32.15 |
| Legacy joint, history | 21848 | 95.26 | 48.48 | 98.26 | 32.14 |

Current-only regions are those trusted in the current image. The history row counts accumulated evidence and cannot be read as instantaneous visibility. Referenced-node evidence means that at least one public instruction-referenced landmark node receives an aligned visual vector. It does not mean the target is visible, successfully grounded, or located. Auxiliary-node evidence covers other public named regions. Grid coverage is cumulative geometric camera footprint, not semantic visibility or valid flight-space coverage.

Nearly every episode encounters some usable named-region evidence, but only about35% of current decisions in independent full seeds update a referenced landmark node. Auxiliary regions occur in58–59% of decisions. History retains more evidence but does not improve aggregate Seen SR over the corresponding current-only legacy model. Together with the zero/shuffled-vector control, this weakens the hypothesis that missing RGB alone explains navigation failure. There is still no demonstrated query-conditioned grounding of target appearance; its usefulness must be tested separately.

Reproduce with `scripts/summarize_observed_evidence.py --run name=completed_run --output fresh_directory`. Original telemetry, prediction SHA, episode manifest SHA, exact script SHA, per-episode CSV and cache counters are saved in `reports/onpolicy_evidence_seen_20261010`. The independent full seeds processed65809/63228 actual new ROI encodes; warm-cache joint rows processed2/0. These execution contexts differ and cannot support isolated model-speed claims.
