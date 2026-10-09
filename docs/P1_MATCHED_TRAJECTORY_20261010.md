# P1 transfer: matched real-trajectory and Stop heads

The three five-epoch heads and full Seen/Unseen navigation evaluations are complete. Both P1 variants harm Seen navigation transfer; Unseen results are mixed, as reported below. Scratch is retained under the original Seen selection rule; no selection uses new Unseen outcomes.

| A3 backbone | Head train seconds | Head development seconds | Selected head epoch |
|---|---:|---:|---:|
| G0 Scratch | 36.3371 | 2.8191 | 5 |
| G1 P1 v2 initialization | 32.8413 | 3.0313 | 5 |
| G1 P1 v3 hard-negative initialization | 33.5927 | 3.2038 | 5 |

The executed fairness audit passes: identical head parameter schemas, five consumed feature schemas/dtypes, optimizer settings, seed0, split seed0, batch256, workers4, five epochs, sample/query SHA, 115559 training states and12950 development states. Each head uses its own A3 checkpoint's immutable features. Best head selection uses the same internal Train query-family development loss, never navigation Unseen. Existing G0 caches additionally contain dense fields for the dynamic module; `FrozenTrajectoryDataset` reads only its five `FEATURE_KEYS` and explicitly discards dense extras, so this metadata difference does not alter head inputs. Selected weights/cache SHA are in `reports/p1_fair_trajectory_training_20261010/fairness_audit.json`.

Trajectory supervision consists of actual remaining human waypoints plus the same recovery/near-stop hard-negative states. Target and future teacher coordinates occur in loss labels only. Runtime uses the same bounded first-waypoint execution, guarded25% learned-path residual and calibrated Stop gate, including the public-altitude repair. Neither G1 row borrows another backbone's head.

P1 pretraining is additional computation and must not be omitted: v2/v3 have67.4038/67.4159s measured pretraining time, before261.7211/269.3189s A3 fine-tuning. Scratch A3 training took265.9938s. Adding the heads gives measured training-only totals302.3309/361.9663/370.3275s for G0/v2/v3. Setup, feature caching and validation are separate; these sums are not end-to-end wall-clock claims. Per-epoch head loss, throughput, memory, configuration and SHA are in `reports/p1_fair_trajectory_training_20261010`. New G1 heads have absolute epoch timestamps; the earlier completed G0 head does not, so its sampled epoch GPU utilization cannot be reconstructed.

The original static transfer experiments are negative on Seen: R@1 declines40.8904→39.1179% for v2 and→36.8508% for v3. See `P1_MATCHED_V2_20261010.md` and `P1_MATCHED_V3_HARD_20261010.md`. The full-navigation controls test whether independently trained trajectory heads change this conclusion, without erasing the static failures. The final candidate list was committed before these new Unseen results; no new Unseen result was consulted during model selection.

Training budget reproduction: `scripts/summarize_training_runs.py --run g0=runs/full_seed0_trajectory --run g1_v2=runs/p1_v2_matched_trajectory --run g1_v3=runs/p1_v3_matched_trajectory --output fresh_directory`. The executed audit source is saved alongside its JSON. Completed rollout artifacts are `runs/p1_matched_trajectory_seen/{g0_trajectory,g1_p1_v2_trajectory,g1_p1_v3_trajectory}`; their immutable source and actual parallel windows are recorded by phase22b.

## Full Seen2470 navigation results

| Backbone + own matched head | SR (%) | SPL (%) | OSR (%) | NE (m) | Moves | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| G0 Scratch | 41.3765 | 40.2144 | 49.1498 | 31.3057 | 7.5623 | 173.12 |
| G1 P1 v2 | 38.2591 | 37.1665 | 47.0445 | 31.5842 | 7.9785 | 183.37 |
| G1 P1 v3 hard negatives | 36.6397 | 35.5842 | 46.4372 | 32.6009 | 9.8283 | 198.34 |

Versus Scratch, v2 SR falls3.1174percentage points (paired episode-bootstrap95% CI[-4.7783,-1.4575]); SPL falls3.0480points (CI[-4.6987,-1.4219]). V3 SR falls4.7368points (CI[-6.5587,-2.9150]), SPL falls4.6303points (CI[-6.3651,-2.8808]), and NE worsens1.2951m (CI[0.3503,2.1807]). These are failed transfer experiments despite extra pretraining computation, not candidates for promotion. Relation-only pretraining appears poorly matched to real target localization; this is a diagnosis hypothesis, not a proven causal explanation.

All three complete action audits pass with zero step/horizon/new-illegal-state/worsening-start violations and legal final heights; the original invalid starts remain. CSV, per-map/episode outcomes, failure counts and2000-draw paired bootstrap(seed20261010) are in `reports/p1_fair_trajectory_seen_20261010`. Constraints are in `reports/p1_fair_trajectory_constraints_seen_20261010`. Reported times are measured under a two-worker evaluation limit; first two runs overlap, third partly overlaps the second, so they are not isolated speed benchmarks. Subsequent Unseen rows are frozen reporting controls; their results cannot be used to change the chosen Scratch backbone.

## Frozen full Unseen2697 navigation results

| Backbone + own matched head | SR (%) | SPL (%) | OSR (%) | NE (m) | Moves |
|---|---:|---:|---:|---:|---:|
| G0 Scratch | 20.8750 | 20.2933 | 29.7367 | 52.5435 | 7.7697 |
| G1 P1 v2 | 20.0593 | 19.4983 | 29.2918 | 50.5713 | 8.1539 |
| G1 P1 v3 hard negatives | 21.8391 | 21.2544 | 31.4423 | 47.8985 | 9.6871 |

The transfer finding is mixed. Versus Scratch, v2 SR changes-0.8157percentage points, paired episode-bootstrap95% CI[-2.1505,+0.4820]; SPL-0.7950points,CI[-2.0678,+0.4888]; NE improves1.9722m,CI for delta[-3.2110,-0.6273]. V3 SR changes+0.9640points,CI[-0.5191,+2.4101]; SPL+0.9611points,CI[-0.4834,+2.3902]; NE improves4.6450m,CI[-5.9786,-3.1910]. Neither SR/SPL interval excludes zero. It would be incorrect to describe both variants as uniformly harmful on Unseen; it would also be incorrect to promote v3 after inspecting Unseen despite its worse Seen results. Scratch remains the originally selected backbone. Newly frozen static Seen/Unseen inference is still queued separately.

All three2697-episode action audits pass with zero step/horizon/new-illegal-state/worsening-start violations and legal final heights, preserving1261 source-invalid starts. Exact metrics, original episode manifests, paired map/episode CSV and all failures are saved in `reports/p1_fair_trajectory_unseen_20261010`; constraints are in `reports/p1_fair_trajectory_constraints_unseen_20261010`. This is a single training-seed transfer control, not three-seed evidence of a P1 benefit.
