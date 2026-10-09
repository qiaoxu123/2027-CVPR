# P1 transfer: matched real-trajectory and Stop heads

The three five-epoch heads are complete. Full Seen navigation evaluations are running; this report does not yet claim a navigation gain or select a model from Unseen.

| A3 backbone | Head train seconds | Head development seconds | Selected head epoch |
|---|---:|---:|---:|
| G0 Scratch | 36.3371 | 2.8191 | 5 |
| G1 P1 v2 initialization | 32.8413 | 3.0313 | 5 |
| G1 P1 v3 hard-negative initialization | 33.5927 | 3.2038 | 5 |

The executed fairness audit passes: identical head parameter schemas, five consumed feature schemas/dtypes, optimizer settings, seed0, split seed0, batch256, workers4, five epochs, sample/query SHA, 115559 training states and12950 development states. Each head uses its own A3 checkpoint's immutable features. Best head selection uses the same internal Train query-family development loss, never navigation Unseen. Existing G0 caches additionally contain dense fields for the dynamic module; `FrozenTrajectoryDataset` reads only its five `FEATURE_KEYS` and explicitly discards dense extras, so this metadata difference does not alter head inputs. Selected weights/cache SHA are in `reports/p1_fair_trajectory_training_20261010/fairness_audit.json`.

Trajectory supervision consists of actual remaining human waypoints plus the same recovery/near-stop hard-negative states. Target and future teacher coordinates occur in loss labels only. Runtime uses the same bounded first-waypoint execution, guarded25% learned-path residual and calibrated Stop gate, including the public-altitude repair. Neither G1 row borrows another backbone's head.

P1 pretraining is additional computation and must not be omitted: v2/v3 have67.4038/67.4159s measured pretraining time, before261.7211/269.3189s A3 fine-tuning. Scratch A3 training took265.9938s. Adding the heads gives measured training-only totals302.3309/361.9663/370.3275s for G0/v2/v3. Setup, feature caching and validation are separate; these sums are not end-to-end wall-clock claims. Per-epoch head loss, throughput, memory, configuration and SHA are in `reports/p1_fair_trajectory_training_20261010`. New G1 heads have absolute epoch timestamps; the earlier completed G0 head does not, so its sampled epoch GPU utilization cannot be reconstructed.

The original static transfer experiments are negative: Seen R@1 declines40.8904→39.1179% for v2 and→36.8508% for v3. See `P1_MATCHED_V2_20261010.md` and `P1_MATCHED_V3_HARD_20261010.md`. The full-navigation controls are retained to test whether independently trained trajectory heads change this conclusion, not to erase the static failures. No new Unseen result has been consulted.

Training budget reproduction: `scripts/summarize_training_runs.py --run g0=runs/full_seed0_trajectory --run g1_v2=runs/p1_v2_matched_trajectory --run g1_v3=runs/p1_v3_matched_trajectory --output fresh_directory`. The executed audit source is saved alongside its JSON. Upcoming rollout artifacts are `runs/p1_matched_trajectory_seen/{g0_trajectory,g1_p1_v2_trajectory,g1_p1_v3_trajectory}`; their immutable source and actual parallel windows are recorded by phase22b.
