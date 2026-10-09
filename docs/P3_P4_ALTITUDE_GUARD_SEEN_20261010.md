# Corrected runtime: five controlled full Seen evaluations

All rows use the same legacy A3 checkpoint, the same trained Trajectory/Stop head, the same 2470 Revised CityNav Seen episodes and the public-altitude Stop repair. Ground truth is accessed only after rollout for scoring and diagnostics. These are navigation metrics, not static recall. The independent full-system seed0 confirmation is still running; no new GraphNav Unseen result is available at this checkpoint.

| Module | SR (%) | SPL (%) | OSR (%) | NE (m) | Moves | Evaluation seconds |
|---|---:|---:|---:|---:|---:|---:|
| A3 + trajectory | 40.8502 | 39.6139 | 48.8259 | 31.4762 | 7.7202 | 169.42 |
| + GAT | 42.0648 | 40.7956 | 50.0000 | 32.2201 | 7.8158 | 181.37 |
| + Visual | 41.5385 | 39.3911 | 51.8623 | 31.7168 | 8.4587 | 571.75 |
| + GAT + Visual, current | 42.4291 | 40.4542 | 51.7004 | 31.5900 | 8.2737 | 642.31 |
| + GAT + Visual + History | 41.9433 | 40.5579 | 50.3239 | 32.0475 | 7.8551 | 644.74 |

GAT versus A3 adds 1.2146 SR percentage points and 1.1816 SPL points, but worsens NE by 0.744m. Paired episode-bootstrap SR CI is [0.0405, 2.3887] points; equal-map CI is [-1.2057, 2.6361]. There is no established map-uniform or multi-seed GAT benefit. Visual alone adds 0.6883 SR points, with CI [0, 1.3765], and lowers SPL by 0.2228 points. Current joint evidence adds 1.5790 SR points, CI [0.7287, 2.3897], while its equal-map SR CI still crosses zero. History lowers aggregate SR compared with current-only inference. Earlier zero/shuffled-image-vector controls remain relevant: these rows do not prove a causal image-appearance benefit.

All five complete action audits report zero horizontal/vertical step violations, horizon violations, new illegal positions/heights, worsening invalid starts or invalid final heights. The unchanged source has 1020 invalid initial heights and one invalid initial XY per row. Direction reversals are reported separately and are not automatically collisions or oscillation. The environment remains the local orthophoto/vertical-depth macro-action benchmark, not swept 3D flight safety.

These five evaluations were isolated, with existing cross-model visual caches. History processed 80408 current observed ROIs: 589 memory hits, 79819 disk hits and zero new backbone encodes. Warm-cache times must not be presented as cold-start encoding throughput or compared as an algorithmic speedup over the independent full-system runs. Upcoming parallel evaluations label concurrent timing separately.

Artifacts:

- `reports/altitude_guard_seen_20261010`: original metrics JSON and `comparison_first_five.csv`; earlier four-row CSV preserved.
- `reports/altitude_guard_five_paired_20261010`: paired bootstrap JSON, full episode/map CSV and failure counts; bootstrap seed20261010, 2000 draws.
- `reports/altitude_guard_five_constraints_20261010`: action/height audit JSON and per-episode CSV.
- Source and checkpoint hashes are recorded in each original metrics JSON and immutable job source manifest under `runs/source_snapshots/97a6e1c96c44_altitude_guard_full_seen_*`.

Reproduce the paired analysis with `scripts/summarize_closedloop.py`, passing `--run name=runs/closedloop_seen_altitude_guard/name` for the five rows in table order and a fresh `--output`. Reproduce constraints with `scripts/audit_rollout_constraints.py`, the same five runs, preserved SBF source, data root and a fresh output directory. Neither script runs model inference or changes success criteria.
