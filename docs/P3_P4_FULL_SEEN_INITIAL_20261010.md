# P3/P4 full Seen controls — initial completed comparisons

This is an intermediate report, not Unseen acceptance. All2470 raw Val Seen episodes, starts and label-independent evaluation manifest are identical; no failed episode is dropped. New GraphNav Val Unseen results have not been measured. Joint current/history branches and the Stop recovery experiment remain in progress.

| Model | SR % | SPL % | OSR % | NE m | nDTW % | sDTW % | Moves | Policy ms | Evaluation s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A3 + guarded learned trajectory/Stop |40.8502|39.6139|48.8259|31.4762|29.9134|14.7727|7.7166|5.9079|162.4|
| + zero-initialized residual GAT |42.0648|40.7956|50.0000|32.2201|29.5695|14.9100|7.8138|6.7308|192.4|
| + current observed landmark visual residual |41.5385|39.3911|51.8623|31.7168|29.7466|14.8676|8.4575|80.7886|1951.0|
| Local frozen SBFNav |43.5223|36.1184|60.4049|31.5172|28.2789|14.7397|15.7652|47.9686|1936.2|

Exact machine-readable values take precedence over rounded table values: `reports/closedloop_full_seen_initial_20261010/{comparison,maps,episodes}.csv` and `summary.json`. A3 field is preserved epoch4, SHA`2008085175d3da5f84fa1fc880d3ef38ba3c3fdd7595bd33a3c856f22d6baf2c`; shared learned head SHA`51d8af7f05a9a718a9dc0ed12721a24ca6dd252e2714cb25948f22cb88cd2dbd`. All Graph controls use25% learned first-waypoint residual, goal-distance taper, threshold0.3 and a gate requiring arrival at the model's own predicted goal. This gate does not read GT and does not change the20m evaluation success definition.

## What is supported

GAT gains132 successes and loses102 relative to A3, net30/2470 (+1.2146pp SR), SPL+1.1816pp. Paired episode bootstrap95% intervals are SR[+0.0405,+2.3887]pp and SPL[+0.0367,+2.3337]pp. However, equal-map-weighted SR delta is+0.6690pp with interval[-1.2057,+2.6361]pp, and NE worsens by0.744m with interval[+0.136,+1.403]m. This is evidence of an aggregate Seen gain, not a uniform map gain or demonstrated Unseen improvement. A single training seed cannot estimate training variability.

Current visual evidence gains47 successes and loses30, net17 (+0.6883pp), while SPL declines0.2228pp. SR interval[0,+1.3765]pp, SPL[-0.9131,+0.4624]pp, NE worsens0.241m. No broad visual benefit is established. Its80.79ms policy latency versus6.73ms GAT warrants retaining a geometry-only option. The visual module matches observed named landmarks; it has no query-conditioned target-region appearance score, which is a concrete remaining hypothesis rather than a proven cause.

Relative to SBFNav, GAT has Seen SR−1.4575pp but SPL+4.6771pp. Static A3 R@8 and P2 cross-frame retrieval are different metrics and do not enter this comparison. The SBFNav reference was a local paper-based implementation whose historical checkpoint selection used a Val Unseen128 proxy; this is disclosed. New GraphNav selection is restricted to training/internal dev and Seen.

## Remaining failures and system limits

GAT fails1411policy Stop episodes (57.13% of all episodes;57.71% of policy stops), whereas SBFNav has no learned policy Stop, so its False-Stop Rate is undefined. A learned Stop at an incorrect predicted goal is still a false stop; the gate does not establish true arrival. Stop recovery therefore receives predicted candidate endpoints and label-independent perturbations, with GT only as a training label. No improvement from that experiment is asserted before real CUDA and Seen validation.

The trajectory head consumes A3 candidate context plus dynamic belief probability/entropy and current pose. Updated GAT node states influence global belief but are not yet directly fused into the trajectory context. Visual history stores positive evidence, public poses and observed grid coverage; pose-indexed history is not an explicit recurrent pose-attention model. Unknown candidates are never marked absent simply for lacking observations.

These are actual closed-loop macro-action rollouts with fresh current-pose observations. The shared simulator provides orthophoto crops and vertical height-derived depth, not a perspective3D rendering or swept collision model. No real-flight safety or complete3D planning claim is justified. Original illegal altitude starts are preserved and corrected through the same bounded vertical moves as SBFNav.

## Reproducibility

Source snapshots, arguments, source/checkpoint/input hashes and per-step diagnostics reside in each local output directory recorded in `summary.json`. The initial slow GAT run interrupted after318 episodes remains intact; optimized rendering has exact pixel/trajectory equivalence checks and distinct output paths. Failed earlier Stop/trajectory controls remain in `reports/trajectory_seed0_20261010/`.

Analysis can be rerun to a new output path with `scripts/summarize_closedloop.py --run A3=runs/closedloop_seen_confirm/a3_guarded_blend --run GAT=runs/closedloop_seen_confirm/dynamic_graph_current_optimized --run Visual=runs/closedloop_seen_confirm/dynamic_visual_current_optimized --run SBFNav=runs/closedloop_baseline/sbf_val_seen_full --output <new_directory>`. Paired bootstrap uses fixed seed20261010 and2000 draws; separate equal-map weighting is reported explicitly.

Three independent legacy-structure A3 scratch training seeds0/1/2 are predeclared in `reports/sbf_protocol/phase14_plan.json`, each5epochs, batch16/BF16, workers4, persistent frozen BERT96, and Seen R@1/NE checkpoint selection. They are backbone replications, not three independently trained complete systems. They queue after the current Seen/Stop controls and do not read Unseen.
