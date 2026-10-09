# Frozen A3 + trajectory: complete Unseen control

This is the first completed frozen GraphNav Unseen control, not the Seen-selected full-system primary. All2697 episodes are retained with unchanged starts, goals, horizon20, bounded50m XY/10m Z actions and XY20m success. The exact episode manifest matches local SBFNav. Its checkpoints/configuration were committed in `fdd9a3d` before any new GraphNav Unseen rollout.

| Model | SR (%) | SPL (%) | OSR (%) | NE (m) | nDTW (%) | sDTW (%) | Moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| Local SBFNav | 21.5054 | 15.8907 | 38.7468 | 47.5653 | 23.5093 | 7.5936 | 16.6507 |
| Legacy A3 + learned trajectory/Stop | 23.3222 | 22.6725 | 32.3693 | 48.7174 | 26.2260 | 8.8368 | 7.7875 |

The point estimates exceed local SBF SR by1.8168percentage points and SPL by6.7818points. Paired episode-bootstrap95% CI is[-0.0751,3.7449] SR points and[5.1225,8.4569] SPL points; SR improvement is not established under this episode-weighted uncertainty. Equal-map estimates/intervals are separate: SR+1.9556points[0.7614,2.7727], SPL+6.6426points[4.8771,8.4082]. There are356 gained versus307 lost episodes, not universal improvement. Do not use this row to replace the Seen-preselected primary after inspecting Unseen.

OSR falls6.3775points and NE worsens1.1521m. There are2068 failures, including2063 failed policy Stops and244 episodes that visit the success region but finish outside it. False-Stop Rate is76.6345%; SBF has no policy Stop and therefore has null False-Stop Rate, not zero. Most nonadjacent XY revisit counts include legitimate vertical recovery and must not be labeled control oscillations without additional inspection.

All2697 episodes and21003 motions pass action/horizon/new-illegal-state checks, with zero illegal final heights.1261 original invalid initial heights are preserved and recovered without changing the benchmark. These checks do not establish terrain collision freedom, swept3D feasibility or physical UAV deployment.

Runtime uses the same frozen A3 language/full-contour geometry path, a multimodal trajectory/Stop head and guarded25% learned waypoint residual. It is a geometry control; it does not consume visual residual/GAT/history evidence. The executed evaluation took203.12s, measured policy latency7.0981ms under a two-worker evaluation limit while Full seed1 ran concurrently. Neither value is an isolated speed comparison against historical SBF.

Artifacts:

- `runs/final_closedloop_unseen/a3_trajectory`: original predictions, episode manifest and full metrics/config/input/checkpoint SHA.
- `reports/final_a3_trajectory_unseen_20261010`: comparison/map/episode CSV, paired bootstrap and failures; seed20261010, 2000 draws.
- `reports/final_a3_trajectory_constraints_unseen_20261010`: public-action audit and per-episode CSV.
- `reports/sbf_protocol/final_frozen_evaluation_plan.json`: fixed primary, all remaining candidates, exact commands and checkpoint hashes.

Level1/2/3 final acceptance remains pending for the full primary, module comparisons, three seeds and all required audits. These point estimates are not a complete-goal claim. Static R@8 and cross-frame visual retrieval are not substituted for closed-loop SR/SPL.
