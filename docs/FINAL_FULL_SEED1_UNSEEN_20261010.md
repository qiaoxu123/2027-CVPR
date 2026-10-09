# Frozen primary Full seed1: complete Unseen evaluation

The Seen-selected primary candidate completes all2697 Val Unseen episodes, with SR22.1728% and SPL21.2317%. Selection and all checkpoint hashes were frozen and pushed in `fdd9a3d17cf648f7a748c5b146ba84fe6498ff2b` before any new GraphNav Unseen evaluation. This is the primary candidate, not a post-Unseen replacement chosen from the highest row.

| Model | SR (%) | SPL (%) | OSR (%) | NE (m) | nDTW (%) | sDTW (%) | Moves | False-Stop (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local SBFNav | 21.5054 | 15.8907 | 38.7468 | 47.5653 | 23.5093 | 7.5936 | 16.6507 | N/A: no policy Stop |
| GraphNav Full seed1 | 22.1728 | 21.2317 | 32.6288 | 51.5006 | 25.7345 | 8.2637 | 8.4564 | 77.8030 |

The primary point estimate exceeds both local SBFNav SR and SPL and the Level2 numerical thresholds. This does not establish full-system acceptance or a reliable SR improvement. Paired episode bootstrap(2000 draws,seed20261010) gives SR difference+0.6674percentage points,95% CI[-1.1874,+2.6706];354 gained and336 lost successes. SPL difference+5.3410points has CI[+3.6849,+7.0715]. OSR falls6.1179points,CI[-8.1572,-3.9674], and NE worsens3.9353m,CI[2.3330,5.5530]. Secondary frozen A3+trajectory has higher SR than Full seed1, but cannot replace the preselected primary after inspecting Unseen.

Complete Seen metrics for the same primary are SR42.6721%,SPL40.8949%,OSR51.7004%,NE30.7278m, versus SBF43.5223%,36.1184%,60.4049%,31.5172m. Seen SR is slightly lower and OSR is materially lower. Stop/safety and architectural requirements must be considered alongside the Unseen point estimates; other full-model seeds and the remaining fixed visual/history controls are still running.

## Actual compute and visual evidence

Unseen evaluation took2019.6585s(33.66min), mean synchronized policy latency73.9900ms, peak allocated GPU4.0200GiB. The evaluation-worker limit was2 and another evaluation overlapped; this is not an isolated speed comparison with SBF. The run made25447 actual observations,79389 trusted ROI requests and79367 actual frozen-backbone encodes, with19 memory hits and no disk hits. This cold-cache run must not be compared as a model-speed regression against warm Seen runs without noting the cache difference.

Actual decision coverage:72.7473% have trusted landmark regions,41.7456% update instruction-referenced nodes,61.1428% have auxiliary-node evidence,96.2551% of episodes have some trusted region. Mean final cumulative grid-FOV coverage is28.7579%. These are geometric/proposal evidence statistics, not target visibility, landmark-recognition accuracy or semantic free-space coverage. Only the actually observed decision prefix is counted.

## Failure and constraint diagnostics

There are2099 failures:1817 never have an observed pose within the unchanged20m target radius,263 have a legal-altitude observed arrival and then fail, and19 only have an illegal-altitude observed arrival.2054 failed policy Stops account for77.8030% of policy Stops, not77.8030% of all episodes. A ground-truth-near visit is a private oracle diagnostic opportunity; the policy does not receive the target coordinate. This decomposition suggests wrong goal belief is the dominant failure count; Stop/goal-gate failures also remain.

All2697 episode traces and22807 movements pass public50m XY/10m Z/20-decision/bounds/nonworsening-invalid-start checks, with legal final altitude and zero newly illegal states. All1261 originally invalid-altitude starts are preserved.1620 direction reversals are measured, without treating every reversal as unsafe oscillation. These checks do not prove swept3D collision freedom or physical flight safety; the shared evaluator is the local orthophoto/vertical-depth macroaction proxy.

Artifacts:

- `reports/final_full_seed1_unseen_20261010`: original metrics/episode manifest, frozen weight audit, comparison CSV, per-map/per-episode CSV, paired uncertainty and all failures.
- `reports/final_full_seed1_constraints_unseen_20261010`: full public action audit.
- `reports/final_full_seed1_evidence_unseen_20261010`: public observed-evidence coverage and cache telemetry.
- `reports/final_full_seed1_stop_unseen_20261010`: private post-rollout Stop diagnosis; no inference changes.
- Original predictions: `runs/final_closedloop_unseen/full_seed1/predictions.jsonl`.

Reproduction uses the unchanged `final_unseen_full_seed1` command in `reports/sbf_protocol/final_frozen_evaluation_plan.json`. All four actual checkpoint SHA values match that frozen plan. No new training, model selection, hyperparameter change or goal/Stop correction uses these Unseen results.
