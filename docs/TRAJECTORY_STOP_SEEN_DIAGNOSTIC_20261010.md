# Trajectory and Stop: controlled Seen diagnosis

Frozen strongest A3 from the preserved validation checkpoint.128 identical quick Seen episodes; benchmark success remains final XY distance≤20m with the original movement limits and horizon. Private labels enter only scoring/training Loss.

The learned head trains5 epochs on115559 Train/12950 query-family-disjoint internal-dev states using genuine human remaining paths plus explicitly augmented recovery states. Batch256,BF16,workers4,seed0. Training40.57s,validation4.06s,peak head allocation54MiB. Checkpoint selection is internal dev total Loss only. Source/cache/config fingerprints,Last/Best and RNG/optimizer resume are recorded. Expert altitude outside the inherited range is masked in vertical Loss without rewriting source z.

| Policy | SR | SPL | OSR | NE(m) | Mean movements | False-Stop Rate |
|---|---:|---:|---:|---:|---:|---:|
| SBFNav |44.53%|36.89%|61.72%|32.96|15.94|N/A(no policy Stop)|
| A3 bounded Top1 |38.28%|37.61%|44.53%|29.76|9.63|N/A(no policy Stop)|
| Joint learned head,calibrated threshold0.3 |18.75%|15.95%|46.09%|43.81|11.35|80.00%|
| Fixed A3 goal + learned path/Stop |30.47%|27.16%|46.09%|32.75|10.01|68.38%|
| Fixed A3 goal + learned path,no Stop |33.59%|23.79%|60.94%|34.23|19.97|N/A(no policy Stop)|
| Fixed A3 direct waypoint + learned Stop |38.28%|37.75%|44.53%|29.72|6.72|53.52%|

Joint head loss falls while real closed-loop SR drops substantially. Raw weighted-BCE sigmoid produced83.59–89.06% failed Stop rates and was corrected analytically by subtracting log(20) from logits. Corrected threshold0.3 wins the predeclared Seen SR/SPL criterion among0.3/0.5/0.7, but fails to restore baseline. The threshold0.7 control reaches OSR67.97% yet SR18.75%, showing loss of previously reached target neighborhoods.

Fixing the static goal recovers11.72pp SR over the joint head, isolating unstable candidate/mode selection as a substantial failure. Learned paths remain worse than direct waypoint execution even with Stop disabled. Human-prefix supervision plus short lateral recovery lacks sufficient wrong-goal/off-policy coverage; repeated re-ranking by current proximity can reinforce the current wrong destination. This explanation is a diagnosis supported by controls, not a proven unique cause. Future training should explicitly separate stable goal ranking from current-pose control and include policy-generated recovery supervision, while protecting the A3 prior with a residual gate.

The selected direct-waypoint/learned-Stop control matches A3 quick SR and slightly improves SPL while reducing movements by30.2%. It is a useful bounded control, not evidence that the learned trajectory decoder adds value. Its failed Stop rate remains53.52% and is reported explicitly. It will receive full Seen validation before any final Unseen run. Do not present it as a validated visual/dynamic-graph navigator.

Full Seen A3 bounded Top1:SR40.89%,SPL39.77%,OSR47.69%,NE31.47m;104.72s. Relative to full SBFNav:SR−2.63pp,SPL+3.65pp. Full raw benchmark denominator2470 differs from the earlier deduplicated static-localization denominator2426; static R@1/R@8 must not substitute for these closed-loop results.

Artifacts:`reports/trajectory_seed0_20261010/`,local checkpoints`runs/a3_trajectory_stop_seed0/{best,last}.pt`,all successful and failed rollout JSONL under`runs/closedloop_baseline/`. No GraphNav Unseen result or Level1/2/3 completion is claimed at this stage.
