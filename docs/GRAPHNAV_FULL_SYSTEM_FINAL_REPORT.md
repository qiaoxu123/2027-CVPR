# GraphNav full system report — work in progress

This report is not a completion claim. The expanded closed-loop goal is active; Level1/2/3 acceptance is not yet met. Unified SBFNav full Seen/Unseen and initial GraphNav full Seen controls are complete; new GraphNav Unseen results remain unmeasured. Static recall and visual retrieval are never substituted for SR/SPL. The selected final architecture and acceptance table will be populated after the predeclared controls finish.

## Implemented architecture and boundaries

```mermaid
flowchart LR
    L[Full frozen BERT96 tokens + provided named map polygons] --> A[A3 per-candidate relative geometry: 784 global cells]
    A --> B0[Static goal logits + dense candidate context]
    O[Current actual native384 RGB + vertical-depth proxy + public pose] --> V[Frozen SigLIP2 Large patch tokens + trained landmark attention]
    V --> N[Name and full-contour node association + confidence]
    N --> H[Causal positive landmark history + pose and observed-area state]
    H --> G[Zero-initialized gated graph and visual residuals]
    B0 --> G
    G --> BT[Global dynamic goal belief]
    BT --> T[Candidate-consistent multimodal trajectory selection]
    B0 --> T
    P[Current pose] --> T
    T --> W[Bounded first waypoint: 50m XY / 10m Z]
    W --> O
    BT --> S[Learned Stop + predicted-goal arrival gate]
```

Evaluator/private labels contain target coordinates and expert trajectories; public policy inputs do not. Teacher future is used only in trajectory training labels and post-rollout diagnostics. The current head receives static A3 dense context with updated belief probability/entropy; updated GAT latent features are not directly fused into it. History records poses but is not recurrent pose attention. These are remaining architecture limitations.

The simulator is the local SBFNav macro-action benchmark with observed orthophoto regions and source height-derived depth. It does not support faithful perspective3D occlusion or swept-path collision physics. Original episode starts and20m XY success rules remain unchanged.

## Completed results and effective changes

Local SBFNav full Seen2470:SR43.5223%,SPL36.1184%; full Unseen2697:SR21.5054%,SPL15.8907%. All5167 paths exactly reproduce the historical trained-source/checkpoint replay. Level2 thresholds are SR20.4301%,SPL14.3016%, subject to pipeline/Seen/Stop/safety checks. See `SHARED_CLOSEDLOOP_PROTOCOL_20261009.md` and `reports/sbf_protocol/baseline_table.csv`.

P0/native projection:54 visual examples and actual384 RGB/depth coordinate checks exist; confidence filters reject holes/clipped/invalid polygons. Projected ROIs are weak geometric proposals, not segmentation GT. P1 matched G0/G1 pretraining shows no transfer improvement; retained strongest historical A3. P2 controlled Large384 versus Base224 gives Unseen landmark retrieval73.04/93.91% versus72.17/92.17% R@1/R@5, only one net identity gain with wide paired uncertainty. This is landmark retrieval, not target localization or navigation.

Same learned trajectory/Stop full Seen: A3 SR40.8502/SPL39.6139%; GAT42.0648/40.7956%; current visual41.5385/39.3911%. GAT has a modest aggregate Seen SR/SPL gain but worse NE and no established map-uniform gain. Visual has no broad benefit. Joint-current full Seen SR42.4291/SPL40.4542%; history41.9433/40.5579%; history has no additional aggregate gain. Candidate-recovery Stop preserves GAT SR42.0648/SPL40.7956% while reducing movements7.8138→7.3607. Top4 exploration adds only2 successes with lower SPL. Fresh public-altitude Stop confirmations for all five module rows now reproduce these displayed SR/SPL values. Their paired/map bootstrap and complete action audits are in `P3_P4_ALTITUDE_GUARD_SEEN_20261010.md`; all five have zero new illegal states and legal final heights. The independent full seed0 confirmation is still running. Earlier Stop-recovery controls and original results remain in `P3_P4_FULL_SEEN_INITIAL_20261010.md`.

Verified engineering improvements include exact-output native rendering/static-GAT caching (Seen12869.12s→14.14s), atomic checkpoint/resume and immutable source/input hashes, causal pixel-keyed feature reuse, node-to-contour binding, no-observation invariance and consistent candidate/path/Stop selection. Earlier harmful rank/trajectory/Stop experiments remain recorded rather than erased. Current Graph regression suite:144passing tests; external SBF test compatibility/config-glob failures are separately disclosed.

## Remaining acceptance work

Three independent A3 scratch seeds and their five-epoch downstream dynamic/trajectory/Stop training are complete. Full Seen Seed1 SR42.6721/SPL40.8949/NE30.7278m and Seed2 SR41.5789/SPL39.2060/NE31.3440m use the altitude-corrected Runtime; Seed0 SR41.7004/SPL40.3309 is preliminary old Runtime and awaits its corrected confirmation. All three measured epoch budgets/configuration/Checkpoint SHA are in `reports/full_system_training_20261010`; old trajectory epochs lack absolute timestamps and their GPU utilization cannot be reconstructed. Current full Graph regression suite144passed.

Matched P1v2 lowers Seen R@1 from40.8904 to39.1179%. The final hard-negative P1v3 lowers it further to36.8508%, with R@8 75.0618→73.2894% and NE31.5274→32.6913m. Seen128 bounded navigation SR41.4063→36.7188%,SPL40.6522→36.1198%. These negative results and the additional67.42s pretraining budget are preserved in `P1_MATCHED_V3_HARD_20261010.md`; neither pretraining variant is promoted. Fair learned-trajectory heads and full navigation controls remain queued.

Actual/zero/shuffled ROI vectors give identical Seen128 SR40.625%, so image-vector benefits remain unproven. Complete the public-altitude Stop confirmations, freeze final candidates and Commit their SHA before any new GraphNav Unseen evaluation. Produce paired/map/seed results and final architecture/weights/config commands. Query-conditioned target appearance in observed regions remains a valuable untested hypothesis: named-landmark retrieval does not directly ground color/roof/shape descriptions to unnamed target houses. No improvement from that proposed mechanism is claimed.

Current input/processing audit: `GRAPHNAV_SBF_INPUT_PROCESSING_AUDIT_20261010.md`. Native SBF spatial RGB features interact directly with its SigLIP language tokens and candidate crops; current Graph landmark retrieval supervision does not directly supervise query-specific target appearance. Public pose+map projection already supplies region identity, explaining why good landmark retrieval need not add navigation information. This is a tested mechanism limitation plus an untested improvement hypothesis, not a new performance claim.

Action audit found4 Graph episodes stopping before source-invalid height recovered. Runtime now requires public ground+10..100m before Stop; original source starts and XY20m success are unchanged. All old results remain available. Pointwise source-DEM clearance proxy found204 SBF versus6 A3/8 joint Seen episodes below a known sampled surface; unknown pixels remain unknown. This does not measure swept collision or establish physical flight safety. Reproducible scripts/reports are committed; final acceptance awaits all frozen final evaluations and checks.


Actual A3 altitude-Stop repair check: all4 previously invalid final-altitude episodes now end inside public ground+10..100m; full2470 Seen has0invalid final heights. Individual SR outcomes are preserved, while the four cases take bounded extra recovery moves. Full A3 SR40.8502/SPL39.6139/NE31.4762m is unchanged at displayed precision. The paired original/current manifest, public heights and post-rollout NE are in `reports/sbf_protocol/altitude_stop_guard_actual_a3_seen.json`; no private target coordinates were used for the correction. Other module confirmations continue before final freezing.


Execution scheduling: following user authorization, final independent evaluations will use at most2GPU processes; training remains serialized. Unique atomic checkpoint/vector temporary files and call-private ROI PNG directories fix demonstrated concurrent-cache races. All144tests pass. The already-running phase22 queue remains untouched; the empty waiting phase22b scheduler was handed over with preserved status/PID evidence. Concurrent latency and device-wide GPU telemetry are labeled and are not isolated speedup evidence against historical SBF timing. See `PARALLEL_EVALUATION_20261010.md`.

Actual on-policy Seen evidence coverage: Full seeds1/2 have trusted regions on70.02%/69.22% of decisions, referenced-node updates on35.78%/35.33%, and any trusted region in98.18%/98.14% of episodes. History raises retained-evidence coverage but not corresponding aggregate SR. These public telemetry statistics are distinct from human-prefix exposure and target visibility; see `ONPOLICY_VISUAL_EVIDENCE_SEEN_20261010.md`.
