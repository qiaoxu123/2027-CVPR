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

Same learned trajectory/Stop full Seen: A3 SR40.8502/SPL39.6139%; GAT42.0648/40.7956%; current visual41.5385/39.3911%. GAT has a modest aggregate Seen SR/SPL gain but worse NE and no established map-uniform gain. Visual has no broad benefit. Joint/history and candidate-recovery Stop controls are still running. Detailed CSV, paired uncertainty and failure analysis are in `P3_P4_FULL_SEEN_INITIAL_20261010.md`.

Verified engineering improvements include exact-output native rendering/static-GAT caching (Seen12869.12s→14.14s), atomic checkpoint/resume and immutable source/input hashes, causal pixel-keyed feature reuse, node-to-contour binding, no-observation invariance and consistent candidate/path/Stop selection. Earlier harmful rank/trajectory/Stop experiments remain recorded rather than erased. Current Graph regression suite:119passing tests; external SBF test compatibility/config-glob failures are separately disclosed.

## Remaining acceptance work

Complete Seen joint/history and Stop controls, assess three independent A3 scratch seeds, then freeze final candidates before any new GraphNav Unseen evaluation. Produce full paired/map/seed results and final architecture/weights/config commands. Query-conditioned target appearance in observed regions remains a valuable untested hypothesis: the current named-landmark visual matching module does not directly ground color/roof/shape descriptions to unnamed target houses. No improvement from that proposed mechanism is claimed.
