# A2 execution ledger

Scope: only the requested A2 seed-0 model, frozen archived A0 including original visual residual and controller. A0/A1 results are immutable. All GPU work is serial. A1 full A3/B3/C5 and Seen2470/Unseen2697 passed `phase1_completion_gate.json` in the A1 checkout; publication e070786d9dbd3f94446e950336814a310fce6bc4.

## Required sequence
1. Audit every CityNav train query, group by (map,target instance), deterministic seed-0 80/10/10 train/dev/calibration partition. No validation benchmark inputs for fitting. Exact public source-linked references; preserve homonyms as ambiguous. Complete token span required within original BERT96.
2. Integrate opt-in frozen-A0 forward hook before unchanged controller. Current native384 RGB only, frozen WALDO and SigLIP2 actual box crops, every box x every linked reference. Remove prior three-peak/32-box filter; record full detection/association funnel. Frozen-source hashes and train-only separate private supervision.
3. CPU tests, CUDA real-observation smoke and alpha-zero A0 parity. Audit world transforms, supervised associations, hard negatives and visual examples. Full train population and fixed legal causal teacher observation times; do not treat unknown detections as negative labels.
4. Cache features resumably, with immutable signatures. Train 20 ranking epochs then 5 local refinement epochs. Freeze all original modules. Select on train-dev only. Save every epoch/log/TensorBoard curve. Fit conservative gate on separate calibration targets, with gate disabled unless supported by independent risk/benefit evidence.
5. Freeze checkpoint and calibration. Internal diagnostic controls and paired functional pilot; no extra trained ablation arms. Complete A0 and A2 Seen2470/Unseen2697 using identical episodes/controller even if gate must remain off (explicit user requirement takes precedence over prototype document's conditional full-evaluation language). No Unseen tuning or reselection.
6. Per-map/paired SR, OSR, SPL, NE, nDTW, sDTW, False-Stop and Recall; repair/break/fallback, candidate coverage, latency and visual cases. Final A0/A1 and A0/A2 report, curves, provenance. Push code and small reports to A2 branch; leave weights/cache server-side.

## Current authoritative state
- A1 done; A2 source audit CPU process launched 2026-10-28.
- New protocol/mentions regressions: 14 passing. Dataset/detector audit and CUDA integration pending; no A2 full training claimed.
- ETA 2–4 days is preliminary scheduling only, pending measured extraction throughput.
