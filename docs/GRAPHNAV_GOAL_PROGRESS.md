# GraphNav goal progress

Branch: `2027-CVPR/graphnav-closedloop-sbf-parity`, based on `11e0409b322f08db4459888c6d38ba08547da730`. Goal remains active: full closed-loop navigation and >=95% SR / >=90% SPL of the local SBFNav baseline on identical Val Unseen evaluation. No acceptance level is achieved yet.

## Unified baseline — in progress

Located genuine local SBFNav epoch-14 best checkpoint, SHA256 `e991b4e46ebc20b9cf7a1e350a0993c75028e863b10b55df166acd4e59510885`, and immutable source/config snapshot. Historical selection used Val Unseen proxy: disclosed, not replicated for new GraphNav tuning. Shared rollout now separates public language/pose/resolved polygons from GT/evaluator labels, uses XY metrics, 20m radius, 20 macro decisions, 50m horizontal and 10m vertical limits, map bounds and ground+10..100m altitude limits. Original SBF has horizon/stagnation termination and no trained Stop. False-Stop Rate is null when no stop decisions exist.

128 evenly spaced Val Seen episodes: SR 44.53%, SPL 36.89%, OSR 61.72%, NE 32.96m, nDTW 27.55%, sDTW 14.68%, mean 15.94 moves, mean policy latency 47.56ms. All 128 trajectories/termination lengths reproduce historical checkpoint outputs exactly. Full Seen/Unseen rerun follows. Closed-loop denominator is 2470/2697 raw episodes; static retrieval denominator 2426/2679 unique map+text queries is intentionally separate.

79 tests pass. SBF tokenizer was incompatible with the prior GraphNav environment (protobuf absent); baseline is run using existing SBFNav environment without modifying either environment. Failed startup output/log retained. Raw depth/collision checks and true-camera intrinsics fidelity remain explicit simulator limitations; no improved safety claims are made.

## P0/P1/P2 — completed first validation, further controlled tests pending

See `BIDIRECTIONAL_VISUAL_LOCAL_VALIDATION_20261009.md`. 54 projection/ROI examples reviewed. Reliable filtered RGB regions and cross-frame pairs exist; source holes/coarse footprints are filtered, not called segmentation GT.

P1 five epochs total train-only 70.02s. G0/G1 matched A3 train-only 226.30/229.32s; G1 total pretrain+fine-tune 299.35s. G0 Unseen R@8=47.14%, NE54.68m; G1 46.40%,56.31m: no demonstrated transfer gain. Historical A3 remains 50.88%,48.42m and is preserved.

P2 five formal epochs 65.62s, peak allocated GPU4.10GiB; Unseen visual retrieval R@1/R@5=69.57/93.91% on115 identities. Frozen Large mean-patch control=66.96/81.74%. Footpath R@1 declines vs frozen baseline (40% vs60%, only10 samples). This is visual matching, not language localization or navigation success. Base/Large control, pose-indexed online cache, distance/visibility-stratified evaluation and aligned native Depth remain pending. No LoRA or Unseen-driven architecture tuning.

## Next experiments

1. Finish unified full SBFNav baseline and source/data/protocol audit.
2. Evaluate preserved A3 with a basic bounded waypoint policy in the same simulator, including all episodes even when landmark binding is unavailable.
3. Select trajectory/Stop training and confidence-gated graph/visual history additions using train/internal dev and Val Seen only, at most3 major configurations per stage.
4. Freeze final candidates before full Val Unseen comparisons, then assess paired episode/map uncertainty and acceptance levels.

## Engineering candidates prepared during full baseline replay

Native RGB-D diagnostic added: 54 cases exactly reproduce the original RGB pixels and use source surface-height-derived vertical depth. Water/bridge source gaps are substantial; this is a benchmark depth proxy, not perspective depth or a 3D occlusion model. Category support/2px sensitivity and image pairs are in `reports/p0_depth_20261009/`.

A3 public-input bounded-waypoint adapter and zero-gated dynamic graph/visual residual candidates pass CPU regression tests. They are untrained integration candidates; no performance gain or navigation completion is claimed. The visual update covers all grid candidates, supports positive landmark history and treats unobserved nodes as unknown. Native RGB/Depth, node association and a learned trajectory policy still require integration/training.

P1/P2 now support atomic Last/Best and explicit resume with optimizer/RNG and input fingerprints. Source or configuration mismatch aborts; existing experiment outputs are protected. P2 epoch-indexed view sampling is worker-independent for fair Base224/Large384 controls. Old P2 results retain their original random-worker sampling and are not overwritten. The new controlled comparison is predeclared in `reports/sbf_protocol/phase1_plan.json` and queued only after full SBFNav replay, preceded by A3 Seen128 smoke. Model choice uses internal train dev; different native encoder resolutions are a disclosed confound. Current regression suite:88 passing tests (before additional queued experiment execution).

Candidate-aligned multimodal Waypoint/Stop head now has label-separated losses and genuine human XY-arclength waypoint interpolation. Tests cover executed-first-step constraints and consistent candidate/mode/Stop selection. It remains untrained and is not integrated into measured policy behavior.

Protocol input audit:1470 Seen and1208 Unseen episodes have different native SBF/GraphNav reference sets, usually multiple genuine same-name road fragments. Both get the same raw annotation assets/reference names; preserve native preprocessing and disclose the difference. The A3 adapter was corrected before its first measured evaluation to retain fragments, with a regression test.1020 Seen/1261 Unseen start poses exceed the old ground+100m altitude cap and remain unchanged. Full episode/schema/raster SHA256 and baseline checkpoint manifests are in `reports/sbf_protocol/`.
