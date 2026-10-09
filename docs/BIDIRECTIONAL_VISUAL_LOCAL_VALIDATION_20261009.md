# Bidirectional visual local validation — 2026-10-09

Base: `11e0409b322f08db4459888c6d38ba08547da730`. Development continues on `2027-CVPR/graphnav-closedloop-sbf-parity`. Previous worktrees, input annotations and checkpoints are preserved. Runs reside in `/home/rental/20260922_1/Workspace/2027-CVPR-bidirectional/runs/`; final counterfactuals are in the new worktree's `runs/p1_counterfactuals/`.

## Data and P0

Reused the audited Revised CityNav files and prior Python environment; annotation SHA256 and prior artifact checks appear in `reports/bidirectional_local_20261009/reused_data_audit.json` and `original_results_unchanged.json`. Source split counts are 21878 / 2470 / 2697. Static query evaluation deduplicates map+instruction, giving 21166 / 2426 / 2679. Closed-loop episodes MUST use the full raw splits; these denominators describe different experiments.

Rendered actual CityFlight pose observations directly from the source rasters at 384px, with sampled episodes per map, stride 8 and maximum source step 64. This is orthographic RGB simulation, not a photorealistic perspective renderer. No 224px image was enlarged to claim native 384px detail. Train/Seen/Unseen produced 18395 / 6242 / 3256 valid frames. The observed frames include intermediate poses; their teacher paths are a training/data diagnostic only and cannot serve as policy-generated evaluation observations.

54 independently inspected projection/ROI examples are committed under `reports/bidirectional_local_20261009/p0_cases/`. No systematic global mirror/rotation/translation was apparent. 38 cases showed no obvious registration error; 16 were flagged for clipping, occlusion, sparse source pixels, coarse boundaries or invalid polygons. This is qualitative registration review, not segmentation accuracy or pixel IoU. Polygon projections are weak geometric regions, never semantic masks.

The quality filter requires valid simple polygons, visibility >=90%, projected area >=256 pixels, minimum bbox side 8 pixels and actual nonblack source-pixel support >=90%. Context is retained within rectangular letterboxed ROIs. Duplicate camera footprints are removed for cross-view pairing. Final retained ROIs: 17493 Train, 6170 Seen, 2619 Unseen. Paired landmark identities: 931 / 794 / 115. No road centerlines were invented: TrafficRoad/Footpath are genuine annotated surface polygons. See coverage JSON for map/category counts and rejection reasons. Native observations currently export RGB; aligned depth generation remains a closed-loop integration task.

## P1 and matched A3 controls

Invalid source polygons and source-family augmentation leakage were found and fixed. All augmentations and both directions remain in one family partition. Final geometry-verified pairs: 9076 Train and 306 Unseen; 8168 train pairs and 908 internal dev pairs. Same-class alternative locations remain ambiguous in 8.32% of base training relations and 12.09% of Unseen relations. Only cardinal inverses are generated; near/between/along are not mechanically inverted. Signed candidate-to-contour geometry and paired displacement consistency are used.

P1 trained 5 epochs, 32 pairs/batch (64 directional samples), hidden 256, 8 heads, relative positional inputs, BERT token limit 128, seed 0, BF16. Synthetic Unseen directional accuracy is 100%; R@1/4/8 = 89.38/99.51/100%, NE 12.15m. These are synthetic geometry metrics, NOT real CityNav navigation metrics. Peak allocated GPU memory 3.30 GiB. See JSON for each epoch loss/time/throughput. Inverse error in the saved training history is normalized SmoothL1; subsequent sample-weighting correction changes only reporting, not model weights.

G0 and G1 both use hidden 256, heads 8, relative geometry, batch 64, workers 4, BERT max length 512, seed 0 and 5 epochs. Only initialization differs; G1 additionally pays P1 pretraining compute. Checkpoints were selected on Val Seen. Real static localization, percentages at 20m:

| Model | Split | Queries | R@1 | R@4 | R@8 | NE m |
|---|---|---:|---:|---:|---:|---:|
| G0 Scratch | val_seen | 2426 | 38.05 | 60.84 | 72.34 | 33.48 |
| G0 Scratch | val_unseen | 2679 | 20.38 | 36.28 | 47.14 | 54.68 |
| G1 P1 Init | val_seen | 2426 | 38.58 | 60.47 | 71.23 | 32.93 |
| G1 P1 Init | val_unseen | 2679 | 17.99 | 34.57 | 46.40 | 56.31 |

G1 Unseen R@8 declines 0.75 percentage points and NE worsens 1.63m. The paired query bootstrap interval for R@8 difference spans zero; there is no supported gain. Retain G0 and the historical strongest A3 checkpoint. Historical A3 50.88% / 48.42m used different width/batch/token/coordinate settings and is not this matched control.

Counterfactuals use 263 controlled directional rows and 812 translated real queries. P1 predictions flip direction correctly in 100% of controlled swaps. Fine-tuned models lose much of that simple directional response; see `p1_counterfactuals.json`. Translating all map geometry by one grid cell or changing world origin leaves aligned logits nearly invariant (approximately 1e-5–1e-4 numerical MAE), arguing against a direct absolute-world-coordinate shortcut. This does not establish absence of all position priors. Multi-solution instructions and synthetic-to-real language mismatch remain plausible transfer failures.

## P2 visual landmark matching

True frozen `google/siglip2-large-patch16-384`, revision `1b426889ea62b5a72bf9839009a1b184bfc9c178`, was tested: actual patch shape [4,576,1024], finite head gradients, no frozen backbone gradients. Official weight SHA256 was verified; no weights are committed. Model download used the existing local proxy and ranged recovery after long-transfer stalls. Details and smoke evidence are in JSON.

Frozen patches use persistent disk/RAM caches keyed by model revision, precision, preprocessing and source file metadata. Wholly black source/letterbox patches are excluded, without treating polygon projections as segmentation. Warm vision microbatch 16 reaches about 84–87 images/s versus microbatch 8 about 77 images/s. Training uses 32 landmark pairs/batch, microbatch 16, workers 4, BF16, hidden 256/8 heads. Same-city same-category distinct instances are hard negatives. 931 eligible identities split into 837 training and 94 internal dev identities, with 836 usable pairs/epoch after dropping a singleton negative group. Internal dev is not CityNav Val Seen.

Best checkpoint: epoch 5 selected on internal dev. Five formal training epochs total 65.62 seconds; first formal epoch reuses cache generated by a preserved aborted metric-format trial. Subsequent epochs encode new sampled views. Epoch train throughput ranges 48–154 pairs/s; peak allocated memory 4.10 GiB. Cold encoding/setup and the aborted trial are excluded from this formal train-only total and retained in logs. DataLoader pinning converted tuple metadata to lists; normalized lookup keys and a regression test fixed the resulting evaluator error. Training restarted at seed 0 with a fresh head; only frozen image features were reused.

Unseen cross-frame gallery has 115 identities on 4 maps:

| Visual representation | R@1 % | R@5 % |
|---|---:|---:|
| Frozen Large mean patch | 66.96 | 81.74 |
| Seed-0 initial matching head | 62.61 | 82.61 |
| Trained matching head | 69.57 | 93.91 |

Compared with frozen mean patches, R@1 gains only 3 identities (+2.61pp), while R@5 gains 14 (+12.17pp). Same-city/type hard-gallery R@1/R@5 is 73.68/97.37% on 114 queries. Building R@1=84.62% (39), TrafficRoad=62.07% (58), Footpath=40% (10), Ground=85.71% (7), Water=100% (1). The water sample is insufficient for category generalization. No qualified Unseen bridge/rail/high-vegetation matching set exists after filtering.

This visual matching evaluation is distinct from language target localization. No A3 localization gain, SR or SPL is demonstrated by P2. Thin roads/footpaths, broad surface regions, context/scale variation and source holes remain weaknesses. Base-vs-Large and distance/visibility-stratified controls are pending under the expanded system goal; LoRA is not justified yet. A confidence-gated, zero-residual dynamic belief integration is worth a controlled Seen/internal-dev experiment, with A3 preserved. Neither P1 nor GAT should become the default without measured gains.

## Fixes, artifacts and limitations

Fixes cover valid surface geometry, inverse-family split isolation, strict same-city/type negatives, camera-footprint deduplication, valid observed patch masks, persistent frozen encoding caches, tuple/list metric keys, incremental Last/Best checkpoints and weighted reporting. Prior invalid-polygon P1/G1 runs and the aborted P2 metric trial remain under `runs/invalid_polygon_attempts/` and `runs/p2_aborted_metric_attempt/` respectively, with original logs retained. Original verification artifacts were not overwritten.

79 unit/regression tests pass including new shared evaluator tests. Training and retrieval CSV/JSON, counterfactuals, model download verification, smoke gradients, GPU timeline and manual review are stored in the report directory. Large ROI data, patch caches and model checkpoints remain local under the documented run paths. The baseline implementation still needs exact resume/provenance enforcement throughout all trainers and complete validation-time telemetry; these are tracked as remaining system work rather than claimed complete.

No closed-loop SR/SPL has been produced for GraphNav. Full navigation, Depth integration, dynamic history, Trajectory/Stop training and SBFNav parity are separate remaining milestones.
