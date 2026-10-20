# Real experiment asset inventory — 2026-10-20

Read-only audit. Existing = asset found, not automatic evidence of comparable evaluation. Need Evaluation = assets/protocol still require execution or verification. Missing = not found in the declared search scope. No training requested or performed.

|Asset|Status|Local source|
|---|---|---|
|Frozen formal metrics|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/reports/graphnav_v1_stage1`|
|Formal checkpoint|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt`|
|Formal full episode predictions|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1`|
|Paper experiments and ablations|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013/reports/paper_experiments`|
|OSM-derived CityRefer landmark polygons|Existing|`/home/rental/20260922_1/Workspace/DATA/sbfnav-data/cityrefer`|
|Original CityNav episodes|Existing|`/home/rental/20260922_1/Workspace/DATA/citynav/extracted/data/citynav`|
|Revised CityNav episodes|Existing|`/home/rental/20260922_1/Workspace/DATA/hett/release/refined_citynav/processed_citynav`|
|Real orthophoto and surface-height rasters|Existing|`/home/rental/20260922_1/Workspace/DATA/rgbd`|
|Observed crop renderer|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013/graphnav/renderer.py`|
|Archived dense belief fields|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/reports/graphnav_v1_stage1/heatmaps`|
|Baseline results/provenance|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/Baselines/comparison`|
|GeomBelief full Unseen profiling|Existing|`/home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/profiling-val-unseen-20261016/profiling_val_unseen/full`|
|Official FlightGPT locally executable full evaluation|Need Evaluation|`/home/rental/20260922_1/Workspace/2027-CVPR/Baselines/comparison/flightgpt/preflight.json`|
|Strict common-protocol official HETT rerun|Need Evaluation|`/home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013/reports/paper_experiments/hett_asset_preflight.json`|
|Verified official trained HTNav checkpoint|Missing|`/home/rental/20260922_1/Workspace/2027-CVPR/Baselines/HTNav/official_trained_checkpoint.pt`|
|Perspective onboard camera captures|Missing|`/home/rental/20260922_1/Workspace/DATA/perspective_onboard`|

The detailed pre-existing recursive inventory is indexed at `/home/rental/20260922_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013/reports/paper_experiments/asset_inventory.csv` and `checkpoint_inventory.csv`; current selected source hashes are revalidated by this bundle.

The paper repository is private and was accessed with existing GitHub credentials. Local `2027-CVPR/paper` is an older non-Git draft and is not the requested current manuscript.

Source experiment branch local HEAD and published branch HEAD differ; both commits are recorded. This bundle uses local saved outputs and versioned source on the requested branch, never silently substitutes the main branch.

No entire orthophoto, CityNav dataset, checkpoint or private credential is included in this bundle. Observed RGB is simulator-rendered orthophoto, not a perspective onboard photo.

## Completed extraction index

|Asset|Status|Evidence|
|---|---|---|
|Nine validation cases and full trace provenance|Existing|cases.jsonl; case_selection_counts.json|
|80 actual observed frame indices and source raster SHA|Existing|observation_index.json|
|Real dense fields, 310 probability-change rows|Existing|fields/; replay_manifest.json; belief_change_statistics.csv|
|Four real derived vector PDF/PNG figures|Existing|figures/ and per-figure JSON sidecars|
|Full RGB versions|Existing, local-only|local_only/figures/ (not committed); IMAGE_USE.md|
|Published/local/official-asset baseline layers, 104 rows|Existing|baseline_results.csv; BASELINE_AUDIT.md|
|Fresh MGP dependency and FlightGPT shard checks|Existing|baseline_preflight.json; sources/mgp_native_preflight.log|
|10,448 offline-rescored episodes and 48 subgroup rows|Existing|episode_statistics.csv; subgroup_statistics.csv; offline_metric_verification.json|
|Full-split same-GPU whole-episode and decision profiling|Existing, different model/control identities|profiling_comparison.csv; profiling_scope.json|
|48 verified bibliographic records|Existing|bibliography_audit.csv; sources/bibliography_metadata.json|
|Strict official matched baseline evaluation|Need Evaluation / Blocked|BASELINE_AUDIT.md and baseline_preflight.json|
|Underlying raster public redistribution confirmation|Missing|IMAGE_USE.md|
|Strict visibility labels for Unseen/Test|Missing|visibility_provenance.json; Seen only|

Every distributable artifact is indexed with size and SHA in PRODUCT_MANIFEST.csv. Original datasets, all weights, pre-existing results and local-only RGB remain in place. Public figures can be redrawn without DATA/checkpoints; lab verification uses REPRODUCE.md.
