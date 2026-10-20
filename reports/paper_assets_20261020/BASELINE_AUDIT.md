# Baseline verification and protocol comparability

Audit date: 2026-10-20. NE is in meters; SR/OSR/SPL are percentages. Empty cells mean not measured/verified; they never receive copied paper values.

## Evidence layers

`published_table`: author/publication aggregates; `official_released_asset`: checkpoint readiness with no score; `local_independent_reproduction`: independent/repaired implementations; `local_substituted_model`: changed API model; `historical_document_only`: no verified episode-level source. MGP historical official-weight claims retain unverified checkpoint linkage. GeomBelief is frozen GraphNav-V1 C epoch1 under the manuscript name.

## Current manuscript table

All 60 revised-table numeric cells match the independently parsed source/archived formal values at manuscript precision. MGP*, FlightGPT*, HTNav* come from [HTNav Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Fan_HTNav_A_Hybrid_Navigation_Framework_with_Tiered_Structure_for_Urban_CVPR_2026_paper.pdf). HETT* matches the [authors Table 2](https://arxiv.org/html/2512.14222); AAAI PDF access failed through the current proxy, so publisher-byte validation is blocked. Original CityNav rows are separately audited; GeoNav original numbers retain the existing secondary-table provenance rather than claiming its paywalled original table was fetched.

## Method readiness and restrictions

|Method|Existing asset / remaining restriction|
|---|---|
|MGP|Recovered official checkpoint; not proven identical to absent historical checkpoint|
|HETT|Official best_val_unseen; clean native complete inference/export not validated|
|FlightGPT|Locked revision 9eb62538741af0e4239cb7394cb9cf9b6dc6c703; incomplete local shards|
|HTNav|No verified official trained checkpoint|
|GeoNav|Original uses API GPT-4o, not a downloadable local checkpoint; local substituted API run separate|

## Exact data and evaluator scope

Local Original counts are 2498/2826/5311; Revised counts 2470/2697/5281. Full file SHA and removed IDs are indexed in the preserved dataset_version_audit.json. Counts alone do not establish author dataset byte identity. Retained episode language/trajectory/target are unchanged in that audit; revised reference annotations differ separately.

Common offline metrics: NE = final XY Euclidean distance; SR = NE <=20m; OSR = min over actually visited XY <=20m; SPL = SR × straight-line start-to-GT distance / max(actual XY path length, straight-line distance). This harmonizes metric definitions only. Input privileges, geometry, controller, step/action budget, stopping and checkpoint-selection protocols remain different. HTNav native teacher-path SPL and GeoNav native 3D evaluator branches must not be conflated with common XY metrics.

There is no newly completed strict official-weight matched evaluation in this bundle. MGP historical refined three-split outputs already exist and are retained; recovered weight identity cannot be retroactively asserted. HETT requires a clean checkpoint-compatible runtime and isolated ID-preserving exporter; FlightGPT lacks complete shards/runtime; HTNav lacks official trained weights; original GeoNav requires the original API protocol. No training is necessary for asset organization and none was started.

No Val-Unseen/Test-Unseen checkpoint selection occurred. Existing release and historical selection provenance is retained, including HETT best_val_unseen. No cross-protocol ranking, significance or speedup claim is made.

Machine-readable evidence: baseline_results.csv, baseline_table_checks.json, sources/upstream_probe.json. Every local source is rehashed; missing weights and predictions remain explicit.

## Fresh native readiness probe

The 2026-10-20 MGP native `main_goal_predictor.py --help` fails in the declared reproduction environment with `ModuleNotFoundError: No module named 'groundingdino'`; exact source, command, exit code and log SHA are in `baseline_preflight.json`. The native entrypoint also loops all three splits and applies terminal goal selection, so it is not an already-validated isolated fixed-ID common-protocol evaluator. This dependency/protocol gap is additional to the missing historical-weight linkage, not a claim that MGP has no public model. FlightGPT's actual locked snapshot has tokenizer/config metadata and zero complete weight shards. No native training or Test inference was launched.
