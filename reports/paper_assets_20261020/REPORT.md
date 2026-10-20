# GeomBelief real paper assets — 2026-10-20

## Completed

- Inspected the requested current manuscript (`22c4fca...`) and published experiment branch `research/visual-residual-20261013` (`2e7a067...`), as well as the local archive, formal GraphNav source, shared CityNav/Revised data, real raster renderer, map annotations, weights, traces and profiling evidence. The older non-Git `2027-CVPR/paper` is not the current manuscript and was not modified.
- Exported nine real validation episode--model cases (eight distinct episodes), 80 observed-frame instances (77 distinct map/pose views) and 310 step/control/comparison statistics. Frozen replay reproduces archived Top-16 coordinates, selected target and Stop probability within the declared tolerances. Checkpoint and original prediction/dataset/code SHA are retained per case. Actual goals, final positions and observed footprints are not invented.
- Produced four vector PDF figures plus PNG previews and JSON sidecars: real four-stage overview; three-condition geometry contrast; correction/reinforcement/degradation with matched-path Real/Zero/Shuffle/No-residual; two navigation successes and four failures. Source arrays and plotting scripts are included. The calibrated visual head and three geometry-ablation heads retain separate identities from the formal main model.
- Audited 110 baseline evidence rows. The 60 revised-table and 48 original-table numeric cells (108 total) in the current benchmark tables match source tables/formal records at manuscript precision. Published aggregates, released-weight readiness, independent local implementations, substituted API models and historical document-only claims are separate layers. No newly verified strict official-weight common-protocol result was manufactured.
- Offline-rescored all 10,448 formal episodes (2470/2697/5281); per-episode and aggregate NE/SR/OSR/SPL match original records. Exported 48 grouped rows for start distance, initial localization error, overlapping lexical relation types and saved Seen visibility labels. No full navigation evaluation was repeated.
- Verified metadata for all 48 BibTeX entries against publisher/DOI/CVF/ACL/PMLR/arXiv records. Updated full names, venues and available DOI fields, preserving original source snapshot. OpenFly's latest version has a changed subtitle and author order; conference-year versus preprint/deposit-year distinctions are recorded.
- Organized same-GPU-model profiling evidence with separate units, components and first-rollout warmup scope. Both models have compatible whole-episode module-span reports; GeomBelief additionally has per-decision timing. The HETT record is a historical local teacherfix checkpoint. No official controlled speedup is claimed.

## Newly extracted real measurements

These are additional analyses of existing data and frozen replay, not new training or a new best model.

|Illustration|Measured result|Interpretation limit|
|---|---|---|
|Seed-0 geometry query `birmingham_block_5:8:5:0`|Name/type error 358.29m; shuffled 22.29m; correct geometry 12.02m|Separately matched trained heads; outcome-selected example, not main-weight intervention|
|Visual correction `birmingham_block_5:8:5`|Top-1 error 22.22→11.95m; GT20 probability 0.237→0.353|Shuffle retains the same Top-1 in this case|
|Visual reinforcement `birmingham_block_5:8:4`|GT20 probability 0.298→0.476; Top-1 remains 11.95m|Changed mass is real despite unchanged Top-1; not evidence that Real beats Shuffle here|
|Visual degradation `birmingham_block_5:33:0`|Top-1 error 14.86→32.05m; GT20 probability 0.229→0.210|Visible negative example retained|
|Formal main-model overview|B0→Bt L1 0.02620; successful NE 11.95m|Visual correction is small; no visual-gain claim|
|Seen initial weak-projection visibility|332 strict-visible (13.44%), 1860 outside (75.30%), 278 uncertain (11.26%); SR 51.51%, 48.23%, 38.85%|Descriptive fixed groups, not causal or human-verified visual grounding|

The formal main checkpoint remains SHA `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`; SR is unchanged at 47.6113% / 22.5436% / 37.1521%. All 27 protected weights/report files are checked before and after work. No new training or Unseen/Test model selection occurred.

## Blocked / missing

- **Public RGB distribution:** CityNav's annotation license is known, but the underlying SensatUrban raster redistribution terms are not conclusively established by a software MIT license. Full real-RGB figures stay in ignored `local_only/figures/`; uploaded manuscript figures show actual map fields/trajectories/observation footprints. Exact local crop indices, source SHA and scripts are provided. This partially blocks replacing the public artwork with RGB-rich scene panels; it does not block real-data analysis.
- **Strict official baseline reruns:** MGP recovered official weight exists but its native readiness probe fails on missing `groundingdino`, its safe fixed-ID/common-metric wrapper remains unvalidated, and the historical local run's missing weight path cannot be retroactively tied to that SHA; HETT lacks a fully validated clean native checkpoint-compatible inference/export environment; HTNav has no verified official trained checkpoint; FlightGPT has incomplete local weight shards and unvalidated full runtime/exporter; GeoNav's local DeepSeek Flash is not the original GPT-4o. Existing local outputs are retained with limitations rather than retrained or relabeled.
- **Source accessibility:** AAAI's HETT publisher PDF was blocked by the current proxy; numeric values were independently verified against the archived author arXiv Table 2. GeoNav original table numbers retain secondary transcription provenance. A DOI metadata match does not by itself verify a numeric experiment table.
- **Matched official timing comparison:** hardware model and whole-episode timing scope can be matched, but HETT checkpoint identity, native controls/step counts and run conditions differ; HETT map geometry is not separately timed. A strictly controlled official HETT/GeomBelief latency ratio remains unavailable. Actual whole-episode rollout means are 396.68ms (GeomBelief) versus 200.82ms (historical local HETT); these do not support claiming a faster complete system.
- **Unseen/Test strict visibility:** only Seen has complete saved weak-projection visibility labels; no labels are guessed for other splits and no large rendering sweep is started.

No extra training is requested: the identified gaps require source/runtime/rights clarification or targeted inference, not speculative retraining.

## Research conclusions

Correctly supplied map geometry can exclude wrong candidates in real model outputs, while the existing matched studies still do not establish a universal full-polygon advantage over centroid/distance representations. Visual residuals can alter normalized probability mass and Top-1 locations, including harmful changes. Individual successes, nearly identical colors, or an unchanged Top-1 are insufficient to establish that the visual branch is absent or semantically effective. Matched-path Shuffle examples caution against attributing every correction to object semantics.

Near/far task bins and weak-projection visibility describe different factors; success is not monotonically determined by initial distance. The grouped analysis is observational and does not establish a cause. Initial R@1 here is episode-weighted, distinct from the paper's deduplicated-query recall.

## Deliverables and validation

`ASSET_MANIFEST.md`, `BASELINE_AUDIT.md`, `baseline_results.csv`, `cases.jsonl`, `belief_change_statistics.csv`, `fields/`, `figures/`, `scripts/`, `subgroup_statistics.csv`, `episode_statistics.csv`, `profiling_comparison.csv`, `bibliography_audit.csv`, `IMAGE_USE.md`, `REPRODUCE.md`. `verification.json` records source preservation, 310 probability-statistic recomputations, nine parity-checked cases, 108 table cells, 10,448 episode metrics and all four figure/source pairs.

The manuscript receives `figures/citynav/` with lightweight standalone source arrays/redraw scripts, `notes/visual_evidence_manifest.md`, `notes/baseline_verification.md` and bibliography audit notes. Old conceptual artwork is preserved with `_conceptual.tex` names and removed from the live figure inputs. Compilation evidence and the final experiment commit are recorded in `latex_build.json` and manuscript notes. The working draft compiles to 13 pages, references beginning on page 10; submission page-budget polishing is still outstanding. Existing compiler package/font warnings are recorded, with no undefined citations/references or overfull boxes.
