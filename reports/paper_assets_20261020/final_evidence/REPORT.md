# GeomBelief P0 final evidence report

2026-10-20. Source snapshot: experiment `16452896e8b169a9c540ce6de7f7d00b17b453a7`; paper `6e7450c7580a61e1158e7cfc5eb5ad66eea05905`. Worktree: `/home/ubuntu5/Workspace/hett-experiments/31-geombelief-p0`. This extension reuses existing real-data assets, archived tensors and navigation summaries. No training, checkpoint modification, new full-dataset evaluation, formal-metric rewrite or regeneration of the original four figure groups was performed.

## Completion and remaining gaps

|Request|Result|Evidence / limitation|
|---|---|---|
|Three cases with start/key/terminal RGB|28 exact model-input crops recovered; nine panels in three English PDF/PNG figures|`RGB_CASE_MANIFEST.md`, `RGB_FRAME_INDEX.json`, local-only figures. These are actual orthophoto-based observations, **not perspective FPV**.|
|Same-step RGB/pose/belief/path alignment|All 28 complete observation NPZ hashes match the archived hashes|Exact pose/step/tensor/footprint checks; no adjacent-frame substitution. Whole source-file bytes differ; no claim of entire-raster identity.|
|Formal and independent CVR mechanism|Completed source/formula/shape/control audit|`MODEL_MECHANISM_AUDIT.md`, `CODE_PROVENANCE.json`; relevant source hashes match archived inference.|
|Quantitative visual corrections and harm|310 original comparisons verified; 164 per-frame/control rows exported|`VISUAL_CAUSAL_CASES.csv`, `CAUSAL_SUMMARY.json`; nine selected model-case instances, eight unique episodes.|
|Real wins over both Zero/Shuffle|88 existing independent-rollout episodes indexed|`REAL_WINS_ZERO_AND_SHUFFLE.csv`; per-step evidence for these extra episodes is not in the selected bundle.|
|Visibility|Projection/support proxy only|Strict optical/semantic visibility remains unknown; no physical FOV/depth-occlusion percentages.|
|Baseline recoverability|Readiness check completed; matched official inference not recovered|`BASELINE_READINESS.json`; no forced reproduction or replacement of official scores.|
|Paper equations/modules/training|Required corrections and already-correct statements listed|`PAPER_METHOD_CORRECTIONS.md`; paper source left unchanged.|
|Existing figure provenance|Four original PDF/PNG groups and paper copies hash-checked|`FIGURE_PROVENANCE_AUDIT.json`; formal, independent CVR and separately trained geometry heads remain distinct.|
|RGB publication clearance|Unresolved; RGB withheld from Git|Primary source checks in `LICENSE_SOURCE_AUDIT.json`; details and links in the RGB manifest.|

The missing original model files at archived `/home/rental/...` locations prevent a fresh neural replay or export of additional uncached gate/encoder tensors on this host. The saved exact observations, existing per-step fields and inference hashes support this audit; they do not substitute for a new checkpoint execution claim. No weights were downloaded or loaded.

## Measured examples

All errors below are horizontal meters. Dense Top-1 is argmax over the saved 784-cell logits; it is not automatically the controller's tie-broken proposal order. Dense-goal errors and executed endpoint NE are listed separately; small coordinate-precision differences are retained. L1 and entropy compare normalized fixed static and terminal Real beliefs on the **Real executed pose prefix**; residual magnitude is the maximum absolute applied terminal logit residual.

|Case / model|Static → final dense Top-1 error|Terminal L1|Max abs residual (logits)|Entropy change (nats)|Independent navigation outcome|
|---|---:|---:|---:|---:|---|
|`birmingham_block_5:8:4`, formal success|11.9533 → 11.9533|0.026204|0.070598|−0.017852|Real succeeds, NE 11.9533; no formal control rollout claimed|
|`birmingham_block_5:8:5`, CVR correction|22.2182 → 11.9533|0.368031|1.459616|−0.436596|Real succeeds; Zero/No-belief fail at NE 22.2182; Shuffle also succeeds|
|`birmingham_block_5:33:0`, CVR degradation|14.8614 → 32.0546|0.366737|1.555966|−0.369647|Real fails, NE 32.0551; Zero/No-belief succeed, NE 14.8621; Shuffle also fails|

The correction first changes dense Top-1 at step3. The degradation first changes it at step7 (29.13m), worsening to32.05m from step8. Both reduce entropy; concentration is not necessarily correctness. GT-20m probability mass rises 0.236793→0.352579 for the correction and falls 0.228730→0.209658 for the degradation. The formal overview's Top-1 never changes, so it must not be used to claim a causal visual gain. The original “reinforced” CVR case has the same initial/final Top-1 but does change Top-1 at intermediate steps; it should not be described as an unchanged ranking throughout the episode.

`INITIAL_ERROR_ANALYSIS.csv` releases initial-error/final-effect pairs for all nine selected model-case instances. The six formal cases retain dense Top-1; one CVR case improves, one worsens, and the reinforced case has zero terminal error improvement. This small outcome-selected set cannot estimate a population correlation between initial error and correction effectiveness. The existing paired navigation rows do not contain full initial static-error fields for all episodes; we do not fabricate that relationship.

## Existing independent navigation summaries

The 15,501 paired rows cover 2,470 Seen and 2,697 Unseen episodes, each with three comparisons, for **the independent CVR head**. These are archived independent rollouts, not the same-path intervention tensors used in the panel figures. Source SHA-256 is `8ca278bbab60577ba9ca70c1c8da21cfa5cc1fbe20ab8543578eb6c3bcb68d16`. All paired branches' SR/SPL/NE aggregates, episode counts and model hashes were checked against their saved metrics. This task counts rows without changing formal results or evaluation protocols.

|Split|Real succeeds, Zero and Shuffle both fail|Real fails, No-belief succeeds|Real succeeds, No-belief fails|
|---|---:|---:|---:|
|Seen|40|87|120|
|Unseen|48|61|110|

All 88 joint-win Episode IDs and three endpoint errors are released. For example, Unseen `birmingham_block_5:73:4` has Real NE15.9693, Zero29.9808, Shuffle24.1025. It is stronger outcome evidence than the chosen correction case, where Shuffle succeeds, but its per-step RGB/gates are not available in this selected evidence bundle. It must not be illustrated with another episode's images. `PAIRED_NAVIGATION_COUNTS.csv` contains all pairwise success/failure counts. Counts describe input contribution under a fixed separately trained head; they are not proof of semantic recognition or robust cross-map generalization.

## Mechanisms verified and semantic limits

The actual chain is pose-based crop → frozen whole-frame SigLIP2 patch tokens → homography-selected candidate/scale pooling → quality-weighted historical features → formal matcher or independent CVR → fixed static logits plus residual → normalized goal belief → recomputed trajectory/Stop. A real per-candidate mask exists. It combines current/past ROI support; CVR adds a nonzero-feature condition. It is not true target visibility. The belief is not recursively accumulated from the previous posterior, although feature memory is causal and recurrent.

CVR additionally receives geometric context, prior scores, ROI qualities/support and UAV distance. Zero changes its meaningful-feature gate; Shuffle changes candidate-image association while preserving pose/quality. Therefore the existing interventions do not separate visual semantics from view-dependent geometry, image texture, location association and support effects. **Independent visual-semantic understanding remains INCONCLUSIVE.** Real input can change ranking and both help and harm navigation; that narrower conclusion is supported.

The manuscript should specify orthophoto acquisition, operational masks, quality-weighted feature memory, the two distinct residual formulas, confidence gating without verified calibration, actual staged training losses and exact controller inputs. Its existing separation of experimental models, fixed-prior formulation, strict Stop guard and matched-path caption limitations are already consistent. The correction list distinguishes omissions from actual wording problems.

## Figures, baseline and licensing

The only new public plot is [the temporal numerical evidence figure](figures/p0_temporal_visual_evidence.pdf), with matching PNG/JSON and redraw script. It contains no scene pixels. Error units are meters, L1 is unitless, entropy is in nats. Controls in its middle/right columns are matched-path belief interventions, not their independent trajectories. The left column is the formal model; the other columns are independent CVR. No original figure was redrawn. The four original captions correctly label formal navigation, independent CVR and three separately trained geometry localizers; their units/colors/data/model identities are recorded in the figure audit.

The historical HETT preflight identifies a 675,603,246-byte official asset (SHA `bb59e46afad00d64f0c69a938037f8b9090e158c51a1b0814de483c0095e277a`) on a different host, but its checkpoint-compatible upstream runtime and isolated matched evaluation were not established. That path is absent here. A local `HETT/weights/best.pt` is a different hash and cannot be promoted to official navigation weights by filename. The [official HETT](https://github.com/crotonyl/HETT) and [official HTNav](https://github.com/tyrofanfan/HTNav) README/release metadata were checked; the HTNav README still indicates incomplete code upload and no verified official HTNav checkpoint was recovered. This is an asset/protocol gap, not evidence that no official weight exists anywhere. Published baseline numbers remain published references.

CityNav's CC BY notice does not resolve the underlying SensatUrban data terms and download-form nonredistribution conditions. See [RGB_CASE_MANIFEST.md](RGB_CASE_MANIFEST.md) for primary sources and the unresolved right to publicly submit derivative screenshots. RGB and RGB-containing figures remain ignored local-only files. No unauthorized RGB, raw rasters, model weights or keys are included in this commit.

## Reproduce and validate

Run from the experiment worktree using the existing interpreter `/home/ubuntu5/miniconda3/envs/AirVLN/bin/python`. The AGENTS-specified `/home/tenant2/miniconda3/envs/AirVLN39/bin/python` is unavailable on this host; this task used CPU-only archive analysis/rendering, not model training or precision-sensitive fresh neural inference. No environment modification was made.

```sh
PY=/home/ubuntu5/miniconda3/envs/AirVLN/bin/python
$PY reports/paper_assets_20261020/final_evidence/scripts/align_rgb.py
$PY reports/paper_assets_20261020/final_evidence/scripts/analyze_evidence.py
$PY reports/paper_assets_20261020/final_evidence/scripts/audit_sources.py
$PY reports/paper_assets_20261020/final_evidence/scripts/plot_evidence.py
$PY reports/paper_assets_20261020/final_evidence/scripts/validate_evidence.py
```

`align_rgb.py` accepts existing-data `--rgbd` / published-map `--ground-table` paths; it fails if any archived crop cannot be reproduced exactly. `audit_sources.py` requests public license/README metadata only. The scripts do not retrain or reevaluate navigation. Public-only reviewers can validate without private observations using `validate_evidence.py --without-local --paper-root <paper-checkout>`; this explicitly omits private RGB checks.

The five relevant existing test files passed **23 tests in0.97s**. `VALIDATION.json` records crop/tensor/pose checks, source matches, preexisting-asset immutability, original/paper figure hashes and RGB Git exclusion. `PROTECTED_INPUTS.json` pins inputs before this extension. A delivery inventory records hashes of public extension files. Those checks establish provenance and internal consistency, not newly measured semantic capability.

P0 mechanism and archived quantitative evidence are complete. Strict perspective FPV, physically verified visibility, screenshot publication clearance, fresh neural replay with the missing original weights and semantic-only causal attribution remain unverified. They should stay explicit limitations in the paper.
