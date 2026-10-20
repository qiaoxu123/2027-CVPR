# RGB case manifest and usage status

2026-10-20. Three real, already selected CityNav episodes were recovered with **28 exact archived model-input observations**, including nine start/key/terminal panels. These are real-data orthophoto crops used by the policies. **Perspective first-person onboard RGB is unavailable in this implementation.** The images must not be captioned as perspective FPV or as newly captured UAV camera imagery.

## Cases and files

All three cases are `val_unseen`, map `birmingham_block_5`, dataset `processed_citynav`. Episode IDs are the original three-part IDs; `step` is zero-based and indexes the observation before that decision. A terminal decision image is the observation at the final recorded pose, not a neighboring frame or a synthesized post-Stop view.

|Purpose|Case / Episode|Start, key, terminal steps|Policy|Local PDF/PNG basename|
|---|---|---|---|---|
|Overview success|`formal_val_unseen_success` / `[birmingham_block_5,8,4]`|0, 4, 7|Formal C-epoch1|`local_only/figures/p0_formal_val_unseen_success_rgb_alignment`|
|Visual correction|`visual_corrected` / `[birmingham_block_5,8,5]`|0, 3, 7|Independent CVR on frozen base|`local_only/figures/p0_visual_corrected_rgb_alignment`|
|Visual degradation|`visual_degraded` / `[birmingham_block_5,33,0]`|0, 7, 11|Independent CVR on frozen base|`local_only/figures/p0_visual_degraded_rgb_alignment`|

The base checkpoint SHA-256 is `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`; the CVR head SHA-256 is `5fc1a3e115b71d14fe1e8fbaf3ed29c1863d8b2bc080455d89f7a4e503d25f11`. Formal success and CVR correction are different models and different instructions, even though they share a nearby target region. The overview is representative of a successful navigation, not proof that vision caused success.

`RGB_FRAME_INDEX.json` records every recovered frame's original instruction, episode, map, exact XYZ/yaw pose, target XYZ, dense predicted goal, actual selected controller goal, four-corner footprint, render parameters, model hashes, source raster path and aligned tensor index. Pitch and roll are `null`: they are not recorded physical camera measurements. It also contains source-raster metadata/hashes and local file paths. Numerical per-frame belief and applied residual are in the original `../fields/<case>.npz` at that index (`real_belief`, `real_residual`); `VISUAL_CAUSAL_CASES.csv` adds normalized-belief differences, entropy and goal errors. Original `../cases.jsonl` supplies the actual executed trajectory, not an expert route. GT is annotation for analysis/plotting only.

Each three-by-three English figure contains the exact input RGB, same-step goal belief, applied residual, executed path prefix and observation footprint. Annotations distinguish the full archived path from the prefix available at that step. The local figure JSON records frame NPZ hashes and model identities. All nine selected RGB PNGs, plus the remaining 19 verified observations, are in `local_only/observations/`; there are no RGB pixels in the public `figures/` directory.

## Exact alignment and renderer

`scripts/align_rgb.py` checks equality of the frame pose in the case trajectory, saved NPZ pose array, field-frame JSON and original observation index, and checks the diagnostic `observation_step`. It renders precisely that pose with the original renderer. The complete compressed observation NPZ (BGR, world footprint, depth-valid support) must have **the same SHA-256 as the archived crop**; any mismatch stops the export. All 28 passed. No nearest-frame lookup, temporal interpolation or adjacent-frame substitution was used.

Current whole-raster PNG/TIF file hashes differ from the archived source-file hashes. This does not establish byte identity of the entire source rasters. Exact reproduction of each complete selected input NPZ independently resolves alignment for these 28 frames only. Both source-file hash sets are retained for review.

Rendering uses `CityNavObservedViewRenderer.current_rgbd` (`graphnav/renderer.py:55–86`): 384×384 BGR, `cv2.INTER_LINEAR`, constant-zero border, no inpainting. The published map ground level is 9.035724612581797 m. A yaw-oriented square footprint has side `2*(camera_z−ground_level)`. Four-corner homography sampling crops the orthophoto; the height raster supplies a vertical depth/support proxy. Black/nodata/above-camera samples are rejected. The observed images include real black unsupported regions; they are retained as model input. There is no calibrated perspective intrinsics matrix, measured pitch/roll, or full 3D ray-occlusion renderer.

The target XY is outside the start footprint for all three episodes and inside supported image pixels at each selected key step. This is **a projected-coverage proxy**, not certified physical target visibility, semantic recognition or facade/sign readability. The two successful cases end with target XY projected inside; the degradation case ends outside. We do not report optical visibility percentages. In particular, a language reference to a facade sign cannot be assumed observable in a nadir orthophoto.

## Licensing: public screenshot clearance is unresolved

The [official CityNav repository](https://github.com/water-cookie/citynav) identifies its dataset as CC BY 4.0 and its code separately as MIT. This does not remove the underlying imagery provider's conditions. The [official SensatUrban repository](https://github.com/QingyongHu/SensatUrban) uses a software license; that alone is not image-data permission. The [official SensatUrban competition terms](https://competitions.codalab.org/competitions/31519#learn_the_details-terms_and_conditions) identify data terms as CC BY-NC-SA 4.0. Its [official download form](https://docs.google.com/forms/d/e/1FAIpQLSfUJVpZ4aBTXwSVhHa5xWaMQ9jeB9EGvo2TsnFovEAykz2eEQ/viewform) additionally imposes research/academic, noncommercial use and dataset nonredistribution conditions. Source retrieval checks/hashes are in `LICENSE_SOURCE_AUDIT.json`; raw pages are local only. No form was submitted and no third party was contacted.

The [CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/) describe conditional noncommercial sharing/adaptation with attribution, change notice and share-alike obligations. However, the download-form restriction and the rights needed for CVF/publisher distribution of these particular rendered screenshots have not been resolved. **We cannot certify that submission/publication of derived synthetic-view screenshots is permitted.** Rendering the same dataset into a synthetic camera view is not an automatic exemption. Retain image-provider attribution and obtain an applicable rights clarification before including RGB in a public submission/artifact.

Accordingly, raw/recovered RGB and all RGB-containing PDF/PNG files remain in ignored `local_only/`; no RGB is committed or pushed. The numerical temporal figure in `figures/` has no RGB, raster or scene pixels. The local multi-panel figures are ready for internal paper review, with public use pending clearance.

## Reproduction

Run `scripts/align_rgb.py` with existing read-only `--rgbd` and `--ground-table` paths, then `scripts/analyze_evidence.py` and `scripts/plot_evidence.py`. No data download, checkpoint load, model training or neural evaluation is required. See `REPORT.md` for the exact interpreter and validation scope. Missing perspective FPV and unverified publication rights remain explicit gaps, not completed deliverables.
