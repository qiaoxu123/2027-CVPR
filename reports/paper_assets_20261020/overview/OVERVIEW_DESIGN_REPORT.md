# GeomBelief first-page real-evidence teaser

## Selected episode and narrative

All three panels use **birmingham_block_5:8:4**, Revised CityNav Val-Unseen, formal frozen GraphNav-V1 / GeomBelief C epoch1. The complete checkpoint SHA and original dataset/prediction SHA are in `OVERVIEW_EVIDENCE.json`. No independent CVR, other episode, geometry-ablation checkpoint, synthetic city image, or newly trained model is included.

Verbatim instruction: “This is a large building just besides a large dirt track on Aldridge Road.”

(a) The real geographically registered scene, actual UAV start, supplied Aldridge Road polygons, and three real candidate grid cells show the search setting. Candidates are selected by descending static score with deterministic 60m spatial separation, without GT. They are candidate locations, not detected object instances. The instruction's dirt track has no separately verified input polygon, so none is invented.

(b) The saved static global belief B0 is normalized over the full 28×28 grid. The nearest supplied road-boundary point to Top-1 is 17.734064m away. The vector from that boundary point to the candidate is 135° from world +X. This is a measured geometric relationship, not a compass bearing or a newly asserted direction in the instruction. No geometry-representation performance comparison is included.

(c) The saved step-7 belief is shown with the actual executed path and true final position. Initial and final Top-1 coincide; the final NE is 11.953304m and SR=1. L1(B0,B7)=0.026203909, KL(B0||B7)=0.000405919 nats. The figure intentionally preserves this small update. It does not claim visual correction, independent visual semantics, or recursive probability fusion. The deployed mechanism is fixed s0 plus a bounded current/history residual.

## Support and image registration

Current ROI support was recomputed using the archived `_small_grid_masks` function on all eight SHA-matched observation NPZs, with no neural inference. The runtime's accepted-history union reproduces the saved formal observed mask at every step. At t=7, **6 candidates** have a valid current ROI at one or more scales; current/history union covers127. The cyan hatched cells show only current support; the dashed cyan outline is the current orthophoto footprint. Neither denotes target visibility.

Full scene imagery uses the original 4000×4000 RGB raster with SHA matching the original observation index. Its pixel-corner affine is `[0.1,0,400,0,-0.1,1200,0,0,1]`. The full-raster extent is x400–800m,y800–1200m. All three panels use the same x560–800m,y830–1070m display crop. All executed UAV positions, GT and predicted goal lie inside. The map is display context only; the policy never receives this entire raster.

The observation inset is the exact step7 384×384 input crop, not a perspective FPV. It is explicitly labeled **Pose-conditioned Orthophoto Observation**. Four-corner homography, valid-depth support, input crop hashes and source paths are recorded in the evidence JSON. Final-evidence metadata also documents a different host's byte-different raster; this figure uses this host's original SHA-matching source and crop files rather than silently substituting that host's full raster.

## Design and public version

One horizontal three-panel composition,7.2×3.0in (2.4:1), white margins, consistent world coordinates, blue road relations/path, orange candidates, red GT. Text, map polygons, candidates, arrows, trajectory, heatmap cells and support outlines are vector objects in PDF/SVG; SVG labels remain text. Actual raster imagery remains raster. Primary annotations are6.3–8.1pt at7.2in width. Footer metadata is deliberately smaller; full provenance is in the sidecar, not dependent on footer readability.

Both beliefs use the same log normalization and opacity. Values below1e-5 share the lowest color; probabilities are neither altered nor renormalized within the display crop. No interpolation of the28×28 belief grid. RGB is not contrast-enhanced, inpainted or fabricated. Path and markers are plotted at exact stored XY; text label offsets do not relocate measurements. Metric scale bars are correct in world XY.

The public version is the identical composition with supplied real vector map geometry and no source RGB pixels. This necessarily omits the photoreal scene. Underlying SensatUrban/CityNav RGB redistribution permission remains unconfirmed. The RGB-containing PDF/PNG/SVG and observations stay in the experiment bundle's ignored `local_only/overview/`. The public map credits CityNav / OpenStreetMap contributors (https://www.openstreetmap.org/copyright).

## Reproduction

Run from this directory with numpy,matplotlib,Pillow and shapely available:

```sh
python scripts/draw_overview.py --output-dir figures
python scripts/draw_overview.py --include-rgb --output-dir ../local_only/overview
```

Public redraw requires only the bundled source and evidence JSON, with no model weights, DATA or network. Local RGB redraw additionally requires the source raster and observation NPZ paths recorded in the evidence JSON. `scripts/prepare_overview.py` reconstructs the small source bundle from the original experiment assets and actual ROI implementation; it additionally needs the repository and its scientific environment. No training or navigation evaluation is performed.

Original sources remain untouched. Each render records its source evidence/script SHA and output hashes. Paper integration and compilation results are recorded in `OVERVIEW_VALIDATION.json` after validation. Only the Overview slot is replaced; other five author placeholders are unchanged.

## Paper integration and validation result

The public PDF replaces only `figures/overview.tex`, preserving `fig:overview`. The other five figure-slot TeX files match the pre-task paper commit4917646 byte-for-byte. Tectonic succeeds:13 pages, Figure1 on page1 beneath the title, complete abstract on page1, no undefined citations/references, and no new overfull boxes. Three existing Method equation overflows at lines45,56,91 were reproduced with the original Overview placeholder and left unchanged.

The public PDF contains zero embedded raster images; the SVG contains zero image nodes and retains editable text. A redraw from the paper's bundled data produces a byte-identical PNG. All97 original asset-manifest files retain their SHA. Full checks are in `OVERVIEW_VALIDATION.json`.

In the paper repository, run `python scripts/draw_overview.py --output-dir figures/author` from its root. Extraction and validation scripts reside in the experiment repository under `reports/paper_assets_20261020/overview/scripts/`; no model execution is needed for a public redraw.
