# Reproduction and scope

Run from the experiment repository root. The public no-RGB figures are fully reproducible from this lightweight bundle, without downloading a model or dataset:

```bash
python reports/paper_assets_20261020/scripts/plot_figures.py
```

Dependencies: Python 3, numpy, matplotlib, shapely. Source arrays, probability scales, coordinate grids, selected validation traces and map context are bundled. This is a deterministic redraw of measured outputs, not newly synthesized scene data.

For lab-side source verification and optional frozen replay use the existing environment:

```bash
export GEOMBELIEF_WORKSPACE=/home/rental/20260922_1/Workspace
export GEOMBELIEF_ARCHIVE="$GEOMBELIEF_WORKSPACE/2027-CVPR/Ablations/GraphNav/visual-residual-20261013"
export GEOMBELIEF_PAPER="$GEOMBELIEF_WORKSPACE/2027-CVPR/GeomBelief-paper"
export GEOMBELIEF_PYTHON="$GEOMBELIEF_WORKSPACE/2027-CVPR/Baselines/.reproduction-env/bin/python"
"$GEOMBELIEF_PYTHON" reports/paper_assets_20261020/scripts/validate.py
```

The following steps use existing read-only CityNav/weights and archived experiment code. They are listed separately to avoid accidental repeated full evaluation or training:

1. `scripts/inventory.py`: inspect source presence and establish/verify original hashes. It never replaces the initial protected-source snapshot.
2. `scripts/prepare_cases.py`: fixed source-order outcome illustrations, validation only.
3. `scripts/replay_cases.py`: frozen inference at the saved poses for 9 cases, real/control matched-prefix comparisons; verifies saved decision outputs. Existing complete case outputs are reused.
4. `scripts/geometry_case.py`: fixed seed-0 first qualifying validation query, replay three already-trained geometry heads; no retraining. Saved Top-1 IDs/errors are verified.
5. `scripts/fetch_sources.py`, `scripts/audit_baselines.py`, then `scripts/baseline_preflight.py`: authoritative-source retrieval, published/local/asset-layer separation, table numeric checks and native import/shard readiness. Full third-party documents remain local; only hashes/derived metadata/numeric transcriptions are published.
6. `scripts/index_observations.py`: hash real scene/crop files and export public named map context.
7. `scripts/plot_figures.py --include-rgb`: real local-only RGB previews; do not redistribute unresolved raw imagery.
8. `scripts/supplementary.py`: offline scoring of the 10,448 already-frozen evaluation records, grouped statistics and profiling scope. The explicitly authorized sealed-Test read is metadata/scoring only, never new Test policy inference, tuning or selection.
9. `scripts/audit_bibliography.py`: original 48-entry manuscript snapshot to verified publisher/arXiv records. Network metadata is cached. It updates the specified paper checkout, preserving unsupported fields instead of guessing.
10. `scripts/sync_paper.py`: copy lightweight figures/source arrays to the specified paper checkout and insert real evidence figures. `scripts/validate.py` performs final checks.

All raw datasets, training logs and original experiment outputs remain in place. Shared DATA is read-only. No model is trained by any command in this bundle. Full navigation is never rerun. The source archive's local HEAD `d86308c2d9d11510cde35e111275538f26a8b707` differs from the published starting branch `2e7a0671c66f4fa4af7c479533b64c7f03ea220b`; exact module SHA and checkpoint SHA, rather than branch labels alone, define provenance.
