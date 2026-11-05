# CityNav TRAIN direction frames

独立离线实验，中文结论见 [REPORT_ZH.md](REPORT_ZH.md)。不训练模型，不修改网络，不读取 Val/Test。

```bash
# Use the existing conda environment's Python directly.
PYTHON=/home/20261024_1/miniconda3/envs/graphnav/bin/python
DATA_ROOT=/home/20261024_1/Workspace/DATA/sbfnav-data
$PYTHON experiments/direction_frames/analyze.py --data-root "$DATA_ROOT"
$PYTHON experiments/direction_frames/statistics.py --data-root "$DATA_ROOT"
$PYTHON experiments/direction_frames/controls.py
$PYTHON experiments/direction_frames/render.py --data-root "$DATA_ROOT"
$PYTHON experiments/direction_frames/report.py
$PYTHON -m pytest experiments/direction_frames/test_analysis.py -q
$PYTHON experiments/direction_frames/verify.py --data-root "$DATA_ROOT"
```

Required packages: numpy, pandas, scipy, shapely, rasterio, matplotlib, Pillow, pytest. No GPU, LLM API, model weights or training. Run from repository root. Explicit input filenames allow only original TRAIN and original CityRefer files; raster reads are limited to TRAIN map names. Additional external information is used only to document coordinate/annotation provenance, never to fit a direction convention.

- `analyze.py`: complete occurrence ledger, conservative scope/name resolution, geometric pairs, hypothesis/group statistics, inverse-text analysis, raster affine audit.
- `statistics.py`: all TRAIN targets/episodes, original annotation alignment, accepted text audit, same-pair contradictions, matched hypothesis comparisons and holdouts.
- `controls.py`: full-polygon confidence intervals, arbitrary-sector upper bounds, weighting controls and flat exclusion counts.
- `render.py`: 320dpi PNG and PDF statistical plots plus actual map crops, GT and complete contours.
- `report.py`: Chinese report generated from exported data, including figure links and limitations.
- `verify.py`: independently recomputes source geometry/pose facts and checks output invariants.

Primary results count whole-word occurrences. Compound and morphological directions are retained separately in `extended_lexemes.json`. Parsing intentionally favors precision over recall: no nearest-GT landmark disambiguation, fuzzy aliases, or flattening of indirect relations. Rates apply only to the selected subset. A fitted direction axis is descriptive evidence, not the speaker's known frame. Initial UAV yaw is a proxy, not annotation-camera ground truth. Landmark axes are unsigned; no D heading is invented. TIFs lack CRS: +Y=image up is verified, +Y=true north is not.

Raw occurrences contain original CityNav/CityRefer English descriptions and derived geometry for reproducibility. Source datasets: [CityNav](https://github.com/water-cookie/citynav), [CityRefer](https://github.com/ATR-DBI/CityRefer); retain original attribution and license conditions. No entire raster or original dataset archive is republished. Figure backgrounds are cropped TRAIN rasters.
