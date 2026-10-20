# Image and map provenance

The local previews use actual crops rendered from the existing CityNav/SensatUrban orthophoto and elevation rasters at executed UAV poses. These are **not** newly captured onboard photographs, perspective-occlusion renders, or generated city images. The elevation-derived depth is a vertical-depth proxy. Black source/nodata areas remain black; no inpainting or city-scene synthesis is performed.

CityNav declares its dataset CC BY 4.0 in the [official README](https://github.com/water-cookie/citynav#license). The underlying [SensatUrban repository](https://github.com/QingyongHu/SensatUrban) has an MIT code license and a separate dataset download form. This audit does not treat a software license as conclusive permission to redistribute every underlying image raster. Until the underlying raster redistribution terms are established, all RGB-containing previews and crop arrays remain ignored under `local_only/` and are not pushed to either repository.

Published `figures/` contain real prediction heatmaps, actual executed paths, named map geometry and measured observation footprints, **no RGB pixels or raw rasters**. Public map context is extracted from CityNav's named landmark annotations. Faint contextual landmarks are for display and were not retroactively supplied as additional model inputs. Referenced input landmarks are highlighted. Attribute CityNav and **OpenStreetMap contributors**, https://www.openstreetmap.org/copyright (OSM data attribution/ODbL applies to the underlying geographic source).

`observation_index.json` records exact scene file SHA, affine/CRS, pose, crop array SHA, frame index and reproduction command. `public_map_context.json` records the named-map source SHA. Every figure has a JSON provenance sidecar and source arrays; no figure asserts semantic identity from color or sharpness.

To inspect full real RGB figures on the lab server:

```bash
python reports/paper_assets_20261020/scripts/plot_figures.py --include-rgb
```

This creates `local_only/figures/figure1_overview.{pdf,png}` and the other local previews. To reconstruct missing local observations, run `scripts/replay_cases.py` using the documented lab environment and shared read-only DATA. That performs frozen inference only at archived poses and verifies parity, not a new navigation evaluation or training run.
