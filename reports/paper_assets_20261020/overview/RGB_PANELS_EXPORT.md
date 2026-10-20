# Separate RGB panels (user-requested private upload)

The user requested independent real RGB panels and remote upload. The destination was verified: `JLU-MCNS-MEC/2027-Qiaoxu-CVPR` is PRIVATE; `qiaoxu123/2027-CVPR` is PUBLIC. Derived RGB panels are therefore retained locally and uploaded only to the private paper repository under `figures/author/rgb_panels/`. This public experiment repository contains only the scripts, manifests and validation metadata. Public imagery redistribution clearance is not asserted.

Three independent map panels (a,b,c), one separate step7 observation, and one separate shared colorbar are exported in PDF/PNG/editable SVG. The navigation panel no longer embeds the observation inset. The original384×384 observation is also retained as an exact RGB PNG. No montage is generated in the new output folder. All elements remain tied to `birmingham_block_5:8:4`, formal C1. No inference, training or data edits occur.

From the experiment overview directory:

```sh
python scripts/draw_overview.py --include-rgb --separate-panels --output-dir ../local_only/overview_panels_rgb
```

The original combined and public files are historical deliverables and remain unchanged. Their existing render manifests refer to the original script version at commit94bc85b. The extended script adds a separate export branch; the new manifest records its current SHA. Source raster/crop paths are in `OVERVIEW_EVIDENCE.json`; no dataset/weights are uploaded. See `RGB_PANELS_MANIFEST.json` and `RGB_PANELS_VALIDATION.json`.
