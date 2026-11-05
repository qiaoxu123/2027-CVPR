# Changelog

## [direction-frames-v1] - 2026-11-06
### Features
- Independent full original CityNav TRAIN spatial-direction audit with original CityRefer contours.
- Conservative relation scope extraction, offline geometric verification, four reference-frame hypotheses, clustered confidence intervals, high-resolution figures and Chinese report.
### Design Rationale
- Keep source network behavior untouched; isolate exploratory directional statistics from model inference.
- Require unique text-based reference binding; preserve indirect/unknown relations instead of using target GT to resolve them.
- Separate fitted XY concentration from independent evidence about an annotator's reference frame.
### Notes & Caveats
- Exact-name/simple-scope coverage is low; rates do not estimate full-corpus frame frequencies.
- Missing CRS and directed heading make true geographic north and intrinsic orientation unavailable; initial trajectory yaw is not annotation yaw.
