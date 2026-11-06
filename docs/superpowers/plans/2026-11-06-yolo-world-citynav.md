# YOLO-World-V2-S CityNav zero-shot experiment

Goal: real GPU frozen yolov8s-worldv2 on120 existing TRAIN HD current-view frames; compare whole640 vs overlapping native640 tiles, seed42. No training/GeomBelief/navigation/Val/Test.

- Reuse G5 review_v3 PNGs (same frame pose/FOV, original raster,800–2048), strip instruction/target ID/old detector boxes from detector input. Recover pose/footprint/depth support from explicitly TRAIN raw frame manifest. Confirm SHA and paired source dimensions.
- Public catalog only for posthoc geometry proxies; never choose detections/prompts/thresholds by GT target. Keep partial large regions. Visibility is geometric only, no independent pixel truth.
- Fixed banks: base car/van/truck/parking lot/parking space/building/road; vehicle color car/van/truck/red car/white car/black car; parking parking lot/parking area/parking space. Class competition changes across banks are a confound, not color accuracy.
- Fixed conf0.25, model NMS IoU0.5, max_det1000; whole imgsz640; tile640, overlap20%, global family NMS0.5. No GT tuning. Save raw tile candidates and global retained boxes. No whole-image boxes mixed into tile-only condition.
- Reuse landmark_projection.world_to_frame/frame_to_world; clip polygons against footprint before projection, retain intersection and whole coverage independently. Public proxy >=16 projected pixels, RGB/depth support>=0.9 primary supported set; unsupported/tiny retained unknown, not automatic detector FN. Partial views remain eligible.
- Global one-to-one maximum-cardinality IoU matching; main proxy BBox IoU0.5 and auxiliary0.3; also rectangle-vs-polygon overlap (not segmentation mIoU). Vehicle class aggregation because public catalog lacks reliable van/truck/color labels.
- TDD projection/roundtrip, tiling borders+overlap, duplicate suppression, maximum-cardinality matching, empty scenes.2-frame GPU Smoke before120 formal frames; persistent resumable jobs.
- Summaries by map/city/resolution/native sampling/density/type/prompt/method, image/map-group paired bootstrap for vehicle recall; no independence assumption for repeated objects across frames.
- HTML120-case atlas + native PNG links, all public polygons/detections/world links. Direct Agent visual checks are observations, not independent human GT. Unmatched detections are unverified new candidates, not proven false positives or correct new instances.
- Tests, checkpoint/package/inputSHA, Chinese report, tracked-artifact hashes, commit/push independent branch; never merge main.
