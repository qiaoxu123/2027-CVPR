# Reproduction

Working directory: repository root. All data are existing TRAIN observations, not generated predictions.

```bash
export CITYNAV_YOLO_RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/yolo_world_v2s_s42_20261106
export CITYNAV_YOLO_PYTHON="$CITYNAV_YOLO_RUN/env/bin/python"
export CITYNAV_REVIEW_QUEUE=/home/20261024_1/Workspace/2027-CVPR-Under-Test/runs/g5/review_v3/queue.json
export CITYNAV_TRAIN_RAW=/home/20261024_1/Workspace/2027-CVPR-Under-Test/runs/original_train_observations_v7/train/raw
export CITYNAV_PUBLIC_OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json
bash scripts/run_yolo_world_citynav.sh
"$CITYNAV_YOLO_PYTHON" -m scripts.profile_yolo_world_postprocess --run "$CITYNAV_YOLO_RUN/formal" --output reports/yolo_world_v2s/postprocess_timing.json
"$CITYNAV_YOLO_PYTHON" -m pytest -q tests/test_yolo_world_diagnostic.py tests/test_bidirectional_visual_stages.py
```

Isolated venv inherits existing graphnav conda PyTorch. Added ultralytics==8.4.176, numpy==1.26.4, opencv-python==4.11.0.86, official Ultralytics CLIP source revision recorded in environment.json. No new CityNav data downloaded. Official detector weights:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt
CLIP ViT-B/32 auto-download is a pretrained model dependency, not an API service.

Model input first-convolution hook separately confirmed CUDA FP32 [1,3,640,640]. Full raw frames/weights remain server-side. Per-image Git JSON retains all prompt banks, pre-global-NMS proposals, final boxes, confidence, world mapping, proxy status and matches. Source SHA identifies lossless PNG, not the published JPEG preview. Published PNG case panels are resized comparisons, not new high-resolution observations.

Serve HTML via HTTP (file:// fetch is restricted):
```bash
python -m http.server 8927 --bind 127.0.0.1 --directory reports/yolo_world_v2s
```
Open http://127.0.0.1:8927/index.html, or use SSH local port forwarding. No changes made to existing web/frpc services.

Optional browser test requires existing Playwright + Chromium:
```bash
python -m scripts.check_yolo_world_atlas
```

No GT-dependent filtering or threshold tuning, no training, no navigation, no Val/Test, no checkpoint overwrite. Posthoc center-in-contour diagnostics were added after observing loose map contours; original pre-specified IoU .5/.3 results are retained unchanged. Do not interpret either proxy as independent human instance accuracy.
