#!/usr/bin/env bash
set -euo pipefail
# Explicit existing TRAIN observations; no Val/Test loader or navigation entry point.
: "${CITYNAV_YOLO_RUN:?Set output/cache root}"
: "${CITYNAV_YOLO_PYTHON:?Set isolated Python interpreter}"
: "${CITYNAV_REVIEW_QUEUE:?Set existing TRAIN HD queue.json}"
: "${CITYNAV_TRAIN_RAW:?Set TRAIN raw directory}"
: "${CITYNAV_PUBLIC_OBJECTS:?Set public objects.json}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
common=(--queue "$CITYNAV_REVIEW_QUEUE" --raw "$CITYNAV_TRAIN_RAW" --catalog "$CITYNAV_PUBLIC_OBJECTS" --weights "$CITYNAV_YOLO_RUN/yolov8s-worldv2.pt")
"$CITYNAV_YOLO_PYTHON" -u -m scripts.yolo_world_citynav "${common[@]}" --output "$CITYNAV_YOLO_RUN/smoke" --limit 2 --smoke
"$CITYNAV_YOLO_PYTHON" -u -m scripts.yolo_world_citynav "${common[@]}" --output "$CITYNAV_YOLO_RUN/formal" --limit 120
"$CITYNAV_YOLO_PYTHON" -m scripts.report_yolo_world_citynav --run "$CITYNAV_YOLO_RUN/formal" --output reports/yolo_world_v2s
"$CITYNAV_YOLO_PYTHON" -m scripts.yolo_world_proxy_diagnostics --run "$CITYNAV_YOLO_RUN/formal" --output reports/yolo_world_v2s
"$CITYNAV_YOLO_PYTHON" -m scripts.yolo_world_extended_analysis --run "$CITYNAV_YOLO_RUN/formal" --output reports/yolo_world_v2s
"$CITYNAV_YOLO_PYTHON" -m scripts.visualize_yolo_world_cases --run "$CITYNAV_YOLO_RUN/formal" --output reports/yolo_world_v2s
