"""P0/P2: register all *visible* CityRefer map polygons to causal CityFlight RGB.

Input: JSONL output of prepare_citynav in NON-static mode (actual rendered
camera prefixes), and same-revision CityRefer objects.json. This script does
NOT load target_xy, future trajectory, semantic image masks or orthophoto.
No fake segmentation IoU is reported: polygon-to-image registration is only
a geometric proposal, requiring separate manual/visual QA.

Example:
 python -m scripts.prepare_landmark_views --records data/revised_dynamic/graphnav_train_seen.jsonl \
   --objects data/revised/cityrefer/objects.json --out-dir runs/p0_train --save-rois
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from graphnav.landmark_projection import (
    project_landmark, draw_projected_roi, make_roi_image, world_to_frame, frame_to_world
)


def objects_by_map(raw, *, types=None):
    """Keep source world-metric polygons, preserving each original object ID."""
    result = {}
    for map_name, entries in raw.items():
        objects = []
        for object_id, o in entries.items():
            name, typ = str(o.get("name", "")), str(o.get("object_type", "Unknown"))
            contour = np.asarray(o.get("contour", []), np.float32)
            if not name or (types and typ not in types) or contour.shape[0] < 3:
                continue
            if contour.ndim != 2 or contour.shape[1] != 2 or not np.isfinite(contour).all():
                continue
            objects.append(dict(
                id=str(object_id), name=name, object_type=typ,
                contour=contour, low=contour.min(0), high=contour.max(0),
            ))
        result[map_name] = objects
    return result


def process(records, objects, output, *, save_rois=False,
            min_visible=.65, min_pixels=40, max_per_frame=32,
            overlay_frames=32, output_size=224, frame_sample_ratio=1.,
            sample_seed=0):
    """Return summary, write one ROI row per source ID and observed frame."""
    if not 0 < frame_sample_ratio <= 1:
        raise ValueError("frame_sample_ratio must be in (0,1]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_dir, overlay_dir = output/"rois", output/"overlays"
    if save_rois:
        image_dir.mkdir(exist_ok=True)
    if overlay_frames:
        overlay_dir.mkdir(exist_ok=True)
    visited_frames = set()
    counts, types = Counter(), Counter()
    roi_path = output/"landmark_rois.jsonl"
    with roi_path.open("w", encoding="utf-8") as out:
        for rec in records:
            map_name = rec["map_name"]
            cutoff = rec.get("observation_cutoff_step")
            if not isinstance(cutoff, int) or cutoff < 0:
                raise ValueError("P2 needs explicit causal observation cutoff")
            for obs in rec.get("rgb_observations", []):
                step = obs.get("step")
                if not isinstance(step, int) or step > cutoff or step < 0:
                    raise ValueError("future/untracked RGB observation rejected")
                frame_path = Path(obs["path"])
                # Avoid duplicating shared observation prefixes. Paths are
                # unique in CityNav rendered data but may be repeated in rows.
                if str(frame_path) in visited_frames:
                    continue
                visited_frames.add(str(frame_path))
                # Hash-based frame sampling is independent of language, object
                # ID and GT destination. Prevents multi-million ROI exports.
                digest = hashlib.sha256(
                    f"{sample_seed}:{map_name}:{frame_path}".encode("utf-8")
                ).digest()
                keep = int.from_bytes(digest[:8], "big") / 2**64
                if keep >= frame_sample_ratio:
                    counts["sampled_out_frames"] += 1
                    continue
                counts["frames"] += 1
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    counts["missing_frame"] += 1
                    continue
                footprint = np.asarray(obs["footprint_world"], np.float32)
                lo, hi = footprint.min(0), footprint.max(0)
                results = []
                for entity in objects.get(map_name, []):
                    if (np.any(entity["high"] < lo) or np.any(entity["low"] > hi)):
                        continue
                    roi = project_landmark(
                        entity, footprint, bgr.shape,
                        min_visible=min_visible, min_pixels=min_pixels,
                    )
                    if roi is not None:
                        results.append((roi.visible_fraction * roi.visible_pixels, entity, roi))
                # Deterministic quality-based cap; NOT selected by target location.
                results.sort(key=lambda t: (-t[0], t[1]["id"]))
                results = results[:max_per_frame]
                if results:
                    counts["frames_with_roi"] += 1
                overlay = bgr.copy() if counts["frames"] <= overlay_frames else None
                for j, (_, entity, roi) in enumerate(results):
                    counts["rois"] += 1
                    types[roi.object_type] += 1
                    saved = None
                    if save_rois:
                        saved = image_dir / f"{counts['frames']:07d}_{j:03d}.png"
                        image = make_roi_image(bgr, roi, output_size=output_size)
                        if not cv2.imwrite(str(saved), image):
                            raise IOError(f"cannot write ROI: {saved}")
                    row = {
                        "map_name": map_name, "object_id": roi.object_id,
                        "name": roi.name, "object_type": roi.object_type,
                        "observed_step": step, "observed_frame": str(frame_path),
                        "roi_path": str(saved.resolve()) if saved else None,
                        "footprint_world": footprint.tolist(),
                        "contour_px": roi.contour_px.tolist(),
                        "visible_fraction": roi.visible_fraction,
                        "visible_pixels": roi.visible_pixels,
                        "projected_area_px": roi.projected_area_px,
                        "bbox_xyxy": roi.bbox_xyxy,
                        "annotation_kind": "geometry_projected_roi_not_verified_segmentation",
                    }
                    out.write(json.dumps(row, ensure_ascii=False)+"\n")
                    if overlay is not None:
                        overlay = draw_projected_roi(overlay, roi)
                if overlay is not None:
                    path = overlay_dir / f"{counts['frames']:07d}.png"
                    if not cv2.imwrite(str(path), overlay):
                        raise IOError(path)
    summary = {
        **counts,
        "frames_with_roi_fraction": counts["frames_with_roi"]/max(1, counts["frames"]),
        "objects_by_type": dict(types),
        "min_visible_fraction": min_visible,
        "frame_sample_ratio": frame_sample_ratio, "sample_seed": sample_seed,
        "min_projected_pixels": min_pixels,
        "roi_quality": "geometric_proposal_only; manually audit semantic alignment",
        "future_RGB_used": False,
        "roi_file": str(roi_path.resolve()),
    }
    (output/"alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True)
    p.add_argument("--objects", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--save-rois", action="store_true")
    p.add_argument("--min-visible", type=float, default=.65)
    p.add_argument("--min-pixels", type=int, default=40)
    p.add_argument("--max-per-frame", type=int, default=32)
    p.add_argument("--overlay-frames", type=int, default=32)
    p.add_argument("--roi-size", type=int, default=224)
    p.add_argument("--frame-sample-ratio", type=float, default=1.,
                   help="stable hash sample; use 0.1 for initial ROI audit")
    p.add_argument("--sample-seed", type=int, default=0)
    args = p.parse_args()
    raw = json.loads(Path(args.objects).read_text(encoding="utf-8"))
    with Path(args.records).open(encoding="utf-8") as handle:
        records = (json.loads(line) for line in handle if line.strip())
        output = process(
            records, objects_by_map(raw), args.out_dir, save_rois=args.save_rois,
            min_visible=args.min_visible, min_pixels=args.min_pixels,
            max_per_frame=args.max_per_frame, overlay_frames=args.overlay_frames,
            output_size=args.roi_size, frame_sample_ratio=args.frame_sample_ratio,
            sample_seed=args.sample_seed,
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
