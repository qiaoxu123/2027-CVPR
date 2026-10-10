"""Conservative P2 input quality filter; never a semantic segmentation label."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from graphnav.semantic_geometry import valid_surface_polygon


def pixel_support(frame, contour):
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 1)
    region = mask > 0
    # Source orthographic point-cloud holes and outside-map pixels are black.
    valid = (frame.max(axis=-1) > 0)
    return float(valid[region].mean()) if region.any() else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    counts = Counter()
    last_path, frame = None, None
    with Path(args.output).open("w") as handle:
        for line in Path(args.input).read_text().splitlines():
            row = json.loads(line)
            counts["input_rois"] += 1
            if not valid_surface_polygon(row["contour_px"]):
                counts["invalid_polygon"] += 1
                continue
            x0,y0,x1,y1 = row["bbox_xyxy"]
            if (row["visible_fraction"] < .9 or row["projected_area_px"] < 256
                    or min(x1-x0,y1-y0) < 8):
                counts["small_or_clipped"] += 1
                continue
            if last_path != row["observed_frame"]:
                last_path = row["observed_frame"]
                frame = cv2.imread(last_path)
                if frame is None or frame.shape != (384,384,3):
                    raise ValueError("P2 requires actual native 384px observed RGB")
            support = pixel_support(frame, row["contour_px"])
            if support < .9:
                counts["source_rgb_holes"] += 1
                continue
            row.update(rgb_pixel_support=support,
                       qa_policy="native384; visible>=.9; area>=256; bbox>=8; source RGB support>=.9")
            handle.write(json.dumps(row) + "\n")
            counts["retained_rois"] += 1
    Path(args.output + ".quality.json").write_text(json.dumps(dict(counts),indent=2))
    print(json.dumps(dict(counts)))


if __name__ == "__main__":
    main()
