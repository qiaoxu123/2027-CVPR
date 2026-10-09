"""Audit source identity, split instructions, map registration and causal RGB."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from graphnav.full_experiment import static_signature
from graphnav.static_localization import StaticLocalizationDataset


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, source, rgbd):
    root, source, rgbd = map(Path, (root, source, rgbd))
    result = {"splits": {}, "maps": {}}
    query_sets = {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        filename = root / f"graphnav_{split}.jsonl"
        dataset = StaticLocalizationDataset(filename)
        records = dataset.records
        query_sets[split] = {(r["map_name"], r["instruction"]) for r in records}
        upstream = json.loads((source / "processed_citynav" / f"citynav_{split}.json").read_text())
        source_queries = {(f'{e["area"]}_block_{e["block"]}', e["descriptions"][0]) for e in upstream}
        assert query_sets[split] <= source_queries, "Converted instruction differs from SBF split"
        provenance = json.loads(Path(str(filename) + ".provenance.json").read_text())
        assert provenance["output_sha256"] == sha(filename)
        for key, path in {
            "objects": source / "cityrefer" / "objects.json",
            "processed": source / "cityrefer" / "processed_descriptions.json",
            "trajectories": source / "processed_citynav" / f"citynav_{split}.json",
        }.items():
            assert provenance["source_sha256"][key] == sha(path)
        bad_target = sum(any(x < 0 or x > 1 for x in r["target_xy"]) for r in records)
        missing, invalid_frames = 0, 0
        if "rgb_observations" in records[0]:
            for r in records:
                observations = r["rgb_observations"]
                if not observations:
                    missing += 1
                    assert r.get("observation_unavailable_reason") == "nonpositive_camera_altitude"
                for frame in observations:
                    assert frame["step"] == 0 == r["observation_cutoff_step"]
                    assert Path(frame["path"]).is_file()
                    invalid_frames += int(np.asarray(frame["footprint_world"]).shape != (4, 2))
        result["splits"][split] = {
            "queries": len(records), "split_text_matches_sbf": True,
            "static_signature": static_signature(dataset), "provenance": provenance,
            "targets_outside_bounds": bad_target,
            "queries_without_valid_initial_camera": missing,
            "invalid_rgb_footprints": invalid_frames,
        }
        for r in records:
            map_name = r["map_name"]
            if map_name in result["maps"]:
                continue
            with rasterio.open(rgbd / f"{map_name}.tif") as raster:
                with Image.open(rgbd / f"{map_name}.png") as rgb:
                    assert rgb.size == (raster.width, raster.height)
                public = np.asarray(r["world_bounds"])
                registered = np.asarray(raster.bounds)
                result["maps"][map_name] = {
                    "public_bounds": public.tolist(), "tif_bounds": registered.tolist(),
                    "max_bound_difference_m": float(abs(public - registered).max()),
                    "rgb_tif_same_dimensions": True, "crs": str(raster.crs),
                    "span_m": (public[2:] - public[:2]).tolist(),
                }
    result["train_validation_query_overlap"] = len(query_sets["train_seen"] &
        (query_sets["val_seen"] | query_sets["val_unseen"]))
    result["seen_unseen_query_overlap"] = len(query_sets["val_seen"] & query_sets["val_unseen"])
    assert result["train_validation_query_overlap"] == 0
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sbf-data-dir", required=True)
    p.add_argument("--rgbd-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = audit(args.data_dir, args.sbf_data_dir, args.rgbd_dir)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print({k: v["queries"] for k, v in result["splits"].items()})
