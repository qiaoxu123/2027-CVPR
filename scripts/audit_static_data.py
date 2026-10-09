"""Reproducible audit of query splits, physical axes, and exact landmark identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from graphnav.data import normalized_name, bind_reference_landmarks
from graphnav.landmark_alias import build_name_index
from graphnav.semantic_geometry import build_entities, parse_spatial_relations


def audit(data_dir, source_dir, output):
    data_dir, source_dir = Path(data_dir), Path(source_dir)
    objects = json.loads((source_dir / "cityrefer/objects.json").read_text())
    processed = json.loads((source_dir / "cityrefer/processed_descriptions.json").read_text())
    indexes = {map_name: build_name_index(items)
               for map_name, items in objects.items()}
    summary, keys, text_keys = {}, {}, {}
    for split in ("train_seen", "val_seen", "val_unseen"):
        path = data_dir / f"graphnav_{split}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        counts = Counter()
        missing = Counter()
        relations = Counter()
        maps = set()
        keys[split], text_keys[split] = set(), set()
        maximum_roundtrip_error = 0.0
        for record in records:
            m = record["map_name"]
            maps.add(m)
            query = record["episode_id"].rsplit(":", 1)[0]
            keys[split].add(query)
            text_keys[split].add((m, record["instruction"]))
            obj_id, ann_id = query.split(":")[-2:]
            bounds = np.asarray(record["world_bounds"])
            span = bounds[2:] - bounds[:2]
            assert np.all(span > 0) and np.allclose(span, record["physical_map_sizes"])
            target = np.asarray(record["target_xy"])
            assert np.isfinite(target).all()
            counts["targets_outside_map"] += int(np.any((target < 0) | (target > 1)))
            counts["queries_with_building"] += any(
                lm.get("object_type", "Building") == "Building"
                for lm in record["landmarks"]
            )
            counts["queries_with_any_surface"] += bool(record["landmarks"])
            counts["queries_with_nonbuilding_surface"] += any(
                lm.get("object_type", "Building") != "Building"
                for lm in record["landmarks"]
            )
            requested = processed[m][obj_id][int(ann_id)].get("landmarks", [])
            for name in requested:
                counts["requested_names"] += 1
                matches = bind_reference_landmarks(
                    [name], objects[m], 64, index=indexes[m]
                )
                buildings = [lm for lm in matches
                             if lm["object_type"] == "Building"]
                counts["requests_matched_any_type"] += bool(matches)
                counts["requests_matched_building"] += bool(buildings)
                counts["requests_nonbuilding_only"] += bool(matches) and not bool(buildings)
                counts["requests_matched_typed_alias"] += any(
                    lm.get("aliases") for lm in matches
                )
                if not matches:
                    missing[name] += 1
            for lm in record["landmarks"]:
                original = objects[m][lm["id"]]
                assert original["object_type"] == lm.get(
                    "object_type", original["object_type"]
                )
                assert lm["name"] == original["name"]
                world = bounds[:2] + np.asarray(lm["contour"]) * span
                err = float(np.abs(world - np.asarray(original["contour"])).max())
                maximum_roundtrip_error = max(maximum_roundtrip_error, err)
                assert err < 0.001
                counts["bound_polygons"] += 1
                counts[f"bound_type_{original['object_type']}"] += 1
            entities = build_entities(record["landmarks"], [], record["world_bounds"])
            parsed = parse_spatial_relations(record["instruction"], entities)
            counts["queries_with_grounded_relation"] += bool(parsed)
            relations.update(r.kind for r in parsed)
            counts["queries_mentioning_road_words"] += bool(
                re.search(
                    r"\b(road|street|avenue|lane|drive|footpath)\b",
                    record["instruction"],
                    re.IGNORECASE,
                )
            )
            counts["queries_with_relation_trigger"] += bool(
                re.search(
                    r"\b(east|west|north|south|northeast|northwest|southeast|southwest|near|beside|close to|next to|between|along|alongside)\b",
                    record["instruction"],
                    re.IGNORECASE,
                )
            )
        summary[split] = {
            "records": len(records),
            "unique_query_ids": len(keys[split]),
            "unique_map_instruction": len(text_keys[split]),
            "maps": sorted(maps),
            "counts": dict(counts),
            "grounded_relation_kinds": dict(relations),
            "unmatched_names_top30": missing.most_common(30),
            "max_polygon_roundtrip_error_m": maximum_roundtrip_error,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    overlaps = {}
    for a, b in [
        ("train_seen", "val_seen"),
        ("train_seen", "val_unseen"),
        ("val_seen", "val_unseen"),
    ]:
        overlaps[a + "__" + b] = {
            "query_id": len(keys[a] & keys[b]),
            "map_instruction": len(text_keys[a] & text_keys[b]),
            "maps": sorted(set(summary[a]["maps"]) & set(summary[b]["maps"])),
        }
        assert not keys[a] & keys[b], "Query identities leaked between splits"
        assert not text_keys[a] & text_keys[b], "Identical map/instruction leaked between splits"
    result = {
        "splits": summary,
        "overlaps": overlaps,
        "roads": {
            "usable_aligned_polylines": False,
            "reason": "CityRefer TrafficRoad/Footpath/Bridge are typed CLOSED surface polygons, never validated navigable road centerlines. No verified centerline/topology provided.",
            "coverage": None,
        },
        "coordinates": "Public HETT MAP_BOUNDS: world XY meters, x east/y north. Normalize without image-row flip.",
    }
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    audit(a.data_dir, a.source_dir, a.output)
