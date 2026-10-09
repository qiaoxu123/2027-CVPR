"""P1: leakage-resistant reversible synthetic spatial-language supervision.

Pair A->B and B->A only when both source polygons are named and their
centroid displacement has an UNAMBIGUOUS cardinal direction. We do not
reverse CityFlight trajectory or modify "between", "along", "left", etc.
The referred TARGET ID/name/contour is NEVER supplied to the model in its
own query; it is only a training label. In the reverse sample B becomes a
legitimate NAMED ANCHOR and A becomes the hidden supervised target.

The generated template includes the actual quantized metric distance to
avoid "east of church" referring to dozens of equally valid buildings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


INVERSE = {"east": "west", "west": "east", "north": "south", "south": "north"}


def _centroid(polygon):
    x = np.asarray(polygon, np.float32)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 3 or not np.isfinite(x).all():
        return None
    return x.mean(0)


def cardinal_relation(a, b, *, min_m=30., max_m=220., transverse_ratio=.20):
    """Return (direction A->B, metric displacement), or None if ambiguous."""
    delta = np.asarray(b, np.float32) - np.asarray(a, np.float32)
    main = int(np.argmax(np.abs(delta)))
    forward = float(abs(delta[main]))
    side = float(abs(delta[1-main]))
    if not min_m <= forward <= max_m or side > min(20., transverse_ratio * forward):
        return None
    direction = ("east" if delta[0] > 0 else "west") if main == 0 else (
        "north" if delta[1] > 0 else "south"
    )
    return direction, forward


def source_landmark(obj_id, entry, bounds):
    """Ground only on known named source outline; no target-selection oracle."""
    name = str(entry.get("name", "")).strip()
    poly = np.asarray(entry.get("contour", []), np.float32)
    center = _centroid(poly)
    if not name or center is None:
        return None
    low, span = bounds[:2], bounds[2:] - bounds[:2]
    if np.any(span <= 0):
        raise ValueError("invalid map bounds")
    return {
        "id": str(obj_id), "name": name, "object_type": entry.get("object_type", "Unknown"),
        "contour": ((poly-low)/span).tolist(),
        "center_world": center, "center_norm": ((center-low)/span).tolist(),
    }


def generate_pairs(
    objects, bounds, source_maps, *,
    max_objects_per_map=300, max_pairs_per_map=800,
    seed=0, augment_shift_count=0,
):
    """Yield exactly TWO records per pair; source_maps is a verified split map set.

    Source objects remain inside their original declared map; augmented
    examples shift BOTH anchor polygon and target center together. Never
    construct a training record for a withheld map.
    """
    rng = np.random.default_rng(seed)
    if max_objects_per_map < 2 or max_pairs_per_map < 1:
        raise ValueError("invalid limits")
    seen_text = set()
    for map_name in sorted(source_maps):
        if map_name not in bounds or map_name not in objects:
            continue
        bbox = np.asarray(bounds[map_name], np.float32)
        entries = [
            source_landmark(object_id, entry, bbox)
            for object_id, entry in sorted(objects[map_name].items(),
                                           key=lambda x:str(x[0]))
        ]
        entries = [x for x in entries if x is not None]
        if len(entries) > max_objects_per_map:
            selected = rng.choice(len(entries), max_objects_per_map, replace=False)
            entries = [entries[int(i)] for i in sorted(selected)]
        attempts = [(i,j) for i in range(len(entries))
                    for j in range(i+1, len(entries))]
        rng.shuffle(attempts)
        emitted = 0
        for i,j in attempts:
            if emitted >= max_pairs_per_map:
                break
            a,b = entries[i],entries[j]
            rel = cardinal_relation(a["center_world"],b["center_world"])
            if rel is None:
                continue
            direction, distance = rel
            rounded = 10 * round(distance/10)
            if rounded <= 0:
                continue
            pair_id = f"{map_name}:{a['id']}:{b['id']}"
            actions = [(a,b,direction,"forward"),(b,a,INVERSE[direction],"reverse")]
            samples=[]
            for anchor,target,direction_word,role in actions:
                desc_type = str(target["object_type"]).replace("TrafficRoad","road surface").lower()
                text = (
                    f"Find a {desc_type} approximately {rounded} meters "
                    f"to the {direction_word} of {anchor['name']}."
                )
                key = (map_name,text)
                if key in seen_text:
                    break
                sample = dict(
                    dataset_version="revised", match_mode="synthetic_verified_geometry",
                    episode_id=f"{pair_id}:{role}",
                    map_name=map_name, source_map_name=map_name,
                    instruction=text,
                    world_bounds=bbox.tolist(),
                    physical_map_sizes=(bbox[2:]-bbox[:2]).tolist(),
                    landmarks=[{key:anchor[key] for key in
                                ("id","name","object_type","contour")}],
                    target_xy=target["center_norm"],
                    # Labels / audit only: never passed to forward().
                    relation_direction=direction_word,
                    pair_id=pair_id, pair_role=role,
                    source_anchor_id=anchor["id"], hidden_target_id=target["id"],
                    quantized_distance_m=rounded,
                )
                samples.append((key,sample))
            if len(samples) != 2:
                continue
            for key,sample in samples:
                seen_text.add(key)
                yield sample
            emitted += 1
            # Paired shift augmentation retains compass direction and distances.
            # No artificial yaw or impossible return trajectory is invented.
            for aug in range(augment_shift_count):
                shift = rng.uniform(-.12, .12, size=2).astype(np.float32)
                new_records=[]
                for _, src in samples:
                    cloned = {**src}
                    ref = {**src["landmarks"][0]}
                    poly = np.asarray(ref["contour"], np.float32)+shift
                    target = np.asarray(src["target_xy"], np.float32)+shift
                    if (np.min(poly)<.02 or np.max(poly)>.98
                        or np.min(target)<.02 or np.max(target)>.98):
                        break
                    ref["contour"]=poly.tolist()
                    cloned["target_xy"]=target.tolist()
                    cloned["landmarks"]=[ref]
                    cloned["map_name"]=f"{map_name}__aug{aug}_{pair_id}"
                    cloned["episode_id"]=src["episode_id"]+f":aug{aug}"
                    cloned["pair_id"]=pair_id+f":aug{aug}"
                    new_records.append(cloned)
                if len(new_records)==2:
                    yield from new_records


def split_map_names(jsonl_path):
    """Map whitelist from the SAME Revised CityNav split used for SBF."""
    names=set()
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                names.add(json.loads(line)["map_name"])
    return names
