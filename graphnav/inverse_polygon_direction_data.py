"""Leakage-safe synthetic inverse relation pairs from TRAIN CityNav source polygons.

No GT annotation or future image is permitted in the model *input*.
CityRefer target contours are training labels ONLY, even when the same
physical object becomes a synthetic anchor in its inverse example.
These are geometrically generated statements, NOT paraphrases of the
original CityNav instruction and NOT benchmark-valid new instructions.

Directions are deliberately MAP-relative: right/left = +/- map X,
up/down = +/- map Y. No unverified geographic north or ego front/back.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

DIR = {"map-right": (1, 0), "map-left": (-1, 0),
       "map-up": (0, 1), "map-down": (0, -1)}
INVERSE = {"map-right": "map-left", "map-left": "map-right",
           "map-up": "map-down", "map-down": "map-up"}
TYPE_NAMES = {"Building": "building", "Car": "car", "Parking": "parking area",
              "Ground": "ground region", "TrafficRoad": "road region",
              "Footpath": "footpath", "Bridge": "bridge"}
GRID_SIZE = 28


def validated_polygon(obj):
    xy = np.asarray(obj.get("contour", []), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1:] != (2,) or len(xy) < 3 or not np.isfinite(xy).all():
        return None
    p = Polygon(xy)
    return p if p.is_valid and p.area > 0 else None


def cardinal_from_whole_outlines(target, anchor, *, margin_m=5., tolerance=.35):
    """Conservative single principal MAP axis from complete footprint bounds.

    A pair must be strictly separated on one axis and approximately aligned
    on its transverse axis; diagonally ambiguous pairs have no gold label.
    Inversion is exact, but *front/behind* is intentionally unsupported.
    """
    ta = np.asarray(target.bounds, float)
    an = np.asarray(anchor.bounds, float)
    tc = np.array(target.centroid.coords[0])
    ac = np.array(anchor.centroid.coords[0])
    candidates = []
    for axis, positive, negative in ((0,"map-right","map-left"),(1,"map-up","map-down")):
        gap_plus=ta[axis]-an[axis+2]
        gap_minus=an[axis]-ta[axis+2]
        gap=max(gap_plus,gap_minus)
        if gap < margin_m:continue
        transverse=abs(float(tc[1-axis]-ac[1-axis]))
        # Cap transverse deviation; retain width/height-dependent tolerance.
        max_transverse=max(8.,tolerance*abs(float(tc[axis]-ac[axis])))
        if transverse > max_transverse:continue
        direction=positive if gap_plus>gap_minus else negative
        candidates.append((direction,gap,transverse))
    if len(candidates)!=1:
        return None
    return candidates[0][0]


def instruction(target_type, anchor_text, word):
    """A fixed template makes swapping the direction the ONLY word change."""
    if word not in DIR:
        raise ValueError("Unsupported frame or unverified front/back")
    return f"Find the {target_type} {word} of {anchor_text}."


def family_id(map_name, a, b):
    """Same unordered pair and SAME map cannot cross FIT/DEV."""
    return f"{map_name}:{':'.join(sorted((str(a),str(b))))}"


def create_pair(map_name, source_episode, target_id, target_obj,
                anchor_id, anchor_obj, world_bounds, *, margin_m=5.):
    target=validated_polygon(target_obj)
    anchor=validated_polygon(anchor_obj)
    if target is None or anchor is None or str(target_id)==str(anchor_id):
        return None
    word=cardinal_from_whole_outlines(target,anchor,margin_m=margin_m)
    if word is None:return None
    inv=INVERSE[word]
    anchor_name=str(anchor_obj.get("name","")).strip()
    if not anchor_name:return None
    target_type=TYPE_NAMES.get(target_obj.get("object_type"),
                               str(target_obj.get("object_type","object")).lower())
    anchor_type=TYPE_NAMES.get(anchor_obj.get("object_type"),
                               str(anchor_obj.get("object_type","object")).lower())
    if not target_type or not anchor_type:return None
    fid=family_id(map_name,target_id,anchor_id)
    source=dict(source_episode_id=source_episode,family_id=fid,
                target_source_id=str(target_id),anchor_source_id=str(anchor_id),
                source_kind="train_polygon_synthetic_not_original_instruction",
                public_map_name=map_name)
    # Original target contour only ever appears as a label when it is the
    # current predicted target. In inverse sample it becomes an explicitly
    # designated (synthetic) PUBLIC anchor input, marked with provenance.
    def row(role, txt, relation, src_id, tgt_id, src_poly, tgt_poly, cf=False):
        return dict(
            family_id=fid,map_name=map_name,world_bounds=list(map(float,world_bounds)),
            input=dict(instruction=txt,relation_phrase=relation,
                       anchor_contour=[list(map(float,p)) for p in src_poly.exterior.coords[:-1]],
                       anchor_id=str(src_id),anchor_provenance=(
                           "original_train_target_promoted_to_synthetic_anchor"
                           if role=="inverse" else "original_public_named_anchor")),
            label=dict(
                # Target contour NEVER passed to model input.
                target_contour=[list(map(float,p)) for p in tgt_poly.exterior.coords[:-1]],
                target_id=str(tgt_id),truth=not cf,synthetic_counterfactual=cf),
            source=dict(source,orientation=role,frame="map_xy_not_geographic",
                        direction_basis="whole_polygon_strict_separation",
                        no_original_instruction_rewrite_claim=True),
        )
    forward=instruction(target_type,anchor_name,word)
    backward=instruction(anchor_type,"the "+target_type,inv)
    # Counterfactual's *original* target region is a negative, never a new GT.
    return [
        row("forward",forward,word,anchor_id,target_id,anchor,target),
        row("inverse",backward,inv,target_id,anchor_id,target,anchor),
        row("forward",instruction(target_type,anchor_name,inv),inv,
            anchor_id,target_id,anchor,target,True),
        row("inverse",instruction(anchor_type,"the "+target_type,word),word,
            target_id,anchor_id,target,anchor,True),
    ]


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def build_dataset(static_path,objects_path,out_dir,*,max_pairs=None,margin_m=5.):
    path=Path(out_dir)
    if path.exists():raise FileExistsError("Choose new output; do not overwrite")
    catalog=json.loads(Path(objects_path).read_text(encoding="utf8"))
    records=[json.loads(s) for s in Path(static_path).read_text(encoding="utf8").splitlines() if s.strip()]
    if not records:raise ValueError("No TRAIN static query records")
    # Reject nonstatic/processed Val tasks, and never operate on test splits.
    if any(x.get("teacher_future") is not None or x.get("uav_pose") is not None for x in records):
        raise ValueError("Use one STATIC-only TRAIN record per query, not trajectory-sampled data")
    unique={};counts=Counter();map_pairs=defaultdict(list)
    for r in records:
        counts["source_train_queries"]+=1
        key=r.get("episode_id","")
        try:
            map_name=r["map_name"]
            if map_name not in catalog or not key.startswith(map_name+":"):
                counts["bad_source_id"]+=1;continue
            target_id=key[len(map_name)+1:].split(":")[0]
            target=catalog[map_name].get(str(target_id))
            if target is None:
                counts["target_id_unmatched"]+=1;continue
            if validated_polygon(target) is None:
                counts["target_polygon_missing"]+=1;continue
            bounds=r["world_bounds"]
            if len(bounds)!=4:
                counts["missing_world_bounds"]+=1;continue
            by_name=defaultdict(list)
            for lm in r.get("landmarks",[]):
                if lm.get("name") and lm.get("id"):
                    by_name[str(lm["name"]).strip().casefold()].append(str(lm["id"]))
            for lm in r.get("landmarks",[]):
                aid=str(lm.get("id",""))
                nm=str(lm.get("name","")).strip()
                if not aid or not nm or len(set(by_name[nm.casefold()]))!=1:
                    counts["anchor_alias_ambiguous"]+=1;continue
                pairkey=family_id(map_name,target_id,aid)
                if pairkey in unique:
                    counts["duplicate_pair_source_query"]+=1;continue
                anchor=catalog[map_name].get(aid)
                if anchor is None:
                    counts["anchor_id_missing"]+=1;continue
                quartet=create_pair(map_name,key,target_id,target,aid,anchor,bounds,margin_m=margin_m)
                if quartet is None:
                    counts["no_single_strict_map_direction_or_invalid_polygon"]+=1;continue
                unique[pairkey]=quartet
                map_pairs[map_name].append(pairkey)
                counts["accepted_pairs"]+=1
        except (ValueError,KeyError,TypeError):
            counts["invalid_query"]+=1
    if not unique:raise ValueError("No strict target+anchor Polygon pairs; inspect provenance and source")
    maps=sorted(map_pairs)
    if len(maps)<2:
        raise ValueError("Need at least two source maps for map-disjoint Fit/Dev")
    # Stable map-disjoint split; if each map has a very different number of
    # source pairs, report the distribution rather than choosing by GT quality.
    dev_map=maps[int(hashlib.sha256(b"direction-map-heldout-v1").hexdigest(),16)%len(maps)]
    dev_maps={dev_map}
    fit_maps=set(maps)-dev_maps
    groups=sorted(unique)
    if max_pairs is not None:
        if max_pairs<1:raise ValueError("max_pairs must be >=1")
        groups=groups[:max_pairs]
    outputs={"fit":[],"dev":[]}
    for group in groups:
        quartet=unique[group]
        part="dev" if quartet[0]["map_name"] in dev_maps else "fit"
        outputs[part].extend(quartet)
    if not outputs["fit"] or not outputs["dev"]:
        raise ValueError("Fit/Dev empty: increase max_pairs or use full source")
    path.mkdir(parents=True)
    for kind,rows in outputs.items():
        with (path/(kind+".jsonl")).open("w",encoding="utf8") as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    inventory=dict(
        dataset="synthetic_world_polygon_inverse_directions_v1",
        source_static_sha256=_sha(static_path),source_objects_sha256=_sha(objects_path),
        source_query_count=len(records),number_of_unique_pairs=len(unique),
        selected_pairs=len(groups),
        fit_samples=len(outputs["fit"]),dev_samples=len(outputs["dev"]),
        fit_maps=sorted(fit_maps),dev_maps=sorted(dev_maps),
        training_target_contour_is_private_label=True,
        inverse_promotes_source_target_to_synthetic_anchor=True,
        positives_and_negative_counterfactuals_per_pair=[2,2],
        direction_labels=list(DIR),world_north_verified=False,
        front_behind_supported=False,original_instruction_preserved=False,
        counts=dict(counts),
        category_counts=dict(Counter(
            x["input"]["relation_phrase"] for x in outputs["fit"] if x["label"]["truth"]
        )),
        training_protocol="no_val_test_no_closedloop_claims",
    )
    (path/"inventory.json").write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding="utf8")
    return inventory
