"""Prepare valid reversible geometry supervision from Revised CityRefer objects.

Run separately for *train_seen* and *val_unseen* map allowlists.
Does NOT convert full CityFlight trajectories into fictitious reverse VLN.
"""

import argparse
import hashlib
import json
from pathlib import Path

from graphnav.bidirectional import generate_pairs, split_map_names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objects", required=True)
    p.add_argument("--bounds", required=True)
    p.add_argument("--source-split", required=True,
                   help="same-revision, previously verified CityNav split JSONL")
    p.add_argument("--output", required=True)
    p.add_argument("--max-objects-per-map", type=int, default=300)
    p.add_argument("--max-pairs-per-map", type=int, default=800)
    p.add_argument("--shift-augmentations", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sources = {
        "objects": Path(a.objects), "bounds": Path(a.bounds),
        "source_split": Path(a.source_split),
    }
    raw = json.loads(sources["objects"].read_text(encoding="utf-8"))
    bounds = json.loads(sources["bounds"].read_text(encoding="utf-8"))
    allowed_maps = split_map_names(sources["source_split"])
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, pairs = 0, set()
    with out.open("w", encoding="utf-8") as handle:
        for rec in generate_pairs(
            raw, bounds, allowed_maps,
            max_objects_per_map=a.max_objects_per_map,
            max_pairs_per_map=a.max_pairs_per_map, seed=a.seed,
            augment_shift_count=a.shift_augmentations,
        ):
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            pairs.add(rec["pair_id"])
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
              for name,path in sources.items()}
    manifest = dict(samples=n, pairs=len(pairs),
                    allowed_source_maps=sorted(allowed_maps),
                    source_sha256=hashes, seed=a.seed,
                    augmentation_count=a.shift_augmentations,
                    note="Reverse samples invert verified geometry, not CityFlight paths")
    Path(str(out)+".manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    print(json.dumps({"samples": n, "pairs":len(pairs),"maps":len(allowed_maps)}))


if __name__ == "__main__":
    main()
