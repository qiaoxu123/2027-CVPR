"""Export public CityNav map bounds and ground levels from a local HETT data helper.

Uses only metadata constants. GraphNav has NO runtime or model dependency on HETT.
 python scripts/export_bounds.py --mapdata /path/to/HETT/multiagent/mapdata.py \
  --bounds-out data/citynav_bounds.json --ground-out data/ground_levels.json
"""

import argparse
import json
import runpy
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapdata", required=True)
    p.add_argument("--bounds-out", required=True)
    p.add_argument("--ground-out", required=True)
    args = p.parse_args()
    data = runpy.run_path(args.mapdata)
    bounds = {name: list(values) for name, values in data["MAP_BOUNDS"].items()}
    ground = dict(data["GROUND_LEVEL"])
    for path, payload in ((args.bounds_out, bounds), (args.ground_out, ground)):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported bounds for {len(bounds)} CityNav map blocks")


if __name__ == "__main__":
    main()
