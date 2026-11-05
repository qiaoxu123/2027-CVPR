"""CLI: create map-disjoint TRAIN-only inverse directional Polygon pairs."""
import argparse
import json
from pathlib import Path

from graphnav.inverse_polygon_direction_data import build_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-static",type=Path,required=True,
                   help="Output of prepare_citynav.py --static-only for TRAIN only")
    p.add_argument("--objects",type=Path,required=True,
                   help="CityRefer objects.json from same TRAIN source")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-pairs",type=int,default=None,
                   help="Optional smoke, no randomly changing split")
    p.add_argument("--margin-m",type=float,default=5.)
    a=p.parse_args()
    stats=build_dataset(a.train_static,a.objects,a.output,
                        max_pairs=a.max_pairs,margin_m=a.margin_m)
    print(json.dumps(stats,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
