"""Convert *already CityNav-aligned* projected XY road GeoJSON to road topology JSON.

GeoJSON coordinates MUST be in the SAME metric coordinates as CityNav map
bounds. EPSG:4326 lon/lat is NOT supported as input. No OSM map matching,
georeferencing, or target-based alignment is assumed. The input must already
have been aligned to the point-cloud map by the dataset producer.
"""

import argparse
import json
from pathlib import Path


def clip_segment(a, b, bounds):
    """Liang-Barsky clip of a polyline segment to a public CityNav world bbox."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = x1-x0, y1-y0
    xlo, ylo, xhi, yhi = bounds
    t0, t1 = 0., 1.
    for p, q in ((-dx,x0-xlo),(dx,xhi-x0),(-dy,y0-ylo),(dy,yhi-y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return None
    return [[x0+t0*dx,y0+t0*dy],[x0+t1*dx,y0+t1*dy]]


def convert_roads(features, map_bounds, single_map=None):
    """Only actual intersecting road polylines can become connected graph nodes."""
    result = {name: [] for name in map_bounds}
    for feature_idx, f in enumerate(features):
        props = f.get("properties", {})
        map_name = props.get("map_name") or single_map
        if map_name not in map_bounds:
            continue
        geom = f.get("geometry") or {}
        kind = geom.get("type")
        if kind == "LineString":
            lines = [geom.get("coordinates", [])]
        elif kind == "MultiLineString":
            lines = geom.get("coordinates", [])
        else:
            continue
        for j, line in enumerate(lines):
            for k, (a, b) in enumerate(zip(line[:-1], line[1:])):
                if len(a) < 2 or len(b) < 2:
                    continue
                clipped = clip_segment(a[:2], b[:2], map_bounds[map_name])
                if clipped is None:
                    continue
                if clipped[0] == clipped[1]:
                    continue
                road_id = str(props.get("osm_id", f.get("id", feature_idx)))
                result[map_name].append({
                    "id": f"{road_id}:{j}:{k}",
                    "name": str(props.get("name") or ""),
                    "polyline": clipped,
                    "coordinate_system": "world",
                })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geojson", required=True)
    p.add_argument("--bounds", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--map-name", default=None)
    p.add_argument("--aligned-citynav-world-xy", action="store_true", required=True,
                   help="Confirm source coordinates are already aligned meter XY, NOT lon/lat")
    args = p.parse_args()
    input_json = json.loads(Path(args.geojson).read_text(encoding="utf-8"))
    if input_json.get("type") != "FeatureCollection":
        raise ValueError("Expected FeatureCollection")
    if input_json.get("crs", {}).get("properties", {}).get("name", "").upper() in (
        "EPSG:4326", "OGC:CRS84",
    ):
        raise ValueError("GPS lon/lat cannot be silently treated as CityNav world XY")
    result = convert_roads(
        input_json["features"], json.loads(Path(args.bounds).read_text(encoding="utf-8")),
        args.map_name,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({name: len(items) for name, items in result.items()})


if __name__ == "__main__":
    main()
