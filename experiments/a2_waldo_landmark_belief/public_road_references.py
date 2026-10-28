"""Opt-in public-name road grouping; legacy V3 resolution remains unchanged."""
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy

from shapely.geometry import Polygon

from .data_protocol import _normalize, public_reference_records
from .public_regions import region_from_contours


RESOLVER_VERSION = 'connected-public-road-v1'


def _connected(polygons):
    reached = {0}
    remaining = set(range(1, len(polygons)))
    while remaining:
        adjacent = {i for i in remaining
                    if any(polygons[i].intersects(polygons[j]) for j in reached)}
        if not adjacent:
            return False
        reached.update(adjacent)
        remaining.difference_update(adjacent)
    return True


def _road_region(matches):
    kinds = {item.get('object_type', '') for item in matches}
    if len(matches) < 2 or 'TrafficRoad' not in kinds or not kinds <= {'TrafficRoad', 'Footpath'}:
        return None
    try:
        # Validate ALL member coordinates before asking Shapely about topology.
        region = region_from_contours([item['contour'] for item in matches])
        polygons = [Polygon(item['contour']) for item in matches]
    except (KeyError, TypeError, ValueError):
        return None
    return region if _connected(polygons) else None


def resolve_road_references(map_name: str, query_names: Sequence[str],
                            named_objects: Sequence[Mapping]) -> list[dict]:
    """Resolve exact public names, grouping only complete connected road sets.

    No target, private label, learned threshold or benchmark result is accepted.
    Group regions have no fake single contour: new consumers must understand
    their explicit region schema, making accidental legacy use fail loudly.
    """
    unique = {}
    for source in named_objects:
        item = deepcopy(dict(source))
        identity = int(item['id'])
        if identity in unique and unique[identity] != item:
            raise ValueError(f'Conflicting public object identity: {identity}')
        unique[identity] = item
    objects = [unique[i] for i in sorted(unique)]
    queries = {}
    for query in query_names:
        queries.setdefault(_normalize(query), str(query))
    refs = []
    for normalized, query in queries.items():
        matches = [item for item in objects if _normalize(item.get('name', '')) == normalized]
        region = _road_region(matches)
        if region is not None:
            ids = [int(item['id']) for item in matches]
            identity = json.dumps([str(map_name), ids], separators=(',', ':'))
            name = str(matches[0]['name'])
            refs.append(dict(map_name=map_name,
                object_id='public-road:'+hashlib.sha256(identity.encode()).hexdigest(),
                name=name, query_name=name, aliases=sorted({str(item['name']) for item in matches}),
                semantic_description='; '.join(sorted({item['object_type'] for item in matches})),
                resolver_version=RESOLVER_VERSION, member_ids=ids,
                member_contours=[deepcopy(item['contour']) for item in matches], region=region,
                height_policy='missing_for_public_road_group'))
        elif matches:
            refs.extend(dict(map_name=map_name, object_id=int(item['id']), name=item['name'],
                             query_name=query, contour=item['contour'],
                             semantic_description=item.get('object_type', '')) for item in matches)
        else:
            refs.append(dict(map_name=map_name, object_id=f'unresolved:{query}', name='',
                             query_name=query, contour=[], semantic_description=''))
    return public_reference_records(refs)
