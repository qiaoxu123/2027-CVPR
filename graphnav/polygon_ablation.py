"""Opt-in, parameter-free geometry views for the Polygon contribution experiment.

Nothing changes the defaults of the archived datasets or networks. Targets,
trajectories and observations are never read by these representation functions.
"""
from copy import deepcopy
import numpy as np
import torch

from .candidate_geometry import relative_entity_features
from .semantic_geometry import (DIRECTIONS, geometry_distance, point_inside_polygon,
                                _range_between, prior_and_rule_fields)

REPRESENTATIONS = ('centroid', 'bounding_box', 'full_polygon')
SCOPES = ('descriptor_only', 'end_to_end')
PAIR_MASKS = {
    'centroid': (1, 1, 1, 1, 1, 0, 0, 0, 0, 1),
    'bounding_box': (1,) * 10,
    'full_polygon': (1,) * 10,
    'no_boundary': (1, 1, 1, 1, 1, 0, 1, 1, 1, 1),
    'no_inside': (1, 1, 1, 1, 1, 1, 0, 1, 1, 1),
    'no_extent': (1, 1, 1, 1, 1, 1, 1, 0, 0, 1),
    'no_inside_information': (1, 1, 1, 1, 1, 1, 0, 1, 1, 1),
    'prior_to_centroid': (1,) * 10,
    'rules_to_centroid': (1,) * 10,
    'nodes_to_centroid': (1,) * 10,
}


def center(entity):
    """Original vertex arithmetic mean, including any supplied closing vertex."""
    return np.asarray(entity['geometry_m'], np.float32).mean(0)


def rectangle(entity):
    p = np.asarray(entity['geometry_m'], np.float32)
    lo, hi = p.min(0), p.max(0)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                     [hi[0], hi[1]], [lo[0], hi[1]]], np.float32)


def pair_features(points, entities, capacity, representation='full_polygon', base_pair=None):
    if representation not in PAIR_MASKS:
        raise ValueError(representation)
    result = (relative_entity_features(points, entities, capacity) if base_pair is None
              else np.asarray(base_pair, np.float32).copy())
    mask = np.broadcast_to(np.asarray(PAIR_MASKS[representation], np.float32),
                           (capacity, 10)).copy()
    if representation == 'bounding_box':
        for i, entity in enumerate(entities):
            if entity['kind'] == 'road':
                # AABB cannot recover an open centerline. Do not invent one or
                # close it into a polygon. Its boundary/inside channels are unavailable.
                mask[i, 5:7] = 0
                continue
            box = rectangle(entity)
            boundary = geometry_distance(points, np.vstack([box, box[0]]), is_road=True)
            inside = point_inside_polygon(points, box).astype(np.float32)
            result[:, i, 5] = np.where(inside > 0, -boundary, boundary) / 400.
            result[:, i, 6] = inside
    if representation == 'no_inside_information':
        result[..., 5] = np.abs(result[..., 5])
    mask[len(entities):] = 0
    return result * mask[None], mask


def region_distance(points, entity, representation):
    if representation == 'full_polygon':
        return geometry_distance(points, entity['geometry_m'], entity['kind'] == 'road')
    if representation == 'centroid' or entity['kind'] == 'road':
        return np.linalg.norm(np.asarray(points) - center(entity), axis=-1)
    return geometry_distance(points, rectangle(entity), is_road=False)


def representation_fields(points, entities, relations, representation, *, sigma_m=25.):
    """Same rule equations, with only geometry permitted by the representation.

    All anchors use the SAME original vertex-mean point. For BBox this public
    shared centroid is retained in addition to the axis-aligned bounding box;
    changing center conventions would confound shape with anchor location.
    A true centerline's along/near support is approximated by its point when
    unavailable, never by a fabricated or closed line. Source type stays fixed.
    """
    if representation == 'full_polygon':
        return prior_and_rule_fields(points, entities, relations, sigma_m=sigma_m)
    if representation not in REPRESENTATIONS:
        raise ValueError(representation)
    prior = (-np.stack([region_distance(points, e, representation) for e in entities], 1).min(1)
             / max(sigma_m, 1.)) if entities else np.zeros(len(points), np.float32)
    fields = []
    for relation in relations:
        if relation.kind == 'between':
            values = [-_range_between(points, center(entities[a]), center(entities[b]))[0] / max(sigma_m, 1.)
                      for a in relation.anchors[0] for b in relation.anchors[1]]
        elif relation.kind in DIRECTIONS:
            direction = np.asarray(DIRECTIONS[relation.kind], np.float32)
            direction /= np.linalg.norm(direction)
            values = [-np.logaddexp(0., -((points - center(entities[i])) @ direction) / max(sigma_m, 1.))
                      for i in relation.anchors[0]]
        else:
            scale = sigma_m * (1.7 if relation.kind == 'along' else 1.)
            values = [-region_distance(points, entities[i], representation) / max(scale, 1.)
                      for i in relation.anchors[0]]
        fields.append(np.max(values, 0) if values else np.zeros(len(points)))
    rel = np.stack(fields, -1).astype(np.float32) if fields else np.zeros((len(points), 0), np.float32)
    return prior.astype(np.float32), rel.sum(-1), rel


def node_features(original, entities, bounds, representation):
    features = np.array(original, copy=True)
    mask = np.ones_like(features, dtype=np.float32)
    if representation == 'full_polygon':
        return features, mask
    span = np.asarray(bounds[2:], np.float32) - np.asarray(bounds[:2], np.float32)
    for i, e in enumerate(entities):
        if representation == 'centroid':
            mask[i, 2:8] = 0
            mask[i, 9] = 0
        elif representation == 'bounding_box':
            xy = rectangle(e)
            wh = xy.max(0) - xy.min(0)
            features[i, 2:4] = wh / span
            if e['kind'] == 'road':
                mask[i, 4:8] = 0
                mask[i, 9] = 0
            else:
                features[i, 4] = 2 * wh.sum() / 400.
                features[i, 5] = wh.prod() / 400.**2
                features[i, 6:8] = wh / max(float(np.linalg.norm(wh)), 1e-6)
                features[i, 9] = .04
        else:
            raise ValueError(representation)
    return features * mask, mask


def geometry_view(item, graph, bounds, *, representation='full_polygon',
                  scope='descriptor_only', prior_sigma_m=25., rule_sigma_m=25.):
    """Transform every MODEL-FACING geometry path for end-to-end controls.

    Graph structure/edges are erased in ALL conditions, including Full Polygon:
    the requested geometry variant has no message passing or topology inputs.
    Public entity masks/names/types and relation membership remain unchanged.
    """
    if scope not in SCOPES:
        raise ValueError(scope)
    if representation in ('prior_to_centroid','rules_to_centroid','nodes_to_centroid'):
        full=geometry_view(item,graph,bounds,representation='full_polygon',scope='descriptor_only',prior_sigma_m=prior_sigma_m,rule_sigma_m=rule_sigma_m)
        point=geometry_view(item,graph,bounds,representation='centroid',scope='end_to_end',prior_sigma_m=prior_sigma_m,rule_sigma_m=rule_sigma_m)
        keys={'prior_to_centroid':('prior','prior_buildings'),'rules_to_centroid':('rules','relation_fields'),'nodes_to_centroid':('nodes','node_feature_mask')}[representation]
        full.update({k:point[k] for k in keys})
        return full
    output = dict(item)
    entities, relations = graph['entities'], graph['relations']
    lo, span = np.asarray(bounds[:2], np.float32), np.asarray(bounds[2:], np.float32)-np.asarray(bounds[:2], np.float32)
    world = lo + item['grid_xy'].numpy() * span
    pair, pmask = pair_features(world, entities, len(item['nodes']), representation,
                                base_pair=item['pair_geometry'].numpy())
    output['pair_geometry'] = torch.from_numpy(pair.astype(np.float16))
    output['pair_feature_mask'] = torch.from_numpy(pmask)
    output['node_feature_mask'] = torch.ones_like(item['nodes'])
    # All three arms keep identical node/name/type packing. Topology is unused.
    output['adjacency'] = torch.zeros_like(item['adjacency'])
    output['edges'] = torch.zeros_like(item['edges'])
    if scope == 'end_to_end' and representation != 'full_polygon':
        nodes, nmask = node_features(item['nodes'].numpy(), entities, bounds, representation)
        output['nodes'], output['node_feature_mask'] = torch.from_numpy(nodes), torch.from_numpy(nmask)
        prior_entities = [e for e in entities if e['kind'] != 'road']
        buildings = [e for e in entities if e['kind'] == 'building']
        for key, es in [('prior', prior_entities), ('prior_buildings', buildings)]:
            output[key] = torch.from_numpy(representation_fields(world, es, [], representation, sigma_m=prior_sigma_m)[0])
        _, rules, fields = representation_fields(world, entities, relations, representation, sigma_m=rule_sigma_m)
        padded = np.zeros_like(item['relation_fields'].numpy())
        padded[:, :fields.shape[1]] = fields
        output['rules'], output['relation_fields'] = torch.from_numpy(rules), torch.from_numpy(padded)
    return output


def geometry_groups(entities, relations):
    """Predeclared public-map strata; multi-type flags can overlap for a query."""
    from shapely.geometry import Polygon
    from .semantic_geometry import canonical
    aspects, fills, areas = [], [], []
    kinds, names = set(), set()
    invalid = 0
    for e in entities:
        names.add(canonical(e['name']))
        if e['kind'] == 'road':
            kinds.add('road_like')
            continue
        polygon = Polygon(e['geometry_m'])
        if not polygon.is_valid or polygon.area <= 0:
            invalid += 1
            continue
        box = polygon.minimum_rotated_rectangle
        sides = np.linalg.norm(np.diff(np.asarray(box.exterior.coords), axis=0), axis=1)
        aspect = float(sides.max() / max(sides.min(), 1e-9))
        areas.append(polygon.area)
        aspects.append(aspect)
        fills.append(polygon.area / max(box.area, 1e-9))
        kinds.add('road_like' if e['kind'] == 'road_surface' else 'point_like' if polygon.area <= 25. else 'area_like')
    result = ['single_landmark_name' if len(names) == 1 else 'multiple_landmark_names' if len(names)>1 else 'no_landmark']
    result += sorted(kinds)
    if areas and not invalid:
        result += ['irregular' if min(fills) < .85 else 'regular',
                   'elongated' if max(aspects) >= 3. else 'compact',
                   'large' if max(areas) > 1000. else 'small']
    else:
        result += ['shape_unknown']
    reliable = [r for r in relations if all(len(group)==1 for group in r.anchors)]
    if reliable:
        result.append('explicit_bound_relation')
    if len(reliable) >= 2:
        result.append('multiple_explicit_bound_relations')
    return result
