"""Frozen-weight context intervention; reference priors/rules stay unchanged.

Only public map name/type/contour fields enter this adapter. The extra map
entities use the same largest-homonym catalog as SBF's contour channel.
No target location, object description, trajectory or learned parameter is used.
"""
import hashlib
import json
import numpy as np
import torch
from .candidate_geometry import relative_entity_features
from .semantic_geometry import build_entities, NODE_TYPES
from .semantic_graph import _physical_node_geometry


MODES = ('references', 'all_named', 'all_named_masked')


def polygon_key(name, contour):
    # Match the normalized source rings, independently of object IDs/targets.
    values = np.round(np.asarray(contour, np.float64), 7)
    return str(name), tuple(map(tuple, values))


def public_context(record, catalog):
    bounds = np.asarray(record['world_bounds'], np.float64)
    span = bounds[2:] - bounds[:2]
    seen = {polygon_key(lm['name'], lm['contour'])
            for lm in record['landmarks']}
    extras = []
    for landmark in sorted(catalog.all_landmarks(record['map_name']), key=lambda lm: (lm.name, lm.object_id)):
        source = catalog.objects[record['map_name']][landmark.object_id]
        contour = (np.asarray(landmark.contour, np.float64)-bounds[:2])/span
        if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3 or not np.isfinite(contour).all():
            continue
        key = polygon_key(landmark.name, contour)
        if key in seen:
            continue
        seen.add(key)
        extras.append(dict(name=landmark.name, object_type=source.get('object_type', 'Unknown'), contour=contour.tolist()))
    return extras


def append_context(inputs, names, record, extras, encoder, mode):
    if mode not in MODES:
        raise ValueError('unknown landmark context')
    if mode == 'references' or not extras:
        return inputs, names
    if getattr(inputs['grid_xy'], 'shape')[-2] != 28*28:
        raise ValueError('Landmark context diagnostic is fixed at 28x28')
    entities = build_entities(extras, [], record['world_bounds'])
    if not entities:
        return inputs, names
    result = dict(inputs)
    device = inputs['nodes'].device
    features = np.stack([_physical_node_geometry(e, record['world_bounds']) for e in entities])
    types = [NODE_TYPES[e['kind']] for e in entities]
    bounds = np.asarray(record['world_bounds'], np.float32)
    world = bounds[:2]+inputs['grid_xy'][0].detach().cpu().numpy()*(bounds[2:]-bounds[:2])
    pair = relative_entity_features(world, entities, len(entities)).astype(np.float16)
    result['nodes'] = torch.cat((inputs['nodes'], torch.as_tensor(features, device=device)[None]), 1)
    result['node_types'] = torch.cat((inputs['node_types'], torch.tensor(types, device=device)[None]), 1)
    valid = torch.ones((1, len(entities)), device=device, dtype=torch.bool)
    result['node_valid'] = torch.cat((inputs['node_valid'], valid), 1)
    result['entity_valid'] = torch.cat((inputs['entity_valid'], valid if mode == 'all_named' else ~valid), 1)
    result['pair_geometry'] = torch.cat((inputs['pair_geometry'], torch.as_tensor(pair, device=device)[None]), 2)
    node_names = [f"{e['source_type']} {e['name']}".strip() for e in entities]
    embeddings = encoder.encode(node_names).to(device).clone()[None]
    names = torch.cat((names, embeddings), 1)
    # Adjacency/edge tensors intentionally remain unused: V1 calls geometry,
    # which performs candidate-to-entity attention and no graph message passing.
    # Prior, rules and relation_fields are the original reference-only tensors.
    return result, names


def context_audit(record, extras, mode):
    payload = json.dumps(extras, sort_keys=True).encode()
    return dict(mode=mode, reference_polygons=len(record['landmarks']),
                extra_named_polygons=len(extras), extra_context_sha256=hashlib.sha256(payload).hexdigest(),
                source='SBF public all_landmarks (largest polygon per exact name), union original reference fragments',
                reference_prior_and_rules_preserved=True, retrained=False)
