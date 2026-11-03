"""Build reference-aligned inputs from public hypotheses and named map only."""
import numpy as np
import torch

from .candidate_geometry import relative_entity_features
from .relation_factor_roles import RELATION_IDS, RELATIONS


def public_reference_inputs(*, world_hypotheses, global_cells, quality,
                            role, text_vectors, map_objects, capacity=8):
    observed = world_hypotheses[global_cells:].numpy()
    count = len(observed)
    dimension = text_vectors.shape[-1]
    references = role['references']
    if len(references) > capacity or len(text_vectors) != len(references):
        raise ValueError('reference vector/role alignment mismatch')
    inputs = dict(reference_text=torch.zeros(capacity, dimension, dtype=torch.float16),
                  relation_ids=torch.zeros(capacity, dtype=torch.long),
                  reference_valid=torch.zeros(capacity, dtype=torch.bool),
                  reference_geometry=torch.zeros(capacity, count, 31, dtype=torch.float16),
                  reference_map_types=torch.zeros(capacity, len(RELATIONS)),
                  reference_support=torch.zeros(capacity, count, dtype=torch.bool))
    for index, reference in enumerate(references):
        inputs['reference_text'][index] = text_vectors[index]
        inputs['relation_ids'][index] = RELATION_IDS[reference['kind']]
        inputs['reference_valid'][index] = True
        geometry = []; kinds = []
        for anchor in reference['anchors']:
            entities = []
            for oid in anchor['public_ids']:
                obj = map_objects[str(oid)]
                # Only IDs obtained from public name mentions are supplied.
                polygon = np.asarray(obj['contour'], np.float32)
                if polygon.ndim == 2 and polygon.shape[-1] == 2 and len(polygon) >= 3 and np.isfinite(polygon).all():
                    entities.append(dict(kind='region', geometry_m=polygon))
            if not entities:
                continue
            features = relative_entity_features(observed, entities, len(entities))
            geometry.extend(features[:, i] for i in range(len(entities)))
            kinds.append(RELATION_IDS[anchor['kind']])
        if geometry:
            pair = np.stack(geometry, axis=1)
            pooled = np.concatenate((pair.mean(1), pair.min(1), pair.max(1),
                                     np.full((count, 1), len(geometry)/16)), -1)
            inputs['reference_geometry'][index] = torch.from_numpy(pooled).half()
            for kind in kinds:
                inputs['reference_map_types'][index, kind] += 1/len(kinds)
        if reference['public_ids']:
            if geometry:
                # Map/RGB overlap is only support, never proof of visibility.
                boundary = np.stack(geometry, 1)[:, :, 5].min(-1)*400
                radius = quality[:, 3].numpy()*100*3
                inputs['reference_support'][index] = torch.from_numpy(boundary <= radius)
        else:
            inputs['reference_support'][index] = True
    return inputs
