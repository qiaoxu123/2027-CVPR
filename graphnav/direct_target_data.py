"""Public geometry features and batching; labels never enter network kwargs."""
import numpy as np
import shapely
import torch
from .landmark_region_supervision import named_union
from .full_experiment import new_model
from .unified_pgvb import UnifiedPGVB
from types import SimpleNamespace

STATIC_KEYS = ('nodes', 'node_types', 'node_valid', 'entity_valid', 'adjacency',
               'edges', 'prior', 'prior_buildings', 'rules', 'relation_fields',
               'pair_geometry', 'grid_xy')


def relation_geometry(task, grid_xy):
    bounds = np.asarray(task['world_bounds'])
    world = bounds[:2]+np.asarray(grid_xy)*(bounds[2:]-bounds[:2])
    points = shapely.points(world)
    output = []
    for ref in task['anchors']:
        poly = named_union(ref['contours'])
        x0, y0, x1, y1 = poly.bounds
        dx = (world[:, 0]-poly.centroid.x)/100
        dy = (world[:, 1]-poly.centroid.y)/100
        distance = shapely.distance(points, poly)/100
        inside = shapely.covers(poly, points).astype(float)
        output.append(np.stack((dx, dy, distance, inside,
            (world[:, 0]-x1)/100, (x0-world[:, 0])/100,
            (world[:, 1]-y1)/100, (y0-world[:, 1])/100,
            np.full(len(world), (x1-x0)/100),
            np.full(len(world), (y1-y0)/100)), -1).astype(np.float32))
    return torch.from_numpy(np.stack(output))


def load_model(bundle, device='cuda'):
    geometry = new_model(SimpleNamespace(**bundle['config']['geometry_args']), torch.device(device))
    del geometry.vision_proj; del geometry.visual_score
    trained = bundle.get('format') == 'unified-pgvb-direct-target-v1'
    model = UnifiedPGVB(geometry=geometry, direct_target=True).to(device)
    result = model.load_state_dict(bundle['model'], strict=False)
    if result.unexpected_keys or (trained and result.missing_keys) or any(
            not k.startswith('direct_target.') for k in result.missing_keys):
        raise ValueError('Checkpoint contains incompatible integrated weights')
    model.direct_target_ready = trained and bundle.get('direct_target_ready') is True
    return model


def batch_inputs(samples, device='cuda', visual=True):
    """Samples have public `inputs` and private `label`; whitelist is explicit."""
    qs = [r['query'] for r in samples]
    static = {k: torch.stack([q['static'][k] for q in qs]).to(device) for k in STATIC_KEYS}
    max_tokens = max(q['tokens'].shape[0] for q in qs)
    tokens = torch.zeros(len(qs), max_tokens, 768)
    mask = torch.zeros(len(qs), max_tokens, dtype=torch.bool)
    max_refs = max(len(q['relation_ids']) for q in qs)
    geometry = torch.zeros(len(qs), max_refs, 784, 10)
    ids = torch.zeros(len(qs), max_refs, dtype=torch.long)
    valid = torch.zeros(len(qs), max_refs, dtype=torch.bool)
    for i, q in enumerate(qs):
        n = len(q['tokens']); tokens[i, :n] = q['tokens']; mask[i, :n] = q['mask']
        n = len(q['relation_ids']); geometry[i, :n] = q['relation_geometry']
        ids[i, :n] = q['relation_ids']; valid[i, :n] = True
    observed = None
    if visual and any(s.get('observed') is not None for s in samples):
        template = next(s['observed'] for s in samples if s.get('observed') is not None)
        observed = {k: torch.stack([s['observed'][k] if s.get('observed') is not None
                                   else torch.zeros_like(v) for s in samples]).to(device)
                    for k, v in template.items()}
        for k in observed:
            if observed[k].is_floating_point(): observed[k] = observed[k].float()
    return dict(static_inputs=static, tokens=tokens.to(device), token_mask=mask.to(device),
                node_emb=torch.stack([q['nodes'] for q in qs]).float().to(device),
                relation_geometry=geometry.to(device), relation_ids=ids.to(device),
                relation_valid=valid.to(device), observed=observed,
                target_text=torch.stack([q['target_text'] for q in qs]).float().to(device))
