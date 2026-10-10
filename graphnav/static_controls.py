"""Opt-in static ablations; original A3 forward and graph packing stay unchanged."""
import torch
from .semantic_geometry import EDGE_TYPES


def graph_control(inputs, control):
    if control == 'full':
        return inputs
    result = dict(inputs)
    adj = inputs['adjacency']
    eye = torch.eye(adj.shape[-1], device=adj.device, dtype=torch.bool)[None]
    if control == 'self':
        result['adjacency'] = adj & eye
    elif control == 'zero-geometry':
        # Preserve typed edges; isolate physical dx/dy/distance features.
        result['edges'] = inputs['edges'].clone()
        result['edges'][..., :8] = 0
    elif control == 'relation':
        edges = inputs['edges']
        language_links = (edges[..., 8+EDGE_TYPES['relation_anchor']] > 0) | (edges[..., 8+EDGE_TYPES['anchor_relation']] > 0)
        result['adjacency'] = adj & (eye | language_links)
    else:
        raise ValueError('unknown graph control')
    return result


def allowed_region_loss(logits, allowed, valid):
    """Penalize mass outside the UNION of legal solutions, not other legal cells.

    Masks are loss-only labels; no target or validation label enters forward.
    Each eligible query has equal mass, even when it contains several relations.
    """
    eligible = valid.bool() & allowed.any(-1) & ~allowed.all(-1)
    if not eligible.any():
        return logits.sum() * 0
    scores = logits[eligible].float()
    good = allowed[eligible]
    return (torch.logsumexp(scores, -1) - torch.logsumexp(scores.masked_fill(~good, -torch.inf), -1)).mean()
