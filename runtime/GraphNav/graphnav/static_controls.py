"""Opt-in static ablations; original A3 forward and graph packing stay unchanged."""
import torch
from .semantic_geometry import EDGE_TYPES, NODE_TYPES, DIRECTIONS


def graph_control(inputs, control, node_names=None):
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
    elif control in ('relation', 'reliable-relation'):
        edges = inputs['edges']
        language_links = (edges[..., 8+EDGE_TYPES['relation_anchor']] > 0) | (edges[..., 8+EDGE_TYPES['anchor_relation']] > 0)
        result['adjacency'] = adj & (eye | language_links)
        if control == 'reliable-relation':
            if node_names is None:
                raise ValueError('public node names required to validate relation links')
            reliable = torch.zeros_like(adj)
            for batch, names in enumerate(node_names):
                for index, kind in enumerate(names):
                    if inputs['node_types'][batch, index] != NODE_TYPES['relation']:
                        continue
                    anchors = ((edges[batch, index, :, 8 + EDGE_TYPES['relation_anchor']] > 0)
                               & adj[batch, index]).nonzero().flatten()
                    expected = 2 if kind == 'between' else 1
                    if kind not in (*DIRECTIONS, 'near', 'between', 'along') or len(anchors) != expected:
                        continue
                    if kind == 'along' and inputs['node_types'][batch, anchors[0]] != NODE_TYPES['road']:
                        continue
                    if kind == 'between' and torch.linalg.vector_norm(
                        inputs['nodes'][batch, anchors[0], :2] - inputs['nodes'][batch, anchors[1], :2]
                    ) < 1e-6:
                        continue
                    reliable[batch, index, anchors] = True
                    reliable[batch, anchors, index] = True
            result['adjacency'] = adj & (eye | (language_links & reliable))
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
