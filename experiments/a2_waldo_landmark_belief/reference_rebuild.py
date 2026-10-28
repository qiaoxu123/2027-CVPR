"""One public-only reference feature adapter for old caches and live perception.

Inputs contain no destinations or supervision labels. Reuse frozen image and
instruction features; regenerate actual name features and pair geometry.
Nothing imports this adapter in the existing V3 execution path.
"""
from copy import deepcopy
import time

import numpy as np
import torch

from .occurrence_mentions import occurrence_mention_spans
from .reference_geometry import box_reference_geometry


VERSION = 'public-reference-rebuild-v1'
PUBLIC_NETWORK_KEYS = {'instruction_tokens','token_valid','mention_mask','landmark_name_tokens',
    'name_valid','link_valid','box_rgb','box_valid','pair_geometry','pair_valid','offset_limit',
    'a0_context','box_kind'}
PUBLIC_SPATIAL_KEYS = {'base_logits','grid_world','candidate_xy','candidate_support','observed'}


def _identity(reference):
    return str(reference['map_name']), str(reference['object_id'])


def _missing_surface(reason):
    return dict(z=None, valid=False, coverage=0., visible_fraction=0., spread=None, reason=reason)


def rebuild_reference_pack(pack, meta, *, instruction, offsets, references, encode_names):
    """Rebuild public features without labels; callable frozen name encoder required.

    Calling this same adapter after legacy live perception gives newly restored
    singleton references the same missing-height policy as cache reconstruction.
    Only legacy-eligible singletons can reuse legacy observed surface summaries.
    """
    tick = time.perf_counter()
    if set(pack) != {'network', 'spatial'}:
        raise ValueError('Reference rebuilding accepts only public network/spatial inputs')
    source = pack['network']
    if set(source)-PUBLIC_NETWORK_KEYS:
        raise ValueError('Unexpected field in public network input')
    if set(pack['spatial'])-PUBLIC_SPATIAL_KEYS:
        raise ValueError('Unexpected field in public spatial input')
    if source['instruction_tokens'].ndim != 2 or len(offsets) != len(source['instruction_tokens']):
        raise ValueError('Reference offsets and frozen instruction token axis differ')
    if source['token_valid'].shape != (len(offsets),):
        raise ValueError('Frozen instruction token mask changed')
    masks, linked = occurrence_mention_spans(instruction, offsets, references)
    masks &= source['token_valid'].detach().cpu().numpy().astype(bool)[None,:]
    linked = masks.any(axis=1)
    selected = np.flatnonzero(linked)
    refs = [deepcopy(references[i]) for i in selected]
    if len(meta['references']) != len(meta['reference_surfaces']):
        raise ValueError('Legacy reference/surface axes differ')
    old_surfaces = {}
    for old, surface in zip(meta['references'], meta['reference_surfaces']):
        key = _identity(old)
        if key in old_surfaces:
            raise ValueError('Duplicate legacy reference surface identity')
        old_surfaces[key] = surface
    heights = []
    for ref in refs:
        if 'region' in ref:
            ref['height_policy'] = 'missing_for_public_road_group'
            heights.append(_missing_surface(ref['height_policy']))
        elif _identity(ref) in old_surfaces:
            heights.append(deepcopy(old_surfaces[_identity(ref)]))
        else:
            ref['height_policy'] = 'missing_newly_available_reference'
            heights.append(_missing_surface(ref['height_policy']))
    candidates = meta['candidates']
    n,l = len(candidates),len(refs)
    if len(source['box_rgb']) != max(n,1) or source['box_valid'].shape != (max(n,1),):
        raise ValueError('Frozen detector candidate axis changed')
    expected = torch.arange(max(n,1), device=source['box_valid'].device) < n
    if not torch.equal(source['box_valid'].bool(), expected):
        raise ValueError('Frozen detector candidate validity changed')
    scale = float(meta['metres_per_pixel'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid public camera footprint scale')
    pair, valid = box_reference_geometry([c['poly'] for c in candidates], refs,
        [c['confidence'] for c in candidates], box_height_m=[c['surface']['z'] for c in candidates],
        landmark_height_m=[s['z'] for s in heights])
    geometry = np.zeros((max(n,1),max(l,1),14), dtype=np.float32)
    geometry[:n,:l,:12] = pair
    geometry[:n,:l,12] = scale/40
    coverage = np.asarray([c['surface']['coverage'] for c in candidates], dtype=np.float32)
    if not np.isfinite(coverage).all() or np.any((coverage<0)|(coverage>1)):
        raise ValueError('Invalid observed candidate surface coverage')
    geometry[:n,:l,13] = coverage[:,None]
    pair_valid = np.zeros((max(n,1),max(l,1)), dtype=bool)
    pair_valid[:n,:l] = valid
    device = source['instruction_tokens'].device
    dimension = source['instruction_tokens'].shape[-1]
    if refs:
        names, name_valid = encode_names([r['name']+'; '+r.get('semantic_description','') for r in refs])
        if (not torch.is_tensor(names) or not torch.is_tensor(name_valid) or names.ndim != 3
            or names.shape[0] != l or names.shape[2] != dimension or names.shape[1] < 1
            or name_valid.shape != names.shape[:2] or not torch.isfinite(names).all()
            or not name_valid.bool().any(-1).all()):
            raise ValueError('Frozen name feature encoder returned misaligned/invalid features')
        mention = masks[selected]
    else:
        names = torch.zeros(1,1,dimension)
        name_valid = torch.zeros(1,1,dtype=torch.bool)
        mention = np.zeros((1,len(offsets)),dtype=bool)
    result, info = deepcopy(pack),deepcopy(meta)
    result['network'].update(
        mention_mask=torch.as_tensor(mention,device=device,dtype=torch.bool),
        landmark_name_tokens=names.detach().to(device=device,dtype=torch.float32).clone(),
        name_valid=name_valid.detach().to(device=device,dtype=torch.bool).clone(),
        link_valid=torch.as_tensor(np.arange(max(l,1))<l,device=device,dtype=torch.bool),
        pair_geometry=torch.as_tensor(geometry,device=device),
        pair_valid=torch.as_tensor(pair_valid,device=device))
    elapsed = time.perf_counter()-tick
    info.update(references=refs,reference_surfaces=heights,public_reference_records=len(references),
                valid_references=l,reference_rebuild_version=VERSION,reference_rebuild_seconds=elapsed,
                extraction_seconds=float(meta.get('extraction_seconds',0.))+elapsed)
    return result, info
