"""New-region TRAIN labels; preserve V3 labels and public-input separation.

This deliberately leaves the frozen V3 builder unchanged. Supervision retains
the same definitions, replacing only its single-contour centroid assumption.
"""
import torch

from .reference_geometry import public_reference_center
from .reference_rebuild import PUBLIC_SPATIAL_KEYS
from .relation_v3_data import NETWORK_KEYS
from .relation_v3_fusion import spatial_utility_labels


def build_reference_training_row(raw, *, target_kind, real):
    if set(raw['public']) != {'network','spatial'} or set(raw['public']['spatial'])-PUBLIC_SPATIAL_KEYS:
        raise ValueError('Unexpected field in public spatial input')
    source = raw['public']['network']
    if set(source)-NETWORK_KEYS-{'a0_context','box_kind'} or not NETWORK_KEYS <= set(source):
        raise ValueError('Unexpected/missing public network input')
    network = {key: source[key] for key in NETWORK_KEYS}
    if any(not torch.is_tensor(value) for value in network.values()):
        raise ValueError('Network inputs must be public tensors')
    n,l = len(network['box_rgb']),len(network['link_valid'])
    candidates,references = raw['meta']['candidates'],raw['meta']['references']
    if len(candidates)>n or len(references)>l:
        raise ValueError('Public candidate/reference axes changed')
    kind = torch.zeros(n,dtype=torch.long)
    for index,candidate in enumerate(candidates):
        if candidate['kind'] not in (0,1):
            raise ValueError('Unsupported public detector category')
        kind[index] = candidate['kind']
    network['box_kind'] = kind
    target = raw['labels']['target_xy'].float().clone()
    if target.shape != (2,) or not torch.isfinite(target).all():
        raise ValueError('Invalid private TRAIN target')
    mentions = network['mention_mask'].bool() & network['token_valid'][None].bool()
    ref_valid = network['link_valid'].bool() & mentions.any(-1) & network['name_valid'].bool().any(-1)
    if ref_valid[len(references):].any():
        raise ValueError('Verified reference has no public geometry')
    reliable = network['box_valid'].bool() & (network['pair_valid'].bool() & ref_valid[None]).any(-1)
    spatial = raw['public']['spatial']
    usable = reliable & spatial['candidate_support'].bool().any(-1)
    geometrical = spatial_utility_labels({key:value[None] for key,value in spatial.items()},target[None])[0]
    compatible = kind == int(target_kind) if target_kind in (0,1) else torch.ones(n,dtype=torch.bool)
    angle,distance,valid = torch.zeros(l,2),torch.zeros(l),torch.zeros(l,dtype=torch.bool)
    for index,reference in enumerate(references):
        center = torch.as_tensor(public_reference_center(reference),dtype=target.dtype)
        delta = target-center
        length = delta.norm()
        angle[index] = delta/length.clamp_min(1e-6)
        distance[index] = torch.log1p(length)/torch.log1p(torch.tensor(40.))
        valid[index] = bool(length>1 and ref_valid[index])
    labels = dict(target_xy=target,target_kind=torch.tensor(int(target_kind)),angle=angle,
        angle_valid=valid,distance=distance,localization_utility=geometrical & compatible & usable,
        utility_valid=usable,geometric_utility=geometrical & usable,
        identity_label_mask=torch.zeros(n,dtype=torch.bool),sample_weight=torch.tensor(8. if real else 1.),
        source_is_real=torch.tensor(bool(real)))
    return dict(public=dict(network=network,spatial=spatial),labels=labels)
