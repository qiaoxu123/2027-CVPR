"""Strict online public feature boundary, independent of private label code."""
import torch
SCHEMA_VERSION=1
NETWORK_KEYS={'tokens','token_mask','candidate_rgb','reference_rgb','names','name_mask',
    'candidate_depth','reference_depth','candidate_geometry','reference_geometry','pair','triad','triad_ids','offset_limit'}


def validate_public(public):
    if set(public)!={'network','spatial'}:raise ValueError('Unexpected public envelope keys')
    if set(public['network'])!=NETWORK_KEYS:raise ValueError('Unexpected/private network keys')
    permitted={'grid_world','base_logits','candidate_xy','candidate_support','local_mask','candidate_base','offset_supported'}
    if not set(public['spatial'])<=permitted:raise ValueError('Unexpected/private spatial keys')
    for group in public.values():
        for name,v in group.items():
            if not torch.is_tensor(v) or not torch.isfinite(v).all():raise ValueError(f'Invalid public tensor: {name}')
    return public


def to_device(public,device):return {k:v.to(device) for k,v in public['network'].items()}
