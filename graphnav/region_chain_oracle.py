"""Strict offline TRAIN oracle, forbidden in deployable model forward."""
import torch


def oracle_region_rejection(global_world,local_world,private_region_labels,local_valid,patch_width_m,expected_region,*,diagnostic=False):
    if diagnostic is not True:raise ValueError('Private region labels require explicit offline diagnostic mode')
    if patch_width_m<=0 or not 0<=expected_region<3:raise ValueError('finite positive patch support and known region required')
    if not torch.isfinite(torch.as_tensor(patch_width_m)):raise ValueError('finite patch width required')
    if local_valid.dtype!=torch.bool or local_world.shape!=(len(local_valid),2) or private_region_labels.shape!=local_valid.shape:
        raise ValueError('aligned local world/validity/private region labels required')
    distance=torch.cdist(global_world.float(),local_world.float()).masked_fill(~local_valid[None],float('inf'))
    nearest,indices=distance.min(-1)
    label=private_region_labels[indices]
    # An inscribed circle guarantees each cell center lies within the native
    # one-patch observation window. Never extend labels to unobserved cells.
    supported=nearest<=patch_width_m/2
    return supported&(label>=0)&(label!=expected_region)
