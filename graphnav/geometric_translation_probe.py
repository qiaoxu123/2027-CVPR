"""Public-only paired translations for offline Geometry equivariance checks."""
import torch


def translate_observable(batch, offset):
    """Translate queries and valid map-node centres in normalized map units.

    Widths, lengths, pair distances, boundary depths, directions and rule fields
    are unchanged when queries and the whole public geometry move together.
    Padding stays zero. Caller compares only the same in-bounds queries in both
    arms, so newly unavailable canvas strips cannot explain paired differences.
    No target, inferred best offset or model weight changes are accepted.
    """
    result=dict(batch)
    delta=torch.as_tensor(offset,device=batch['grid_xy'].device,dtype=batch['grid_xy'].dtype)
    result['grid_xy']=batch['grid_xy']+delta
    result['nodes']=batch['nodes'].clone()
    result['nodes'][...,:2]+=delta*batch['node_valid'][...,None]
    valid=((result['grid_xy']>=0)&(result['grid_xy']<=1)).all(-1)
    return result,valid


def average_aligned_logits(base, branches):
    """Fixed uniform nuisance averaging, with public common-query gauge alignment.

    Out-of-canvas branch queries are unknown, not negative scores. The original
    field always contributes, and a query receives only available branch votes.
    An exactly equivariant scorer (up to additive constants) stays unchanged.
    """
    total=base.float().clone();count=torch.ones_like(total)
    for logits,valid in branches:
        difference=logits.float()-base.float()
        offset=(difference*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        total+=torch.where(valid,logits.float()-offset[:,None],torch.zeros_like(total))
        count+=valid
    return total/count
