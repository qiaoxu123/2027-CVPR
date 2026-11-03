"""Geometry-only support and strict offline bounds for local world evidence."""
import torch
from .world_likelihood_evidence import world_evidence_logits


def local_evidence_coverage(local_uv,local_valid,global_cells):
    base=torch.zeros(len(local_uv),global_cells,device=local_uv.device)
    updated=world_evidence_logits(base,local_uv,torch.ones_like(local_valid,dtype=torch.float32),local_valid)
    return updated>base.log_softmax(-1)


def position_support_bound(prior_logits,coverage,success_mask,*,diagnostic=False):
    if diagnostic is not True:raise ValueError('GT success region is offline diagnostic only')
    if prior_logits.ndim!=1 or coverage.shape!=prior_logits.shape or success_mask.shape!=coverage.shape or coverage.dtype!=torch.bool or success_mask.dtype!=torch.bool:
        raise ValueError('aligned one-dimensional prior, coverage and diagnostic success mask required')
    if not success_mask.any():return dict(possible=False,kind='no-success-grid-cell')
    if (coverage&success_mask).any():return dict(possible=True,kind='covered-success-cell-optimistic')
    outside=~coverage
    good=outside&success_mask
    if good.any() and prior_logits[good].max()>=prior_logits[outside].max():
        return dict(possible=True,kind='unchanged-prior-best-correct')
    return dict(possible=False,kind='immutable-uncovered-prior-order')
