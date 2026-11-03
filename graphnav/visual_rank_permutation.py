"""Controlled ranking intervention: preserve frozen score magnitudes exactly.

The ordinary native-order function accepts only public visual scores. A separate
weak-box oracle requires explicit diagnostic authorization and is never wired
into normal inference.
"""
import torch
from graphnav.world_likelihood_evidence import world_evidence_logits


def native_rank_permutation(visual,native,valid):
    if visual.shape!=native.shape or valid.shape!=visual.shape or valid.dtype!=torch.bool:
        raise ValueError('aligned current visual/native scores and validity required')
    if visual.ndim!=1 or not torch.isfinite(visual).all() or not torch.isfinite(native).all():
        raise ValueError('one finite candidate set required')
    out=visual.clone();ids=valid.nonzero().flatten()
    order=ids[torch.argsort(native[ids],descending=True,stable=True)]
    out[order]=visual[ids].sort(descending=True).values
    return out


def weak_box_oracle_permutation(visual,positive,valid,*,diagnostic=False):
    if not diagnostic:raise PermissionError('weak-box oracle is offline diagnosis only')
    if positive.shape!=visual.shape or positive.dtype!=torch.bool:raise ValueError('aligned weak-box mask required')
    out=visual.clone();ids=valid.nonzero().flatten()
    old_order=ids[torch.argsort(visual[ids],descending=True,stable=True)]
    order=torch.cat((old_order[positive[old_order]],old_order[~positive[old_order]]))
    out[order]=visual[ids].sort(descending=True).values
    return out


def replace_visual_world_field(logprob,uv,old_visual,new_visual,valid):
    """Linearity of fixed bilinear evidence splats; all spatial terms stay fixed."""
    if logprob.ndim!=1:raise ValueError('one fixed world field required')
    uniform=torch.zeros_like(logprob)[None]
    delta=world_evidence_logits(uniform,uv[None],(new_visual-old_visual)[None],valid[None])-uniform.log_softmax(-1)
    return (logprob+delta[0]).log_softmax(-1)
