"""Standalone observed-domain visual belief; no geometry prior or polygon input.

Public coordinates map current image evidence onto the world grid. Unknown
unobserved cells have zero conditional mass, not evidence of target absence.
An empty observation abstains rather than returning a fake uniform prediction.
"""
import torch
from .world_likelihood_evidence import world_evidence_logits


def visual_only_world_belief(*, local_uv, scores, valid, grid_size=28):
    uniform=torch.zeros((len(scores),grid_size*grid_size),device=scores.device,dtype=torch.float32)
    baseline=uniform.log_softmax(-1)
    energy=world_evidence_logits(uniform,local_uv,scores,valid)-baseline
    support=world_evidence_logits(uniform,local_uv,torch.ones_like(scores),valid)-baseline
    covered=support>.5
    has_view=covered.any(-1)
    logits=(baseline+energy).masked_fill(~covered,float('-inf'))
    safe=torch.where(has_view[:,None],logits,uniform)
    probabilities=safe.softmax(-1)*has_view[:,None]
    return dict(logits=logits,probabilities=probabilities,covered=covered,has_view=has_view)
