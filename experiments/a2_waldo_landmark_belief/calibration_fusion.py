"""Offline calibration uses the exact public runtime gate and fusion formula."""
import math
import torch
from .observed_geometry import gaussian_belief_mix


def calibrated_logits(packet,parameters):
    alpha=float(parameters['alpha']);temperature=float(parameters['temperature']);threshold=float(parameters['threshold'])
    if not all(math.isfinite(v) for v in (alpha,temperature,threshold)) or not 0<=alpha<=.5 or temperature<=0 or not 0<=threshold<=1:
        raise ValueError('Invalid calibration parameters')
    s,out=packet['spatial'],packet['output']
    if not parameters['enabled'] or float(torch.sigmoid(out['match_logit']/temperature))<threshold:
        alpha=0.
    trusted=out['reliable'] & (torch.sigmoid(out['candidate_match_logits']/temperature)>=threshold)
    return gaussian_belief_mix(s['base_logits'],s['grid_world'],s['candidate_xy']+out['offset'],
                               out['scores'],s['candidate_support'],trusted,alpha)
