"""Frozen residual-input interventions; not a claim of fair retraining."""
import torch

def residual_input_mask(x,arm):
    if x.shape[-1]!=106:raise ValueError('Unexpected calibrated residual interface')
    y=x.clone()
    if arm=='geometry_pose':
        y[...,16:98]=0  # both RGB/text interactions and all image-quality channels
        y[...,100:102]=0  # current/history appearance scores
    elif arm=='rgb_text':
        y[...,:16]=0  # spatial-language candidate context
        y[...,98:100]=0  # static relative logits and belief probabilities
        y[...,104]=0  # pose-to-candidate distance
    else:raise ValueError(arm)
    # Shared observed-support indicators/gate remain: the test isolates semantic
    # RGB descriptors, not the physical ability to know which map cells were seen.
    return y

def install_input_intervention(policy,arm):
    if arm=='current_rgb':
        policy.use_history=False
        return
    head=policy.model.calibrated_residual
    head.network[0].register_forward_pre_hook(lambda module,args:(residual_input_mask(args[0],arm),))
