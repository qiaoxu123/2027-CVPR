"""Causal bounded residual averaging on a fixed world-coordinate grid."""
from collections import deque
import torch


class CausalResidualWindow:
    def __init__(self,window=1):
        if not isinstance(window,int) or not 1<=window<=32:
            raise ValueError('Residual history window must be an integer in [1,32]')
        self.frames=deque(maxlen=window)

    def reset(self):
        self.frames.clear()

    def update(self,delta,observed):
        if delta.shape!=observed.shape:raise ValueError('Residual/support shape mismatch')
        self.frames.append((delta.detach().clone(),observed.detach().bool().clone()))
        weights=torch.stack([mask.float() for _,mask in self.frames]).sum(0)
        return torch.stack([value*mask for value,mask in self.frames]).sum(0)/weights.clamp_min(1)
