"""Nine-parameter bounded matching calibration; all large towers stay frozen."""
import torch
from .inference import DEFAULT_WEIGHTS

class Matcher(torch.nn.Module):
    def __init__(self,rgb=True):
        super().__init__();self.rgb=rgb
        default=torch.tensor(DEFAULT_WEIGHTS);self.theta=torch.nn.Parameter(torch.log(torch.expm1(default)))
    def weights(self):
        w=torch.nn.functional.softplus(self.theta)
        if not self.rgb:w=w*torch.tensor([1,1,1,1,1,1,0,0,1],device=w.device)
        return torch.clamp(w,max=20)
