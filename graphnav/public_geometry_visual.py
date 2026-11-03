"""Public map-conditioned global field plus observed-only visual likelihood."""
import torch
from torch import nn
from torch.nn import functional as F
from graphnav.world_likelihood_evidence import world_evidence_logits


class PublicGeometryVisualHead(nn.Module):
    grid_output=True
    def __init__(self,base,mode='conditioned',dimension=1024,hidden=32):
        super().__init__()
        if mode not in ('map','unconditioned','conditioned'):raise ValueError('explicit controlled mode required')
        self.base=base.eval().requires_grad_(False);self.mode=mode
        self.geometry=nn.Linear(33,hidden)
        self.instruction=nn.Linear(dimension,hidden,bias=False)
        self.target=nn.Linear(dimension,hidden,bias=False)
        self.image=nn.Linear(dimension*3,hidden,bias=False)
        self.map_score=nn.Linear(hidden,1,bias=False)
        self.visual_score=nn.Linear(hidden,1,bias=False)
        nn.init.zeros_(self.map_score.weight);nn.init.zeros_(self.visual_score.weight)

    def train(self,mode=True):
        super().train(mode);self.base.eval();return self

    def forward(self,*,prior_logits,local_uv,image_features,target_text,context_text,
                public_geometry,quality,local_valid,world_geometry,instruction_text,
                control='real',return_base=False):
        if control not in ('real','zero','shuffle'):raise ValueError('unknown new-branch control')
        cells=prior_logits.shape[-1];count=image_features.shape[1]
        if world_geometry.shape!=(len(prior_logits),cells+count,33):raise ValueError('aligned public geometry required')
        with torch.no_grad():
            base=self.base(prior_logits=prior_logits,local_uv=local_uv,image_features=image_features,
                target_text=target_text,context_text=context_text,public_geometry=public_geometry,
                quality=quality,local_valid=local_valid)
        g=torch.tanh(self.geometry(world_geometry.float()))
        text=torch.tanh(self.instruction(F.normalize(instruction_text.float(),dim=-1)))
        map_field=self.map_score(g[:,:cells]*text[:,None]).squeeze(-1)
        if self.mode=='map':visual=torch.zeros_like(local_valid,dtype=torch.float32)
        else:
            image=image_features
            if control=='zero':image=torch.zeros_like(image)
            elif control=='shuffle':
                image=image.clone()
                for b in range(len(image)):
                    ids=local_valid[b].nonzero().flatten();image[b,ids]=image[b,ids.flip(0)].clone()
            v=torch.tanh(self.image(F.normalize(image.float(),dim=-1).flatten(-2)))
            target=torch.tanh(self.target(F.normalize(target_text.float(),dim=-1)))
            matched=v*(target+text)[:,None]
            if self.mode=='conditioned':matched=matched*(1+g[:,cells:])
            visual=self.visual_score(matched).squeeze(-1)
        valid=local_valid&(local_uv>=0).all(-1)&(local_uv<=1).all(-1)
        uniform=torch.zeros_like(prior_logits)
        visual_field=world_evidence_logits(uniform,local_uv,visual,valid)-uniform.float().log_softmax(-1)
        # Remove the helper's uniform log-prior; only its evidence is added.
        output=torch.cat((base[:,:cells]+map_field+visual_field,base[:,cells:]),-1)
        return (output,base,map_field,visual_field) if return_base else output
