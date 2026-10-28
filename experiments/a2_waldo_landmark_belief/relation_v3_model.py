"""Independent RGB/landmark evidence ranking; A0 is used only by risk gating."""
import torch
from torch import nn
from .model import LandmarkBeliefNetwork


class RelationV3Network(nn.Module):
    def __init__(self,text_dim=768,vision_dim=1024,hidden=128,heads=4):
        super().__init__();self.config=dict(text_dim=text_dim,vision_dim=vision_dim,hidden=hidden,heads=heads)
        self.base=LandmarkBeliefNetwork(**self.config)
        for parameter in self.base.offset.parameters():parameter.requires_grad_(False)
        self.category=nn.Sequential(nn.LayerNorm(text_dim),nn.Linear(text_dim,hidden),nn.GELU(),nn.Linear(hidden,2))
        self.risk=nn.Sequential(nn.Linear(15,32),nn.GELU(),nn.Linear(32,3))
        self.register_buffer('risk_class_weights',torch.tensor([1.,8.,6.]))

    def forward(self,box_kind,a0_context=None,**network):
        # a0_context is accepted for old public-cache compatibility, but NEVER
        # enters candidate scores or matching. This is tested explicitly.
        out=self.base(**network)
        tokens=network['instruction_tokens'].float();valid=network['token_valid'].float()
        sentence=(tokens*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        category=self.category(sentence)
        consistency=category.log_softmax(-1).gather(1,box_kind.long().clamp(0,1))
        scores=(out['scores']+consistency).masked_fill(~out['reliable'],-1e4)
        winner=scores.argmax(-1,keepdim=True)
        match=out['candidate_match_logits'].gather(1,winner).squeeze(-1).masked_fill(~out['reliable'].any(-1),-1e4)
        return dict(out,scores=scores,category_logits=category,match_logit=match,offset=torch.zeros_like(out['offset']))

    def predict_risk(self,public_features):
        if public_features.ndim!=2 or public_features.shape[-1]!=15 or not torch.isfinite(public_features).all():raise ValueError('Invalid public risk features')
        # Risk classification must not change evidence ranking through an
        # outcome-label shortcut. Actor has its separate spatial/map losses.
        return self.risk(public_features.detach().float())

    def corrected_risk_logits(self,logits):return logits-self.risk_class_weights.log()

    def warmstart(self,state):
        if state['model_config']!=self.config:raise ValueError('Evidence warmstart dimensions changed')
        previous=state['model'];copied=[]
        for prefix,module in (('base',self.base),('category',self.category)):
            weights={key:previous[prefix+'.'+key] for key in module.state_dict()}
            module.load_state_dict(weights,strict=True);copied.extend(prefix+'.'+key for key in weights)
        return copied
