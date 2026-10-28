"""Fresh revision network: public A0 context, full-pool category and benefit."""
import torch
from torch import nn
from .model import LandmarkBeliefNetwork
from .revision_fusion import MAX_GATE


class RevisionNetwork(nn.Module):
    def __init__(self,text_dim=768,vision_dim=1024,hidden=128,heads=4):
        super().__init__()
        self.config=dict(text_dim=text_dim,vision_dim=vision_dim,hidden=hidden,heads=heads)
        self.base=LandmarkBeliefNetwork(**self.config)
        # This revision deliberately retains detector centers exactly.
        for parameter in self.base.offset.parameters():parameter.requires_grad_(False)
        self.category=nn.Sequential(nn.LayerNorm(text_dim),nn.Linear(text_dim,hidden),nn.GELU(),nn.Linear(hidden,2))
        self.rank_context=nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.trust_context=nn.Sequential(nn.Linear(9,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.benefit=nn.Sequential(nn.Linear(9,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.rank_context[-1].weight);nn.init.zeros_(self.rank_context[-1].bias)
        nn.init.zeros_(self.trust_context[-1].weight);nn.init.zeros_(self.trust_context[-1].bias)
        nn.init.constant_(self.benefit[-1].bias,-2.)

    def forward(self,a0_context,box_kind,**network):
        out=self.base(**network)
        tokens=network['instruction_tokens'].float();valid=network['token_valid'].float()
        sentence=(tokens*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        category=self.category(sentence)
        consistency=category.log_softmax(-1).gather(1,box_kind.long().clamp(0,1))
        raw=out['scores'];scores=raw+self.rank_context(a0_context.float()).squeeze(-1)+consistency
        scores=scores.masked_fill(~out['reliable'],-1e4)
        centered=(raw-raw.max(-1,keepdim=True).values).clamp(-30,0)/10
        extra=torch.cat((a0_context.float(),consistency[...,None],centered[...,None],
                         out['candidate_match_logits'].clamp(-20,20)[...,None]/10),-1)
        trust=out['candidate_match_logits']+self.trust_context(extra).squeeze(-1)
        supported=out['reliable'] & (a0_context[...,5]>0)
        weights=scores.masked_fill(~supported,-1e4).softmax(-1)*supported
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        pooled=(extra*weights[...,None]).sum(1)
        benefit=self.benefit(pooled).squeeze(-1)
        gate=MAX_GATE*benefit.sigmoid()*supported.any(-1)
        winner=scores.argmax(-1,keepdim=True)
        match=trust.gather(1,winner).squeeze(-1).masked_fill(~out['reliable'].any(-1),-1e4)
        return dict(out,scores=scores,candidate_match_logits=trust,match_logit=match,
                    offset=torch.zeros_like(out['offset']),category_logits=category,
                    benefit_logit=benefit,gate=gate)
