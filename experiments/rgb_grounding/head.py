"""Lightweight semantic VEN. Geometry/targets are not encoder inputs."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Evidence:
    score: torch.Tensor
    confidence: torch.Tensor
    observed: torch.Tensor


class VisualEvidenceNetwork(nn.Module):
    PUBLIC_KEYS={'text','images','patch_tokens','language_tokens','language_valid','quality','valid'}

    def __init__(self,dimension=1024,hidden=64,token_attention=False):
        super().__init__();self.dimension=dimension;self.token_attention=token_attention
        self.image_adapter=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),nn.Linear(hidden,dimension))
        self.text_adapter=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),nn.Linear(hidden,dimension))
        for adapter in [self.image_adapter,self.text_adapter]:nn.init.zeros_(adapter[-1].weight);nn.init.zeros_(adapter[-1].bias)
        self.log_temperature=nn.Parameter(torch.tensor(3.0))
        if token_attention:
            self.image_tokens=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,hidden))
            self.word_tokens=nn.Sequential(nn.LayerNorm(dimension),nn.Linear(dimension,hidden))
            self.attention=nn.MultiheadAttention(hidden,4,batch_first=True)
            self.token_score=nn.Linear(hidden,1);nn.init.zeros_(self.token_score.weight);nn.init.zeros_(self.token_score.bias)
        self.confidence_head=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,public):
        if set(public)!=self.PUBLIC_KEYS:raise ValueError('VEN public contract; private geometry/labels/IDs forbidden')
        text,images=public['text'].float(),public['images'].float();valid=public['valid']
        if images.ndim!=3 or text.shape!=(images.shape[0],self.dimension) or images.shape[-1]!=self.dimension:
            raise ValueError('native same-checkpoint shared space required')
        if valid.shape!=images.shape[:2] or valid.dtype!=torch.bool:raise ValueError('explicit observed boolean mask required')
        if public['quality'].shape!=(*valid.shape,3):raise ValueError('ROI quality mismatch')
        if not all(torch.isfinite(public[k]).all() for k in ['text','images','quality','patch_tokens','language_tokens']):
            raise ValueError('nonfinite actual features')
        language=F.normalize(text+.2*self.text_adapter(text),dim=-1)
        visual=F.normalize(images+.2*self.image_adapter(images),dim=-1)
        cosine=(visual*language[:,None]).sum(-1);score=cosine*self.log_temperature.exp().clamp(1,100)
        if self.token_attention:
            patches=public['patch_tokens'];words=public['language_tokens'];word_valid=public['language_valid']
            if patches.shape[:2]!=valid.shape or patches.shape[-1]!=self.dimension or word_valid.shape!=words.shape[:2]:
                raise ValueError('real token alignment mismatch')
            b,c,p,_=patches.shape;l=words.shape[1]
            query=self.word_tokens(words.float())[:,None].expand(-1,c,-1,-1).reshape(b*c,l,-1)
            key=self.image_tokens(patches.float()).reshape(b*c,p,-1)
            attended,_=self.attention(query,key,key,need_weights=False)
            weights=word_valid[:,None,:,None].expand(-1,c,-1,-1).reshape(b*c,l,1).float()
            pooled=(attended*weights).sum(1)/weights.sum(1).clamp_min(1)
            score=score+self.token_score(pooled).reshape(b,c)
        known=valid.float();mean=(cosine*known).sum(-1,keepdim=True)/known.sum(-1,keepdim=True).clamp_min(1)
        spread=(((cosine-mean)**2*known).sum(-1,keepdim=True)/known.sum(-1,keepdim=True).clamp_min(1)).sqrt()
        confidence_inputs=torch.cat([public['quality'].float(),cosine[...,None],(cosine-mean)[...,None],spread[:,None].expand(-1,images.shape[1],-1)],-1)
        confidence=self.confidence_head(confidence_inputs).squeeze(-1).sigmoid()
        return Evidence(torch.where(valid,score,torch.zeros_like(score)),torch.where(valid,confidence,torch.zeros_like(confidence)),valid)


def grounding_loss(evidence,positive,judged,quality,hard_weight=.2,confidence_weight=.1):
    """Multi-positive labels passed only to loss; unknown never a negative."""
    positive=positive.bool() & evidence.observed;judged=judged.bool() & evidence.observed
    if (positive & ~judged).any():raise ValueError('positive must have trusted judgment')
    negative=judged & ~positive;usable=positive.any(-1)&negative.any(-1)
    if not usable.any():return dict(total=evidence.score.sum()*0,usable=0)
    scores=evidence.score
    logpos=torch.logsumexp(scores.masked_fill(~positive,-1e4),-1)
    logall=torch.logsumexp(scores.masked_fill(~judged,-1e4),-1)
    row_quality=(quality[...,1]*quality[...,2]*positive).sum(-1)/positive.sum(-1).clamp_min(1)
    region=((logall-logpos)[usable]*row_quality[usable]).sum()/row_quality[usable].sum().clamp_min(1e-6)
    hard=F.relu(.2+scores.masked_fill(~negative,-1e4).max(-1).values-scores.masked_fill(~positive,-1e4).max(-1).values)[usable].mean()
    confidence_loss=F.binary_cross_entropy(evidence.confidence[judged].clamp(1e-6,1-1e-6),positive[judged].float())
    return dict(total=region+hard_weight*hard+confidence_weight*confidence_loss,region_text=region,hard_negative=hard,confidence=confidence_loss,usable=int(usable.sum()))


def update_belief(static_logits,evidence,association,*,gain=1.0,threshold=.5):
    """Public ROI-to-cell map only; all unobserved logits stay bitwise exact."""
    if association.shape!=(*evidence.score.shape,static_logits.shape[-1]) or association.dtype!=torch.bool:
        raise ValueError('boolean public ROI/candidate geometry association required')
    accepted=evidence.observed & (evidence.confidence>=threshold)
    weights=association & accepted[...,None]
    counts=weights.sum(1);known=counts>0
    scores=(weights*evidence.score[...,None]*evidence.confidence[...,None]).sum(1)/counts.clamp_min(1)
    center=(scores*known).sum(-1,keepdim=True)/known.sum(-1,keepdim=True).clamp_min(1)
    updated=static_logits+float(gain)*(scores-center)
    return torch.where(known,updated,static_logits),known
