"""Small instance VEN: full words, target phrase, RGB patches and context.

Identity, goal positions, static scores and map names are forbidden inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.rgb_grounding.head import VisualEvidenceNetwork

class InstanceVEN(nn.Module):
    KEYS={'full','phrase','images','context','patch_tokens','language_tokens','language_valid','quality','valid','relative'}
    def __init__(self,d=1024,h=64,relative_attention=False):
        super().__init__();self.relative_attention=relative_attention;self.base=VisualEvidenceNetwork(dimension=d)
        if relative_attention:
            self.relation_coefficients=nn.Linear(d,12)
            nn.init.zeros_(self.relation_coefficients.weight);nn.init.zeros_(self.relation_coefficients.bias)
        self.visual=nn.Linear(d,h);self.words=nn.Linear(d,h);self.context=nn.Linear(d,h)
        self.word_patch=nn.MultiheadAttention(h,4,batch_first=True)
        self.references=nn.MultiheadAttention(h,4,batch_first=True)
        self.identity=nn.Sequential(nn.Linear(h*3+3,h),nn.GELU(),nn.Linear(h,1))
        nn.init.zeros_(self.identity[-1].weight);nn.init.zeros_(self.identity[-1].bias)
        self.judgment=nn.Sequential(nn.Linear(h*3+6,h),nn.GELU(),nn.Linear(h,3))
    def forward(self,p):
        if set(p)!=self.KEYS:raise ValueError('private information forbidden by public contract')
        if p['valid'].dtype!=torch.bool:raise ValueError('observed mask required')
        if not all(torch.isfinite(v).all() for v in p.values()):raise ValueError('nonfinite features')
        b,n,d=p['images'].shape
        old=dict(text=p['full'],images=p['images'],patch_tokens=p['patch_tokens'],language_tokens=p['language_tokens'],language_valid=p['language_valid'],quality=p['quality'],valid=p['valid'])
        full=self.base(old).score;old['text']=p['phrase'];phrase=self.base(old).score
        vis=self.visual(p['images'].float());ctx=self.context(p['context'].float());words=self.words(p['language_tokens'].float())
        query=words[:,None].expand(-1,n,-1,-1).reshape(b*n,words.shape[1],-1)
        patches=self.visual(p['patch_tokens'].float()).reshape(b*n,p['patch_tokens'].shape[2],-1)
        attended,_=self.word_patch(query,patches,patches,need_weights=False)
        word_valid=p['language_valid'][:,None,:,None].expand(-1,n,-1,-1).reshape(b*n,-1,1).float()
        wp=((attended*word_valid).sum(1)/word_valid.sum(1).clamp_min(1)).reshape(b,n,-1)
        # Causally observed memory evidence, never source-object identities. A
        # zero dummy token prevents all-masked empty-observation NaNs.
        refs=torch.cat([ctx,torch.zeros(b,1,ctx.shape[-1],device=ctx.device)],1)
        invalid=torch.cat([~p['valid'],torch.zeros(b,1,dtype=torch.bool,device=ctx.device)],1)
        mask=None
        if self.relative_attention:
            if p['relative'].shape!=(b,n,n+1,3):raise ValueError('public relative observed offsets required')
            coefficients=self.relation_coefficients(p['full'].float()).reshape(b,4,3)
            bias=torch.einsum('bhk,bijk->bhij',coefficients,p['relative'].float())
            bias=bias.masked_fill(invalid[:,None,None],float('-inf'));mask=bias.reshape(b*4,n,n+1)
        rel,_=self.references(vis,refs,refs,key_padding_mask=invalid if mask is None else None,attn_mask=mask,need_weights=False)
        scores=torch.stack([full,phrase,full-phrase],-1)
        interaction=torch.cat([vis*wp,ctx*wp,rel*wp,scores],-1)
        score=(full+phrase)/2+self.identity(interaction).squeeze(-1)
        tri=self.judgment(torch.cat([interaction,p['quality']],-1))
        score=torch.where(p['valid'],score,torch.zeros_like(score))
        prob=tri.softmax(-1);prob=torch.where(p['valid'][...,None],prob,torch.tensor([0.,0.,1.],device=prob.device))
        return score,prob,tri

def loss(score,tri,positive,judged,valid,quality,class_balanced=True):
    pos=positive&valid;neg=judged&~positive&valid;use=pos.any(-1)&neg.any(-1)
    # 0 Match, 1 Mismatch, 2 Unknown. Only actual observed entries trained.
    targets=torch.full_like(positive,2,dtype=torch.long);targets[pos]=0;targets[neg]=1
    # Balance observed judgment classes: the many Unknown ROIs must not
    # make predicting Unknown everywhere a cheap classification optimum.
    if class_balanced:
        active=[F.cross_entropy(tri[valid & (targets==c)],targets[valid & (targets==c)]) for c in range(3) if (valid & (targets==c)).any()]
        ce=torch.stack(active).mean() if active else score.sum()*0
    else:ce=F.cross_entropy(tri[valid],targets[valid]) if valid.any() else score.sum()*0
    ranking=score.sum()*0
    if use.any():
        a=torch.logsumexp(score.masked_fill(~pos,-1e4),-1);z=torch.logsumexp(score.masked_fill(~(pos|neg),-1e4),-1)
        q=(quality[...,1]*quality[...,2]*pos).sum(-1)/pos.sum(-1).clamp_min(1)
        ranking=((z-a)[use]*q[use]).sum()/q[use].sum().clamp_min(1e-8)
        hard=F.relu(.2+score.masked_fill(~neg,-1e4).max(-1).values-score.masked_fill(~pos,-1e4).max(-1).values)[use].mean()
        ranking=ranking+.2*hard
    return ranking+.2*ce
