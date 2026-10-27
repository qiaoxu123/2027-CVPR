"""Trainable local relation head over frozen public RGB/language features."""
import torch
from torch import nn
from torch.nn import functional as F
from geometry import EDGE_DIM


def mlp(a,b):return nn.Sequential(nn.Linear(a,b),nn.GELU(),nn.Linear(b,b),nn.LayerNorm(b))


class Cross(nn.Module):
    def __init__(self,h):
        super().__init__();self.attention=nn.MultiheadAttention(h,4,dropout=.1,batch_first=True)
        self.norm=nn.LayerNorm(h);self.ff=mlp(h,h)
    def forward(self,query,memory,mask=None):
        if not len(query):return query
        a,_=self.attention(query[None],memory[None],memory[None],key_padding_mask=None if mask is None else ~mask[None],need_weights=False)
        x=self.norm(query+a[0]);return x+self.ff(x)


class RelationLayer(nn.Module):
    def __init__(self,h):
        super().__init__();self.edge=mlp(h*3,h);self.language=Cross(h)
        self.compatibility=nn.Linear(h,1);self.update=mlp(h*2,h)
        self.triad=mlp(4+h,h);self.triad_language=Cross(h);self.triad_weight=nn.Linear(h,1)
    def forward(self,c,r,e,triad,ids,words,word_mask):
        n,l=e.shape[:2];h=c.shape[-1]
        attention=c.new_empty(n,l)
        if n and l:
            edge=self.edge(torch.cat((c[:,None].expand(-1,l,-1),r[None].expand(n,-1,-1),e),-1))
            edge=self.language(edge.reshape(n*l,h),words,word_mask).reshape(n,l,h)
            attention=self.compatibility(edge).squeeze(-1).softmax(-1)
            pooled=(edge*attention[...,None]).sum(1)
        else:pooled=c.new_zeros(n,h)
        if n and len(ids):
            # Symmetric representation for an unordered A/B 'between' relation.
            t=triad.clone();t[...,0]=torch.minimum(t[...,0],1-t[...,0])
            anchors=(r[ids[:,0]]+r[ids[:,1]])[None].expand(n,-1,-1)
            t=self.triad(torch.cat((t,anchors),-1))
            t=self.triad_language(t.reshape(-1,h),words,word_mask).reshape(n,-1,h)
            valid=triad[...,3]>.5;weights=self.triad_weight(t).squeeze(-1).masked_fill(~valid,-1e4).softmax(-1)*valid
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            pooled=pooled+(t*weights[...,None]).sum(1)
        return c+self.update(torch.cat((c,pooled),-1)),attention


class LocalRelationNetwork(nn.Module):
    def __init__(self,rgb_dim=1024,hidden=256):
        super().__init__();self.rgb_dim=rgb_dim;self.hidden=hidden
        self.word=nn.Linear(768,hidden);self.rgb=nn.Linear(rgb_dim,hidden)
        self.depth=nn.Sequential(nn.Conv2d(2,32,3,2,1),nn.GroupNorm(4,32),nn.GELU(),
            nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,128,3,2,1),nn.GroupNorm(8,128),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,hidden))
        self.missing_depth=nn.Parameter(torch.zeros(hidden));self.node=mlp(hidden*2+12,hidden)
        self.name=nn.Linear(768,hidden);self.c_language=Cross(hidden);self.r_language=Cross(hidden)
        self.expected=nn.Linear(hidden,4);self.edge=mlp(EDGE_DIM+4,hidden)
        self.layers=nn.ModuleList([RelationLayer(hidden) for _ in range(2)])
        self.score=nn.Linear(hidden,1);self.offset=nn.Linear(hidden,2);self.match=nn.Linear(hidden*2,1)

    def nodes(self,rgb,depth,geometry):
        if not len(rgb):return rgb.new_empty((0,self.hidden))
        valid=depth[:,1:2]>.5
        safe=torch.cat((torch.where(valid,depth[:,:1],0),valid.float()),1)
        z=self.depth(safe);known=valid.flatten(1).any(1)
        z=torch.where(known[:,None],z,self.missing_depth[None])
        return self.node(torch.cat((self.rgb(rgb.float()),z,geometry.float()),-1))

    def forward(self,*,tokens,token_mask,candidate_rgb,reference_rgb,names,name_mask,
                candidate_depth,reference_depth,candidate_geometry,reference_geometry,
                pair,triad,triad_ids,offset_limit,control='real'):
        allowed=['real','no_language','no_depth','no_geometry','no_height','no_offset','shuffle_depth']
        if control not in allowed:raise ValueError(control)
        if not token_mask.any():raise ValueError('At least one valid language token is required')
        words=self.word(tokens.float() if control!='no_language' else torch.zeros_like(tokens).float())
        n,l=pair.shape[:2];cg=candidate_geometry.float().clone();rg=reference_geometry.float().clone()
        cd=candidate_depth.float().clone();rd=reference_depth.float().clone();p=pair.float().clone();t=triad.float().clone()
        p[...,8:11]=torch.where(p[...,12,None]>.5,p[...,8:11],0.)
        if control=='no_depth':
            cd.zero_();rd.zero_();p[...,8:11]=0;p[...,12]=0;p[...,14:16]=0;cg[:,10:]=0;rg[:,10:]=0
        if control=='no_height':p[...,8:11]=0;p[...,12]=0
        if control=='no_geometry':cg.zero_();rg.zero_();p.zero_();t.zero_()
        if control=='shuffle_depth' and n:
            cd[:,0]=cd[:,0].roll(1,0)*cd[:,1];p[...,8:11]=p[...,8:11].roll(1,0)*p[...,12,None]
        c=self.c_language(self.nodes(candidate_rgb,cd,cg),words,token_mask)
        r=self.nodes(reference_rgb,rd,rg)
        if l:
            name_words=names.float() if control!='no_language' else torch.zeros_like(names).float()
            name=(name_words*name_mask[...,None]).sum(1)/name_mask.sum(1,keepdim=True).clamp_min(1)
            r=self.r_language(r+self.name(name),words,token_mask)
        expected=self.expected(r);direction=F.normalize(expected[:,:2],dim=-1,eps=1e-6)
        distance=expected[:,2];height=expected[:,3]
        if n and l:
            residual=torch.cat((p[...,[3,2]]-direction[None],(p[...,4]-distance[None])[...,None],
                                ((p[...,8]-height[None])*p[...,12])[...,None]),-1)
            e=self.edge(torch.cat((p,residual),-1))
        else:e=c.new_empty(n,l,self.hidden)
        attention=c.new_empty(n,l)
        for layer in self.layers:c,attention=layer(c,r,e,t,triad_ids,words,token_mask)
        scores=self.score(c).squeeze(-1);offset=self.offset(c).tanh()*offset_limit
        if control=='no_offset':offset=offset*0
        language=(words*token_mask[:,None]).sum(0)/token_mask.sum()
        pooled=(c*scores.softmax(0)[:,None]).sum(0) if n else language*0
        match=self.match(torch.cat((pooled,language))).squeeze(-1)
        return dict(scores=scores,offset=offset,match_logit=match,direction=direction,
                    distance=distance,height=height,reference_attention=attention)
