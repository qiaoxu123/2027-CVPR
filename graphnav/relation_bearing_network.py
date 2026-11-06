"""Language-conditioned public-Polygon correspondence and angular uncertainty.

No target coordinates/IDs, true bearings or frame rules in forward. Known
anchor indices are supplied solely by public text binding. Intermediate
objects stay probabilistic; TRAIN target-ID matching is a loss only.
"""
import math
import torch
from torch import nn


class RolePolygonBearing(nn.Module):
    def __init__(self,*,text_dim=768,shape_dim=39,hidden=128,bins=8,types=10):
        super().__init__();self.bins=bins;self.hidden=hidden
        self.name=nn.Linear(text_dim,hidden);self.shape=nn.Linear(shape_dim,hidden);self.type_embed=nn.Embedding(types,hidden)
        self.node_norm=nn.LayerNorm(hidden);self.edge=nn.Sequential(nn.Linear(10,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.node_q=nn.Linear(hidden,hidden);self.node_k=nn.Linear(hidden,hidden);self.node_v=nn.Linear(hidden,hidden);self.edge_bias=nn.Linear(hidden,1)
        self.node_update=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.text_k=nn.Linear(text_dim,hidden);self.text_v=nn.Linear(text_dim,hidden)
        self.entity=nn.Linear(text_dim,hidden);self.relation=nn.Linear(text_dim,hidden);self.unspecified=nn.Linear(hidden,hidden)
        self.object_q=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.free=nn.Sequential(nn.Linear(hidden*2+1,hidden),nn.GELU(),nn.Linear(hidden,bins))
        self.coverage=nn.Sequential(nn.Linear(hidden*2+1,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,*,tokens,token_mask,node_names,node_shape,node_types,node_valid,edges,anchor_indices,anchor_valid,entity_steps,relation_steps,step_valid,angle_kernel,use_relations=True,emit_steps=False):
        h=self.hidden;b,n,_=node_shape.shape;r=anchor_indices.shape[1]
        if edges.shape!=(b,n,n,10) or angle_kernel.shape!=(b,r,n,self.bins):raise ValueError('Geometry alignment')
        if not node_valid.any(-1).all() or not token_mask.any(-1).all():raise ValueError('Need valid context masks')
        x=self.node_norm(self.name(node_names.float())+self.shape(node_shape.float())+self.type_embed(node_types))
        edge=self.edge(edges.float());scores=torch.einsum('bih,bjh->bij',self.node_q(x),self.node_k(x))/math.sqrt(h)+self.edge_bias(edge).squeeze(-1)
        scores=scores.masked_fill(~node_valid[:,None,:],-1e4);attn=scores.softmax(-1)
        x=x+self.node_update(torch.cat((x,attn@self.node_v(x)),-1))
        anchors=x.gather(1,anchor_indices[...,None].expand(-1,-1,h))
        word_score=torch.einsum('brh,blh->brl',anchors,self.text_k(tokens.float()))/math.sqrt(h)
        text=word_score.masked_fill(~token_mask[:,None,:],-1e4).softmax(-1)@self.text_v(tokens.float())
        logp=x.new_full((b,r,n),-1e4);logp.scatter_(2,anchor_indices[...,None],0.)
        final=logp;steps=step_valid.shape[-1] if use_relations else 1;step_history=[]
        for t in range(steps):
            active=step_valid[:,:,t]&anchor_valid if use_relations else anchor_valid
            if not bool(active.any()):
                if emit_steps:step_history.append(logp.softmax(-1))
                continue
            ent=self.entity(entity_steps[:,:,t].float()) if use_relations else text
            rel=self.relation(relation_steps[:,:,t].float()) if use_relations else self.unspecified(text)
            query=self.object_q(torch.cat((text,ent),-1))
            semantic=torch.einsum('brh,bnh->brn',query,x)/math.sqrt(h)
            geometric=torch.einsum('bijh,brh->brij',edge,rel)/math.sqrt(h)
            propagated=torch.logsumexp(geometric+logp[:,:,None,:],dim=-1)
            logits=(semantic+propagated).masked_fill(~node_valid[:,None,:],-1e4)
            new=logits.log_softmax(-1);logp=torch.where(active[...,None],new,logp);final=logp
            if emit_steps:step_history.append(logp.softmax(-1))
        object_p=final.softmax(-1)
        entropy=-(object_p*object_p.clamp_min(1e-9).log()).sum(-1,keepdim=True)/math.log(max(n,2))
        context=torch.cat((text,anchors,entropy),-1)
        free=self.free(context).softmax(-1);coverage_logit=self.coverage(context).squeeze(-1)
        geographic=torch.einsum('brn,brnk->brk',object_p,angle_kernel.float())
        alpha=coverage_logit.sigmoid()[...,None]
        p=alpha*geographic+(1-alpha)*free;p=p/p.sum(-1,keepdim=True).clamp_min(1e-9)
        p=torch.where(anchor_valid[...,None],p,torch.full_like(p,1/self.bins))
        result=dict(probabilities=p,object_probabilities=object_p,object_logits=final,coverage_logits=coverage_logit,entropy=-(p*p.clamp_min(1e-9).log()).sum(-1),free_probabilities=free,geometric_probabilities=geographic)
        if emit_steps:result['step_object_probabilities']=torch.stack(step_history,dim=2)
        return result
