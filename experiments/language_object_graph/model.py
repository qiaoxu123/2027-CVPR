"""One-layer language-conditioned GATv2; no object/map ID embeddings."""
import torch
from torch import nn
from torch.nn import functional as F
from .objects import FEATURE_DIM,EDGE_DIM


class LanguageGAT(nn.Module):
    KEYS={'nodes','edge_index','edge_attr','language','node_prior','kernels','static'}
    def __init__(self,hidden=32,heads=2,message=True):
        super().__init__();self.h=hidden;self.heads=heads;self.d=hidden//heads;self.message=message
        self.input=nn.Linear(FEATURE_DIM,hidden);self.condition=nn.Linear(96,hidden)
        self.source=nn.Linear(hidden,hidden,bias=False);self.target=nn.Linear(hidden,hidden,bias=False)
        self.edge=nn.Linear(EDGE_DIM,hidden,bias=False);self.attention=nn.Parameter(torch.randn(heads,self.d)*.02)
        self.output=nn.Sequential(nn.Linear(hidden*2+1,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,p):
        if set(p)!=self.KEYS:raise ValueError('strict public-only GAT contract')
        x=self.input(p['nodes'])+self.condition(p['language'])[None]
        src,dst=p['edge_index'];pair=self.source(x[src])+self.target(x[dst])+self.edge(p['edge_attr'])
        pair=F.leaky_relu(pair.reshape(-1,self.heads,self.d),.2)
        scores=(pair*self.attention).sum(-1);maxima=torch.full((len(x),self.heads),-torch.inf,device=x.device)
        maxima.scatter_reduce_(0,dst[:,None].expand(-1,self.heads),scores,reduce='amax',include_self=True)
        w=(scores-maxima[dst]).exp();denom=torch.zeros_like(maxima);denom.scatter_add_(0,dst[:,None].expand(-1,self.heads),w);w=w/denom[dst].clamp_min(1e-8)
        message=(x[src].reshape(-1,self.heads,self.d)*w[...,None]).reshape(-1,self.h)
        aggregated=torch.zeros_like(x);aggregated.index_add_(0,dst,message)
        if not self.message:aggregated=torch.zeros_like(aggregated)
        prior=p['node_prior'];delta=self.output(torch.cat([x,aggregated,prior[:,None]],-1)).squeeze(-1)
        belief=(prior+delta).softmax(0);heat=belief@p['kernels'];heat=.5*heat+.5*p['static'];heat=heat/heat.sum().clamp_min(1e-8)
        return {'object_belief':belief,'heatmap':heat,'delta':delta}


def goal_loss(output,positive_cells):
    if not positive_cells.any():raise ValueError('no positive original grid candidate')
    return -output['heatmap'][positive_cells].sum().clamp_min(1e-8).log()
