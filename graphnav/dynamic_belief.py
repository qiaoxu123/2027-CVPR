"""A3-preserving residual branches and causal landmark evidence memory.

Unobserved nodes are unknown. Missing observations never create negative evidence.
This module is an implementation candidate, not an experimentally selected model.
"""
import copy
from dataclasses import dataclass
import torch
from torch import nn
from .full_localization import GoalField


@dataclass
class LandmarkHistory:
    vectors: torch.Tensor
    confidence: torch.Tensor
    last_step: int = -1

    @classmethod
    def empty(cls,batch,nodes,vision_dim,device):
        return cls(torch.zeros(batch,nodes,vision_dim,device=device),
                   torch.zeros(batch,nodes,device=device))

    def update(self,vectors,confidence,*,step):
        if step <= self.last_step:
            raise ValueError('history observations must advance causal step')
        if vectors.shape!=self.vectors.shape or confidence.shape!=self.confidence.shape:
            raise ValueError('landmark evidence dimensions mismatch')
        if not torch.isfinite(vectors).all() or not torch.isfinite(confidence).all():
            raise ValueError('nonfinite visual evidence')
        if (confidence<0).any() or (confidence>1).any():
            raise ValueError('confidence must be in [0,1]')
        observed=confidence>0
        total=self.confidence+confidence
        mean=(self.vectors*self.confidence[...,None]+vectors*confidence[...,None])/total.clamp_min(1e-8)[...,None]
        self.vectors=torch.where(observed[...,None],mean,self.vectors)
        self.confidence=torch.where(observed,total.clamp_max(1),self.confidence)
        self.last_step=step


class DynamicGoalBelief(nn.Module):
    def __init__(self,a3,vision_dim=256,freeze_a3=True):
        super().__init__()
        self.a3=a3
        if freeze_a3:self.a3.requires_grad_(False)
        h=a3.hidden
        self.graph=copy.deepcopy(a3.gnn)
        self.graph.requires_grad_(True)
        self.visual=nn.Sequential(nn.Linear(vision_dim,h),nn.GELU(),nn.LayerNorm(h))
        self.relative=nn.Sequential(nn.Linear(10,h),nn.GELU(),nn.Linear(h,h))
        self.score=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),nn.Linear(h,1))
        self.graph_gate=nn.Parameter(torch.zeros(()))
        self.visual_gate=nn.Parameter(torch.zeros(()))

    def train(self,mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.a3.parameters()):self.a3.eval()
        return self

    def forward(self,batch,*,tokens,token_mask,node_emb,visual_vectors=None,
                visual_confidence=None,use_graph=False,use_visual=False):
        base=self.a3(batch,variant='geometry',tokens=tokens,token_mask=token_mask,node_emb=node_emb)
        b,q=base.belief_logits.shape
        dense=base.dense_context
        geo=self.relative(batch['pair_geometry'].to(dense.dtype))
        logits=base.belief_logits
        valid=batch['entity_valid'].bool()
        graph_nodes=None
        if use_graph:
            graph_nodes=self.graph(batch['nodes'],node_emb,batch['node_types'],
                batch['adjacency'],batch['edges'],batch['node_valid'])
            weights=valid.to(dense.dtype)
            context=((graph_nodes[:,None]+geo)*weights[:,None,:,None]).sum(2)/weights.sum(1).clamp_min(1)[:,None,None]
            logits=logits+torch.tanh(self.graph_gate)*self.score(torch.cat((dense,context),-1)).squeeze(-1)
        if use_visual:
            if visual_vectors is None or visual_confidence is None:
                raise ValueError('visual branch requires actual observed landmark evidence')
            if visual_confidence.shape!=valid.shape or visual_vectors.shape[:2]!=valid.shape:
                raise ValueError('visual evidence must align with packed landmark nodes')
            if not torch.isfinite(visual_vectors).all() or not torch.isfinite(visual_confidence).all():
                raise ValueError('nonfinite visual evidence')
            if (visual_confidence<0).any() or (visual_confidence>1).any():
                raise ValueError('confidence must be in [0,1]')
            weights=visual_confidence.to(dense.dtype)*valid
            nodes=self.visual(visual_vectors.to(dense.dtype))
            # When enabled, messages contain current visual node state rather than static-only nodes.
            if use_graph:
                nodes=self.graph.layer1(nodes, batch['edges'],batch['adjacency'],batch['node_valid'])
                nodes=self.graph.layer2(nodes,batch['edges'],batch['adjacency'],batch['node_valid'])
            context=((nodes[:,None]+geo)*weights[:,None,:,None]).sum(2)/weights.sum(1).clamp_min(1e-8)[:,None,None]
            present=(weights.sum(1)>0).to(dense.dtype)[:,None]
            delta=self.score(torch.cat((dense,context),-1)).squeeze(-1)*present
            logits=logits+torch.tanh(self.visual_gate)*delta
        ids=logits.topk(min(self.a3.top_k,q),dim=1).indices
        return GoalField(logits,ids,batch['grid_xy'].gather(1,ids[...,None].expand(-1,-1,2)),
            dense.gather(1,ids[...,None].expand(-1,-1,self.a3.hidden)),dense_context=dense)
