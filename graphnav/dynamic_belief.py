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
    total_weight: torch.Tensor | None = None

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
        prior_weight=self.total_weight if self.total_weight is not None else self.confidence
        total=prior_weight+confidence
        mean=(self.vectors*prior_weight[...,None]+vectors*confidence[...,None])/total.clamp_min(1e-8)[...,None]
        self.vectors=torch.where(observed[...,None],mean,self.vectors)
        self.total_weight=total
        # Registration quality is not repeated independent semantic evidence.
        self.confidence=torch.maximum(self.confidence,confidence)
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
                visual_confidence=None,use_graph=False,use_visual=False,base_field=None,
                observed_vectors=None,observed_confidence=None,observed_geometry=None):
        if base_field is not None and any(p.requires_grad for p in self.a3.parameters()):
            raise ValueError('precomputed A3 field requires a frozen backbone')
        base=base_field if base_field is not None else self.a3(batch,variant='geometry',tokens=tokens,token_mask=token_mask,node_emb=node_emb)
        b,q=base.belief_logits.shape
        dense=base.dense_context
        geo=self.relative(batch['pair_geometry'].to(dense.dtype))
        logits=base.belief_logits
        valid=batch['entity_valid'].bool()
        graph_nodes=None
        if use_graph:
            graph_nodes=(self.graph.node_geometry(batch['nodes'])+self.graph.name_projection(node_emb)+self.graph.type_embedding(batch['node_types']))
            if use_visual and visual_vectors is not None:
                if visual_confidence is None or visual_confidence.shape!=valid.shape or visual_vectors.shape[:2]!=valid.shape:
                    raise ValueError('visual node association dimensions mismatch')
                graph_nodes=graph_nodes+self.visual(visual_vectors.to(dense.dtype))*visual_confidence[...,None]*valid[...,None]
            graph_nodes=graph_nodes*batch['node_valid'][...,None]
            graph_nodes=self.graph.layer1(graph_nodes,batch['edges'],batch['adjacency'],batch['node_valid'])
            graph_nodes=self.graph.layer2(graph_nodes,batch['edges'],batch['adjacency'],batch['node_valid'])
            weights=valid.to(dense.dtype)
            context=(torch.tanh(graph_nodes[:,None]+geo)*weights[:,None,:,None]).sum(2)/weights.sum(1).clamp_min(1)[:,None,None]
            logits=logits+torch.tanh(self.graph_gate)*self.score(torch.cat((dense,context),-1)).squeeze(-1)
        if use_visual:
            if visual_vectors is None and observed_vectors is not None:
                visual_vectors=dense.new_zeros(b,valid.shape[1],self.visual[0].in_features)
                visual_confidence=dense.new_zeros(b,valid.shape[1])
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
            visual_geo=geo
            if observed_vectors is not None:
                # Public visible landmarks outside the static reference set may
                # contribute evidence anywhere on the map, not only Top-K cells.
                if observed_geometry is None or observed_confidence is None:
                    raise ValueError('observed auxiliary landmarks require relative geometry and confidence')
                m=observed_vectors.shape[1]
                if observed_vectors.shape!=(b,m,self.visual[0].in_features) or observed_confidence.shape!=(b,m) or observed_geometry.shape!=(b,q,m,10):
                    raise ValueError('auxiliary observed landmark dimensions mismatch')
                if not all(torch.isfinite(x).all() for x in (observed_vectors,observed_confidence,observed_geometry)):
                    raise ValueError('nonfinite auxiliary observed evidence')
                if (observed_confidence<0).any() or (observed_confidence>1).any():
                    raise ValueError('auxiliary confidence must be in [0,1]')
                nodes=torch.cat((nodes,self.visual(observed_vectors.to(dense.dtype))),1)
                weights=torch.cat((weights,observed_confidence.to(dense.dtype)),1)
                visual_geo=torch.cat((geo,self.relative(observed_geometry.to(dense.dtype))),2)
            # Fuse before pooling: averaging visual and geometry separately loses
            # which landmark owns each visual vector, even with perfect registration.
            context=(torch.tanh(nodes[:,None]+visual_geo)*weights[:,None,:,None]).sum(2)/weights.sum(1).clamp_min(1e-8)[:,None,None]
            present=weights.amax(1).clamp(0,1)[:,None]
            delta=self.score(torch.cat((dense,context),-1)).squeeze(-1)*present
            logits=logits+torch.tanh(self.visual_gate)*delta
        ids=logits.topk(min(self.a3.top_k,q),dim=1).indices
        return GoalField(logits,ids,batch['grid_xy'].gather(1,ids[...,None].expand(-1,-1,2)),
            dense.gather(1,ids[...,None].expand(-1,-1,self.a3.hidden)),dense_context=dense)
