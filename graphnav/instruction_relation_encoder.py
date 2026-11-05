"""Trainable two-hop text+typed-geometry graph encoder, independent of PGVB.

IMPORTANT: consumes a structured (possibly imperfect) instruction graph
compiled with instruction_relation_graph.py and node/edge phrase embeddings.
This is NOT a trained end-to-end raw-English parser. Every unknown physical
coordinate is gated by an explicit validity mask. No GT position is an input.
"""
from __future__ import annotations

import torch
from torch import nn

from .instruction_relation_graph import (
    ROLES, PREDICATES, FRAMES, NODE_GEO_DIM, EDGE_GEO_DIM,
)


class InstructionRelationGraphEncoder(nn.Module):
    """Encode known spatial-constraint graphs; readout target and each relation.

    Pass BERT phrase embeddings (not sentence mean) so target/anchor identities
    remain separate. Each layer exchanges messages in BOTH directions while
    retaining predicate direction and native endpoint ordering.
    """

    def __init__(self, text_dim=768, hidden=128, layers=4, dropout=0.1):
        super().__init__()
        if hidden < 8 or layers < 1:
            raise ValueError("Hidden>=8 and at least one graph layer required")
        self.hidden=hidden
        self.node_text=nn.Sequential(nn.Linear(text_dim,hidden),nn.LayerNorm(hidden))
        self.edge_text=nn.Linear(text_dim,hidden)
        self.role=nn.Embedding(len(ROLES),hidden)
        self.predicate=nn.Embedding(len(PREDICATES),hidden)
        self.frame=nn.Embedding(len(FRAMES),hidden)
        self.node_geom=nn.Sequential(nn.Linear(NODE_GEO_DIM,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.edge_geom=nn.Sequential(nn.Linear(EDGE_GEO_DIM,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.edge_direction=nn.Linear(2,hidden)
        self.forward_message=nn.ModuleList()
        self.reverse_message=nn.ModuleList()
        self.update=nn.ModuleList()
        for _ in range(layers):
            self.forward_message.append(nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden)))
            self.reverse_message.append(nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden)))
            self.update.append(nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.LayerNorm(hidden)))
        self.edge_readout=nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.graph_readout=nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.LayerNorm(hidden))

    def forward(self, packed: dict, node_text: torch.Tensor, edge_text: torch.Tensor):
        """packed contains fixed-size tensors with leading batch dimension.

        Unknown geometry contributes exactly zero even if the geometry
        projection has a bias. Invalid padded edges contribute zero messages.
        A graph with no target returns only node/masked pooled representations.
        """
        if node_text.ndim!=3 or edge_text.ndim!=3:
            raise ValueError("Expected B,N,D and B,E,D phrase embeddings")
        b,n,_=node_text.shape;be,e,_=edge_text.shape
        if b!=be:
            raise ValueError("Batch mismatch")
        valid=packed["node_mask"].bool()
        edge_valid=packed["edge_mask"].bool()
        if valid.shape!=(b,n) or edge_valid.shape!=(b,e):
            raise ValueError("Node/edge tensor shape mismatch")
        indices=packed["edge_index"].long()
        if indices.shape!=(b,e,2) or (indices<0).any() or (indices>=n).any():
            raise ValueError("Invalid edge index")
        nm=valid.unsqueeze(-1)
        em=edge_valid.unsqueeze(-1)
        h=(self.node_text(node_text)
           +self.role(packed["node_roles"].long())
           +self.node_geom(packed["node_geometry"].float())
             *packed["node_geo_mask"].bool().unsqueeze(-1))
        h=h*nm
        rel=(self.edge_text(edge_text)
             +self.predicate(packed["predicates"].long())
             +self.frame(packed["frames"].long())
             +self.edge_geom(packed["edge_geometry"].float())
               *packed["edge_geo_mask"].bool().unsqueeze(-1)
             +self.edge_direction(packed["edge_direction"].float())
               *packed["edge_direction_mask"].bool().unsqueeze(-1))
        rel=rel*em
        src,dst=indices[:,:,0],indices[:,:,1]
        bs=torch.arange(b,device=h.device)[:,None]
        endpoints_ok=valid[bs,src] & valid[bs,dst] & edge_valid
        for fwd,rev,update in zip(self.forward_message,self.reverse_message,self.update):
            hs=h[bs,src];hd=h[bs,dst]
            # fwd transfers source -> destination; rev transfers destination
            # -> source without inverting semantic predicate labels.
            mf=fwd(torch.cat((hs,rel,hd),dim=-1))*endpoints_ok.unsqueeze(-1)
            mr=rev(torch.cat((hd,rel,hs),dim=-1))*endpoints_ok.unsqueeze(-1)
            total=torch.zeros_like(h)
            degree=torch.zeros((b,n,1),device=h.device,dtype=h.dtype)
            total.scatter_add_(1,dst.unsqueeze(-1).expand(-1,-1,self.hidden),mf)
            total.scatter_add_(1,src.unsqueeze(-1).expand(-1,-1,self.hidden),mr)
            degree.scatter_add_(1,dst.unsqueeze(-1),endpoints_ok.unsqueeze(-1).to(h.dtype))
            degree.scatter_add_(1,src.unsqueeze(-1),endpoints_ok.unsqueeze(-1).to(h.dtype))
            h=(h+update(torch.cat((h,total/degree.clamp_min(1)),dim=-1)))*nm
        src_repr=h[bs,src];dst_repr=h[bs,dst]
        relation_states=self.edge_readout(torch.cat((src_repr,rel,dst_repr),dim=-1))*em
        pooled=(h*nm).sum(1)/nm.sum(1).clamp_min(1)
        target=packed["target_index"].long()
        if target.shape!=(b,):
            raise ValueError("target_index must be shape B")
        target_present=(target>=0)&(target<n)&valid[torch.arange(b,device=h.device),target.clamp(0,n-1)]
        target_state=h[torch.arange(b,device=h.device),target.clamp(0,n-1)]*target_present[:,None]
        graph=self.graph_readout(torch.cat((target_state,pooled),dim=-1))
        return dict(
            node_states=h, edge_states=relation_states,
            target_state=target_state, graph_state=graph,
            target_present=target_present,
        )


def batch_relation_graphs(graphs, device=None):
    """Tensorize a batch, retaining phrase metadata for frozen BERT encoding."""
    import numpy as np
    from .instruction_relation_graph import tensorize_relation_graph
    packed=[tensorize_relation_graph(x) for x in graphs]
    return {
        key:torch.from_numpy(np.stack([x[key] for x in packed])).to(device)
        if isinstance(packed[0][key],np.ndarray)
        else torch.tensor([x[key] for x in packed],device=device)
        for key in packed[0]
        if key not in ("node_phrases","edge_phrases")
    }
