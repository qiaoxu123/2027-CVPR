"""Two-layer typed instruction graph -> candidate-specific PUBLIC Polygon Belief.

IMPORTANT: this is a residual on the existing COMPLETE-INSTRUCTION Unified V2
Geometry+Relation Belief. Never replace its full BERT tokens, public Polygon,
controller, Stop, or target-feature signal. Parsed graphs are weak proposals.

No GT, future trajectory, or privileged target-selected anchor enters forward.
A target node without a verified public *path endpoint* produces no residual.
The geometry of target/intermediate nodes stays unknown, not copied from GT.

No positional embeddings tied to arbitrary node ordering; scatter aggregation
over typed directed edges makes the encoder permutation equivariant. A source
sentence reordering is invariant ONLY IF parsing yields the same graph.
"""
from __future__ import annotations

import torch
from torch import nn

from .instruction_relation_graph import ROLES
from .instruction_relation_encoder import InstructionRelationGraphEncoder


class UnifiedInstructionGraphBelief(nn.Module):
    def __init__(self, *, hidden=128, text_dim=768, layers=2,
                 max_logit_delta=1.0):
        super().__init__()
        if hidden < 8 or layers < 1 or max_logit_delta <= 0:
            raise ValueError("Invalid graph Belief capacity")
        self.max_logit_delta=float(max_logit_delta)
        self.encoder=InstructionRelationGraphEncoder(
            text_dim=text_dim,hidden=hidden,layers=layers,dropout=0.0)
        self.candidate=nn.Sequential(nn.LayerNorm(hidden),
                                     nn.Linear(hidden,hidden))
        self.geo=nn.Sequential(nn.LayerNorm(10),nn.Linear(10,hidden),
                               nn.GELU(),nn.LayerNorm(hidden))
        self.fuse=nn.Sequential(
            nn.LayerNorm(4*hidden),
            nn.Linear(4*hidden,hidden),nn.GELU(),nn.LayerNorm(hidden),
        )
        self.anchor_weight=nn.Linear(hidden,1)
        self.anchor_score=nn.Linear(hidden,1)
        # Zero residual on day 1; existing full-text Unified V2 exactly
        # reproduces its verified score before graph learning.
        nn.init.zeros_(self.anchor_score.weight)
        nn.init.zeros_(self.anchor_score.bias)

    def forward(self, *, packed, node_text, edge_text,
                dense_context, pair_geometry, node_anchor, base_logits,
                return_diagnostics=False):
        """Inputs:
            packed: PR19 graph tensors, e.g. node roles, directed edge index.
            node_text [B,N,768], edge_text [B,E,768]: frozen BERT *phrases*.
            dense_context [B,Q,128]: frozen existing Unified V2 context.
            pair_geometry [B,Q,M,10]: existing candidate-to-public-Polygon.
            node_anchor [B,N,M]: binary public-ID binding of path TERMINAL
              landmark nodes to current source public Polygon slots.
            base_logits [B,Q]: existing complete-instruction Goal Belief.

        Target/anchor public XY not supplied separately; all candidate geometry
        is from the existing candidate-to-Polygon field, never from GT.
        """
        if (dense_context.ndim!=3 or pair_geometry.ndim!=4 or
                node_anchor.ndim!=3 or base_logits.ndim!=2):
            raise ValueError("Invalid Unified graph/Polygon tensor rank")
        b,q,h=dense_context.shape
        n=node_text.shape[1]
        m=pair_geometry.shape[2]
        if (h!=self.candidate[0].normalized_shape[0] or
                base_logits.shape!=(b,q) or pair_geometry.shape!=(b,q,m,10)
                or node_anchor.shape!=(b,n,m) or
                node_text.shape[0]!=b or edge_text.shape[0]!=b):
            raise ValueError("Misaligned Unified candidate/graph/Polygon tensors")
        if (not torch.isfinite(base_logits).all() or
                not torch.isfinite(pair_geometry).all() or
                not torch.isfinite(node_anchor).all()):
            raise ValueError("Nonfinite model inputs")
        if (node_anchor<0).any():
            raise ValueError("Anchor assignment must be nonnegative")
        # Only public landmarks at the END of an actual target-anchored
        # path are marked by the data adapter. Never score a nearby public
        # polygon merely because it appears in the map catalog.
        roles=packed["node_roles"].long()
        node_valid=packed["node_mask"].bool()
        is_landmark=roles==ROLES.index("public_landmark")
        count=node_anchor.sum(-1)
        eligible=(count>0)&is_landmark&node_valid
        support=eligible.any(-1)&(packed["target_index"].long()>=0)
        if not support.any():
            zeros=torch.zeros_like(base_logits.float())
            result=dict(logits=base_logits.float(),delta=zeros,supported=support)
            if return_diagnostics:
                result["anchor_weights"]=base_logits.new_zeros(b,q,n)
            return result
        graph=self.encoder(packed,node_text.float(),edge_text.float())
        support=support&graph["target_present"]
        candidate=self.candidate(dense_context.float())
        target=graph["target_state"].float()
        node=graph["node_states"].float()
        # Multiple public same-name polygons are averaged as candidates, not
        # selected by GT proximity. All existing M public anchors are usable.
        weights=node_anchor.float()/count.clamp_min(1)[...,None]
        geo=torch.einsum("bqmc,bnm->bqnc",pair_geometry.float(),weights)
        geo=self.geo(geo)
        fused=self.fuse(torch.cat((
            candidate[:,:,None,:].expand(-1,-1,n,-1),
            target[:,None,None,:].expand(-1,q,n,-1),
            node[:,None,:,:].expand(-1,q,-1,-1),
            geo,
        ),dim=-1))
        valid=eligible.clone()
        valid[~valid.any(-1),0]=True
        anchor_logits=self.anchor_weight(fused).squeeze(-1)
        anchor_logits=anchor_logits.masked_fill(~valid[:,None,:],-1e4)
        attention=anchor_logits.softmax(-1)*eligible[:,None,:].float()
        local=self.anchor_score(fused).squeeze(-1)
        residual=(attention*local).sum(-1)
        delta=self.max_logit_delta*torch.tanh(residual)*support[:,None].float()
        result=dict(logits=base_logits.float()+delta,delta=delta,
                    supported=support)
        if return_diagnostics:
            result.update(anchor_weights=attention,
                          target_present=graph["target_present"])
        return result
