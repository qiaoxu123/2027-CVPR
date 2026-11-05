"""Span-to-relation neural parser head trained with HUMAN/TEACHER corrected graphs.

Inputs: frozen BERT token features and candidate noun-phrase token masks.
Outputs: target/landmark/intermediate roles + DIRECTED relation predicate
and reference-frame labels for each ordered pair of spans.

The candidate spans come from a separate noun-phrase proposer; this module
does not yet discover spans end-to-end from raw text. No target coordinates,
RGB or GT identity enter forward().
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .instruction_relation_graph import ROLES, PREDICATES, FRAMES

# Separate absence-of-edge from genuine "unknown predicate".
EDGE_CLASSES = ("none",) + PREDICATES


class InstructionSpanGraphParser(nn.Module):
    def __init__(self, text_dim=768, hidden=192, dropout=0.1):
        super().__init__()
        self.phrase=nn.Sequential(nn.Linear(text_dim,hidden),nn.GELU(),
                                  nn.LayerNorm(hidden),nn.Dropout(dropout))
        self.role=nn.Linear(hidden,len(ROLES))
        self.edge=nn.Sequential(nn.Linear(hidden*4,hidden),nn.GELU(),
                                nn.LayerNorm(hidden),nn.Dropout(dropout),
                                nn.Linear(hidden,len(EDGE_CLASSES)))
        self.frame=nn.Sequential(nn.Linear(hidden*4,hidden),nn.GELU(),
                                 nn.Linear(hidden,len(FRAMES)))

    def forward(self, token_hidden, span_token_mask, span_valid):
        if token_hidden.ndim!=3 or span_token_mask.ndim!=3:
            raise ValueError("Expected B,T,D tokens and B,N,T span masks")
        b,t,_=token_hidden.shape
        if span_token_mask.shape[0]!=b or span_token_mask.shape[2]!=t:
            raise ValueError("Span-token alignment mismatch")
        if span_valid.shape!=span_token_mask.shape[:2]:
            raise ValueError("Invalid span_valid shape")
        valid=span_valid.bool()
        mask=span_token_mask.bool()&valid[:,:,None]
        if (valid&~mask.any(-1)).any():
            raise ValueError("Valid noun phrase with zero aligned BERT tokens")
        weights=mask.to(token_hidden.dtype)
        pooled=torch.bmm(weights,token_hidden)/weights.sum(-1,keepdim=True).clamp_min(1)
        h=self.phrase(pooled)*valid[:,:,None]
        src=h[:,:,None,:].expand(-1,-1,h.shape[1],-1)
        dst=h[:,None,:,:].expand(-1,h.shape[1],-1,-1)
        pair=torch.cat((src,dst,src-dst,src*dst),-1)
        pair_mask=valid[:,:,None]&valid[:,None,:]
        # Self-loops never express a reference relation.
        pair_mask=pair_mask&~torch.eye(h.shape[1],device=h.device,dtype=torch.bool)[None]
        return dict(role_logits=self.role(h),edge_logits=self.edge(pair),
                    frame_logits=self.frame(pair),valid_nodes=valid,
                    valid_pairs=pair_mask)


def supervised_parser_loss(output, role_labels, edge_labels, frame_labels,
                           *, none_weight=0.2):
    """Teacher-gold labels, with -100 only on missing/unannotated entries.

    Role labels: 0..len(ROLES)-1, edge: 0=none, 1..=predicate+1,
    frame: 0..len(FRAMES)-1 for annotated positive edge, -100 otherwise.
    """
    r=role_labels.long()
    e=edge_labels.long()
    f=frame_labels.long()
    valid_node=output["valid_nodes"]&(r>=0)
    valid_pair=output["valid_pairs"]&(e>=0)
    valid_frame=valid_pair&(e>0)&(f>=0)
    if not valid_node.any() or not valid_pair.any():
        raise ValueError("No supervised nodes/edges")
    role=F.cross_entropy(output["role_logits"][valid_node],r[valid_node])
    weights=torch.ones(len(EDGE_CLASSES),device=e.device)
    weights[0]=none_weight
    edge=F.cross_entropy(output["edge_logits"][valid_pair],e[valid_pair],weight=weights)
    frame=(F.cross_entropy(output["frame_logits"][valid_frame],f[valid_frame])
           if valid_frame.any() else edge*0)
    return dict(total=role+edge+0.3*frame,role=role,edge=edge,frame=frame)
