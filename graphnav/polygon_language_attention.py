"""Lightweight candidate-conditioned language attention over PUBLIC Polygon geometry.

Input is the already frozen Geometry+Relation Belief and its dense context.
Every 28x28 candidate separately attends to full BERT tokens, guided by
candidate-to-PUBLIC-contour geometry and public name embeddings. No RGB, no
target location or label enters forward. A bounded zero-initialized residual
protects the verified static baseline at initialization.

Token attention is NOT a causal attribution score. Phrase swap/masking
controls are necessary before asserting a spatial word is "most important".
"""
from __future__ import annotations

import math
import torch
from torch import nn


class PolygonGuidedLanguageAttention(nn.Module):
    def __init__(self, context_dim=128, text_dim=768, geometry_dim=10,
                 hidden=96, max_delta=0.5):
        super().__init__()
        if context_dim<1 or hidden<8 or geometry_dim<1 or max_delta<=0:
            raise ValueError("Invalid Polygon Language Attention settings")
        self.max_delta=float(max_delta)
        self.geom=nn.Sequential(nn.LayerNorm(geometry_dim),
                                nn.Linear(geometry_dim,hidden),
                                nn.GELU(),nn.LayerNorm(hidden))
        self.names=nn.Sequential(nn.Linear(text_dim,hidden),nn.LayerNorm(hidden))
        self.candidate=nn.Sequential(nn.Linear(context_dim,hidden),
                                    nn.LayerNorm(hidden))
        self.lang_keys=nn.Linear(text_dim,hidden,bias=False)
        self.lang_values=nn.Linear(text_dim,hidden,bias=False)
        self.query=nn.Sequential(nn.Linear(hidden*2,hidden),nn.LayerNorm(hidden))
        self.score=nn.Sequential(nn.LayerNorm(context_dim+2*hidden),
                                 nn.Linear(context_dim+2*hidden,hidden),
                                 nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    @staticmethod
    def _content_tokens(token_mask):
        """Exclude BERT [CLS] and terminal [SEP], not the actual words."""
        content=token_mask.bool().clone()
        b,l=content.shape
        lengths=content.long().sum(-1)
        content[:,0]=False
        end=(lengths-1).clamp(min=0)
        content[torch.arange(b,device=content.device),end]=False
        return content

    def forward(self, *, base_logits, dense_context, pair_geometry,
                entity_valid, node_emb, tokens, token_mask,
                return_attention=False):
        """Return corrected [B,Q] logits; no label or GT candidate filtering.

        pair_geometry: [B,Q,N,10] public candidate-to-Polygon features.
        node_emb: [B,N,768] public name BERT features.
        tokens: [B,L,768] frozen instruction BERT sequence features.
        entity_valid: [B,N] public geometry support, padded references False.
        """
        if (base_logits.ndim!=2 or dense_context.ndim!=3 or
                pair_geometry.ndim!=4 or tokens.ndim!=3 or
                node_emb.ndim!=3 or token_mask.ndim!=2 or
                entity_valid.ndim!=2):
            raise ValueError("Invalid public geometry or text feature ranks")
        b,q=base_logits.shape
        n=pair_geometry.shape[2]
        if (dense_context.shape[:2]!=(b,q) or
                dense_context.shape[-1]!=self.candidate[0].in_features or
                pair_geometry.shape!=(b,q,n,self.geom[0].normalized_shape[0]) or
                entity_valid.shape!=(b,n) or
                node_emb.shape!=(b,n,self.names[0].in_features) or
                tokens.shape[0]!=b or tokens.shape[-1]!=self.lang_keys.in_features or
                token_mask.shape!=tokens.shape[:2]):
            raise ValueError("Mismatched candidate / entity / token dimensions")
        if not entity_valid.any():
            result=dict(logits=base_logits.float(),delta=torch.zeros_like(base_logits.float()),
                        supported=torch.zeros(b,dtype=torch.bool,device=base_logits.device))
            if return_attention:
                result.update(token_weights=base_logits.new_zeros(b,q,tokens.shape[1]),
                              entity_weights=base_logits.new_zeros(b,q,n))
            return result
        if not torch.isfinite(base_logits).all() or not torch.isfinite(pair_geometry).all():
            raise ValueError("Nonfinite candidate/Polygon geometry")
        names=self.names(node_emb.float())
        geometry=self.geom(pair_geometry.float())
        query=self.candidate(dense_context.float())
        # Each candidate picks its own public anchor(s), without GT selection.
        entity_scores=((geometry+query[:,:,None,:])*names[:,None,:,:]).sum(-1)/math.sqrt(names.shape[-1])
        has_anchor=entity_valid.any(-1)
        safe_entities=entity_valid.bool().clone()
        safe_entities[~has_anchor,0]=True
        entity_scores=entity_scores.masked_fill(~safe_entities[:,None,:],-1e4)
        entity_weights=entity_scores.softmax(-1)
        entity_weights=entity_weights*entity_valid[:,None,:].float()
        polygon_context=(entity_weights.unsqueeze(-1)*geometry).sum(-2)
        # Token weights vary across candidate positions and landmark contours.
        content=self._content_tokens(token_mask)
        content_ok=content.any(-1)
        safe_content=content.clone()
        safe_content[~content_ok,0]=True
        k=self.lang_keys(tokens.float());v=self.lang_values(tokens.float())
        language_query=self.query(torch.cat((query,polygon_context),-1))
        scores=torch.bmm(language_query,k.transpose(1,2))/math.sqrt(k.shape[-1])
        scores=scores.masked_fill(~safe_content[:,None,:],-1e4)
        token_weights=scores.softmax(-1)*content[:,None,:].float()
        lang_context=torch.bmm(token_weights,v)
        features=torch.cat((dense_context.float(),polygon_context,lang_context),-1)
        raw=self.score(features).squeeze(-1)
        supported=has_anchor&content_ok
        delta=self.max_delta*torch.tanh(raw)*supported[:,None]
        result=dict(logits=base_logits.float()+delta,delta=delta,supported=supported)
        if return_attention:
            result.update(token_weights=token_weights,entity_weights=entity_weights)
        return result
