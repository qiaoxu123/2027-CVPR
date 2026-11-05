"""Standalone minimalist CityNav goal-belief model.

ONLY inputs:
  - public landmark names (frozen BERT embeddings);
  - directly adjacent geometric words (frozen BERT embeddings);
  - corresponding public Polygon contours and regular 28x28 candidate grid.

No original instruction, target type, appearance, map imagery, intermediate
objects, original pretrained Geometry, dynamic maps or teacher trajectories.
GT target_xy is used externally ONLY to compute training/evaluation loss.

A missing named landmark produces a UNIFORM spatial belief rather than a
hallucinated target-specific output. This may have a low overall R20 when
many instructions have no public anchor: that is an honest scope limitation.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .geometry import candidate_polygon_geometry


class LandmarkGeometryWordsOnly(nn.Module):
    def __init__(self, text_dim=768, hidden=96, max_anchors=16):
        super().__init__()
        if hidden<8 or max_anchors<1:
            raise ValueError("Invalid compact model configuration")
        self.max_anchors=max_anchors
        self.name=nn.Sequential(nn.Linear(text_dim,hidden),nn.LayerNorm(hidden))
        self.words=nn.Sequential(nn.Linear(text_dim,hidden),nn.LayerNorm(hidden))
        self.polygon=nn.Sequential(nn.Linear(9,hidden),nn.GELU(),
                                   nn.LayerNorm(hidden))
        self.merge=nn.Sequential(nn.LayerNorm(hidden*3),
                                 nn.Linear(hidden*3,hidden),nn.GELU(),
                                 nn.Linear(hidden,1))

    def forward(self, *, name_embedding, words_embedding, landmark_contours,
                landmark_valid, words_valid, candidate_xy):
        """[B,Q] trainable target heatmap from ONLY landmark/geometry-word pairs."""
        if (name_embedding.ndim!=3 or words_embedding.shape!=name_embedding.shape
                or landmark_contours.ndim!=4 or candidate_xy.ndim!=3):
            raise ValueError("Expected B,M,text; B,M,points,XY; B,Q,XY")
        b,m,dim=name_embedding.shape
        if (m>self.max_anchors or
                landmark_contours.shape[:2]!=(b,m) or
                landmark_contours.shape[-1]!=2 or
                landmark_valid.shape!=(b,m) or words_valid.shape!=(b,m) or
                candidate_xy.shape!=(b,candidate_xy.shape[1],2)):
            raise ValueError("Invalid landmark/candidate capacity or shapes")
        if not torch.isfinite(landmark_contours).all():
            raise ValueError("Nonfinite public contours")
        if not torch.isfinite(candidate_xy).all():
            raise ValueError("Nonfinite candidate XY")
        # The 9d geometry is calculated for ALL 28x28 candidates, not chosen
        # from GT-near cells. Padded polygon parts are masked below.
        geometry=candidate_polygon_geometry(
            candidate_xy.float(),landmark_contours.float()
        )
        q=geometry.shape[1]
        name=self.name(name_embedding.float())
        words=self.words(words_embedding.float())*words_valid[...,None].float()
        polygon=self.polygon(geometry.float())
        combined=torch.cat((
            polygon,
            name[:,None,:,:].expand(-1,q,-1,-1),
            words[:,None,:,:].expand(-1,q,-1,-1),
        ),dim=-1)
        each=self.merge(combined).squeeze(-1)
        valid=landmark_valid.bool()
        scores=each.masked_fill(~valid[:,None,:],-1e4)
        fused=torch.logsumexp(scores,-1)-valid.sum(-1).clamp_min(1).log()[:,None]
        # Empty or unrecognized map-reference input has NO learned global
        # goal shortcut: the model is correctly unable to localize it.
        return torch.where(valid.any(-1)[:,None],fused,torch.zeros_like(fused))
