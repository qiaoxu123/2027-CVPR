"""Neural MAP-XY direction heatmap conditioned on full anchor Polygon + text.

Use a frozen pretrained BERT sentence vector (optional) and train the relation
embedding plus outline->grid candidate scorer. The model NEVER accepts the
target contour. The ground-truth target polygon is used only by losses.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .inverse_polygon_direction_data import DIR
from .polygon_conditioned_relation import (
    PublicPolygonOutlineEncoder, candidate_polygon_vectors, outline_tokens,
)


def map_grid(bounds, size=28):
    b=np.asarray(bounds,float)
    if b.shape!=(4,) or (b[2:]<=b[:2]).any():
        raise ValueError("Invalid world bounds")
    u=(np.arange(size,dtype=float)+.5)/size
    gx,gy=np.meshgrid(u,u,indexing="xy")
    return np.column_stack((b[0]+gx.ravel()*(b[2]-b[0]),
                            b[1]+gy.ravel()*(b[3]-b[1]))).astype(np.float32)


def polygon_target_distribution(contour, world_grid, *, sigma_m=12.):
    from shapely.geometry import Point,Polygon
    p=Polygon(contour)
    if not p.is_valid or p.area<=0:
        raise ValueError("Target Polygon is invalid")
    if sigma_m<=0:
        raise ValueError("Invalid soft-label scale")
    d=np.array([Point(float(x),float(y)).distance(p) for x,y in world_grid])
    # A wide target gets uniform interior mass; other cells decay smoothly.
    score=-.5*(d/sigma_m)**2
    score-=score.max()
    weights=np.exp(score)
    return (weights/weights.sum()).astype(np.float32)


def example_features(row, *, size=28):
    source=row["input"]
    bounds=row["world_bounds"]
    grid=map_grid(bounds,size=size)
    contour=source["anchor_contour"]
    polygon=outline_tokens([contour],bounds)
    geometry=candidate_polygon_vectors(grid,[contour],bounds)
    label=polygon_target_distribution(row["label"]["target_contour"],grid)
    return dict(
        polygon_outline=polygon["outline"],polygon_parts=polygon["part_mask"],
        polygon_areas=polygon["area_ratio"],geometry=geometry,
        target_distribution=label,world_grid=grid,
        word=source["relation_phrase"],text=source["instruction"],
        truth=row["label"]["truth"],
        family_id=row["family_id"],orientation=row["source"]["orientation"],
        bounds=bounds,
    )


class DirectionalPolygonBelief(nn.Module):
    """Predict full grid logits from input relation text and full source outline.

    A relation embedding is a trainable text-phrase vocabulary adapter.
    Frozen BERT sentence embeddings can additionally condition the
    score. This prototype uses map-relative unambiguous words only.
    """

    def __init__(self,hidden=96,text_dim=768):
        super().__init__()
        self.hidden=hidden
        self.direction=nn.Embedding(len(DIR),hidden)
        self.outline=PublicPolygonOutlineEncoder(hidden)
        self.candidate=nn.Sequential(nn.Linear(8,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.text=nn.Sequential(nn.Linear(text_dim,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.gate=nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),
                                nn.Linear(hidden,hidden),nn.Sigmoid())
        self.scorer=nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),
                                  nn.Linear(hidden,1))
        self.register_buffer("axes",torch.tensor(list(DIR.values()),dtype=torch.float32))

    def forward(self, geometry, outline, part_mask, areas, direction_id,
                text_emb=None):
        if geometry.ndim!=3 or outline.ndim!=4:
            raise ValueError("Expected B,Q,8 and B,P,V,4")
        b,q,c=geometry.shape
        if c!=8 or direction_id.shape!=(b,):
            raise ValueError("Invalid geometry/direction input")
        base=self.outline(outline,part_mask,areas)
        direction=self.direction(direction_id)
        text=(self.text(text_emb) if text_emb is not None
              else torch.zeros_like(base))
        cond=self.gate(torch.cat((base,direction,text),-1))
        attrs=(base+direction+text)*cond
        geo=self.candidate(geometry)
        scores=self.scorer(torch.cat((geo,attrs[:,None].expand(-1,q,-1),
                                       geo*attrs[:,None]),-1)).squeeze(-1)
        return scores


def collate_direction_features(rows,device):
    feats=[example_features(row) for row in rows]
    def array(key,dtype=None):
        return torch.tensor(np.stack([x[key] for x in feats]),dtype=dtype,device=device)
    return dict(
        geometry=array("geometry",torch.float32),
        outline=array("polygon_outline",torch.float32),
        part_mask=array("polygon_parts",torch.bool),
        areas=array("polygon_areas",torch.float32),
        direction_id=torch.tensor([list(DIR).index(x["word"]) for x in feats],
                                  device=device),
        truth=torch.tensor([x["truth"] for x in feats],dtype=torch.bool,device=device),
        soft_target=array("target_distribution",torch.float32),
        text=[x["text"] for x in feats],
        world_grid=array("world_grid",torch.float32),
        family_id=[x["family_id"] for x in feats],
        orientation=[x["orientation"] for x in feats],
        bounds=[x["bounds"] for x in feats],
    )
