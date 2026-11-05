"""Polygon-conditioned *grounding*, separate from pure language parsing.

Public full contours are sampled into masked spatial tokens; every disconnected
part is kept (or rejected if above capacity) instead of collapsing to a
centroid/bbox. Candidate -> Polygon signed distance/bearing is computed from
PUBLIC map geometry and candidate XY only. No GT, no privately rendered RGB.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def _normalized_bounds(world_bounds):
    b=np.asarray(world_bounds,dtype=np.float64)
    if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<=b[:2]).any():
        raise ValueError("Finite, increasing XY map bounds required")
    return b[:2], b[2:]-b[:2]


def _parts(contours):
    regions=[]
    for c in contours:
        p=Polygon(np.asarray(c,dtype=np.float64))
        if not p.is_valid or p.area<=0:
            raise ValueError("Invalid public Polygon ring")
        regions.append(p)
    if not regions:
        raise ValueError("Public Polygon requires at least one part")
    return regions


def outline_tokens(contours, world_bounds, *, max_parts=16, vertices=24):
    """Fixed capacity [P,V,4]: XY map-frame + normalized forward tangent.

    Preserve world-map orientation; never equate XY axes with human 'left/right'.
    A polygon with more parts than max_parts fails explicitly (no truncation).
    """
    if max_parts<1 or vertices<3:
        raise ValueError("Invalid polygon token capacity")
    lo,span=_normalized_bounds(world_bounds)
    parts=_parts(contours)
    if len(parts)>max_parts:
        raise ValueError("Public Polygon exceeds part capacity; cannot truncate")
    tokens=np.zeros((max_parts,vertices,4),np.float32)
    valid=np.zeros(max_parts,dtype=bool)
    area=np.zeros(max_parts,np.float32)
    for idx,p in enumerate(parts):
        boundary=p.exterior
        # Sample the original true boundary, not just its bounding box.
        xy=np.array([boundary.interpolate(j/vertices,normalized=True).coords[0]
                     for j in range(vertices)],dtype=np.float64)
        # Canonical start gives stable cyclic shift across different ring starts.
        pivot=np.lexsort((xy[:,1],xy[:,0]))[0]
        xy=np.roll(xy,-int(pivot),axis=0)
        tang=np.roll(xy,-1,axis=0)-xy
        tang/=np.maximum(np.linalg.norm(tang,axis=-1,keepdims=True),1e-7)
        tokens[idx,:,:2]=(xy-lo)/span
        tokens[idx,:,2:]=tang
        valid[idx]=True
        area[idx]=float(p.area/np.prod(span))
    return dict(outline=tokens,part_mask=valid,
                area_ratio=area,has_polygon=True)


def named_public_outline(public_ids,public_objects,world_bounds,**kwargs):
    """Full same-name public Polygon fragments with no target oracle binding."""
    contours=[]
    for identifier in public_ids:
        obj=public_objects.get(str(identifier))
        if obj is None:
            raise ValueError("Requested public landmark ID not in catalog")
        contours.append(obj["contour"])
    if not contours:
        raise ValueError("No public polygon IDs")
    return outline_tokens(contours,world_bounds,**kwargs)


def candidate_polygon_vectors(candidate_xy_world,contours,world_bounds):
    """[Q,8] physically interpretable candidate-to-whole-Polygon descriptors.

    Signed boundary proximity distinguishes inside and outside. Bearing is raw
    map XY, NOT verified geographic north or an egocentric left/right frame.
    A named multi-part polygon is a union; no GT-selected component.
    """
    lo,span=_normalized_bounds(world_bounds)
    p=unary_union(_parts(contours))
    candidate=np.asarray(candidate_xy_world,dtype=np.float64)
    if candidate.ndim!=2 or candidate.shape[-1]!=2 or not np.isfinite(candidate).all():
        raise ValueError("Candidate XY must be finite Qx2")
    center=np.array([p.centroid.x,p.centroid.y])
    diagonal=np.linalg.norm(span)
    out=np.zeros((len(candidate),8),dtype=np.float32)
    for i,xy in enumerate(candidate):
        pt=Point(float(xy[0]),float(xy[1]))
        inside=p.covers(pt)
        bd=float(pt.distance(p.boundary))
        dist=float(pt.distance(p))
        vec=center-xy
        r=float(np.linalg.norm(vec))
        unit=vec/max(r,1e-7)
        # signed distance to nearest boundary, negative when inside
        out[i]=[( -bd if inside else bd)/diagonal,
                dist/diagonal,float(inside),
                float(unit[0]),float(unit[1]),
                float(vec[0]/span[0]),float(vec[1]/span[1]),
                float(p.area/np.prod(span))]
    return out


class PublicPolygonOutlineEncoder(nn.Module):
    """Pool each public Polygon part separately; attention over all valid parts."""

    def __init__(self,hidden=128):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(4,hidden),nn.GELU(),
                                 nn.Linear(hidden,hidden))
        self.area=nn.Linear(1,hidden)
        self.part_attn=nn.Linear(hidden,1)
        self.norm=nn.LayerNorm(hidden)

    def forward(self,outlines,part_mask,area_ratios):
        if outlines.ndim!=4 or outlines.shape[-1]!=4:
            raise ValueError("Expected [B,P,V,4] public outline tokens")
        b,p,v,_=outlines.shape
        if part_mask.shape!=(b,p) or area_ratios.shape!=(b,p):
            raise ValueError("Polygon part masks/areas do not align")
        valid=part_mask.bool()
        if not valid.any(-1).all():
            raise ValueError("Every encoded landmark needs a public Polygon")
        point=self.point(outlines.float()).mean(-2)
        part=point+self.area(area_ratios.float().unsqueeze(-1))
        weights=self.part_attn(part).squeeze(-1).masked_fill(~valid,-1e9).softmax(-1)
        pooled=(part*weights.unsqueeze(-1)).sum(-2)
        return self.norm(pooled)


class PolygonConditionedCandidateHead(nn.Module):
    """Explicit public-geometry scorer for an ALREADY grounded relation.

    Gate complete_paths=False when an indirect reference is still an unknown
    parking lot/car: these scores must remain *exactly zero* even if a
    geometrically close building exists. No hallucinated chain shortening.
    """

    def __init__(self, hidden=128, geometry_dim=8):
        super().__init__()
        self.outline=PublicPolygonOutlineEncoder(hidden)
        self.candidate=nn.Sequential(nn.Linear(geometry_dim,hidden),nn.GELU(),
                                     nn.LayerNorm(hidden))
        self.score=nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),
                                 nn.Linear(hidden,1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self,graph_target,public_outline,part_mask,area_ratios,
                candidate_geometry,complete_paths):
        if graph_target.ndim!=2 or candidate_geometry.ndim!=3:
            raise ValueError("B,H graph state and B,Q,D geometry expected")
        b,q,_=candidate_geometry.shape
        if graph_target.shape[0]!=b or complete_paths.shape!=(b,):
            raise ValueError("Batch/path mask mismatch")
        shape=self.outline(public_outline,part_mask,area_ratios)
        repeat=lambda z:z[:,None].expand(-1,q,-1)
        raw=self.score(torch.cat((repeat(graph_target.float()),
                                   repeat(shape),self.candidate(candidate_geometry)),-1)).squeeze(-1)
        return raw * complete_paths.bool()[:,None]
