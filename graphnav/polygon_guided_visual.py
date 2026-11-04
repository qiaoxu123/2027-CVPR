"""Standalone Polygon-Guided Visual Belief scoring, without a map-position shortcut."""
import numpy as np
import shapely
import torch
from torch import nn
from torch.nn import functional as F
from .landmark_region_supervision import named_union
from .landmark_projection import world_to_frame


def polygon_shape_tokens(contours,footprint,vertices=16):
    boundary=named_union(contours).boundary
    if boundary.length<=0:raise ValueError('Nonempty public polygon boundary required')
    points=np.asarray([boundary.interpolate(t,normalized=True).coords[0] for t in np.arange(vertices)/vertices])
    pixels=world_to_frame(points,footprint,(384,384,3))/383.
    # Keep metric scale/orientation relative to observation, remove ALL translation.
    shape=pixels-pixels.mean(0)
    delta=np.roll(shape,-1,axis=0)-shape
    return np.concatenate((shape,delta),-1).astype(np.float32)


class PolygonGuidedVisualBelief(nn.Module):
    def __init__(self,dimension=1024,hidden=64,heads=4):
        super().__init__()
        self.image=nn.Sequential(nn.Linear(dimension,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.text=nn.Linear(dimension*2,hidden)
        self.polygon=nn.Sequential(nn.Linear(4,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.attention=nn.MultiheadAttention(hidden,heads,batch_first=True,dropout=0.)
        self.score=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.log_scale=nn.Parameter(torch.tensor(2.3))

    def forward(self,*,image_features,name_text,type_text,polygon_tokens,valid):
        if image_features.ndim!=4 or valid.shape!=image_features.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('Observed multiscale image features and bool validity required')
        if not all(torch.isfinite(x).all() for x in (image_features,name_text,type_text,polygon_tokens)):
            raise ValueError('Nonfinite visual input')
        image=F.normalize(image_features.float(),dim=-1)
        vi=self.image(image).mean(2)
        text=self.text(torch.cat((F.normalize(name_text.float(),dim=-1),F.normalize(type_text.float(),dim=-1)),-1))
        queries=text[:,None]+self.polygon(polygon_tokens.float())
        safe=valid.clone();empty=~safe.any(-1);safe[empty,0]=True
        vi=vi.masked_fill(~valid[...,None],0)
        context,_=self.attention(queries,vi,vi,key_padding_mask=~safe,need_weights=False)
        context=context.mean(1)
        residual=self.score(torch.cat((vi*text[:,None],vi*context[:,None]),-1)).squeeze(-1)
        native=(image[:,:,0]*F.normalize((name_text+3*type_text).float(),dim=-1)[:,None]).sum(-1)
        return (self.log_scale.exp().clamp(max=30)*native+residual).masked_fill(~valid,0)
