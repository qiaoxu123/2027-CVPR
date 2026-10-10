"""Multi-scale pooling of an actual 384px SigLIP2 observation's patch tokens.

This reuses contextual patch tokens: it is NOT an independently re-encoded ROI
or a higher resolution view. No source raster or target labels are accepted.
"""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .landmark_projection import world_to_frame, frame_to_world


def observed_region_masks(observation, centers_world, *, patch_grid=24,
                          extents=(32.,64.,128.), min_support=.9):
    rgb = observation['bgr']
    if rgb.shape != (384,384,3):
        raise ValueError('native current 384px observation required; no upsampling')
    support = (rgb.max(-1)>0) & np.asarray(observation['depth_valid'],bool)
    centers = np.asarray(centers_world,np.float32)
    if centers.ndim!=2 or centers.shape[1]!=2 or not np.isfinite(centers).all():
        raise ValueError('finite public candidate XY required')
    if support.shape!=(384,384) or patch_grid!=24:
        raise ValueError('expected Large/384 patch16 observation geometry')
    positions = (np.arange(patch_grid)+.5)*(384./patch_grid)-.5
    xx,yy = np.meshgrid(positions,positions)
    pixels = np.stack((xx.ravel(),yy.ravel()),-1)
    world = frame_to_world(pixels,observation['footprint_world'],rgb.shape)
    patch_support = support.reshape(24,16,24,16).mean((1,3)).ravel()>=min_support
    masks = np.zeros((len(centers),len(extents),576),bool)
    quality = np.zeros((len(centers),len(extents),3),np.float32)
    offset = np.array([[-1,-1],[1,-1],[1,1],[-1,1]],np.float32)
    for scale,extent in enumerate(extents):
        if extent<=0:raise ValueError('positive metric ROI extents required')
        quads = centers[:,None]+offset[None]*extent/2
        projected = world_to_frame(quads.reshape(-1,2),observation['footprint_world'],rgb.shape).reshape(-1,4,2)
        inside = ((projected>=0)&(projected<=383)).all((1,2))
        for i in np.flatnonzero(inside):
            span = abs(cv2.contourArea(projected[i]))**.5
            if span<24:continue
            roi = np.zeros((384,384),np.uint8)
            cv2.fillConvexPoly(roi,np.rint(projected[i]).astype(np.int32),1)
            fraction = float(support[roi.astype(bool)].mean())
            if fraction<min_support:continue
            mask = (np.abs(world-centers[i])<=extent/2).all(-1) & patch_support
            if not mask.any():continue
            masks[i,scale] = mask
            quality[i,scale] = [span/384.,fraction,extent/128.]
    return masks,quality


def masked_native_pool(head, patches, masks):
    """Exact native SigLIP2 attention pooling, sharing its K/V projections.

patches: [P,D], masks:[R,P], every row nonempty. All-ones matches HF head.
The frozen patch transformer attends the current frame, including ROI context.
"""
    if patches.ndim!=2 or masks.ndim!=2 or masks.shape[1]!=patches.shape[0]:
        raise ValueError('patch and region dimensions differ')
    if masks.dtype!=torch.bool or not masks.any(-1).all():
        raise ValueError('only nonempty boolean region masks can be pooled')
    attention = head.attention
    if attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn:
        raise ValueError('unsupported native attention configuration')
    dim = patches.shape[-1];heads = attention.num_heads;part = dim//heads
    weights = attention.in_proj_weight;bias = attention.in_proj_bias
    query = F.linear(head.probe[0],weights[:dim],None if bias is None else bias[:dim])
    key = F.linear(patches,weights[dim:2*dim],None if bias is None else bias[dim:2*dim])
    value = F.linear(patches,weights[2*dim:],None if bias is None else bias[2*dim:])
    query = query.reshape(heads,part)
    key = key.reshape(-1,heads,part).transpose(0,1)
    value = value.reshape(-1,heads,part).transpose(0,1)
    logits = (query[:,None].float()*key.float()).sum(-1)/(part**.5)
    logits = logits[None].expand(len(masks),-1,-1).masked_fill(~masks[:,None],float('-inf'))
    probability = logits.softmax(-1).to(value.dtype)
    pooled = torch.einsum('rhp,hpd->rhd',probability,value).reshape(len(masks),dim)
    pooled = attention.out_proj(pooled)
    return pooled + head.mlp(head.layernorm(pooled))


class ObservedPatchScaleEncoder:
    def __init__(self,native):
        self.native = native

    @torch.inference_mode()
    def encode(self,observations,centers_world):
        if len(observations)!=len(centers_world):raise ValueError('observation/geometry mismatch')
        native = self.native;device=native.device
        batch = native.processor(images=[cv2.cvtColor(o['bgr'],cv2.COLOR_BGR2RGB) for o in observations],
                                 return_tensors='pt').to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            vision = native.model.vision_model(**batch)
            output=[]
            for observation,centers,patches in zip(observations,centers_world,vision.last_hidden_state):
                masks,quality = observed_region_masks(observation,centers)
                valid = masks.any(-1)
                values = torch.zeros(*valid.shape,native.dimension,device=device,dtype=torch.float32)
                if valid.any():
                    region_masks = torch.from_numpy(masks[valid]).to(device)
                    pooled = masked_native_pool(native.model.vision_model.head,patches,region_masks)
                    values[torch.from_numpy(valid).to(device)] = F.normalize(pooled.float(),dim=-1)
                output.append((values.detach().cpu().half().clone(),torch.from_numpy(valid),
                               torch.from_numpy(quality)))
        return output
