"""Explicit OFFLINE TRAIN-only Oracle helpers. No deployed policy imports this."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,time
import cv2,numpy as np,torch
from .checkpointing import atomic_save,file_sha256
from .citynav_box_diagnostics import BoxAuthorization,box_world_centers
from .landmark_projection import frame_to_world


@dataclass
class PublicEvidence:
    features: torch.Tensor
    text: torch.Tensor
    text_mask: torch.Tensor
    boxes: torch.Tensor
    centers: torch.Tensor
    quality: torch.Tensor
    prior: torch.Tensor
    geometry: torch.Tensor
    native_scores: torch.Tensor
    base: torch.Tensor
    grid: torch.Tensor
    observed: torch.Tensor


def prepare_public_evidence(features,text,text_mask,boxes,centers,quality,base,grid,observed):
    """Explicit public-only whitelist; there is no label/GT/instance argument."""
    scores=(features.float()@text.float().T)[:,text_mask].sigmoid().amax(-1)
    nearest=torch.cat([torch.cdist(grid.float(),x.float()).argmin(0) for x in centers.split(64)])
    prior=base[nearest]
    geometry=torch.stack(((prior-base.max()).clamp(-20,0)/5,
        torch.logit(scores.clamp(1e-5,1-1e-5))/5,quality,boxes[:,2],boxes[:,3],boxes[:,0],boxes[:,1],boxes[:,2]*boxes[:,3]),-1)
    return PublicEvidence(features,text,text_mask,boxes,centers,quality,prior,geometry,scores,base,grid,observed)


def pack_public(e,ids,width):
    device=e.base.device;n=len(ids)
    if width<n:raise ValueError('Do not silently truncate')
    f=torch.zeros(width,256,device=device);g=torch.zeros(width,8,device=device)
    world=torch.zeros(width,2,device=device);valid=torch.zeros(width,dtype=torch.bool,device=device)
    f[:n]=e.features[ids].half().float();g[:n]=e.geometry[ids];world[:n]=e.centers[ids];valid[:n]=True
    return dict(features=f[None],geometry=g[None],world=world[None],valid=valid[None],
                text=e.text.half().float()[None],text_mask=e.text_mask[None],ids=ids)


@torch.no_grad()
def predict_packed(native,e,x):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=e.base.is_cuda):
        q,null=native(x['features'],x['text'],x['text_mask'],x['geometry'],x['valid'])
    kernel=torch.exp(-torch.cdist(e.grid[None],x['world']).square()/(2*8.**2))*e.observed[None,:,None]*x['valid'][:,None]
    belief,delta=native.update_belief(e.base[None],q,null,kernel,e.observed[None],x['valid'],bound=2.)
    return dict(q=q.float(),null=null.float(),kernel=kernel,belief=belief[0],delta=delta[0],pack=x)


class OracleRegionEncoder:
    def __init__(self,directory,model_sha):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.encoder=None;self.model_sha=model_sha

    @torch.inference_mode()
    def encode(self,authorization,bgr,depth,footprint,instruction,image_sha):
        if not isinstance(authorization,BoxAuthorization):raise TypeError('Explicit Oracle authorization required')
        x0,y0,x1,y1=authorization.bbox
        x0,y0,x1,y1=int(np.floor(x0)),int(np.floor(y0)),int(np.ceil(x1)),int(np.ceil(y1))
        crop=bgr[y0:y1,x0:x1].copy();support=depth[y0:y1,x0:x1].copy()
        if min(crop.shape[:2])<2:return dict(available=False,reason='crop_too_small')
        digest=hashlib.sha256(crop.tobytes()+support.tobytes()+instruction.encode()+self.model_sha.encode()).hexdigest()
        path=self.directory/(digest+'.pt')
        if path.exists():return torch.load(path,weights_only=True)
        if self.encoder is None:
            from .native_object_runtime import NativeObjectEncoder
            self.encoder=NativeObjectEncoder('cuda')
        corners=frame_to_world(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),footprint,bgr.shape)
        obs=dict(bgr=crop,depth_valid=support,footprint_world=corners)
        torch.cuda.synchronize();start=time.perf_counter();out=self.encoder.encode(obs,instruction);torch.cuda.synchronize()
        scores=(out['features']@out['text'].T)[:,out['text_mask']].amax(-1)
        good=(out['quality']>=.6)&torch.isfinite(scores)
        if not good.any():result=dict(available=False,reason='no_supported_query_in_real_crop')
        else:
            index=int(scores.masked_fill(~good,-1e4).argmax())
            result=dict(available=True,feature=out['features'][index].cpu().float(),
                source_image_sha=image_sha,crop_pixel_sha=hashlib.sha256(crop.tobytes()).hexdigest(),
                model_sha=self.model_sha,crop_bbox_xyxy=[x0,y0,x1,y1],selected_crop_query=index,
                selected_crop_query_box=out['boxes'][index].cpu().tolist(),seconds=time.perf_counter()-start,
                provenance='REAL_CURRENT_RGB_CROP_FROZEN_AERIALVG_QUERY_NOT_FABRICATED',oracle_only=True)
        atomic_save(result,path);return result


def append_oracle_region(e,normal_pack,authorization,real_crop_feature,footprint,shape,depth_valid):
    if not isinstance(authorization,BoxAuthorization):raise TypeError('Oracle-only box authorization required')
    from .citynav_box_diagnostics import bbox_cxcywh
    feature=real_crop_feature.to(e.base.device).float()
    if feature.shape!=(256,) or not torch.isfinite(feature).all():raise ValueError('A real finite256D crop feature required')
    box=bbox_cxcywh(authorization.bbox,shape).to(e.base.device)
    world=box_world_centers(box[None],footprint,shape).to(e.base.device)[0]
    prior=e.base[(e.grid-world).norm(dim=-1).argmin()]
    score=(feature@e.text.T)[e.text_mask].sigmoid().max()
    from .object_proposals import box_valid_fraction
    actual_quality=torch.tensor(float(box_valid_fraction(box[None].cpu().numpy(),depth_valid)[0]),device=e.base.device)
    geometry=torch.stack(((prior-e.base.max()).clamp(-20,0)/5,torch.logit(score.clamp(1e-5,1-1e-5))/5,
                          actual_quality,box[2],box[3],box[0],box[1],box[2]*box[3]))
    result=dict(normal_pack)
    result['features']=torch.cat([normal_pack['features'],feature.half().float()[None,None]],1)
    result['geometry']=torch.cat([normal_pack['geometry'],geometry[None,None]],1)
    result['world']=torch.cat([normal_pack['world'],world[None,None]],1)
    result['valid']=torch.cat([normal_pack['valid'],torch.ones(1,1,dtype=torch.bool,device=e.base.device)],1)
    return result


@torch.no_grad()
def force_selected(native,e,pred,index,authorization):
    """Select actual candidate but KEEP its learned absolute score and null."""
    if not isinstance(authorization,BoxAuthorization):raise TypeError('Offline Oracle authorization required')
    q=torch.full_like(pred['q'],-1e4);q[:,index]=pred['q'][:,index]
    belief,delta=native.update_belief(e.base[None],q,pred['null'],pred['kernel'],e.observed[None],pred['pack']['valid'],bound=2.)
    mass=float(torch.cat([q,pred['null'][:,None]],-1).softmax(-1)[0,index])
    return belief[0],mass


@torch.no_grad()
def ideal_position(native,e,position,authorization):
    """Explicit ideal evidence, no query feature is invented; bounded/unbounded."""
    if not isinstance(authorization,BoxAuthorization):raise TypeError('Offline Oracle authorization required')
    distance=(e.grid-position[:2]).norm(dim=-1)
    kernel=torch.exp(-distance.square()/(2*8.**2))[None,:,None]*e.observed[None,:,None]
    belief,_=native.update_belief(e.base[None],torch.tensor([[20.]],device=e.base.device),torch.tensor([-20.],device=e.base.device),
                                 kernel,e.observed[None],torch.ones(1,1,dtype=torch.bool,device=e.base.device),bound=2.)
    return belief[0],-distance.square()/(2*8.**2)
