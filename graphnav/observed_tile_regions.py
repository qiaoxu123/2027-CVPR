"""Fixed current-view tiles; world sites and support never consume target labels."""
import cv2
import numpy as np
from .landmark_projection import frame_to_world


def split_current_tiles(bgr):
    if bgr.shape!=(768,768,3) or bgr.dtype!=np.uint8:
        raise ValueError('actual uint8 768x768 current view required')
    return [bgr[y:y+384,x:x+384] for y in (0,384) for x in (0,384)]


def tile_region_inputs(observation384,observation768,*,dense=False):
    low,high=observation384,observation768
    if low['bgr'].shape!=(384,384,3) or high['bgr'].shape!=(768,768,3):
        raise ValueError('paired actual 384/768 views required')
    if low['depth_valid'].shape!=(384,384) or high['depth_valid'].shape!=(768,768):
        raise ValueError('matching depth validity required')
    if not np.allclose(low['footprint_world'],high['footprint_world'],atol=1e-6,rtol=0) or not np.allclose(low['pose'],high['pose'],atol=1e-6,rtol=0):
        raise ValueError('same camera pose and world footprint required')
    support=(low['bgr'].max(-1)>0)&np.asarray(low['depth_valid'],bool)
    common=cv2.resize(support.astype(np.uint8),(768,768),interpolation=cv2.INTER_NEAREST).astype(bool)
    common&=np.asarray(high['depth_valid'],bool)&(high['bgr'].max(-1)>0)
    common&=cv2.resize(low['bgr'],(768,768),interpolation=cv2.INTER_LINEAR).max(-1)>0
    token_pixels=[];token_valid=[]
    axis=(np.arange(24)+.5)*16-.5;xx,yy=np.meshgrid(axis,axis)
    for y in (0,384):
        for x in (0,384):
            token_pixels.append(np.stack((xx.ravel()+x,yy.ravel()+y),-1))
            block=common[y:y+384,x:x+384]
            token_valid.append(block.reshape(24,16,24,16).mean((1,3)).ravel()>=.9)
    token_pixels=np.concatenate(token_pixels)*383/767
    token_valid=np.concatenate(token_valid)
    axis=np.arange(24) if dense else np.arange(1,24,2)
    gy,gx=np.meshgrid(axis,axis,indexing='ij');centers=np.stack((gx.ravel(),gy.ravel()),-1)
    pixels=(centers+.5)*16-.5
    masks=np.zeros((len(centers),3,2304),bool)
    patch=support.reshape(24,16,24,16).mean((1,3))>=.9
    quality=np.zeros((len(centers),6),np.float32)
    for i,(x,y) in enumerate(centers):
        for scale,radius in enumerate((0,1,3)):
            half=(radius*2+1)*8
            masks[i,scale]=(np.abs(token_pixels-pixels[i])<=half).all(-1)&token_valid
            quality[i,scale]=patch[max(0,y-radius):min(24,y+radius+1),max(0,x-radius):min(24,x+radius+1)].mean()
    valid=patch[centers[:,1],centers[:,0]]&masks.any(-1).all(-1)
    masks&=valid[:,None,None]
    footprint=np.asarray(low['footprint_world'],np.float32);span=float(np.linalg.norm(footprint[1]-footprint[0]))
    quality[:,3]=span/24/100;quality[:,4]=float(support.mean());quality[:,5]=span/200
    world=frame_to_world(pixels,low['footprint_world'],low['bgr'].shape)
    return dict(world=np.asarray(world,np.float32),masks=masks,valid=valid,quality=quality,
                common_token_support=token_valid,source_patch_indices=centers)
