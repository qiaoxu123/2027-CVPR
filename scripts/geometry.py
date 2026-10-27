"""Observed geometry only; world units are metres, angles use explicit frames."""
import cv2
import numpy as np

EDGE_DIM=18
TRIAD_DIM=4


def centroid(poly):
    p=np.asarray(poly,np.float32);m=cv2.moments(p)
    if abs(m['m00'])>1e-8:return np.array([m['m10']/m['m00'],m['m01']/m['m00']],np.float32)
    return (p.min(0)+p.max(0))/2


def sample_surface(surface,valid,polygon):
    """Same weak visible-interior statistic for boxes, references and label masks."""
    p=np.asarray(polygon,np.float32);lo=np.floor(p.min(0)).astype(int);hi=np.ceil(p.max(0)).astype(int)+1
    result=dict(z=0.,valid=False,coverage=0.,visible_fraction=0.,spread=0.)
    if not np.isfinite(p).all() or min(hi-lo)<=0:return result
    # Bound allocation for invalid projections before creating a polygon mask.
    if max(hi-lo)>max(surface.shape)*16:return result
    mask=np.zeros((hi[1]-lo[1],hi[0]-lo[0]),np.uint8)
    cv2.fillPoly(mask,[np.rint(p-lo).astype(np.int32)],1)
    radius=max(1,int(round(min(hi-lo)*.05)))
    mask=cv2.erode(mask,np.ones((2*radius+1,2*radius+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    full=int(mask.sum())
    if not full:return result
    x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[surface.shape[1],surface.shape[0]])
    if x1<=x0 or y1<=y0:return result
    local=mask[y0-lo[1]:y1-lo[1],x0-lo[0]:x1-lo[0]]>0
    z=surface[y0:y1,x0:x1];ok=valid[y0:y1,x0:x1]&np.isfinite(z)&local
    visible=int(local.sum());coverage=float(ok.sum()/max(visible,1))
    result.update(coverage=coverage,visible_fraction=visible/full)
    if visible/full<.8 or coverage<.7 or ok.sum()<4:return result
    low,high=np.quantile(z[ok],[.1,.9]);spread=float(high-low)
    result['spread']=spread
    if spread>8:return result
    result.update(z=float(np.median(z[ok])),valid=True)
    return result


def depth_patch(surface,valid,polygon,origin):
    p=np.asarray(polygon,np.float32);h,w=surface.shape
    lo=np.maximum(np.floor(p.min(0)).astype(int),0);hi=np.minimum(np.ceil(p.max(0)).astype(int)+1,[w,h])
    out=np.zeros((2,32,32),np.float32)
    if (hi<=lo).any():return out
    x0,y0=lo;x1,y1=hi
    mask=np.zeros((y1-y0,x1-x0),np.uint8);cv2.fillPoly(mask,[np.rint(p-lo).astype(np.int32)],1)
    good=valid[y0:y1,x0:x1]&np.isfinite(surface[y0:y1,x0:x1])&(mask>0)
    z=np.where(good,np.clip((surface[y0:y1,x0:x1]-origin)/20,-5,5),0).astype(np.float32)
    # Weighted resampling avoids invalid zeros becoming a legitimate surface.
    den=cv2.resize(good.astype(np.float32),(32,32),interpolation=cv2.INTER_AREA)
    num=cv2.resize(z,(32,32),interpolation=cv2.INTER_AREA)
    out[1]=den>=.9;out[0]=np.where(out[1]>0,num/np.maximum(den,1e-6),0)
    return out


def relation_features(cxy,rxy,cz,rz,cv,rv,*,rpolys=None,carea=None,rarea=None,image_direction=None,radius=40.):
    cxy=np.asarray(cxy,np.float32).reshape(-1,2);rxy=np.asarray(rxy,np.float32).reshape(-1,2)
    d=cxy[:,None]-rxy[None];dist=np.linalg.norm(d,axis=-1);unit=d/np.maximum(dist[...,None],1e-6)
    out=np.zeros((*dist.shape,EDGE_DIM),np.float32)
    out[...,:2]=d/radius;out[...,2]=unit[...,1];out[...,3]=unit[...,0]
    out[...,4]=np.log1p(dist)/np.log1p(radius);out[...,5]=out[...,4]
    if rpolys is not None:
        for j,poly in enumerate(rpolys):
            p=np.asarray(poly,np.float32)
            for i,point in enumerate(cxy):
                v=cv2.pointPolygonTest(p,tuple(map(float,point)),True)
                out[i,j,5]=abs(v)/radius;out[i,j,6]=float(v>=0)
        out[...,13]=1.
    if carea is not None and rarea is not None:
        out[...,7]=np.log(np.maximum(np.asarray(carea)[:,None],1)/np.maximum(np.asarray(rarea)[None],1))/5
    known=np.asarray(cv,bool)[:,None]&np.asarray(rv,bool)[None]
    dz=np.asarray(cz,np.float32)[:,None]-np.asarray(rz,np.float32)[None]
    dz=np.where(known,dz,0);elev=np.arctan2(dz,dist)
    out[...,8]=dz/20;out[...,9]=np.where(known,np.sin(elev),0);out[...,10]=np.where(known,np.cos(elev),0)
    out[...,11]=dist>1;out[...,12]=known;out[...,14]=np.asarray(cv)[:,None];out[...,15]=np.asarray(rv)[None]
    out[...,16:18]=unit if image_direction is None else image_direction
    return out


def triad_features(cxy,rxy):
    c=np.asarray(cxy,np.float32).reshape(-1,2);r=np.asarray(rxy,np.float32).reshape(-1,2)
    pairs=np.array([(a,b) for a in range(len(r)) for b in range(a+1,len(r))],np.int64).reshape(-1,2)
    out=np.zeros((len(c),len(pairs),4),np.float32)
    for k,(a,b) in enumerate(pairs):
        ab=r[b]-r[a];length=float(np.linalg.norm(ab))
        if length<1:continue
        ac=c-r[a];t=(ac*ab).sum(-1)/length**2
        out[:,k,0]=t;out[:,k,1]=np.linalg.norm(ac-np.clip(t,0,1)[:,None]*ab,axis=-1)/length
        out[:,k,2]=(np.linalg.norm(c-r[a],axis=-1)+np.linalg.norm(c-r[b],axis=-1))/length
        out[:,k,3]=1
    return out,pairs


def choose_candidates(xy,base_scores,semantic_scores,limit):
    if not len(xy):return []
    xy=np.asarray(xy);routes=[list(np.argsort(-np.asarray(base_scores),kind='stable')),list(np.argsort(-np.asarray(semantic_scores),kind='stable'))]
    selected=[]
    while len(selected)<min(limit,len(xy)):
        before=len(selected)
        for route in routes:
            while route and route[0] in selected:route.pop(0)
            if route and len(selected)<limit:selected.append(route.pop(0))
        remaining=[i for i in range(len(xy)) if i not in selected]
        if remaining and len(selected)<limit:
            distance=np.linalg.norm(xy[remaining,None]-xy[selected][None],axis=-1).min(-1)
            selected.append(remaining[int(distance.argmax())])
        if len(selected)==before:break
    return selected


def local_peaks(grid,logits,count=3,radius=40.):
    grid=np.asarray(grid);order=np.argsort(-np.asarray(logits),kind='stable');peaks=[]
    unique=np.unique(grid[:,0]);step=float(np.median(np.diff(unique))) if len(unique)>1 else radius
    for i in order:
        if not peaks or np.linalg.norm(grid[i]-grid[peaks],axis=-1).min()>=step*1.4:
            peaks.append(int(i))
        if len(peaks)==count:break
    return grid[peaks],np.linalg.norm(grid[:,None]-grid[peaks][None],axis=-1).min(-1)<=radius


def grid_box_support(grid,polys,observed,local,padding=2.):
    masks=np.zeros((len(polys),len(grid)),bool)
    available=np.asarray(observed,bool)&np.asarray(local,bool)
    for i,poly in enumerate(polys):
        p=np.asarray(poly,np.float32)
        masks[i]=np.array([cv2.pointPolygonTest(p,tuple(map(float,xy)),True)>=-padding for xy in grid])&available
        if not masks[i].any() and available.any():
            ids=np.flatnonzero(available);j=ids[np.linalg.norm(grid[ids]-centroid(p),axis=-1).argmin()]
            # At most one grid diagonal of quantization, not an arbitrary distant cell.
            unique=np.unique(np.asarray(grid)[:,0]);step=float(np.median(np.diff(unique))) if len(unique)>1 else 1.
            if np.linalg.norm(grid[j]-centroid(p))<=max(step*1.5,8.):masks[i,j]=True
    return masks
