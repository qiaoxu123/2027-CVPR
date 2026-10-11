"""Observed-only polygon node attributes: measurable colors and frozen vectors."""
from dataclasses import dataclass
import cv2
import numpy as np
from graphnav.landmark_projection import project_landmark


COLORS=('black','white','gray','red','orange','yellow','green','blue','brown','beige')


def color_measurements(bgr,mask):
    mask=np.asarray(mask,bool)
    if mask.shape!=bgr.shape[:2] or mask.sum()<64:return None
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)[mask].astype(float)
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)[mask].astype(float)
    h,s,v=hsv.T;chroma=np.hypot(lab[:,1]-128,lab[:,2]-128)
    # Fixed interpretable rules, NOT human roof/semantic ground truth.
    flags={'black':v<60,'white':(v>190)&(s<45),'gray':(v>=60)&(v<=190)&(s<45),
           'red':((h<10)|(h>170))&(s>=70)&(v>=60),'orange':(h>=10)&(h<23)&(s>=70),
           'yellow':(h>=23)&(h<36)&(s>=60),'green':(h>=36)&(h<90)&(s>=60),
           'blue':(h>=90)&(h<140)&(s>=60),'brown':(h>=5)&(h<25)&(s>=50)&(v<170),
           'beige':(h>=5)&(h<35)&(s>=20)&(s<80)&(v>=170)}
    return dict(fractions={k:float(a.mean()) for k,a in flags.items()},
                median_lab=np.median(lab,axis=0).tolist(),median_hsv=np.median(hsv,axis=0).tolist(),
                chroma=float(np.median(chroma)),support=int(mask.sum()),truth='measured pixels, semantic scope unverified')


class CausalAttributes:
    def __init__(self):self.nodes={};self.last_step=-1
    def observe(self,observation,graph,step,cutoff,*,control='real',permutation=None):
        if step>cutoff or step<=self.last_step:raise ValueError('future/out-of-order node update')
        if control not in ('real','black','shuffled'):raise ValueError('RGB control')
        self.last_step=step;bgr=np.asarray(observation['bgr']);valid=np.asarray(observation['depth_valid'],bool)
        if bgr.shape[:2]!=valid.shape:raise ValueError('aligned RGB/depth required')
        rows=[]
        for i,o in enumerate(graph.objects):
            if not graph.valid[i]:continue
            roi=project_landmark(o,observation['footprint_world'],bgr.shape,min_visible=.9,min_pixels=256)
            if roi is None:continue
            mask=roi.mask.astype(bool);depth=float(valid[mask].mean());rgb=float((bgr.max(-1)>0)[mask].mean())
            if min(roi.bbox_xyxy[2]-roi.bbox_xyxy[0],roi.bbox_xyxy[3]-roi.bbox_xyxy[1])<16 or depth<.9 or rgb<.9:continue
            foreground=cv2.erode(roi.mask,np.ones((5,5),np.uint8)).astype(bool)
            pixels=bgr.copy()
            if control=='black':pixels[:]=0
            # Shuffle crops only after eligibility is decided from original RGB.
            box=roi.bbox_xyxy;rows.append((i,roi,foreground,pixels[box[1]:box[3],box[0]:box[2]].copy(),pixels))
        crops=[r[3] for r in rows]
        if control=='shuffled':
            if permutation is None:permutation=np.roll(np.arange(len(rows)),1)
            if sorted(map(int,permutation))!=list(range(len(rows))):raise ValueError('same candidate shuffle required')
            crops=[crops[int(j)] for j in permutation]
        for k,(i,roi,foreground,crop,pixels) in enumerate(rows):
            a,b,c,d=roi.bbox_xyxy
            if control=='shuffled':pixels=pixels.copy();pixels[b:d,a:c]=cv2.resize(crops[k],(c-a,d-b))
            measured=color_measurements(pixels,roi.mask);clean=color_measurements(pixels,foreground)
            quality=float(foreground.sum()/max(roi.visible_pixels,1))*float(valid[roi.mask.astype(bool)].mean())
            old=self.nodes.get(i)
            if old is None or quality>old['quality']:
                self.nodes[i]=dict(step=step,quality=quality,polygon=measured,foreground=clean,
                                   weak_segmentation=True,observed=True,control=control)
        return self.nodes

    def evidence(self,entity,count):
        score=np.zeros(count,float)
        if len(entity.get('colors',[]))!=1:return score
        color=entity['colors'][0]
        if color=='grey':color='gray'
        for i,r in self.nodes.items():
            m=r['foreground']
            if m is None or r['quality']<.5 or color not in m['fractions']:continue
            f=m['fractions'][color]
            # Low purity remains Unknown; negative color truth is not invented.
            if f>=.7:score[i]=r['quality']*f
        return score


def frozen_phrase_evidence(images,phrase_vector,observed,quality):
    """Cached same-checkpoint SigLIP2/RemoteCLIP vectors; no model downloads."""
    a=np.asarray(images,float);t=np.asarray(phrase_vector,float)
    if a.ndim!=2 or t.shape!=(a.shape[1],):raise ValueError('same image/text embedding dimension')
    score=(a@t)/(np.linalg.norm(a,axis=1)*max(np.linalg.norm(t),1e-12)).clip(1e-12)
    use=np.asarray(observed,bool)&(np.asarray(quality)>=.9)
    # Similarity is evidence, not semantic certainty; caller must Dev-calibrate acceptance.
    return np.where(use,score,0.),use
