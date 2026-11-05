"""Frozen offline Polygon/RGB scores; no target GT and no navigation imports."""
import cv2
import numpy as np
from shapely.geometry import Polygon
from graphnav.local_landmark_anchor import local_landmark_anchor, local_anchor_tokens

CONFIG = dict(min_positive_patches=2,min_negative_patches=2,min_visible_pixels=256,
              local_area_ratio=[.5,2.],visible_fraction_difference=.2,max_real_negatives=3,
              pixel_shifts=[16,32],rotation_degrees=[-30,30],scale_factors=[.75,1.25],
              explicit_weights=[.5,.3,.2],appearance_fusion=[.8,.2],boundary_tolerance_px=3,
              reference_min_translation_m=1.,reference_min_yaw_degrees=5.)


def candidate_inputs(contours, observation):
    anchor=local_landmark_anchor(contours,observation)
    tokens=local_anchor_tokens(contours,observation['footprint_world']) if not anchor['local_geometry'].is_empty else None
    reasons=[]
    if anchor['stable_valid_pixels']<CONFIG['min_visible_pixels']:reasons.append('insufficient_pixels')
    if anchor['positive_patches']<2:reasons.append('insufficient_positive_patches')
    if anchor['negative_patches']<2:reasons.append('insufficient_negative_patches')
    return dict(anchor=anchor,tokens=tokens,eligible=not reasons,reasons=reasons,
                geometry_score=float(anchor['whole_fraction']))


def comparable(base, other):
    if not base['eligible'] or not other['eligible']:return False
    a=base['anchor'];b=other['anchor'];ratio=b['local_geometry'].area/a['local_geometry'].area
    return .5<=ratio<=2 and abs(a['whole_fraction']-b['whole_fraction'])<=.2


def image_evidence(bgr, support):
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    scale=max(float(np.percentile(gradient[support],95)),1.) if support.any() else 1.
    gradient=np.clip(gradient/scale,0,1)
    return dict(gradient=gradient,lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(float),support=support)


def explicit_score(mask, evidence):
    """Boundary edge support, two-sided Lab contrast and boundary/background gain.

    Tolerance is fixed at 3px. Artificial camera clipping edges are excluded.
    This evaluates alignment, not which proper-name instance is requested.
    """
    support=evidence['support'];mask=mask.astype(np.uint8)
    inner=cv2.erode(mask,np.ones((5,5),np.uint8)).astype(bool)&support
    outer=(cv2.dilate(mask,np.ones((9,9),np.uint8)).astype(bool)&~mask.astype(bool))&support
    boundary=(cv2.dilate(mask,np.ones((3,3),np.uint8))!=cv2.erode(mask,np.ones((3,3),np.uint8)))&support
    boundary[:4]=False;boundary[-4:]=False;boundary[:,:4]=False;boundary[:,-4:]=False
    if not boundary.any() or not inner.any() or not outer.any():return None
    gradient=evidence['gradient'];tolerant=cv2.dilate(gradient,np.ones((7,7),np.uint8))
    edge=float(tolerant[boundary].mean())
    contrast=float(np.clip(np.linalg.norm(evidence['lab'][inner].mean(0)-evidence['lab'][outer].mean(0))/100.,0,1))
    gain=float(np.clip(.5+.5*(gradient[boundary].mean()-gradient[outer].mean()),0,1))
    return dict(score=.5*edge+.3*contrast+.2*gain,edge=edge,color_contrast=contrast,boundary_background_gain=gain)


def score_contrast(scores, labels, valid):
    pos=scores[(labels==1)&valid];neg=scores[(labels==0)&valid]
    if len(pos)<2 or len(neg)<2:return None
    return float(pos.mean()-neg.mean())


def rank_credit(scores, positive=0):
    """Ties get fractional Top1 and midrank, never privilege A being first."""
    values=np.asarray(scores,float);target=values[positive]
    tied=np.isclose(values,target,atol=1e-8,rtol=0)
    rank=1+int((values>target+1e-8).sum())+.5*(tied.sum()-1)
    top1=float(1/tied.sum()) if not (values>target+1e-8).any() else 0.
    return dict(top1=top1,rank=float(rank),margin=float(target-np.max(np.delete(values,positive))))
