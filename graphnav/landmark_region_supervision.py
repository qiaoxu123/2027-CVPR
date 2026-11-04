"""Conservative public-map region proxies; never true pixel visibility labels."""
import numpy as np
import shapely
from shapely.geometry import Polygon


def named_union(contours):
    polygons=[]
    for c in contours:
        poly=Polygon(c)
        if not poly.is_valid or poly.area<=0:raise ValueError('invalid named polygon')
        polygons.append(poly)
    if not polygons:raise ValueError('no named polygons')
    return shapely.union_all(polygons)


def weak_region_labels(windows,region,valid,*,margin=.5):
    windows=np.asarray(windows,float);valid=np.asarray(valid,bool)
    poly=shapely.polygons(windows);area=shapely.area(poly)
    supported=valid&shapely.is_valid(poly)&(area>1e-6)
    inner=region.buffer(-margin);outer=region.buffer(margin)
    coverage=np.zeros((len(windows),2),np.float32)
    for i,reg in enumerate((inner,outer)):
        if not reg.is_empty:
            coverage[supported,i]=shapely.area(shapely.intersection(poly[supported],reg))/area[supported]
    label=np.full(len(windows),-1,np.int8)
    label[supported&(coverage[:,1]<=.01)]=0
    label[supported&(coverage[:,0]>=.9)]=1
    return label,coverage


def public_type(types):
    types=set(types)
    if types and types<={'TrafficRoad','Footpath','Bridge'}:return 'Road'
    if len(types)==1 and next(iter(types)) in {'Building','Ground','Parking'}:return next(iter(types))
    return None
