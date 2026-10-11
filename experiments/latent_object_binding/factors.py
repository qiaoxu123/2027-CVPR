"""Symmetric true-polygon factors; no target identity/coordinates as inputs."""
import numpy as np
import shapely
from shapely.geometry import Point


def between_region(a,b):
    if a.equals(b):return None
    hull=shapely.convex_hull(shapely.union(a,b));corridor=shapely.difference(hull,shapely.union(a,b))
    return corridor if not corridor.is_empty else None


def between_points(xy,a,b,scale=15.):
    region=between_region(a,b)
    if region is None:return np.zeros(len(xy))
    points=shapely.points(xy);distance=shapely.distance(points,region)
    out=np.exp(-.5*np.square(distance/scale))
    # Inside an anchor is not between anchors. Boundaries remain legal.
    interior=shapely.contains(a,points)|shapely.contains(b,points);out[interior]=0
    return out


def unary_features(e,g,prior):
    n=len(g.ids);types=np.array(g.types);typ=e['object_type'];same=types==typ if typ else np.ones(n,bool)
    names=np.zeros(n);names[list(e['named_ids'])]=1
    area=np.log1p(g.areas);den=np.std(area[same]) if same.any() else 1;area=(area-(np.mean(area[same]) if same.any() else np.mean(area)))/max(den,1.)
    aspect=np.log(np.maximum(g.aspects,1));small=-np.clip(area,-3,3) if e['size'] in ('small','tiny') else np.zeros(n)
    large=np.clip(area,-3,3) if e['size'] in ('large','big','huge') else np.zeros(n)
    shape=-np.abs(aspect) if e['shape']=='square' else aspect if e['shape'] in ('long','narrow') else np.zeros(n)
    # Normalized positions and dimensions are public geometry; no ID feature.
    prior=np.log(np.maximum(prior,1e-30));prior-=np.max(prior)
    return np.stack([prior,np.where(same,0.,-1.) if typ else np.zeros(n),names,small,large,shape,np.zeros(n),np.zeros(n)],axis=-1).astype(np.float32)


class Geometry:
    def __init__(self,graph,world):
        self.g=graph;self.world=np.asarray(world);self.positions=np.array([o['position'][:2] for o in graph.objects],float)
        self.points=np.concatenate([self.positions,self.world]);self.near_cache={};self.between_cache={}
        d=np.square(self.positions[:,None]-self.world[None]).sum(-1);k=np.exp(-.5*d/100);self.kernels=(k/np.maximum(k.sum(1,keepdims=True),1e-300)).astype(np.float32)
    def field(self,kind,anchors):
        if kind=='between':
            if len(anchors)!=2 or anchors[0]==anchors[1]:return np.zeros(len(self.points),np.float32)
            key=tuple(sorted(anchors))
            if key not in self.between_cache:
                self.between_cache[key]=between_points(self.points,self.g.shapes[key[0]],self.g.shapes[key[1]]).astype(np.float32)
            return self.between_cache[key]
        i=anchors[0];key=(kind,i)
        if key not in self.near_cache:
            if kind in ('near','along'):
                distance=shapely.distance(shapely.points(self.points),self.g.shapes[i]);value=np.exp(-distance/(40 if kind=='along' else 25))
            elif kind in ('east','west','north','south'):
                # Compare against actual polygon bounding edge, not centroid.
                bounds=self.g.shapes[i].bounds;axis=0 if kind in ('east','west') else 1;sign=1 if kind in ('east','north') else -1;edge=bounds[axis+2] if sign>0 else bounds[axis];delta=sign*(self.points[:,axis]-edge);value=1/(1+np.exp(np.clip(-delta/10,-60,60)))
            else:return np.ones(len(self.points),np.float32)  # relative frame Unknown
            self.near_cache[key]=np.asarray(value,np.float32)
        return self.near_cache[key]
    def object_relation(self,kind,subject,anchors):
        if kind in ('near','along'):
            # Real boundary-to-boundary distance handles long roads/irregulars.
            distance=float(shapely.distance(self.g.shapes[subject],self.g.shapes[anchors[0]]));return np.exp(-distance/(40 if kind=='along' else 25))
        return float(self.field(kind,anchors)[subject])
